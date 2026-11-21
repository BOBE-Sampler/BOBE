# This module manages the samplers used to run HMC/Nested sampling using the GP model as a surrogate for the objective function
# It contains two functions, one for the Dynesty nested sampler and the other for the HMC sampler using NUTS from numpyro
import time
from typing import Any, List, Optional, Dict, Union
import jax.numpy as jnp
import jax.random as random
import numpy as np
import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
from .surrogate import Surrogate, ClassifierMixin
from .utils.log import get_logger
from .utils.seed import get_new_jax_key, get_numpy_rng
from .utils.core import is_cluster_environment, renormalise_log_weights, resample_equal
log = get_logger("sampler")

from dynesty import NestedSampler as StaticNestedSampler, DynamicNestedSampler
import math

# dynesty utility function for computing evidence
def compute_integrals(logl=None, logvol=None, reweight=None,squared=False):
    assert logl is not None
    assert logvol is not None
    loglstar_pad = np.concatenate([[-1.e300], logl])
    # we want log(exp(logvol_i)-exp(logvol_(i+1)))
    # assuming that logvol0 = 0
    # log(exp(LV_{i})-exp(LV_{i+1})) =
    # = LV{i} + log(1-exp(LV_{i+1}-LV{i}))
    # = LV_{i+1} - (LV_{i+1} -LV_i) + log(1-exp(LV_{i+1}-LV{i}))
    dlogvol = np.diff(logvol, prepend=0)
    logdvol = logvol - dlogvol + np.log1p(-np.exp(dlogvol))
    if squared:
        logdvol = 2 * logdvol
    # logdvol is log(delta(volumes)) i.e. log (X_i-X_{i-1})
    logdvol2 = logdvol + math.log(0.5)
    # These are log(1/2(X_(i+1)-X_i))
    dlogvol = -np.diff(logvol, prepend=0)
    # this are delta(log(volumes)) of the run
    # These are log((L_i+L_{i_1})*(X_i+1-X_i)/2)
    saved_logwt = np.logaddexp(loglstar_pad[1:], loglstar_pad[:-1]) + logdvol2
    if reweight is not None:
        saved_logwt = saved_logwt + reweight
    saved_logz = np.logaddexp.accumulate(saved_logwt)
    return saved_logz

def prior_transform(x):
    return x
        
def nested_sampling_Dy(gp: Surrogate,
                       mode: str = 'acq',
                       ndim: int = 1,
                       dlogz: float = 0.1,
                       dynamic: bool = False,
                       maxcall: Optional[int] = int(5e6),
                       print_progress: Optional[bool] = True,
                       equal_weights: bool = False,
                       sample_method: str = 'rwalk',
                       rng=None,
                       ) -> tuple[np.ndarray,Dict,bool]:
    """
    Nested Sampling using Dynesty

    Arguments
    ---------
    gp : saas_fbgp
        Gaussian Process model
    ndim : int
        Number of dimensions
    dlogz : float
        Log evidence goal
    dynamic : bool
        Use dynamic nested sampling, see Dynesty documentation for more details
    logz_std : bool
        Compute the upper and lower bounds on logZ using the GP uncertainty
    maxcall : int
        Maximum number of function calls
    boost_maxcall : int
        Boost the maximum number of function calls
    print_progress : bool, optional
        Print progress of the nested sampling run. If None, automatically disables 
        progress printing in cluster environments and enables it otherwise.
    equal_weights : bool
        Resample to obtain equal weights
    sample_method : str
        Sampling method for dynesty
    rng : random number generator
        Random number generator

    Returns
    -------
    samples : ndarray
        Equally weighted samples from the nested sampler
    logz_dict : dict
        Dictionary containing the mean, upper and lower bounds on logZ and the logZ error
        from the nested sampler
    success : bool
        Whether the nested sampling run was successful
    """

    log.info("Running Nested Sampling using Dynesty...")

    # Auto-detect cluster environment if print_progress not explicitly set
    if is_cluster_environment():
        print_progress = False

    @jax.jit
    def loglike(x):
        mu = gp.predict_mean(x)
        return mu

    start = time.time()

    if mode == 'acq': # a bit lower precision settings for acquisition
        nlive = max(100, min(500, 20 * ndim))
        dlogz = 0.1
        maxcall = int(2e6)
        equal_weights = True
    else:
        nlive = max(500, 40 * ndim)

    rng = rng if rng is not None else get_numpy_rng()

    if isinstance(gp, ClassifierMixin):
        maxtries = 1000
        nlogl = 5000 * ndim
        x = rng.uniform(low=0., high=1., size=(nlogl, ndim))
        logl = jax.lax.map(loglike,x,batch_size=200)
        logl = np.array(logl)
        success = False
        for i in range(maxtries):
            live_indices = rng.choice(nlogl, size=nlive, replace=False)
            live_logl = logl[live_indices]
            if np.all(live_logl == live_logl[0]):
                log.debug(f" All logl values are the same on try {i+1}/{maxtries}. Retrying...")
            else:
                log.debug(f" Successful live points on try {i+1}/{maxtries}.")
                success = True
                break
        live_points = x[live_indices]
        live_logl = logl[live_indices]
        if not success:
            valid_point = gp.get_random_point(rng=rng,nstd=1.0)
            valid_logl = float(loglike(valid_point))
            live_points[0] = valid_point
            live_logl[0] = valid_logl
    else:
        live_points = rng.uniform(low=0., high=1., size=(nlive, ndim))
        live_logl = jax.lax.map(loglike,live_points,batch_size=200)
        live_logl = np.array(live_logl)

    sampler = StaticNestedSampler(loglike, prior_transform, ndim=ndim, blob=False
                                  ,live_points=[live_points,live_points,live_logl]
                                  ,sample=sample_method, nlive=nlive, rstate=rng)
    sampler.run_nested(print_progress=print_progress,dlogz=dlogz,maxcall=maxcall)
    
    res = sampler.results
    mean = res['logz'][-1]
    logz_err = res['logzerr'][-1]
    samples_x = res['samples']
    logl = res['logl']
    success = ~np.all(logl == logl[0]) # in case of failure do not check convergence
    log.debug(f" Nested Sampling took {time.time() - start:.2f}s")
    log.debug(" Log Z evaluated using {} points".format(np.shape(logl))) 
    log.debug(f" Dynesty made {np.sum(res['ncall'])} function calls, max value of logl = {np.max(logl):.4f}")

    var = jax.lax.map(gp.predict_var, samples_x, batch_size=100)
    std = np.sqrt(var)
    logl_lower,logl_upper = logl - std, logl + std
    logvol = res['logvol']
    upper = compute_integrals(logl=logl_upper,logvol=logvol)
    lower = compute_integrals(logl=logl_lower,logvol=logvol)

    var = np.clip(var,a_min=1e-12,a_max=1e12)
    varintegrand = 2*logl + np.log(var)
    log_var_delta = compute_integrals(logl=varintegrand,logvol=logvol,squared=True)[-1]
    log_var_logz = log_var_delta - 2*mean 
    log_var_logz = np.clip(log_var_logz, a_min=-100, a_max=100)  # Avoid numerical issues with very small or large variances
    var_logz = np.exp(log_var_logz)
    logz_dict = {'mean': mean, 'dlogz_sampler': logz_err, 'upper': upper[-1], 'lower': lower[-1], 'var': var_logz, 'std': 2*np.sqrt(var_logz)}
    best_pt = samples_x[np.argmax(logl)]
    weights = renormalise_log_weights(res['logwt'])
    if equal_weights: #for MC points
        samples_x, logl = resample_equal(samples_x, logl, weights=weights)
        weights = np.ones(samples_x.shape[0])  # Equal weights after resampling
    samples_dict = {'x': samples_x,'weights': weights,'logl': logl,'best': best_pt,'method': 'nested'}
    samples_dict['x'] = samples_x
    samples_dict['weights'] = weights    
    return (samples_dict, logz_dict, success)

def sample_GP_NUTS(gp: Surrogate, 
                   np_rng=None, 
                   rng_key=None, 
                   num_chains=4, 
                   temp=1., 
                   **kwargs):
    """
    Obtain samples from the posterior represented by the GP mean as the logprob.
    This is a unified function that works for both GP and ClassifierGP.
    
    Parameters
    ----------
    gp : Union[GP, ClassifierGP]
        The Gaussian Process model to sample from.
    np_rng : np.random.Generator, optional
        NumPy random number generator. Default is None.
    rng_key : jax.random.PRNGKey, optional
        JAX random key. Default is None.
    num_chains : int, optional
        Number of parallel HMC chains. Default is 4.
    temp : float, optional
        Temperature parameter for tempering. Default is 1.0.
    **kwargs : dict
        Additional keyword arguments. Can include:
        - warmup_steps : int, optional
            Number of warmup steps for HMC. If not provided, defaults based on dimensionality.
        - num_samples : int, optional
            Number of samples to draw from each chain. If not provided, defaults based on dimensionality.
        - thinning : int, optional
            Thinning factor for samples. If not provided, defaults to 4.
        - dense_mass : bool, optional
            Whether to use dense mass matrix in NUTS. Default is True.
        - max_tree_depth : int, optional
            Maximum tree depth for NUTS. Default is 6.
            
    Returns
    -------
    samples_dict : dict
        Dictionary containing:
        - 'x': samples array of shape (num_chains * num_samples / thinning, ndim)
        - 'logp': log probabilities for each sample
        - 'best': best sample found
        - 'method': 'MCMC'
    """
    # Extract HMC settings from kwargs with simple fallback defaults
    # Note: Dimension-based defaults are now handled centrally in bo.py
    warmup_steps = kwargs.get('warmup_steps', 512)
    num_samples = kwargs.get('num_samples', 1024)
    thinning = kwargs.get('thinning', 4)
    dense_mass = kwargs.get('dense_mass', True)
    max_tree_depth = kwargs.get('max_tree_depth', 6)
    

    shape = gp.train_x.shape[1]
    
    def model():
        x = numpyro.sample('x', dist.Uniform(
            low=jnp.zeros(shape),
            high=jnp.ones(shape)
        ))
        
        mean = gp.predict_mean(x)
        numpyro.factor('y', mean/temp)
        numpyro.deterministic('logp', mean)
    
    @jax.jit
    def run_single_chain(rng_key,init_x):
        init_strategy = init_to_value(values={'x': init_x})
        kernel = NUTS(model, dense_mass=dense_mass, max_tree_depth=max_tree_depth, 
                        init_strategy=init_strategy)
        mcmc = MCMC(kernel, num_warmup=warmup_steps, num_samples=num_samples,
                    num_chains=1, progress_bar=False, thinning=thinning)
        mcmc.run(rng_key)
        samples_x = mcmc.get_samples()['x']
        logps = mcmc.get_samples()['logp']
        return samples_x, logps
    
    num_devices = jax.device_count()
    
    rng_key = rng_key if rng_key is not None else get_new_jax_key()
    rng_keys = jax.random.split(rng_key, num_chains)
    
    # Generate initialization points if needed
    if num_chains == 1: 
        inits = jnp.array([gp.get_random_point(rng=np_rng)])
    else:
        inits = jnp.vstack([gp.get_random_point(rng=np_rng) for _ in range(num_chains-1)])
        inits = jnp.vstack([inits, gp.train_x[jnp.argmax(gp.train_y)]])  # Add best training point as one init

    log.debug(f"Running MCMC with {num_chains} chains on {num_devices} devices.")

    # Adaptive method selection based on device/chain configuration
    if num_devices == 1:
        # Sequential method for single device
        log.debug("Using sequential method (single device)")
        samples_x = []
        logps = []
        for i in range(num_chains):
            samples_x_i, logps_i = run_single_chain(rng_keys[i], inits[i])
            samples_x.append(samples_x_i)
            logps.append(logps_i)
        samples_x = jnp.concatenate(samples_x)
        logps = jnp.concatenate(logps)
        
    elif num_devices >= num_chains and num_chains > 1:
        # Direct pmap method when devices >= chains
        log.debug("Using direct pmap method (devices >= chains)")
        pmapped = jax.pmap(run_single_chain, in_axes=(0, 0), out_axes=(0, 0))
        samples_x, logps = pmapped(rng_keys, inits)
        samples_x = jnp.concatenate(samples_x, axis=0)
        logps = jnp.concatenate(logps, axis=0)
        logps = jnp.reshape(logps, (samples_x.shape[0],))
        
    elif 1 < num_devices < num_chains:
        # Chunked method when devices < chains (but > 1 device)
        log.debug(f"Using chunked pmap method ({num_devices} devices < {num_chains} chains)")
        
        # Process chains in chunks of device count using the existing run_single_chain
        pmapped_chunked = jax.pmap(run_single_chain, in_axes=(0, 0), out_axes=(0, 0))
        
        all_samples = []
        all_logps = []
        
        for i in range(0, num_chains, num_devices):
            end_idx = min(i + num_devices, num_chains)
            chunk_keys = rng_keys[i:end_idx]
            chunk_inits = inits[i:end_idx]
            
            # Run chunk (pmap handles variable chunk sizes automatically)
            chunk_samples, chunk_logps = pmapped_chunked(chunk_keys, chunk_inits)
            
            all_samples.append(chunk_samples)
            all_logps.append(chunk_logps)
        
        # Concatenate all chunks
        samples_x = jnp.concatenate([jnp.concatenate(chunk, axis=0) for chunk in all_samples], axis=0)
        logps = jnp.concatenate([jnp.concatenate(chunk, axis=0) for chunk in all_logps], axis=0)

    samples_dict = {
        'x': samples_x,
        'logp': logps,
        'best': samples_x[jnp.argmax(logps)],
        'method': "MCMC"
    }

    log.debug(f"Max logl found in HMC = {np.max(logps):.4f}")

    return samples_dict