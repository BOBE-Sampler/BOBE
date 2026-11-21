# This module manages the samplers used to run HMC/Nested sampling using the GP model as a surrogate for the objective function.
# It contains two functions: one for the Dynesty nested sampler and one for the NumPyro NUTS HMC sampler.
import math
import time
from typing import Any, Dict, Optional
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
from dynesty import NestedSampler as StaticNestedSampler

from .surrogate import Surrogate, ClassifierMixin
from .utils.log import get_logger
from .utils.seed import get_new_jax_key, get_numpy_rng
from .utils.core import (
    get_R,
    is_cluster_environment,
    renormalise_log_weights,
    resample_equal,
    scale_from_unit,
)

log = get_logger("sampler")


# Dynesty utility function for computing evidence integrals
def compute_integrals(logl=None, logvol=None, reweight=None, squared=False):
    assert logl is not None
    assert logvol is not None
    loglstar_pad = np.concatenate([[-1.e300], logl])
    # log(exp(LV_i) - exp(LV_{i+1})) = LV_{i+1} - (LV_{i+1} - LV_i) + log(1 - exp(LV_{i+1} - LV_i))
    dlogvol = np.diff(logvol, prepend=0)
    logdvol = logvol - dlogvol + np.log1p(-np.exp(dlogvol))
    if squared:
        logdvol = 2 * logdvol
    logdvol2 = logdvol + math.log(0.5)
    saved_logwt = np.logaddexp(loglstar_pad[1:], loglstar_pad[:-1]) + logdvol2
    if reweight is not None:
        saved_logwt = saved_logwt + reweight
    saved_logz = np.logaddexp.accumulate(saved_logwt)
    return saved_logz


def prior_transform(u):
    return np.asarray(u, dtype=np.float64)


def nested_sampling_Dy(
    gp: Surrogate,
    mode: str = 'acq',
    ndim: int = 1,
    dlogz: float = 0.1,
    dynamic: bool = False,
    maxcall: Optional[int] = int(5e6),
    print_progress: Optional[bool] = True,
    sample_method: str = 'rwalk',
    rng=None,
    param_bounds=None,
    logz_err_method: str = 'mean_std',
) -> tuple[Dict[str, Any], Dict[str, Any], bool]:
    """
    Nested Sampling using Dynesty in the GP unit cube ``[0, 1]^ndim``.

    Parameters
    ----------
    gp : Surrogate
        Gaussian Process (or ClassifierGP) surrogate model.
    mode : str, optional
        ``'acq'`` uses lower-precision settings for acquisition point generation;
        any other value (e.g. ``'conv'`` or ``'convergence'``) uses convergence
        settings. Default is ``'acq'``.
    ndim : int, optional
        GP dimensionality (used for ``nlive`` / ``n_init`` scaling).
    dlogz : float, optional
        Log-evidence stopping tolerance for Dynesty.
    dynamic : bool, optional
        Kept for API compatibility (static nested sampling is used).
    maxcall : int, optional
        Maximum log-likelihood calls for Dynesty.
    print_progress : bool, optional
        Print Dynesty progress bar. Auto-disabled in cluster/non-TTY environments.
    sample_method : str, optional
        Dynesty sampling method (default ``'rwalk'``).
    rng : np.random.Generator, optional
        Random number generator. Defaults to global BOBE NumPy RNG.
    param_bounds : array-like, shape (2, ndim), optional
        Physical parameter bounds. When provided, ``samples_dict['x']`` and
        ``samples_dict['eq_x']`` are scaled to physical coordinates while
        ``'u'`` and ``'eq_u'`` remain in the unit cube. When ``None``, ``'x'``
        and ``'eq_x'`` match the unit-cube coordinates.
    logz_err_method : {'mean_std', 'integral'}, optional
        How to estimate GP-induced uncertainty on ``logZ``:
        - ``'mean_std'`` (default): Mean GP predictive std over up to 512
          equal-weighted posterior samples, setting ``upper = mean + std``,
          ``lower = mean - std``, ``var = std**2``.
        - ``'integral'``: Integrate ``logl ± std`` and the second-order variance
          integrand across all NS samples via :func:`compute_integrals`.

    Returns
    -------
    samples_dict : dict
        ``'x'``, ``'u'``, ``'weights'``, ``'logl'`` — weighted samples (physical and unit cube).
        ``'eq_x'``, ``'eq_u'``, ``'eq_logl'`` — equal-weighted resampled samples.
        ``'best'`` — best unit-cube point by ``logl``.
        ``'method'`` — ``'nested'``.
    logz_dict : dict
        ``'mean'``, ``'dlogz_sampler'``, ``'upper'``, ``'lower'``, ``'var'``, ``'std'``.
    success : bool
        ``False`` when all ``logl`` values are identical (degenerate run).
    """
    if logz_err_method not in ('mean_std', 'integral'):
        raise ValueError(
            f"logz_err_method must be 'mean_std' or 'integral', got {logz_err_method!r}"
        )

    log.info("Running Nested Sampling using Dynesty...")

    if is_cluster_environment():
        print_progress = False

    @jax.jit
    def loglike(u):
        return gp.predict_mean(u)

    start = time.time()

    if mode == 'acq':
        nlive = max(100, min(500, 20 * ndim))
        dlogz = 0.1
        maxcall = int(2e6)
    else:
        nlive = max(500, 40 * ndim)

    rng = rng if rng is not None else get_numpy_rng()

    # Initial live points in GP unit cube
    if isinstance(gp, ClassifierMixin):
        n_init = 5000 * ndim
        init_u = rng.uniform(0.0, 1.0, size=(n_init, ndim))
        init_logl = np.array(
            jax.lax.map(loglike, jnp.array(init_u, dtype=jnp.float64), batch_size=200)
        )
        live_idx = rng.choice(n_init, size=nlive, replace=False)
        found_diverse = False
        for i in range(1000):
            sel_logl = init_logl[live_idx]
            if not np.all(sel_logl == sel_logl[0]):
                log.debug(f" Successful live points on try {i + 1}/1000.")
                found_diverse = True
                break
            live_idx = rng.choice(n_init, size=nlive, replace=False)
        if not found_diverse:
            log.info(" Could not find diverse live points; injecting GP fallback point.")
            valid_unit = gp.get_random_point(rng=rng, nstd=1.0)
            init_u[live_idx[0]] = valid_unit
            init_logl[live_idx[0]] = float(loglike(jnp.array(valid_unit, dtype=jnp.float64)))
        live_pts = init_u[live_idx]
        live_logl = init_logl[live_idx]
    else:
        live_pts = rng.uniform(0.0, 1.0, size=(nlive, ndim))
        live_logl = np.array(
            jax.lax.map(loglike, jnp.array(live_pts, dtype=jnp.float64), batch_size=200)
        )

    sampler = StaticNestedSampler(
        loglike,
        prior_transform,
        ndim=ndim,
        blob=False,
        live_points=[live_pts, live_pts, live_logl],
        sample=sample_method,
        nlive=nlive,
        rstate=rng,
    )
    sampler.run_nested(print_progress=print_progress, dlogz=dlogz, maxcall=maxcall)

    res = sampler.results
    mean = float(res['logz'][-1])
    logz_err = float(res['logzerr'][-1])
    logl = np.asarray(res['logl'])
    success = bool(~np.all(logl == logl[0]))
    log.debug(f" Nested Sampling took {time.time() - start:.2f}s")
    log.debug(" Log Z evaluated using {} points".format(np.shape(logl)))
    log.debug(f" Dynesty made {np.sum(res['ncall'])} function calls, max value of logl = {np.max(logl):.4f}")

    u_out = np.asarray(res['samples'])
    weights_all = renormalise_log_weights(res['logwt'])
    eq_unit, eq_logl = resample_equal(u_out, logl, weights=weights_all)
    eq_unit = np.asarray(eq_unit)
    eq_logl = np.asarray(eq_logl)
    best_pt = eq_unit[np.argmax(eq_logl)]

    if logz_err_method == 'mean_std':
        n_std = min(512, len(eq_unit))
        gp_var = np.asarray(
            jax.lax.map(gp.predict_var, jnp.array(eq_unit[:n_std], dtype=jnp.float64), batch_size=100)
        )
        mean_gp_std = float(np.mean(np.sqrt(np.clip(gp_var, 1e-12, None))))
        logz_dict = {
            'mean': mean,
            'dlogz_sampler': logz_err,
            'upper': mean + mean_gp_std,
            'lower': mean - mean_gp_std,
            'var': mean_gp_std ** 2,
            'std': mean_gp_std,
        }
    else:
        var = np.asarray(
            jax.lax.map(gp.predict_var, jnp.array(u_out, dtype=jnp.float64), batch_size=100)
        )
        std = np.sqrt(np.clip(var, 1e-12, None))
        logl_lower, logl_upper = logl - std, logl + std
        logvol = res['logvol']
        upper = compute_integrals(logl=logl_upper, logvol=logvol)
        lower = compute_integrals(logl=logl_lower, logvol=logvol)

        var_clipped = np.clip(var, a_min=1e-12, a_max=1e12)
        varintegrand = 2 * logl + np.log(var_clipped)
        log_var_delta = compute_integrals(logl=varintegrand, logvol=logvol, squared=True)[-1]
        log_var_logz = np.clip(log_var_delta - 2 * mean, a_min=-100, a_max=100)
        var_logz = float(np.exp(log_var_logz))
        logz_dict = {
            'mean': mean,
            'dlogz_sampler': logz_err,
            'upper': float(upper[-1]),
            'lower': float(lower[-1]),
            'var': var_logz,
            'std': float(2.0 * np.sqrt(var_logz)),
        }

    if param_bounds is not None:
        x_out = np.asarray(scale_from_unit(u_out, param_bounds))
        eq_x_out = np.asarray(scale_from_unit(eq_unit, param_bounds))
    else:
        x_out = u_out
        eq_x_out = eq_unit

    samples_dict = {
        'x': x_out,
        'u': u_out,
        'weights': weights_all,
        'logl': logl,
        'eq_u': eq_unit,
        'eq_x': eq_x_out,
        'eq_logl': eq_logl,
        'best': best_pt,
        'method': 'nested',
    }

    return (samples_dict, logz_dict, success)


def sample_GP_NUTS(
    gp: Surrogate,
    np_rng=None,
    rng_key=None,
    num_chains: int = 4,
    temp: float = 1.0,
    flat: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Obtain samples from the posterior represented by the GP mean as the logprob.
    Works for both ``GP`` and ``ClassifierGP`` surrogates.

    Parameters
    ----------
    gp : Surrogate
        The Gaussian Process model to sample from.
    np_rng : np.random.Generator, optional
        NumPy random number generator.
    rng_key : jax.random.PRNGKey, optional
        JAX random key. Defaults to a fresh key from :func:`get_new_jax_key`.
    num_chains : int, optional
        Number of parallel HMC chains. Default is 4.
    temp : float, optional
        Temperature parameter for tempering. Default is 1.0.
    flat : bool, optional
        If ``True`` (default), concatenate chains into shape
        ``(num_chains * n_samples_per_chain, ndim)``.
        If ``False``, preserve chain axis with shape
        ``(num_chains, n_samples_per_chain, ndim)``.
    **kwargs : dict
        Optional NUTS settings: ``warmup_steps`` (512), ``num_samples`` (1024),
        ``thinning`` (4), ``dense_mass`` (True), ``max_tree_depth`` (6).

    Returns
    -------
    samples_dict : dict
        Dictionary containing:
        - ``'x'``: unit-cube samples (flat or per-chain depending on ``flat``)
        - ``'logp'``: GP mean log probabilities for each sample
        - ``'best'``: best unit-cube sample found
        - ``'R'``: Gelman-Rubin R-hat diagnostic across chains (``np.nan`` if ``num_chains < 2``)
        - ``'method'``: ``'MCMC'``
    """
    warmup_steps = kwargs.get('warmup_steps', 512)
    num_samples = kwargs.get('num_samples', 1024)
    thinning = kwargs.get('thinning', 4)
    dense_mass = kwargs.get('dense_mass', True)
    max_tree_depth = kwargs.get('max_tree_depth', 6)

    shape = gp.train_x.shape[1]

    def model():
        x = numpyro.sample('x', dist.Uniform(
            low=jnp.zeros(shape),
            high=jnp.ones(shape),
        ))
        mean = gp.predict_mean(x)
        numpyro.factor('y', mean / temp)
        numpyro.deterministic('logp', mean)

    @jax.jit
    def run_single_chain(key, init_x):
        init_strategy = init_to_value(values={'x': init_x})
        kernel = NUTS(
            model,
            dense_mass=dense_mass,
            max_tree_depth=max_tree_depth,
            init_strategy=init_strategy,
        )
        mcmc = MCMC(
            kernel,
            num_warmup=warmup_steps,
            num_samples=num_samples,
            num_chains=1,
            progress_bar=False,
            thinning=thinning,
        )
        mcmc.run(key)
        samples = mcmc.get_samples()
        return samples['x'], samples['logp']

    num_devices = jax.device_count()

    rng_key = rng_key if rng_key is not None else get_new_jax_key()
    rng_keys = jax.random.split(rng_key, num_chains)

    if num_chains == 1:
        inits = jnp.array([gp.get_random_point(rng=np_rng)])
    else:
        inits = jnp.vstack([gp.get_random_point(rng=np_rng) for _ in range(num_chains - 1)])
        inits = jnp.vstack([inits, gp.train_x[jnp.argmax(gp.train_y)]])

    log.debug(f"Running MCMC with {num_chains} chains on {num_devices} devices.")

    if num_devices == 1 or num_chains == 1:
        log.debug("Using sequential method (single device or single chain)")
        chain_x_list = []
        chain_logp_list = []
        for i in range(num_chains):
            sx_i, lp_i = run_single_chain(rng_keys[i], inits[i])
            chain_x_list.append(sx_i)
            chain_logp_list.append(lp_i)
        chains_x = jnp.stack(chain_x_list, axis=0)
        chains_logp = jnp.stack(chain_logp_list, axis=0)
    elif num_devices >= num_chains:
        log.debug("Using direct pmap method (devices >= chains)")
        pmapped = jax.pmap(run_single_chain, in_axes=(0, 0), out_axes=(0, 0))
        chains_x, chains_logp = pmapped(rng_keys, inits)
    else:
        log.debug(f"Using chunked pmap method ({num_devices} devices < {num_chains} chains)")
        pmapped_chunked = jax.pmap(run_single_chain, in_axes=(0, 0), out_axes=(0, 0))
        all_samples = []
        all_logps = []
        for i in range(0, num_chains, num_devices):
            end_idx = min(i + num_devices, num_chains)
            chunk_samples, chunk_logps = pmapped_chunked(rng_keys[i:end_idx], inits[i:end_idx])
            all_samples.append(chunk_samples)
            all_logps.append(chunk_logps)
        chains_x = jnp.concatenate(all_samples, axis=0)
        chains_logp = jnp.concatenate(all_logps, axis=0)

    # Measure Gelman-Rubin R-hat across chains before optional flattening
    r_hat = get_R(np.asarray(chains_x))

    flat_x = jnp.reshape(chains_x, (-1, shape))
    flat_logp = jnp.reshape(chains_logp, (-1,))
    best_pt = flat_x[jnp.argmax(flat_logp)]

    if flat:
        samples_x = flat_x
        logps = flat_logp
    else:
        samples_x = chains_x
        logps = chains_logp

    samples_dict = {
        'x': samples_x,
        'logp': logps,
        'best': best_pt,
        'R': r_hat,
        'method': 'MCMC',
    }

    log.info(f"NUTS sampling: max logl = {float(jnp.max(flat_logp)):.4f}, R = {r_hat:.4f}")

    return samples_dict
