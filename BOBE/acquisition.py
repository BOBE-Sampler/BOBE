from typing import Any, List, Optional, Dict, Tuple
import jax
import jax.numpy as jnp
from jax import lax,jit
import numpy as np
from scipy.stats import qmc
from jax.scipy.stats import norm
from jax import config
import tensorflow_probability.substrates.jax as tfp
from .optim import optimize_optax, optimize_optax_vmap, optimize_scipy
from .utils.log import get_logger
from .utils.seed import get_numpy_rng
from .samplers import nested_sampling_Dy, sample_GP_NUTS
from .gp import GP
config.update("jax_enable_x64", True)
log = get_logger("acq")

#------------------Helper functions-------------------------
# These are jax versions of the BoTorch functions. 

def _scaled_improvement(mu, sigma, best_f):
    """u = (mu - best_f) / sigma, safe for sigma=0."""
    return (mu - best_f) / sigma

def _log_phi(u):
    """log of standard normal PDF"""
    return -0.5 * (u**2 + jnp.log(2 * jnp.pi))

def _ei_helper(u):
    """EI = phi(u) + u * Phi(u), stable for large |u|."""
    return norm.pdf(u) + u * norm.cdf(u)

def _log_abs_u_Phi_div_phi(u):
    """
    log(|u| * Phi(u) / phi(u)), valid for u < 0.
    Uses erfcx for numerical stability in tail.
    """
    neg_inv_sqrt2 = -1.0 / jnp.sqrt(2.0)
    log_sqrt_pi_div_2 = 0.5 * jnp.log(jnp.pi / 2.0)

    erfcx_val = tfp.math.erfcx(neg_inv_sqrt2 * u)
    return jnp.log(jnp.abs(u) * erfcx_val) + log_sqrt_pi_div_2

def _log_ei_helper(u):
    """
    Accurately computes log(phi(u) + u * Phi(u)).
    Matches BoTorch branching for stability, based on Ament et al., [arxiv: 2310.20708].
    """
    if u.dtype not in [jnp.float32, jnp.float64]:
        raise TypeError(
            f"LogExpectedImprovement only supports float32 and float64, got {u.dtype}."
        )

    bound = -1.0
    neg_inv_sqrt_eps = -1e6 if u.dtype == jnp.float64 else -1e3

    # u > bound: directly log(EI)
    u_upper = jnp.where(u < bound, bound, u)
    log_ei_upper = jnp.log(_ei_helper(u_upper))

    # u <= bound: use asymptotic expansion
    u_lower = jnp.where(u > bound, bound, u)
    u_eps = jnp.where(u_lower < neg_inv_sqrt_eps, neg_inv_sqrt_eps, u_lower)

    w = _log_abs_u_Phi_div_phi(u_eps)
    log_phi_u = _log_phi(u)

    second_term = jnp.where(
        u > neg_inv_sqrt_eps,
        tfp.math.log1mexp(w),
        -2.0 * jnp.log(jnp.abs(u_lower))
    )
    log_ei_lower = log_phi_u + second_term

    return jnp.where(u > bound, log_ei_upper, log_ei_lower)

#------------------The acquisition function classes-------------------------

class AcquisitionFunction:
    """Base class for acquisition functions.
    
    Acquisition functions guide the selection of new points to evaluate by balancing
    exploration and exploitation. Subclasses must implement the `fun` and `get_next_point` methods.
    
    Attributes
    ----------
    name : str
        Name of the acquisition function.
    optimizer : str
        Optimizer to use ('scipy' or 'optax').
    optimizer_options : dict
        Additional options for the optimizer.
    acq_optimize : callable
        Optimization function (optimize_scipy or optimize_optax).
    """

    name: str = "BaseAcquisitionFunction"

    def __init__(self, optimizer: str = "scipy", optimizer_options: Optional[Dict[str, Any]] = {}):
        self.optimizer = optimizer
        self.optimizer_options = optimizer_options
        if self.optimizer == "scipy":
            self.acq_optimize = optimize_scipy
        else:
            self.acq_optimize = optimize_optax

    def fun(self, x):
        raise NotImplementedError

    def get_next_point(self, gp: GP,
                       acq_kwargs: Dict[str, Any] = {},
                       maxiter: int = 500,
                       n_restarts: int = 8,
                       verbose: bool = True,
                       early_stop_patience: int = 25,
                       rng=None) -> Tuple[np.ndarray, float]:
        """
        Optimize the acquisition function to obtain the next point to sample.

        Parameters
        ----------
        gp : GP
            Gaussian process model.
        acq_kwargs : dict, optional
            Additional arguments for the acquisition function. Default is {}.
        maxiter : int, optional
            Maximum number of optimization iterations. Default is 500.
        n_restarts : int, optional
            Number of random restarts for optimization. Default is 8.
        verbose : bool, optional
            Whether to print optimization progress. Default is True.
        early_stop_patience : int, optional
            Patience for early stopping. Default is 25.
        rng : np.random.Generator, optional
            Random number generator. Default is None.
            
        Returns
        -------
        tuple
            (best_point, best_value) where best_point is shape (ndim,) and
            best_value is the acquisition function value.

        """

        raise NotImplementedError("Base class get_next() not implemented")

    def get_next_batch(self, gp: GP,
                       n_batch: int = 1,
                       acq_kwargs: Dict[str, Any] = {},
                       maxiter: int = 500,
                       n_restarts: int = 8,
                       verbose: bool = True,
                       early_stop_patience: int = 25,
                       rng=None) -> Tuple[np.ndarray, float]:

        """
        Get the next batch of points to sample.
        """

        rng = rng if rng is not None else get_numpy_rng()

        x_batch, acq_vals = [], []

        x_next, acq_val_next = self.get_next_point(gp, acq_kwargs=acq_kwargs,
                                        maxiter=maxiter,
                                        n_restarts=n_restarts,
                                        verbose=verbose,
                                        early_stop_patience=early_stop_patience,
                                        rng=rng)
        x_batch.append(x_next)
        acq_vals.append(acq_val_next)

        if n_batch > 1:
            # Create dummy GP without classifier functionality, for now we do not use batching for EI/LogEI
            state = gp.state_dict()
            dummy_gp = GP.from_state_dict(state)
            # dummy_gp = GP(train_x=gp.train_x, 
            #              train_y=gp.train_y*gp.y_std + gp.y_mean,
            #              noise=gp.noise,
            #              kernel=gp.kernel_name,
            #              lengthscales=gp.lengthscales,
            #              kernel_variance=gp.kernel_variance,)
                        
            dummy_gp.update(x_next, dummy_gp.predict_mean_single(x_next))
            for i in range(1,n_batch):
                x_next, acq_val_next = self.get_next_point(dummy_gp, acq_kwargs=acq_kwargs,
                                                        maxiter=maxiter,
                                                        n_restarts=n_restarts,
                                                        verbose=verbose,
                                                        early_stop_patience=early_stop_patience,
                                                        rng=rng)
                x_batch.append(x_next)
                acq_vals.append(acq_val_next)

                mu = dummy_gp.predict_mean_single(x_next)
                dummy_gp.update(x_next, mu)

        return np.array(x_batch), np.array(acq_vals)


class EI(AcquisitionFunction):
    """Expected Improvement acquisition function.
    
    EI measures the expected improvement over the current best observed value.
    It balances exploitation (high mean) and exploration (high uncertainty).
    
    The EI criterion is defined as:
        EI(x) = E[max(f(x) - f_best - zeta, 0)]
    
    where f_best is the best observed value and zeta is an exploration bonus.
    
    Parameters
    ----------
    optimizer : str, optional
        Optimizer to use ('scipy' or 'optax'). Default is 'scipy'.
    optimizer_options : dict, optional
        Additional options for the optimizer. Default is {}.
    """

    name: str = "EI"

    def __init__(self, optimizer: str = "scipy", optimizer_options: Optional[Dict[str, Any]] = {}):
        super().__init__(optimizer=optimizer, optimizer_options=optimizer_options)

        if optimizer == 'optax':
            self.acq_optimize = optimize_optax_vmap

    def fun(self, x, gp, best_y, zeta):
        """
        Compute Expected Improvement at point x.
        
        Parameters
        ----------
        x : jnp.ndarray
            Point at which to evaluate EI, shape (ndim,).
        gp : GP
            Gaussian process model.
        best_y : float
            Best observed function value.
        zeta : float
            Exploration bonus parameter.
            
        Returns
        -------
        float
            Negative expected improvement (for minimization).
        """
        mu, var = gp.predict_single(x)
        var = jnp.clip(var, a_min=1e-20)  # prevent zero variance
        sigma = jnp.sqrt(var)

        u = _scaled_improvement(mu - zeta, sigma, best_y)
        ei = _ei_helper(u) * sigma

        return jnp.reshape(-ei, ())  # optimizer minimizes this
    
    def get_next_point(self, gp, 
                       acq_kwargs,
                 maxiter: int = 250,
                 n_restarts: int = 20,
                 verbose: bool = True,
                 early_stop_patience: int = 25,
                 rng=None):

        rng = rng if rng is not None else get_numpy_rng()
        zeta = acq_kwargs.get('zeta', 0.)
        best_y = acq_kwargs.get('best_y', max(gp.train_y.flatten()))
        fun_args = (gp, best_y, zeta)
        fun_kwargs = {}
        best_x = gp.train_x[jnp.argmax(gp.train_y)]

        # For Classifier GP, we make sure to get points inside the positive region
        if n_restarts > 1:
            n_random_restarts = int(n_restarts/2)
            x0_acq = jnp.vstack([gp.get_random_point(rng,nstd=5) for _ in range(n_random_restarts)])
            n_best_restarts = n_restarts - n_random_restarts
            # print(f'shape x0_acq: {x0_acq.shape}, best_x shape: {best_x.shape}, nrestarts: {n_restarts}, n_random: {n_random_restarts}, n_best: {n_best_restarts}')
            x0_acq = jnp.vstack([x0_acq, jnp.full((n_best_restarts, gp.ndim), best_x)])
        else:
            x0_acq = best_x
        jitter = rng.normal(0.,0.005,size=x0_acq.shape)
        x0_acq = jnp.clip(x0_acq + jitter, 0., 1.)
        pts, vals =  self.acq_optimize(fun=self.fun,
                            fun_args=fun_args,
                            fun_kwargs=fun_kwargs,
                            num_params=gp.ndim,
                            x0=x0_acq,
                            bounds = [0,1],
                            optimizer_options=self.optimizer_options,
                            maxiter=maxiter,
                            n_restarts=n_restarts,
                            verbose=verbose)
        return pts, -vals # we minimize -EI so return -vals

class LogEI(EI):
    """Log Expected Improvement acquisition function.
    
    LogEI computes the logarithm of the Expected Improvement, providing better 
    numerical stability compared to EI, especially when EI values are very small.
    Uses advanced numerical techniques for accurate computation in extreme cases.
    
    Parameters
    ----------
    optimizer : str, optional
        Optimizer to use ('scipy' or 'optax'). Default is 'scipy'.
    optimizer_options : dict, optional
        Additional options for the optimizer. Default is {}.
        
    References
    ----------
    [1] Ament, S., et al. (2023). "Unexpected Improvements to Expected Improvement
        for Bayesian Optimization." arXiv:2310.20708.
    """

    name: str = "LogEI"

    def __init__(self, optimizer: str = "scipy", optimizer_options: Optional[Dict[str, Any]] = {}):
        super().__init__(optimizer=optimizer, optimizer_options=optimizer_options)

    def fun(self, x, gp, best_y, zeta):
        """
        Log Expected Improvement in pure JAX.
        Returns *positive* log-EI, so you can maximize directly.
        """
        mu, var = gp.predict_single(x)
        var = jnp.clip(var, a_min=1e-18)  # prevent zero variance
        sigma = jnp.sqrt(var)

        u = _scaled_improvement(mu - zeta, sigma, best_y)
        log_ei = _log_ei_helper(u) + jnp.log(sigma)

        return jnp.reshape(-log_ei, ())  # optimizer minimizes this


class WeightedIntegratedPosteriorBase(AcquisitionFunction):
    """Base class for Weighted Integrated Posterior acquisition functions.
    
    This base class provides common functionality for acquisition functions that
    integrate over MC samples from the GP posterior, such as WIPV and WIPStd.
    
    Parameters
    ----------
    optimizer : str, optional
        Optimizer to use ('scipy' or 'optax'). Default is 'scipy'.
    optimizer_options : dict, optional
        Additional options for the optimizer. Default is {}.
    """

    def __init__(self, optimizer: str = "scipy", optimizer_options: Optional[Dict[str, Any]] = {}):
        super().__init__(optimizer=optimizer, optimizer_options=optimizer_options)

    def get_next_point(self, gp,
                 acq_kwargs,
                 maxiter: int = 100,
                 n_restarts: int = 1,
                 verbose: bool = True,
                 early_stop_patience: int = 25,
                 rng=None):
        """
        Optimize the acquisition function to obtain the next point.
        
        This method is shared between WIPV and WIPStd as they follow the same
        optimization procedure but differ only in their objective function.
        
        Parameters
        ----------
        gp : GP
            Gaussian process model.
        acq_kwargs : dict
            Additional arguments containing 'mc_samples' and optionally 'mc_points_size'.
        maxiter : int, optional
            Maximum optimization iterations. Default is 100.
        n_restarts : int, optional
            Number of optimization restarts. Default is 1.
        verbose : bool, optional
            Whether to print progress. Default is True.
        early_stop_patience : int, optional
            Early stopping patience. Default is 25.
        rng : np.random.Generator, optional
            Random number generator. Default is None.
            
        Returns
        -------
        tuple
            (best_point, best_value) where best_point is shape (ndim,).
        """
        mc_samples = acq_kwargs.get('mc_samples')
        mc_points_size = acq_kwargs.get('mc_points_size', 128)
        mc_points = get_mc_points(mc_samples, mc_points_size=mc_points_size, rng=rng)
        k_train_mc = gp.kernel.covariance(gp.train_x, mc_points, include_noise=False)

        @jax.jit
        def mapped_fn(x):
            return self.fun(x, gp, mc_points=mc_points, k_train_mc=k_train_mc)

        acq_vals = lax.map(mapped_fn, mc_points)
        acq_val_min = jnp.min(acq_vals)
        log.debug(f"{self.name} acquisition min value on MC points: {float(acq_val_min):.4e}")
        best_x = mc_points[jnp.argmin(acq_vals)]
        x0_acq = best_x

        if gp.train_x.shape[0] > 500:
            return x0_acq, float(acq_val_min)
        else:
            return self.acq_optimize(fun=self.fun,
                                  fun_args=(gp,),
                                  fun_kwargs={'mc_points': mc_points, 'k_train_mc': k_train_mc},
                                  num_params=gp.ndim,
                                  x0=x0_acq,
                                  bounds = [0,1],
                                  optimizer_options=self.optimizer_options,
                                  maxiter=maxiter,
                                  n_restarts=n_restarts,
                                  verbose=verbose)


class WIPV(WeightedIntegratedPosteriorBase):
    """Weighted Integrated Posterior Variance acquisition function.
    
    WIPV focuses on reducing uncertainty in regions weighted by posterior probability.
    It integrates the posterior variance over MC samples drawn from the GP posterior,
    making it particularly effective for Bayesian evidence estimation.
    
    The criterion is defined as:
        WIPV(x) = E_{x' ~ p(x' | D)}[Var[f(x) | D]]
    
    where the expectation is over MC samples x' from the posterior.
    
    Parameters
    ----------
    optimizer : str, optional
        Optimizer to use ('scipy' or 'optax'). Default is 'scipy'.
    optimizer_options : dict, optional
        Additional options for the optimizer. Default is {}.
        
    """

    name: str = "WIPV"

    def fun(self, x, gp,  mc_points=None, k_train_mc = None):
        var = gp.fantasy_var(new_x=x, mc_points=mc_points,k_train_mc=k_train_mc)
        return jnp.mean(var)


class WIPStd(WeightedIntegratedPosteriorBase):
    """Weighted Integrated Posterior Standard Deviation acquisition function.
    
    WIPStd is similar to WIPV but uses standard deviation instead of variance,
    which can provide different exploration characteristics. It integrates the
    posterior standard deviation over MC samples from the GP posterior.
    
    The criterion is defined as:
        WIPStd(x) = E_{x' ~ p(x' | D)}[Std[f(x) | D]]
    
    Parameters
    ----------
    optimizer : str, optional
        Optimizer to use ('scipy' or 'optax'). Default is 'scipy'.
    optimizer_options : dict, optional
        Additional options for the optimizer. Default is {}.
    """

    name: str = "WIPStd"

    def fun(self, x, gp,  mc_points=None, k_train_mc = None):
        std = jnp.sqrt(gp.fantasy_var(new_x=x, mc_points=mc_points,k_train_mc=k_train_mc))
        return jnp.mean(std)


def get_mc_samples(gp: GP,warmup_steps=512, num_samples=1024, thinning=4,method="NUTS",num_chains=4,np_rng=None,rng_key=None):
    if method=='NUTS':
        mc_samples = sample_GP_NUTS(gp=gp, warmup_steps=warmup_steps,
            num_samples=num_samples, thinning=thinning, num_chains=num_chains,np_rng=np_rng,rng_key=rng_key
        )
    elif method=='NS':
        mc_samples, logz, success = nested_sampling_Dy(gp=gp, ndim=gp.ndim, mode = 'acq', maxcall=int(2e6),
                                            dynamic=False, dlogz=0.02,equal_weights=True, rng=np_rng)
    elif method=='uniform':
        mc_samples = {}
        points = qmc.Sobol(gp.ndim, scramble=True, rng=np_rng).random(num_samples)
        mc_samples['x'] = points
    else:
        raise ValueError(f"Unknown method {method} for sampling GP")
    return mc_samples


def get_mc_points(mc_samples, mc_points_size=128, rng=None):
    mc_size = max(mc_samples['x'].shape[0], mc_points_size)
    rng = rng if rng is not None else get_numpy_rng()   
    idxs = rng.choice(mc_size, size=mc_points_size, replace=False)
    return mc_samples['x'][idxs]