from typing import Any, List, Optional, Dict, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc
from jax.scipy.stats import norm
from .optim import optimize_optax, optimize_optax_vmap, optimize_scipy
from .utils.log import get_logger
from .utils.seed import get_numpy_rng
from .samplers import nested_sampling_Dy, sample_GP_NUTS
from .gp import GP
log = get_logger("acq")

# ---------------------------------------------------------------------------
# Optional TFP import — not compatible with all JAX versions.
# If unavailable we fall back to pure-JAX implementations of erfcx / log1mexp.
# ---------------------------------------------------------------------------
try:
    import tensorflow_probability.substrates.jax as tfp
    _TFP_AVAILABLE = True
except Exception:
    _TFP_AVAILABLE = False
    log.warning(
        "tensorflow_probability not available (may be incompatible with current JAX version). "
        "LogEI will use a pure-JAX fallback for erfcx / log1mexp."
    )


def _erfcx_jax(x):
    """
    Pure-JAX erfcx(x) = exp(x^2) * erfc(x), stable for large positive x.

    TODO: add ref
    """
    x_small = jnp.where(x <= 25.0, x, 0.0)
    direct = jnp.exp(x_small ** 2) * jax.scipy.special.erfc(x_small)

    x_large = jnp.where(x > 25.0, x, 25.0)
    inv_2x2 = 0.5 / (x_large ** 2)
    asymp = (
        1.0
        + inv_2x2 * (
            -1.0
            + inv_2x2 * (
                3.0
                + inv_2x2 * (
                    -15.0
                    + inv_2x2 * (
                        105.0
                        + inv_2x2 * (-945.0 + inv_2x2 * 10395.0)
                    )
                )
            )
        )
    ) / (jnp.sqrt(jnp.pi) * x_large)
    return jnp.where(x <= 25.0, direct, asymp)


def _log1mexp_jax(x):
    """
    Pure-JAX log(1 - exp(x)) for x < 0, numerically stable.
    """
    return jnp.where(
        x > -jnp.log(2.0),
        jnp.log(-jnp.expm1(x)),
        jnp.log1p(-jnp.exp(x)),
    )

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

    if _TFP_AVAILABLE:
        erfcx_val = tfp.math.erfcx(neg_inv_sqrt2 * u)
    else:
        erfcx_val = _erfcx_jax(neg_inv_sqrt2 * u)
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

    log1mexp_fn = tfp.math.log1mexp if _TFP_AVAILABLE else _log1mexp_jax

    second_term = jnp.where(
        u > neg_inv_sqrt_eps,
        log1mexp_fn(w),
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

    def __init__(self, optimizer: str = "scipy", optimizer_options: Optional[Dict[str, Any]] = None):
        self.optimizer = optimizer
        # Private copy: the optimizers must never see (or alias) a caller's dict.
        self.optimizer_options = {} if optimizer_options is None else dict(optimizer_options)
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
            # Reconstruct explicitly as a base GP so fantasy observations do
            # not use classifier gating, while retaining the complete fitted
            # GP state (including any kernel input-space transform).
            dummy_gp = GP.from_state_dict(gp.state_dict())

            dummy_gp.update(x_next, dummy_gp.predict_mean(x_next))
            for i in range(1,n_batch):
                x_next, acq_val_next = self.get_next_point(dummy_gp, acq_kwargs=acq_kwargs,
                                                        maxiter=maxiter,
                                                        n_restarts=n_restarts,
                                                        verbose=verbose,
                                                        early_stop_patience=early_stop_patience,
                                                        rng=rng)
                x_batch.append(x_next)
                acq_vals.append(acq_val_next)

                mu = dummy_gp.predict_mean(x_next)
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

    def __init__(self, optimizer: str = "scipy", optimizer_options: Optional[Dict[str, Any]] = None):
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
        mu, var = gp.predict(x, standardized=True)
        var = jnp.clip(var, min=1e-20)  # prevent zero variance
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

    def __init__(self, optimizer: str = "scipy", optimizer_options: Optional[Dict[str, Any]] = None):
        super().__init__(optimizer=optimizer, optimizer_options=optimizer_options)

    def fun(self, x, gp, best_y, zeta):
        """
        Log Expected Improvement in pure JAX.
        Returns *positive* log-EI, so you can maximize directly.
        """
        mu, var = gp.predict(x, standardized=True)
        var = jnp.clip(var, min=1e-18)  # prevent zero variance
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
        Additional options for the optimizer. Default is None.
    share_mc_points : bool, optional
        If False (default), each point of a batch is chosen on a freshly drawn
        subset of MC points, so no single subset dominates the batch.
        If True, one subset is drawn per batch and later selections reuse it
        via O(M^2) rank-1 cache updates.
    optimize_until_size : int or None, optional
        Maximum number of conditioning points (training + fantasised points)
        for which continuous optimization refines the best MC candidate.
        Above this size, the best MC point is returned directly. Set to ``0``
        to always skip continuous refinement, or ``None`` to always refine.
        Default is 1000.
    """

    def __init__(
        self,
        optimizer: str = "scipy",
        optimizer_options: Optional[Dict[str, Any]] = None,
        share_mc_points: bool = False,
        optimize_until_size: Optional[int] = 1000,
    ):
        super().__init__(optimizer=optimizer, optimizer_options=optimizer_options)
        if optimize_until_size is not None and optimize_until_size < 0:
            raise ValueError(
                f"optimize_until_size must be >= 0 or None, got {optimize_until_size}"
            )
        self.share_mc_points = share_mc_points
        self.optimize_until_size = optimize_until_size

    def _reduce(self, var):
        """Collapse fantasy variances over the MC points (last axis) into the acquisition value."""
        raise NotImplementedError

    def fun(self, x, gp, mc_points=None, cache=None):
        return self._reduce(gp.fantasy_var(new_x=x, mc_points=mc_points, cache=cache))

    @staticmethod
    def _draw_mc_points(acq_kwargs, rng):
        mc_samples = acq_kwargs.get('mc_samples')
        mc_points_size = acq_kwargs.get('mc_points_size', 128)
        return jnp.asarray(get_mc_points(mc_samples, mc_points_size=mc_points_size, rng=rng))

    def _select_point(self, gp, mc_points, cache, n_conditioning, maxiter, n_restarts, verbose):
        """
        Score every MC point as a candidate, then (when ``optimize_until_size``
        is ``None`` or ``n_conditioning <= optimize_until_size``) refine the
        best one with the continuous optimizer.
        """
        acq_vals = self._reduce(gp.fantasy_var_scan(mc_points, cache))

        best_idx = int(jnp.argmin(acq_vals))
        acq_val_min = float(acq_vals[best_idx])
        log.debug(f"{self.name} acquisition min value on MC points: {acq_val_min:.4e}")
        x0_acq = mc_points[best_idx]

        if self.optimize_until_size is not None and n_conditioning > self.optimize_until_size:
            return x0_acq, acq_val_min
        return self.acq_optimize(
            fun=self.fun,
            fun_args=(gp,),
            fun_kwargs={'mc_points': mc_points, 'cache': cache},
            num_params=gp.ndim,
            x0=x0_acq,
            bounds=[0, 1],
            optimizer_options=self.optimizer_options,
            maxiter=maxiter,
            n_restarts=n_restarts,
            verbose=verbose,
        )

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
        mc_points = self._draw_mc_points(acq_kwargs, rng)
        cache = gp.precompute_fantasy_cache(mc_points)
        return self._select_point(gp, mc_points, cache, gp.npoints, maxiter, n_restarts, verbose)

    def get_next_batch(
        self,
        gp: GP,
        n_batch: int = 1,
        acq_kwargs: Optional[Dict[str, Any]] = None,
        maxiter: int = 500,
        n_restarts: int = 8,
        verbose: bool = True,
        early_stop_patience: int = 25,
        rng=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Greedy (kriging-believer) batch selection.

        After each selection the cache is conditioned on the chosen point via
        ``gp.condition_fantasy_cache``, which is exact because fantasy variances
        do not depend on the fantasised value. This avoids copying the GP and
        refactorising its Cholesky for every batch point.

        By default (``share_mc_points=False``) a fresh MC subset is drawn for
        every batch point and the conditioned cache is built directly for it.
        With ``share_mc_points=True`` the first subset is kept and later
        selections cost only O(M^2) each.

        Surrogates without ``supports_fantasy_conditioning`` fall back to the
        base-class copy-and-update path.
        """
        acq_kwargs = {} if acq_kwargs is None else acq_kwargs
        if not getattr(gp, 'supports_fantasy_conditioning', False):
            return super().get_next_batch(
                gp,
                n_batch=n_batch,
                acq_kwargs=acq_kwargs,
                maxiter=maxiter,
                n_restarts=n_restarts,
                verbose=verbose,
                early_stop_patience=early_stop_patience,
                rng=rng,
            )

        rng = rng if rng is not None else get_numpy_rng()
        mc_points = self._draw_mc_points(acq_kwargs, rng)
        cache = gp.precompute_fantasy_cache(mc_points)

        x_batch, acq_vals = [], []
        base_npoints = gp.npoints
        for i in range(n_batch):
            x_next, acq_val_next = self._select_point(
                gp, mc_points, cache, base_npoints + i, maxiter, n_restarts, verbose
            )
            x_batch.append(np.asarray(x_next))
            acq_vals.append(float(acq_val_next))
            if i + 1 < n_batch:
                if self.share_mc_points:
                    cache = gp.condition_fantasy_cache(cache, x_next, mc_points)
                else:
                    mc_points = self._draw_mc_points(acq_kwargs, rng)
                    cache = gp.condition_fantasy_cache(cache, x_next, new_mc_points=mc_points)

        return np.array(x_batch), np.array(acq_vals)


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
        Additional options for the optimizer. Default is None.
    share_mc_points : bool, optional
        Whether to share one MC subset across a batch. Default is False.
    """

    name: str = "WIPV"

    def _reduce(self, var):
        return jnp.mean(var, axis=-1)


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
        Additional options for the optimizer. Default is None.
    share_mc_points : bool, optional
        Whether to share one MC subset across a batch. Default is False.
    """

    name: str = "WIPStd"

    def _reduce(self, var):
        return jnp.mean(jnp.sqrt(var), axis=-1)


def get_mc_samples(
    gp: GP,
    warmup_steps=512,
    num_samples=1024,
    thinning=4,
    method="NUTS",
    num_chains=4,
    np_rng=None,
    rng_key=None,
    param_bounds=None,
):
    if method == 'NUTS':
        mc_samples = sample_GP_NUTS(
            gp=gp,
            warmup_steps=warmup_steps,
            num_samples=num_samples,
            thinning=thinning,
            num_chains=num_chains,
            np_rng=np_rng,
            rng_key=rng_key,
        )
    elif method == 'NS':
        ns_samples, logz, success = nested_sampling_Dy(
            gp=gp,
            ndim=gp.ndim,
            mode='acq',
            maxcall=int(2e6),
            dynamic=False,
            dlogz=0.02,
            rng=np_rng,
            param_bounds=param_bounds,
        )
        mc_samples = {
            'x': np.asarray(ns_samples['eq_u']),
            'logp': np.asarray(ns_samples['eq_logl']),
            'weights': np.ones(len(ns_samples['eq_u'])),
            'best': ns_samples['best'],
            'method': 'nested',
        }
        if param_bounds is not None:
            mc_samples['x_physical'] = np.asarray(ns_samples['eq_x'])
    elif method == 'uniform':
        mc_samples = {}
        points = qmc.Sobol(gp.ndim, scramble=True, rng=np_rng).random(num_samples)
        mc_samples['x'] = points
    else:
        raise ValueError(f"Unknown method {method} for sampling GP")

    # Filter out NaN samples early
    samples = mc_samples['x']
    if np.any(np.isnan(samples)):
        valid_mask = ~np.any(np.isnan(samples), axis=1)
        n_bad = np.sum(~valid_mask)
        log.warning(f"Filtered {n_bad} NaN samples from MC chain ({n_bad/len(samples)*100:.1f}%)")
        mc_samples['x'] = samples[valid_mask]
        for key in ('x_physical', 'weights', 'logp', 'logl'):
            if key in mc_samples and len(mc_samples[key]) == len(valid_mask):
                mc_samples[key] = np.asarray(mc_samples[key])[valid_mask]

    return mc_samples


def get_mc_points(mc_samples, mc_points_size=128, rng=None):
    """Select a subset of MC samples for acquisition function evaluation."""
    samples = mc_samples['x']
    n_available = samples.shape[0]

    # Check for NaN in samples
    if np.any(np.isnan(samples)):
        log.warning(f"NaN detected in mc_samples! Filtering out {np.sum(np.any(np.isnan(samples), axis=1))} bad samples.")
        valid_mask = ~np.any(np.isnan(samples), axis=1)
        samples = samples[valid_mask]
        n_available = samples.shape[0]
        if n_available == 0:
            raise ValueError("All MC samples contain NaN - cannot proceed.")

    rng = rng if rng is not None else get_numpy_rng()
    n_select = min(n_available, mc_points_size)
    idxs = rng.choice(n_available, size=n_select, replace=False)
    return samples[idxs]