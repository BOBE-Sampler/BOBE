"""
Gaussian Process (GP) surrogate model implementation using exact GP inference.
"""

from math import pi
from typing import Any, List, NamedTuple, Optional
import jax.numpy as jnp
import numpy as np
import jax
from jax.scipy.linalg import cho_solve, solve_triangular
from .utils.log import get_logger
log = get_logger("gp")
from .optim import optimize_optax, optimize_scipy
from .utils.seed import get_new_jax_key, get_numpy_rng
from .kernels import Kernel
from .surrogate import Surrogate
from .transforms import PrincipalAxesTransform


safe_noise_floor = 1e-12


@jax.jit
def gp_mll(k, train_y, num_points):
    """
    Computes the marginal log likelihood of the GP.
    """
    L = jnp.linalg.cholesky(k)
    alpha = cho_solve((L, True), train_y)
    mll = -0.5 * jnp.einsum("ij,ji", train_y.T, alpha) - jnp.sum(jnp.log(jnp.diag(L))) - 0.5 * num_points * jnp.log(2 * pi)
    return mll


def _clip_var(var):
    """Replace NaN / sub-floor variances (numerical round-off) by ``safe_noise_floor``."""
    var = jnp.where(jnp.isnan(var), safe_noise_floor, var)
    return jnp.where(var < safe_noise_floor, safe_noise_floor, var)


def _extend_cholesky_row(L: jnp.ndarray, v: jnp.ndarray, sqrt_s: jnp.ndarray) -> jnp.ndarray:
    """Append bottom row ``[v, sqrt_s]`` to lower-triangular Cholesky factor ``L``."""
    n = L.shape[0]
    L_new = jnp.zeros((n + 1, n + 1), dtype=L.dtype)
    return L_new.at[:n, :n].set(L).at[n, :n].set(v).at[n, n].set(sqrt_s)


@jax.jit
def fast_update_cholesky(L: jnp.ndarray, k: jnp.ndarray, k_self: float):
    # solve L v = k  -> v has shape (n,)
    v = solve_triangular(L, k, lower=True)

    # new diagonal entry (clamped against numerical round-off)
    diag = jnp.sqrt(jnp.maximum(k_self - jnp.dot(v, v), safe_noise_floor))

    return _extend_cholesky_row(L, v, diag)


class FantasyCache(NamedTuple):
    """
    Exact-GP posterior restricted to a fixed set of M MC integration points.

    Built by ``GP.precompute_fantasy_cache`` and consumed by ``fantasy_var``,
    ``fantasy_var_scan`` and ``condition_fantasy_cache``. Everything is in
    standardised units and independent of the training targets, so fantasised
    batch points can be conditioned on without touching ``y``.

    Attributes
    ----------
    L : jnp.ndarray, shape (N', N')
        Lower Cholesky factor of ``k(X, X) + noise`` for the conditioning set.
    X : jnp.ndarray, shape (N', D)
        Conditioning inputs: the training points plus any fantasised points.
    V : jnp.ndarray, shape (N', M)
        ``L^{-1} k(X, mc_points)``.
    var0 : jnp.ndarray, shape (M,)
        Predictive variance at the MC points given ``X`` (noise included).
    C : jnp.ndarray, shape (M, M)
        Posterior covariance among the MC points given ``X`` (noise excluded).
    """
    L: jnp.ndarray
    X: jnp.ndarray
    V: jnp.ndarray
    var0: jnp.ndarray
    C: jnp.ndarray


@jax.jit
def _fantasy_scan_var(C, var0, y_std):
    """
    Fantasy variances with every MC point as the candidate.

    ``out[j, m]`` is the physical-scale predictive variance at MC point ``m``
    after adding MC point ``j`` to the conditioning set:
    ``var0[m] - C[j, m]**2 / var0[j]``. Shapes depend only on M, so this
    compiles once per run.
    """
    s = jnp.maximum(var0, safe_noise_floor)
    return _clip_var(var0[None, :] - C**2 / s[:, None]) * y_std**2


class GP(Surrogate):

    #: Implements ``condition_fantasy_cache``, so WIPV/WIPStd can select whole
    #: batches by conditioning the cache instead of copying and refactorising the GP.
    supports_fantasy_conditioning = True

    def __init__(
        self,
        kernel: Kernel,
        train_x=None,
        train_y=None,
        optimizer: str = "scipy",
        optimizer_options: dict = None,
        param_names: List[str] = None,
    ):
        """
        Initialize the Gaussian Process surrogate model.

        Kernel hyperparameters, priors, bounds, observation noise, and any
        kernel-metric transform are configured directly on the ``kernel``
        instance (e.g. ``RBFKernel(...)`` or ``MaternKernel(...)``).

        Parameters
        ----------
        kernel : Kernel
            Kernel instance to use.
        train_x : jnp.ndarray, optional
            Training inputs in unit-cube coordinates, shape ``(N, D)``.
            If ``None``, dimension-dependent setup is deferred until
            ``initialize(ndim)`` or the first call to ``update()``.
        train_y : jnp.ndarray, optional
            Objective function values at training points, shape ``(N, 1)``
            or ``(N,)``.
        optimizer : str, optional
            Optimizer for hyperparameter tuning (``'scipy'`` or ``'optax'``).
            Default is ``'scipy'``.
        optimizer_options : dict, optional
            Keyword arguments forwarded to the hyperparameter optimizer.
        param_names : list of str, optional
            Parameter names for logging hyperparameters.
        """
        if not isinstance(kernel, Kernel):
            raise TypeError(
                f"kernel must be a Kernel instance, got {type(kernel).__name__}"
            )

        self.kernel = kernel
        self.optimizer_method = optimizer
        self.mll_optimize = optimize_scipy if optimizer == "scipy" else optimize_optax
        self.optimizer_options = {} if optimizer_options is None else dict(optimizer_options)
        self._param_names_arg = list(param_names) if param_names is not None else None

        if train_x is not None:
            self._setup_training_data(train_x, train_y)
            self.initialize()
        else:
            # Fully deferred — nothing ndim-dependent is set yet
            self.ndim = None
            self.train_x = None
            self.train_y = None
            self.y_mean = 0.0
            self.y_std = 1.0

    def initialize(self, ndim=None, param_bounds=None, param_names=None):
        """
        Complete deferred initialisation (dimensionality, parameter names,
        kernel, and any deferred kernel input transform).

        Called immediately from ``__init__`` when training data are provided,
        from ``BOBE`` with likelihood bounds/names, or from ``update()`` on
        first data arrival when constructed with ``train_x=None``.
        """
        if ndim is None and param_bounds is not None:
            ndim = int(np.asarray(param_bounds).shape[1])

        if self.ndim is None:
            if ndim is None:
                raise ValueError("Provide ndim or param_bounds to initialize GP.")
            self.ndim = int(ndim)
            self.train_x = jnp.empty((0, self.ndim))
            self.train_y = jnp.empty((0, 1))
            self.y_mean = 0.0
            self.y_std = 1.0
        elif ndim is not None and int(ndim) != self.ndim:
            raise ValueError(
                f"GP was initialized for ndim={self.ndim}, cannot use with ndim={ndim}."
            )

        if self._param_names_arg is None and param_names is not None:
            self._param_names_arg = list(param_names)

        self.param_names = (
            list(self._param_names_arg)
            if self._param_names_arg is not None
            else ['x_' + str(i) for i in range(self.ndim)]
        )

        self.kernel.initialize(
            self.ndim, param_bounds=param_bounds, param_names=param_names
        )
        self.recompute_cholesky()

    @property
    def kernel_rotation(self):
        transform = getattr(self.kernel, "input_transform", None)
        if isinstance(transform, PrincipalAxesTransform):
            return transform
        return None

    @property
    def has_adaptive_kernel_rotation(self):
        return self.kernel_rotation is not None and self.kernel_rotation.adaptive

    def try_update_kernel_rotation(
        self,
        mc_samples,
        acq_val,
        step,
        acq_threshold=None,
        current_evals=None,
        min_evals=None,
    ):
        """Update the kernel rotation from posterior samples, if gated in."""
        transform = self.kernel_rotation
        if transform is None:
            return False
        if min_evals is not None:
            if current_evals is None:
                raise ValueError(
                    "current_evals is required when min_evals is provided"
                )
            if current_evals < min_evals:
                log.debug(
                    "Skipping adaptive kernel rotation before min_evals: "
                    f"{current_evals} < {min_evals}"
                )
                return False
        # Axes are learned in the coordinates the transform receives, i.e.
        # after any kernel pre_transform.
        x = np.asarray(mc_samples["x"], dtype=float)
        mc_samples = {**mc_samples, "x": np.asarray(self.kernel.pre_transform(x))}
        try:
            did_update = transform.try_update(
                mc_samples=mc_samples,
                acq_val=acq_val,
                step=step,
                acq_threshold=acq_threshold,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            log.warning(f"[Kernel rotation update] {exc}; skipping.")
            return False
        if did_update:
            self.recompute_cholesky()
            kl_text = (
                "first posterior baseline"
                if transform.last_update_kl is None
                else f"symmetric KL={transform.last_update_kl:.4f}"
            )
            log.info(
                f"[Kernel rotation update] accepted at iteration {step}: "
                f"count={transform.update_count}, {kl_text}"
            )
        return did_update

    def _setup_training_data(self, train_x, train_y):
        """
        Setup and validate training data, compute standardization parameters.
        """

        # Check x and y sizes
        if train_x.shape[0] != train_y.shape[0]:
            raise ValueError("train_x and train_y must have the same number of points")
        if train_y.ndim != 2:
            train_y = train_y.reshape(-1, 1)
        if train_x.ndim != 2:
            raise ValueError("train_x must be 2D")

        self.ndim = train_x.shape[1]

        # Compute standardization parameters (and handle the case of 0 initialisation points)
        self.y_mean = jnp.mean(train_y) if train_y.size > 0 else 0
        self.y_std = jnp.std(train_y) if train_y.size > 0 else 1.0

        # Handle edge case where std is zero (all values identical or only 1 point)
        if self.y_std == 0:
            log.warning("Training targets have zero variance. Setting std to 1.0 to avoid division by zero.")
            self.y_std = 1.0

        # Store standardized training data
        self.train_x = jnp.array(train_x)
        self.train_y = (train_y - self.y_mean) / self.y_std
        log.debug(f"GP training size = {self.train_x.shape[0]}")

    def neg_mll(self, log_params):
        """
        Computes the negative log marginal likelihood for the GP with given hyperparameters.
        """
        self.kernel.set_hyperparams(log_params)
        K = self.kernel.covariance(self.train_x, self.train_x, include_noise=True)
        mll = gp_mll(K, self.train_y, self.train_y.shape[0])
        mll += self.kernel.log_prior()
        return -mll

    def fit(self, x0: np.ndarray = None, maxiter: int = 500) -> dict:
        """
        Performs a serial fit for a given batch of starting points (x0).
        This method is called by each MPI process on its assigned chunk.

        Arguments
        ---------
        x0 : np.ndarray
            Array of shape (n_restarts_chunk, n_params) containing starting points for optimization (in log space).
        maxiter : int
            Maximum number of iterations for the optimizer. Defaults to 500.

        Returns
        -------
        result : dict
            Dictionary containing the best 'mll' and corresponding 'params' (log space) found.
        """
        if x0 is None:  # set to current hyperparameters
            x0 = jnp.log(self.kernel.get_hyperparams())[None, :]

        optimizer_options = self.optimizer_options.copy()

        best_params_log, best_loss = self.mll_optimize(
            fun=self.neg_mll,
            num_params=self.kernel.num_hyperparams,
            bounds=self.kernel.hyperparam_bounds,
            x0=x0,
            maxiter=maxiter,
            n_restarts=x0.shape[0],
            optimizer_options=optimizer_options,
        )

        # Apply best parameters and recompute Cholesky so the kernel always
        # holds concrete values after fit() returns.
        self.update_hyperparams(best_params_log)

        return {
            'mll': -best_loss,
            'params': best_params_log,
        }

    def update_hyperparams(self, hyperparams):
        """
        Update the GP hyperparameters and recompute the Cholesky and alphas.
        """
        self.kernel.set_hyperparams(hyperparams)
        self.recompute_cholesky()

    def predict_mean(self, x):
        """
        Predict GP mean in physical (unstandardised) space for a single point
        ``(D,)`` or a batch ``(M, D)``.
        """
        x = jnp.asarray(x)
        single = x.ndim == 1
        x_2d = jnp.atleast_2d(x)
        k12 = self.kernel.covariance(self.train_x, x_2d, include_noise=False)  # shape (N, M)
        mean = (k12.T @ self.alphas).squeeze(-1) * self.y_std + self.y_mean    # shape (M,)
        return mean[0] if single else mean

    def predict(self, x, *, standardized: bool = False):
        """
        Predict ``(mean, var)`` for a single point ``(D,)`` or a batch ``(M, D)``.

        Parameters
        ----------
        x : jnp.ndarray, shape (D,) or (M, D)
        standardized : bool, optional
            If ``False`` (default), return physical mean and variance.
            If ``True``, return standardised mean and variance (for EI / LogEI).
        """
        x = jnp.asarray(x)
        single = x.ndim == 1
        x_2d = jnp.atleast_2d(x)
        k12 = self.kernel.covariance(self.train_x, x_2d, include_noise=False)  # shape (N, M)
        k22 = self.kernel.diagonal(x_2d, include_noise=True)                   # shape (M,)
        mean = (k12.T @ self.alphas).squeeze(-1)                               # shape (M,)
        vv = solve_triangular(self.cholesky, k12, lower=True)                  # shape (N, M)
        var = _clip_var(k22 - jnp.sum(vv * vv, axis=0))                        # shape (M,)
        if not standardized:
            mean = mean * self.y_std + self.y_mean
            var = var * self.y_std**2
        if single:
            return mean[0], var[0]
        return mean, var

    def update(self, new_x, new_y):
        """
        Add new training points, re-standardise targets, and recompute Cholesky factors.
        """
        new_x = jnp.atleast_2d(new_x)
        new_y = jnp.array(new_y)
        if new_y.ndim != 2:
            new_y = new_y.reshape(-1, 1)

        # Complete deferred initialisation on first data arrival
        if self.ndim is None:
            self.initialize(ndim=new_x.shape[1])

        # Filter NaN inputs / targets
        if jnp.any(jnp.isnan(new_x)):
            log.warning(f"NaN detected in new_x! Filtering out {jnp.sum(jnp.any(jnp.isnan(new_x), axis=1))} bad points.")
            valid_mask = ~jnp.any(jnp.isnan(new_x), axis=1)
            new_x = new_x[valid_mask]
            new_y = new_y[valid_mask]
            if new_x.shape[0] == 0:
                log.warning("All new points contain NaN - skipping update.")
                return
        if jnp.any(jnp.isnan(new_y)):
            log.warning(f"NaN detected in new_y! Filtering out {jnp.sum(jnp.any(jnp.isnan(new_y), axis=1))} bad points.")
            valid_mask = ~jnp.any(jnp.isnan(new_y), axis=1)
            new_x = new_x[valid_mask]
            new_y = new_y[valid_mask]
            if new_x.shape[0] == 0:
                log.warning("All new points contain NaN - skipping update.")
                return

        new_pts_to_add = []
        new_vals_to_add = []

        # Check for duplicates and collect new points
        for i in range(new_x.shape[0]):
            if jnp.any(jnp.all(jnp.isclose(self.train_x, new_x[i], atol=1e-6, rtol=1e-4), axis=1)):
                log.debug(f"Point {new_x[i]} already exists in the training set, not updating")
            else:
                new_pts_to_add.append(new_x[i])
                new_vals_to_add.append(new_y[i])

        # Add new points if any
        if new_pts_to_add:
            new_pts_to_add = jnp.array(new_pts_to_add)
            new_vals_to_add = jnp.array(new_vals_to_add)

            # Add to training data
            self.train_x = jnp.vstack([self.train_x, new_pts_to_add])
            train_y_original = jnp.vstack([self.train_y * self.y_std + self.y_mean, new_vals_to_add])

            self.y_mean = jnp.mean(train_y_original)
            self.y_std = jnp.std(train_y_original)

            if self.y_std == 0:
                log.warning("Training targets have zero variance. Setting std to 1.0 to avoid division by zero.")
                self.y_std = 1.0

            self.train_y = (train_y_original - self.y_mean) / self.y_std

            self.recompute_cholesky()

    def recompute_cholesky(self):
        """
        Recomputes the Cholesky decomposition and alphas. Useful if hyperparameters are changed manually.
        """
        if self.train_x is None or self.train_x.shape[0] == 0:
            return
        K = self.kernel.covariance(self.train_x, self.train_x, include_noise=True)
        self.cholesky = jnp.linalg.cholesky(K)
        self.alphas = cho_solve((self.cholesky, True), self.train_y)

    def precompute_fantasy_cache(self, mc_points: jnp.ndarray) -> FantasyCache:
        """
        Precompute the GP posterior on the MC integration points once, so each
        candidate in ``fantasy_var`` only costs O(N^2 + N M).

        Costs one O(N^2 M) triangular solve plus O(N M^2) for the MC-point
        posterior covariance ``C``.

        Parameters
        ----------
        mc_points : jnp.ndarray, shape (M, D)

        Returns
        -------
        FantasyCache
            ``L``, ``X`` (current Cholesky factor and training inputs),
            ``V = L^{-1} k(X, mc_points)``, ``var0`` and ``C``.
        """
        return self._build_fantasy_cache(self.cholesky, jnp.asarray(self.train_x), mc_points)

    def _build_fantasy_cache(self, L, X, mc_points) -> FantasyCache:
        """Build a ``FantasyCache`` for ``mc_points`` given conditioning ``(L, X)``."""
        mc_points = jnp.atleast_2d(jnp.asarray(mc_points))
        k_x_mc = self.kernel.covariance(X, mc_points, include_noise=False)            # (N, M)
        V = solve_triangular(L, k_x_mc, lower=True)                                   # (N, M)
        var0 = self.kernel.diagonal(mc_points, include_noise=True) - jnp.sum(V * V, axis=0)
        C = self.kernel.covariance(mc_points, mc_points, include_noise=False) - V.T @ V
        return FantasyCache(L=L, X=X, V=V, var0=var0, C=C)

    def _cholesky_step(self, new_x, cache: FantasyCache):
        """
        Compute ``(v, s)`` for appending ``new_x`` (shape ``(1, D)``) to ``cache``:

        - ``v = L^{-1} k(X, x*)``          (new Cholesky row prefix)
        - ``s = k(x*, x*) + noise - v.v``  (predictive variance at ``x*``, floored)
        """
        k_x_new = self.kernel.covariance(cache.X, new_x, include_noise=False)[:, 0]   # (N',)
        v = solve_triangular(cache.L, k_x_new, lower=True)                            # O(N'^2)
        s = self.kernel.diagonal(new_x, include_noise=True)[0] - v @ v
        s = jnp.maximum(s, safe_noise_floor)
        return v, s

    def _fantasy_terms(self, new_x, mc_points, cache: FantasyCache):
        """
        Rank-1 Schur terms ``(v, s, c)`` for adding ``new_x`` (shape ``(1, D)``)
        to the conditioning set in ``cache``, where ``c = k(mc, x*) - V^T v`` is
        the posterior covariance between ``mc_points`` and ``new_x``.
        """
        v, s = self._cholesky_step(new_x, cache)
        c = self.kernel.covariance(new_x, mc_points, include_noise=False)[0] - v @ cache.V  # O(N' M)
        return v, s, c

    def fantasy_var(self, new_x, mc_points, cache: FantasyCache):
        """
        Variance of the GP at ``mc_points`` if a single point ``new_x`` were
        added to the training set.

        Uses the exact rank-1 Schur-complement update
        ``Var_new = var0 - c**2 / s`` (O(N^2 + N M) per candidate).

        Parameters
        ----------
        new_x : jnp.ndarray, shape (D,) or (1, D)
            Candidate point.
        mc_points : jnp.ndarray, shape (M, D)
            Integration points (the same ones the cache was built for).
        cache : FantasyCache
            Output of ``precompute_fantasy_cache(mc_points)``, optionally
            conditioned on fantasised points via ``condition_fantasy_cache``.

        Returns
        -------
        jnp.ndarray, shape (M,)
            Fantasy variances in physical (unstandardised) scale.
        """
        new_x = jnp.atleast_2d(new_x)
        _, s, c = self._fantasy_terms(new_x, mc_points, cache)
        return _clip_var(cache.var0 - c**2 / s) * self.y_std**2

    def fantasy_var_scan(self, mc_points: jnp.ndarray, cache: FantasyCache):
        """
        Fantasy variances with every MC point of ``cache`` as the candidate.

        Returns
        -------
        jnp.ndarray, shape (M, M)
            Row ``j`` equals ``fantasy_var(mc_points[j], mc_points, cache)``.
            Costs O(M^2) given the cache, versus M separate O(N^2 M) solves.
        """
        return _fantasy_scan_var(cache.C, cache.var0, self.y_std)

    def condition_fantasy_cache(
        self,
        cache: FantasyCache,
        new_x: jnp.ndarray,
        mc_points: Optional[jnp.ndarray] = None,
        *,
        new_mc_points: Optional[jnp.ndarray] = None,
    ) -> FantasyCache:
        """
        Condition the cached MC-point posterior on a fantasised observation at
        ``new_x`` (kriging believer), without copying the GP or refactorising.

        - If ``new_mc_points`` is provided, only ``(L, X)`` are extended (O(N^2))
          and ``(V, var0, C)`` are built directly for ``new_mc_points``
          (O(N'^2 M + N' M^2)), avoiding wasted work on the discarded subset.
        - Otherwise ``mc_points`` must be provided; one row is appended to
          ``L`` / ``X`` / ``V`` (O(N^2 + N M)) and ``var0`` / ``C`` are updated
          in place via rank-1 downdates (O(M^2)).
        """
        new_x = jnp.atleast_2d(jnp.asarray(new_x))
        X_new = jnp.vstack([cache.X, new_x])
        if new_mc_points is not None:
            v, s = self._cholesky_step(new_x, cache)
            L_new = _extend_cholesky_row(cache.L, v, jnp.sqrt(s))
            return self._build_fantasy_cache(L_new, X_new, new_mc_points)

        if mc_points is None:
            raise ValueError("Provide either mc_points or new_mc_points to condition_fantasy_cache.")
        mc_points = jnp.atleast_2d(jnp.asarray(mc_points))
        v, s, c = self._fantasy_terms(new_x, mc_points, cache)
        sqrt_s = jnp.sqrt(s)
        L_new = _extend_cholesky_row(cache.L, v, sqrt_s)
        return FantasyCache(
            L=L_new,
            X=X_new,
            V=jnp.vstack([cache.V, (c / sqrt_s)[None, :]]),
            var0=cache.var0 - c**2 / s,
            C=cache.C - jnp.outer(c, c) / s,
        )

    def get_random_point(self, rng=None, nstd=None):
        """
        Returns a random point in the unit cube.
        """
        log.debug("Getting random point in unit cube")
        rng = rng if rng is not None else get_numpy_rng()
        pt = rng.uniform(0, 1, size=self.train_x.shape[1])
        return pt

    def state_dict(self):
        """
        Returns a dictionary containing the complete state of the GP.
        This can be used for saving, loading, or copying the GP.

        Returns
        -------
        state : dict
            Dictionary containing all necessary information to reconstruct the GP.
        """
        return {
            # Training data (original, unstandardized)
            'train_x': np.array(self.train_x),
            'train_y': np.array(self.train_y * self.y_std + self.y_mean),

            # Standardization parameters
            'y_mean': float(self.y_mean),
            'y_std': float(self.y_std),

            # Kernel (owns all hyperparameters, bounds, priors, noise, transforms)
            'kernel_state': self.kernel.state_dict(),

            # GP-level configuration
            'optimizer_method': self.optimizer_method,
            'optimizer_options': self.optimizer_options,
            'param_names': self.param_names,

            # Computed state
            'cholesky': np.array(self.cholesky) if hasattr(self, 'cholesky') else None,
            'alphas': np.array(self.alphas) if hasattr(self, 'alphas') else None,

            # Dimensions & class identifier
            'ndim': self.ndim,
            'gp_class': 'GP',
        }

    @classmethod
    def from_state_dict(cls, state):
        """
        Create a GP instance from a state dictionary returned by ``state_dict()``.
        """
        kernel = Kernel.from_state_dict(state['kernel_state'])
        gp = cls(
            train_x=state['train_x'],
            train_y=state['train_y'],
            kernel=kernel,
            optimizer=state['optimizer_method'],
            optimizer_options=state['optimizer_options'],
            param_names=state.get('param_names'),
        )

        if state.get('cholesky') is not None and state.get('alphas') is not None:
            gp.cholesky = jnp.array(state['cholesky'])
            gp.alphas = jnp.array(state['alphas'])

        return gp

    @property
    def npoints(self):
        if self.train_x is None:
            return 0
        return self.train_x.shape[0]

    def get_hyperparams(self):
        return self.kernel.get_hyperparams()

    def get_hyperparam_bounds(self):
        return self.kernel.hyperparam_bounds

    def hyperparams_dict(self):
        return self.kernel.hyperparams_dict(self.param_names)
