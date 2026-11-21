"""
Kernel implementations for Gaussian Process models.

All kernels inherit from the base Kernel class and implement the covariance() method.
JAX JIT compilation is handled at higher levels (acquisition functions, optimization).
"""

from abc import ABC, abstractmethod
from math import sqrt
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from .transforms import PrincipalAxesTransform

KERNEL_REGISTRY = {}

# ----------------------------------------------------------------------
# Hyperparameter helpers (shared by Kernel and any caller that needs them)
# ----------------------------------------------------------------------


class DummyDistribution:
    """A dummy distribution that always returns log_prob = 0.0."""
    def log_prob(self, x):
        return 0.0


def make_distribution(spec: dict) -> dist.Distribution:
    """
    Turn a dictionary specification into a NumPyro distribution.

    Parameters
    ----------
    spec : dict
        Dictionary with a ``'name'`` key (e.g. ``'Normal'``) plus any
        keyword arguments accepted by that distribution.

    Returns
    -------
    dist.Distribution
    """
    dist_class = getattr(dist, spec["name"], None)
    if dist_class is None:
        raise ValueError(f"Distribution {spec['name']} not found in numpyro.distributions.")
    kwargs = {k: v for k, v in spec.items() if k != "name"}
    return dist_class(**kwargs)


def saas_prior_logprob(lengthscales, kernel_variance, tausq):
    """Log probability under SAAS priors."""
    logprior = dist.LogNormal(0., 1.).log_prob(kernel_variance)
    logprior += dist.HalfCauchy(0.1).log_prob(tausq)
    inv_lengthscales_sq = 1 / (tausq * lengthscales ** 2)
    logprior += jnp.sum(dist.HalfCauchy(1.).log_prob(inv_lengthscales_sq))
    return logprior


# Constants for Matérn kernel
sqrt2 = sqrt(2.)
sqrt3 = sqrt(3.)
sqrt5 = sqrt(5.)


def rbf_from_sqdist(dsq):
    """Unit-variance RBF correlation from squared scaled distances."""
    return jnp.exp(-0.5 * dsq)


def matern52_from_sqdist(dsq):
    """Unit-variance Matérn-5/2 correlation from squared scaled distances."""
    # Safe sqrt to avoid an infinite gradient at zero distance
    d = jnp.sqrt(jnp.where(dsq < 1e-30, 1e-30, dsq))
    return (1. + d * (sqrt5 + d * 5. / 3.)) * jnp.exp(-sqrt5 * d)


STATIONARY_CORRELATIONS = {"rbf": rbf_from_sqdist, "matern": matern52_from_sqdist}

# Relative tolerance for initial hyperparameters at a bound (log-space round-off).
BOUND_RTOL = 1e-9


def _initial_in_bounds(name, value, bounds):
    """Validate an initial positive hyperparameter (scalar or array).

    Fitted values come back as exp(log-space optimum); one at a bound can
    exceed it by round-off (e.g. 100.00000000000004 for 100). Accept that and
    clip; reject only values genuinely outside.
    """
    lo, hi = map(float, bounds)
    if not 0 < lo < hi:
        raise ValueError(f"{name}_bounds require 0 < lower < upper.")
    value = np.asarray(value, dtype=float)
    if not np.all((lo * (1 - BOUND_RTOL) <= value) & (value <= hi * (1 + BOUND_RTOL))):
        raise ValueError(f"Initial {name}={value} lies outside its bounds {list(bounds)}.")
    return np.clip(value, lo, hi)


class Kernel(ABC):
    """
    Abstract base class for all kernels in BOBE.

    Each kernel owns its own hyperparameters (including observation noise),
    bounds, prior specifications, optional kernel-metric input transform,
    and state serialization.

    When constructed without ``lengthscales`` or ``ndim``, dimension-dependent
    setup (default lengthscales, dimension-dependent priors such as DSLP, and
    log-space optimization bounds) is deferred until ``initialize(ndim)`` is
    called (e.g. automatically by ``GP`` once training data arrive).

    Subclasses with additional positive hyperparameters list their attribute
    names in ``extra_hyperparam_names`` and set ``extra_hyperparam_bounds``
    (name -> [lo, hi]). They are optimised in log space after
    ``kernel_variance``, with a flat (bounded uniform) prior unless
    ``extra_log_prior`` is overridden. A vector-valued one also sets
    ``extra_hyperparam_sizes[name]`` to its length; all its entries share the
    bounds.
    """

    extra_hyperparam_names = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        KERNEL_REGISTRY[cls.__name__] = cls

    def __init__(
        self,
        lengthscales=None,
        kernel_variance: float = 1.0,
        noise: float = 1e-6,
        *,
        lengthscale_bounds=(0.01, 5.0),
        kernel_variance_bounds=(1e-4, 1e4),
        noise_bounds=(1e-8, 1e1),
        lengthscale_prior=None,
        kernel_variance_prior=None,
        noise_prior="fixed",
        tausq: float = 1.0,
        tausq_bounds=(1e-4, 1e4),
        input_transform=None,
        ndim: int = None,
    ):
        """
        Initialize kernel with hyperparameters, bounds, priors, and optional
        kernel-metric transform.

        Parameters
        ----------
        lengthscales : array-like, optional
            Initial lengthscales for each input dimension, shape ``(D,)``.
            If ``None``, defaults to ``jnp.ones(ndim)`` when initialized.
        kernel_variance : float, optional
            Initial kernel variance/amplitude parameter. Default is 1.0.
        noise : float, optional
            Observation noise level added to the diagonal. Default is 1e-6.
        lengthscale_bounds : sequence of two floats, optional
            ``[lo, hi]`` bounds for lengthscales. Default is ``(0.01, 5.0)``.
        kernel_variance_bounds : sequence of two floats, optional
            ``[lo, hi]`` bounds for kernel variance. Default is ``(1e-4, 1e4)``.
        noise_bounds : sequence of two floats, optional
            ``[lo, hi]`` bounds for observation noise when ``noise_prior != 'fixed'``.
            Default is ``(1e-8, 1e1)``.
        lengthscale_prior : str, dict, or None, optional
            ``None`` -> Uniform over ``lengthscale_bounds``.
            ``'DSLP'`` -> DSLP log-normal prior.
            ``'SAAS'`` -> SAAS prior (uses ``tausq``).
            dict -> passed to ``make_distribution``.
        kernel_variance_prior : str, dict, or None, optional
            ``None`` -> Uniform over ``kernel_variance_bounds``.
            ``'fixed'`` -> kernel variance is not optimized.
            dict -> passed to ``make_distribution``.
        noise_prior : str, dict, or None, optional
            ``'fixed'`` (default) -> noise is not optimized.
            ``None`` -> Uniform over ``noise_bounds``.
            dict -> passed to ``make_distribution``.
        tausq : float, optional
            Initial global shrinkage parameter for SAAS prior. Default is 1.0.
        tausq_bounds : sequence of two floats, optional
            ``[lo, hi]`` bounds for ``tausq``. Default is ``(1e-4, 1e4)``.
        input_transform : InputTransform, optional
            Transform applied only while evaluating the kernel metric
            (e.g. ``PrincipalAxesTransform``).
        ndim : int, optional
            Input dimensionality. If provided (or inferred from ``lengthscales``),
            dimension-dependent setup completes immediately; otherwise it is
            deferred until ``initialize(ndim)`` is called.
        """
        self._lengthscales_init = (
            None if lengthscales is None else jnp.asarray(lengthscales, dtype=float)
        )
        if self._lengthscales_init is not None and self._lengthscales_init.ndim != 1:
            raise ValueError(
                f"lengthscales must be a 1D array, got shape {self._lengthscales_init.shape}."
            )

        self.kernel_variance = float(kernel_variance)
        self.noise = float(noise)
        self.lengthscale_bounds = [float(b) for b in lengthscale_bounds]
        self.kernel_variance_bounds = [float(b) for b in kernel_variance_bounds]
        self.noise_bounds = [float(b) for b in noise_bounds]
        self.tausq = float(tausq)
        self.tausq_bounds = [float(b) for b in tausq_bounds]

        self._lengthscale_prior_arg = lengthscale_prior
        self._kernel_variance_prior_arg = kernel_variance_prior
        self._noise_prior_arg = noise_prior

        self.lengthscale_prior_spec = lengthscale_prior
        self.kernel_variance_prior_spec = kernel_variance_prior
        self.noise_prior_spec = noise_prior
        self.fixed_kernel_variance = (kernel_variance_prior == "fixed")
        self.fixed_noise = (noise_prior == "fixed")

        self.input_transform = input_transform
        self.lengthscales = None
        self.ndim = None
        self._initialized = False

        if self._lengthscales_init is not None:
            inferred_ndim = len(self._lengthscales_init)
            if ndim is not None and int(ndim) != inferred_ndim:
                raise ValueError(
                    f"lengthscales has length {inferred_ndim}, which does not match ndim={ndim}."
                )
            self.initialize(inferred_ndim)
        elif ndim is not None:
            self.initialize(int(ndim))

    def _validate_input_transform(self, ndim):
        """Validate that an attached (bound) input transform matches ``ndim``."""
        transform = self.input_transform
        if transform is None or getattr(transform, "rotation", None) is None:
            return
        rotation_dims = getattr(transform, "rotation_dims", None)
        if rotation_dims is not None and len(rotation_dims) > 0:
            if int(jnp.max(rotation_dims)) >= ndim:
                raise ValueError(
                    "rotation_dims contains indices outside the GP input dimensions"
                )
        expected_ndim = ndim if rotation_dims is None else len(rotation_dims)
        if transform.rotation.shape != (expected_ndim, expected_ndim):
            raise ValueError(
                f"Rotation dimensions are incompatible with the GP input dimensions. "
                f"Expected rotation shape {(expected_ndim, expected_ndim)}, got {transform.rotation.shape}."
            )

    def bind_space(self, param_bounds, param_names=None):
        """
        Bind any deferred physical-space kernel transform using likelihood bounds
        and parameter names. No-op if no transform requires binding.
        """
        if self.input_transform is not None and hasattr(self.input_transform, "bind_space"):
            self.input_transform.bind_space(param_bounds, param_names=param_names)
            if self.ndim is not None:
                self._validate_input_transform(self.ndim)

    def initialize(self, ndim: int):
        """
        Complete dimension-dependent initialization (default lengthscales,
        priors, and optimization bounds). Safe to call repeatedly with the
        same ``ndim``.
        """
        ndim = int(ndim)
        if self._initialized:
            if self.ndim != ndim:
                raise ValueError(
                    f"Kernel was initialized for ndim={self.ndim}, cannot use with ndim={ndim}."
                )
            self._validate_input_transform(ndim)
            return

        self.ndim = ndim
        if self._lengthscales_init is not None:
            if len(self._lengthscales_init) != ndim:
                raise ValueError(
                    f"lengthscales has length {len(self._lengthscales_init)}, expected {ndim}."
                )
            self.lengthscales = jnp.array(self._lengthscales_init, dtype=float)
        else:
            self.lengthscales = jnp.ones(ndim, dtype=float)

        self._validate_input_transform(ndim)
        self._setup_kernel_variance_prior(self._kernel_variance_prior_arg)
        self._setup_noise_prior(self._noise_prior_arg)
        self._setup_lengthscale_prior(self._lengthscale_prior_arg, ndim)
        self._setup_optimization_parameters(ndim)
        self._initialized = True

    def set_input_transform(self, transform):
        """Set a transform applied only while evaluating the kernel."""
        self.input_transform = transform
        if self.ndim is not None:
            self._validate_input_transform(self.ndim)

    def pre_transform(self, x):
        """Hyperparameter-dependent input map applied before the input transform."""
        return x

    def _transform_inputs(self, x):
        x = self.pre_transform(x)
        if self.input_transform is None:
            return x
        return self.input_transform.forward(x)

    def extra_log_prior(self):
        """Log prior of the extra hyperparameters (flat by default)."""
        return 0.0

    def sq_dist(self, xa, xb):
        """
        Compute squared Euclidean distance between two sets of points.

        This utility method is used by many kernel implementations.

        Parameters
        ----------
        xa : jnp.ndarray
            First set of points, shape (n1, D)
        xb : jnp.ndarray
            Second set of points, shape (n2, D)

        Returns
        -------
        sq_dist : jnp.ndarray
            Squared distances, shape (n1, n2)
        """
        return jnp.sum(jnp.square(xa[:, None, :] - xb[None, :, :]), axis=-1)

    @abstractmethod
    def covariance(self, xa, xb, include_noise=True):
        """
        Compute covariance matrix between two sets of points.

        Parameters
        ----------
        xa : jnp.ndarray
            First set of points, shape (n1, D)
        xb : jnp.ndarray
            Second set of points, shape (n2, D)
        include_noise : bool, optional
            Whether to add noise to diagonal (only when xa is xb). Default is True.

        Returns
        -------
        K : jnp.ndarray
            Covariance matrix of shape (n1, n2)
        """
        pass

    def diagonal(self, x, include_noise=True):
        """
        Compute only the diagonal of the kernel matrix K(x,x).

        For stationary kernels, the diagonal is constant: kernel_variance (+ noise).
        Override this method if your kernel has a non-constant diagonal.

        Parameters
        ----------
        x : jnp.ndarray
            Points at which to compute diagonal, shape (n, D)
        include_noise : bool, optional
            Whether to include noise in diagonal. Default is True.

        Returns
        -------
        diag : jnp.ndarray
            Diagonal values, shape (n,)
        """
        diag = self.kernel_variance * jnp.ones(x.shape[0])
        if include_noise:
            diag += self.noise
        return diag

    def update_hyperparams(self, lengthscales=None, kernel_variance=None, noise=None):
        """
        Update kernel hyperparameters.

        Parameters
        ----------
        lengthscales : jnp.ndarray, optional
            New lengthscale values
        kernel_variance : float, optional
            New kernel variance
        noise : float, optional
            New noise level
        """
        if lengthscales is not None:
            self.lengthscales = jnp.array(lengthscales)
        if kernel_variance is not None:
            self.kernel_variance = kernel_variance
        if noise is not None:
            self.noise = noise

    def __call__(self, xa, xb, include_noise=True):
        """Convenience method - same as covariance()"""
        return self.covariance(xa, xb, include_noise=include_noise)

    # ------------------------------------------------------------------
    # Hyperparameter management
    # ------------------------------------------------------------------

    def _setup_kernel_variance_prior(self, spec):
        self.kernel_variance_prior_spec = spec
        if self.kernel_variance_prior_spec is None:
            self.kernel_variance_prior_spec = {
                "name": "Uniform",
                "low": self.kernel_variance_bounds[0],
                "high": self.kernel_variance_bounds[1],
            }
        self.fixed_kernel_variance = (self.kernel_variance_prior_spec == "fixed")
        if not self.fixed_kernel_variance:
            self.kernel_variance_prior_dist = make_distribution(self.kernel_variance_prior_spec)
        else:
            self.kernel_variance_prior_dist = DummyDistribution()

    def _setup_noise_prior(self, spec):
        self.noise_prior_spec = spec
        self.fixed_noise = (self.noise_prior_spec == "fixed")
        if self.fixed_noise:
            self.noise_prior_dist = DummyDistribution()
            return
        if self.noise_prior_spec is None:
            self.noise_prior_spec = {
                "name": "Uniform",
                "low": self.noise_bounds[0],
                "high": self.noise_bounds[1],
            }
        self.noise_prior_dist = make_distribution(self.noise_prior_spec)

    def _setup_lengthscale_prior(self, spec, ndim):
        self.lengthscale_prior_spec = spec
        if self.lengthscale_prior_spec is None:
            self.lengthscale_prior_spec = {
                "name": "Uniform",
                "low": self.lengthscale_bounds[0],
                "high": self.lengthscale_bounds[1],
            }
        if self.lengthscale_prior_spec == "DSLP":
            self.lengthscale_prior_dist = dist.LogNormal(
                loc=sqrt2 + 0.5 * jnp.log(ndim), scale=sqrt3
            )
            self.prior_func = self._standard_prior_logprob
        elif self.lengthscale_prior_spec == "SAAS":
            self.lengthscale_prior_dist = None
            self.prior_func = self._saas_prior_logprob
        else:
            self.lengthscale_prior_dist = make_distribution(self.lengthscale_prior_spec)
            self.prior_func = self._standard_prior_logprob

    def _setup_optimization_parameters(self, ndim):
        self.hyperparam_names = ["lengthscales"]
        bounds = [self.lengthscale_bounds] * ndim
        if not self.fixed_kernel_variance:
            self.hyperparam_names.append("kernel_variance")
            bounds.append(self.kernel_variance_bounds)
        for name in self.extra_hyperparam_names:
            self.hyperparam_names.append(name)
            bounds.extend([self.extra_hyperparam_bounds[name]] * self._extra_size(name))
        if not self.fixed_noise:
            self.hyperparam_names.append("noise")
            bounds.append(self.noise_bounds)
        if self.lengthscale_prior_spec == "SAAS":
            self.hyperparam_names.append("tausq")
            bounds.append(self.tausq_bounds)
        self.hyperparam_bounds = jnp.log(jnp.array(bounds).T)
        self.num_hyperparams = self.hyperparam_bounds.shape[1]

    # --- prior functions --------------------------------------------------

    def _standard_prior_logprob(self, lengthscales, kernel_variance, tausq=None, noise=None):
        logprior = self.kernel_variance_prior_dist.log_prob(kernel_variance)
        if self.lengthscale_prior_dist is not None:
            logprior += self.lengthscale_prior_dist.log_prob(lengthscales).sum()
        if not self.fixed_noise:
            logprior += self.noise_prior_dist.log_prob(noise)
        return logprior

    def _saas_prior_logprob(self, lengthscales, kernel_variance, tausq, noise=None):
        logprior = saas_prior_logprob(lengthscales, kernel_variance, tausq)
        if not self.fixed_noise:
            logprior += self.noise_prior_dist.log_prob(noise)
        return logprior

    def log_prior(self):
        """Return log-prior evaluated at the current hyperparameter values."""
        return (
            self.prior_func(self.lengthscales, self.kernel_variance, self.tausq, self.noise)
            + self.extra_log_prior()
        )

    def _extra_size(self, name):
        """Number of entries of an extra hyperparameter (1 for scalars)."""
        return getattr(self, "extra_hyperparam_sizes", {}).get(name, 1)

    # --- flat-vector hyperparameter interface ----------------------------

    def get_hyperparams(self):
        """Return the current optimisable hyperparameters as a flat jnp array."""
        hp = self.lengthscales
        if not self.fixed_kernel_variance:
            hp = jnp.hstack([hp, self.kernel_variance])
        for name in self.extra_hyperparam_names:
            hp = jnp.hstack([hp, getattr(self, name)])
        if not self.fixed_noise:
            hp = jnp.hstack([hp, self.noise])
        if self.lengthscale_prior_spec == "SAAS":
            hp = jnp.hstack([hp, self.tausq])
        return hp

    def set_hyperparams(self, log_params):
        """
        Parse a flat vector of log-space hyperparameters and update the
        kernel's internal state (``lengthscales``, ``kernel_variance``,
        any extra hyperparameters, ``noise``, and ``tausq``).

        Parameters
        ----------
        log_params : jnp.ndarray
            Log-space parameter vector as produced by the optimiser.
        """
        ndim = len(self.lengthscales)
        hyperparams = jnp.exp(log_params)
        self.lengthscales = hyperparams[:ndim]
        idx = ndim
        if not self.fixed_kernel_variance:
            self.kernel_variance = hyperparams[idx]
            idx += 1
        for name in self.extra_hyperparam_names:
            size = self._extra_size(name)
            if name in getattr(self, "extra_hyperparam_sizes", {}):
                setattr(self, name, hyperparams[idx:idx + size])
            else:
                setattr(self, name, hyperparams[idx])
            idx += size
        if not self.fixed_noise:
            self.noise = hyperparams[idx]
            idx += 1
        if self.lengthscale_prior_spec == "SAAS":
            self.tausq = hyperparams[idx]

    # --- human-readable dict ---------------------------------------------

    def hyperparams_dict(self, param_names=None):
        """
        Return a human-readable dictionary of the current hyperparameters.

        Parameters
        ----------
        param_names : list of str, optional
            Names for each lengthscale dimension.
        """
        ndim = len(self.lengthscales)
        if param_names is None:
            param_names = [f"x_{i}" for i in range(ndim)]
        ls_str = {
            name: f"{float(val):.4f}"
            for name, val in zip(param_names, self.lengthscales)
        }
        d = {
            "lengthscales": ls_str,
            "kernel_variance": f"{float(self.kernel_variance):.4f}",
        }
        for name in self.extra_hyperparam_names:
            if name in getattr(self, "extra_hyperparam_sizes", {}):
                d[name] = {n: f"{float(v):.4g}" for n, v in zip(param_names, getattr(self, name))}
            else:
                d[name] = f"{float(getattr(self, name)):.4g}"
        if not self.fixed_noise:
            d["noise"] = f"{float(self.noise):.4g}"
        if self.lengthscale_prior_spec == "SAAS":
            d["tausq"] = f"{float(self.tausq):.4f}"
        return d

    # --- serialization ---------------------------------------------------

    def state_dict(self):
        """Return a serializable dictionary capturing the kernel configuration and state."""
        return {
            "kernel_class": type(self).__name__,
            "ndim": self.ndim,
            "lengthscales": (
                np.array(self.lengthscales) if self.lengthscales is not None else None
            ),
            "kernel_variance": float(self.kernel_variance),
            "noise": float(self.noise),
            "tausq": float(self.tausq),
            "lengthscale_bounds": list(self.lengthscale_bounds),
            "kernel_variance_bounds": list(self.kernel_variance_bounds),
            "noise_bounds": list(self.noise_bounds),
            "tausq_bounds": list(self.tausq_bounds),
            "lengthscale_prior": (
                self.lengthscale_prior_spec
                if self._initialized
                else self._lengthscale_prior_arg
            ),
            "kernel_variance_prior": (
                self.kernel_variance_prior_spec
                if self._initialized
                else self._kernel_variance_prior_arg
            ),
            "noise_prior": (
                self.noise_prior_spec if self._initialized else self._noise_prior_arg
            ),
            "input_transform_state": (
                self.input_transform.state_dict()
                if self.input_transform is not None
                else None
            ),
        }

    @classmethod
    def from_state_dict(cls, state):
        """Reconstruct a Kernel instance from a state dict produced by ``state_dict()``."""
        kernel_class_name = state["kernel_class"]
        target_cls = KERNEL_REGISTRY.get(kernel_class_name)
        if target_cls is None:
            raise ValueError(
                f"Unknown kernel class '{kernel_class_name}'. "
                f"Registered kernels: {list(KERNEL_REGISTRY)}."
            )

        transform_state = state.get("input_transform_state")
        transform = None
        if transform_state is not None:
            if transform_state["type"] != "PrincipalAxesTransform":
                raise ValueError(f"Unknown input transform '{transform_state['type']}'.")
            transform = PrincipalAxesTransform.from_state_dict(transform_state)

        return target_cls(
            lengthscales=state.get("lengthscales"),
            kernel_variance=state["kernel_variance"],
            noise=state["noise"],
            lengthscale_bounds=state["lengthscale_bounds"],
            kernel_variance_bounds=state["kernel_variance_bounds"],
            noise_bounds=state["noise_bounds"],
            lengthscale_prior=state.get("lengthscale_prior"),
            kernel_variance_prior=state.get("kernel_variance_prior"),
            noise_prior=state.get("noise_prior", "fixed"),
            tausq=state.get("tausq", 1.0),
            tausq_bounds=state.get("tausq_bounds", (1e-4, 1e4)),
            input_transform=transform,
            ndim=state.get("ndim"),
        )


class RBFKernel(Kernel):
    """
    Radial Basis Function (RBF) / Squared Exponential kernel.

    k(x, x') = σ² * exp(-0.5 * ||x - x'||²/ℓ²)

    where σ² is kernel_variance and ℓ is lengthscale.
    """

    def covariance(self, xa, xb, include_noise=True):
        """
        Compute RBF covariance matrix.

        Parameters
        ----------
        xa : jnp.ndarray
            First set of input points, shape (n1, d).
        xb : jnp.ndarray
            Second set of input points, shape (n2, d).
        include_noise : bool, optional
            Whether to include noise on diagonal. Default is True.

        Returns
        -------
        jnp.ndarray
            Kernel matrix of shape (n1, n2).
        """
        xa = self._transform_inputs(xa)
        xb = self._transform_inputs(xb)

        # Scale inputs by lengthscales
        xa_scaled = xa / self.lengthscales
        xb_scaled = xb / self.lengthscales

        # Compute squared distances
        sq_dist = self.sq_dist(xa_scaled, xb_scaled)

        # Apply RBF kernel
        K = self.kernel_variance * rbf_from_sqdist(sq_dist)

        # Add noise to diagonal if needed
        if include_noise:
            K += self.noise * jnp.eye(K.shape[0])

        return K


class MaternKernel(Kernel):
    """
    Matérn-5/2 kernel.

    k(x, x') = σ² * (1 + √5*d + 5*d²/3) * exp(-√5*d)

    where d = ||x - x'||/ℓ, σ² is kernel_variance, and ℓ is lengthscale.
    """

    def covariance(self, xa, xb, include_noise=True):
        """
        Compute Matérn-5/2 covariance matrix.

        Parameters
        ----------
        xa : jnp.ndarray
            First set of input points, shape (n1, d).
        xb : jnp.ndarray
            Second set of input points, shape (n2, d).
        include_noise : bool, optional
            Whether to include noise on diagonal. Default is True.

        Returns
        -------
        jnp.ndarray
            Kernel matrix of shape (n1, n2).
        """
        xa = self._transform_inputs(xa)
        xb = self._transform_inputs(xb)

        # Scale inputs by lengthscales
        xa_scaled = xa / self.lengthscales
        xb_scaled = xb / self.lengthscales

        # Compute squared distances
        dsq = self.sq_dist(xa_scaled, xb_scaled)

        # Matérn-5/2 formula (safe sqrt inside the helper)
        K = self.kernel_variance * matern52_from_sqdist(dsq)

        # Add noise to diagonal if needed
        if include_noise:
            K += self.noise * jnp.eye(K.shape[0])

        return K
