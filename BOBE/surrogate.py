"""
Abstract base classes for BOBE surrogate models.

Defines the ``Surrogate`` ABC and the optional ``ClassifierMixin`` ABC that
together specify the full interface required by the BOBE optimisation loop.
Custom surrogate implementations should subclass ``Surrogate`` (and optionally
``ClassifierMixin``) and implement all abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np

SURROGATE_REGISTRY: Dict[str, type] = {}


class Surrogate(ABC):
    """
    Abstract base class defining what BOBE requires from a surrogate model.

    Subclass this to create a custom surrogate. All ``@abstractmethod``
    members must be implemented; instantiation will raise ``TypeError``
    otherwise. ``isinstance(obj, Surrogate)`` works via normal inheritance.

    Concrete subclasses must also set the following instance attributes
    during ``__init__`` (or deferred initialization):

    Attributes
    ----------
    train_x : jnp.ndarray, shape (N, D)
        Training inputs in the unit cube.
    train_y : jnp.ndarray, shape (N, 1)
        Standardised training targets.
    y_mean : float
        Mean of the raw (unstandardised) training targets.
    y_std : float
        Std of the raw (unstandardised) training targets.
    ndim : int
        Input dimensionality.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        SURROGATE_REGISTRY[cls.__name__] = cls

    def initialize(self, ndim=None, param_bounds=None, param_names=None) -> None:
        """
        Optional deferred initialization hook called by ``BOBE`` with problem
        dimensionality, parameter bounds, and parameter names before initial
        data are added. Default is a no-op.
        """
        return None

    # --- Properties ---

    @property
    @abstractmethod
    def npoints(self) -> int:
        """Number of training points currently in the surrogate."""
        ...

    # --- Predictions ---

    @abstractmethod
    def predict_mean(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Predict the mean in *physical* (unstandardised) space.

        Accepts either a single point ``(D,)`` (returning a scalar) or a batch
        ``(M, D)`` (returning shape ``(M,)``). Kept separate from ``predict``
        so samplers (NUTS / Nested Sampling) can evaluate means without paying
        the cost of predictive variances.

        Parameters
        ----------
        x : jnp.ndarray, shape (D,) or (M, D)

        Returns
        -------
        jnp.ndarray
            Scalar for 1D input, or shape ``(M,)`` for 2D input.
        """
        ...

    @abstractmethod
    def predict(
        self,
        x: jnp.ndarray,
        *,
        standardized: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict ``(mean, var)`` for a single point ``(D,)`` or batch ``(M, D)``.

        Parameters
        ----------
        x : jnp.ndarray, shape (D,) or (M, D)
        standardized : bool, optional
            If ``False`` (default), return mean and variance in physical
            (unstandardised) space. If ``True``, return standardised mean and
            variance (used by EI / LogEI acquisition functions).

        Returns
        -------
        (mean, var) : tuple of jnp.ndarray
            Scalars for 1D input, or arrays of shape ``(M,)`` for 2D input.
        """
        ...

    def predict_var(
        self,
        x: jnp.ndarray,
        *,
        standardized: bool = False,
    ) -> jnp.ndarray:
        """
        Predict variance for a single point ``(D,)`` or batch ``(M, D)``.

        Defaults to ``self.predict(x, standardized=standardized)[1]``.
        """
        return self.predict(x, standardized=standardized)[1]

    # --- Fantasy predictions for acquisition functions ---

    @abstractmethod
    def precompute_fantasy_cache(self, mc_points: jnp.ndarray) -> Any:
        """
        Precompute whatever the surrogate needs to efficiently evaluate
        ``fantasy_var`` for many candidate points against the same set of
        MC integration points.

        For an exact GP this is ``kernel(train_x, mc_points)`` (the
        cross-covariance matrix).  For a sparse GP it might be
        inducing-point cross-covariances.  For a non-GP surrogate this
        can return ``None``.

        Parameters
        ----------
        mc_points : jnp.ndarray, shape (M, D)

        Returns
        -------
        Any
            Opaque cache object passed straight through to ``fantasy_var``.
        """
        ...

    @abstractmethod
    def fantasy_var(
        self,
        new_x: jnp.ndarray,
        mc_points: jnp.ndarray,
        cache: Any,
    ) -> jnp.ndarray:
        """
        Predictive variance at *mc_points* after hypothetically adding *new_x*
        to the training set (fantasy / one-step-lookahead prediction).

        Parameters
        ----------
        new_x : jnp.ndarray, shape (D,) or (1, D)
            Candidate point to add.
        mc_points : jnp.ndarray, shape (M, D)
            Points at which to evaluate the fantasy variance.
        cache : Any
            Output of ``precompute_fantasy_cache(mc_points)``.

        Returns
        -------
        jnp.ndarray, shape (M,)
            Fantasy variances in *physical* scale.
        """
        ...

    @abstractmethod
    def get_random_point(self, rng=None, nstd=None) -> np.ndarray:
        """
        Return a random point in the unit cube, optionally biased toward
        high-likelihood regions (used for acquisition warm-starts and
        sampler initialisation).

        Parameters
        ----------
        rng : numpy.random.Generator, optional
        nstd : float, optional
            Number of std deviations for biased sampling.

        Returns
        -------
        np.ndarray, shape (D,)
        """
        ...

    # --- Mutation ---

    @abstractmethod
    def update(self, new_x: jnp.ndarray, new_y: jnp.ndarray) -> None:
        """
        Add new training points, re-standardise targets, and recompute
        internal caches (e.g. Cholesky factor).
        """
        ...

    @abstractmethod
    def fit(self, x0: np.ndarray = None, maxiter: int = 500) -> dict:
        """
        Optimise hyperparameters.

        Returns
        -------
        dict
            Must contain at least ``'params'`` (best hyperparameters in log
            space) and ``'mll'`` (marginal log-likelihood at those params).
        """
        ...

    @abstractmethod
    def update_hyperparams(self, hyperparams: dict) -> None:
        """Set hyperparameters from a vector returned by ``fit()``."""
        ...

    # --- Serialisation ---

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Return a complete, serialisable state dict for checkpointing."""
        ...

    @classmethod
    @abstractmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "Surrogate":
        """
        Reconstruct a surrogate instance from a state dict returned by
        ``state_dict()``.

        When called directly on ``Surrogate``, dispatches to the concrete
        subclass recorded in ``state['gp_class']``.
        """
        if cls is Surrogate:
            gp_class = state.get("gp_class")
            target_cls = SURROGATE_REGISTRY.get(gp_class)
            if target_cls is None:
                raise ValueError(
                    f"Unknown surrogate class '{gp_class}'. "
                    f"Registered surrogates: {list(SURROGATE_REGISTRY)}."
                )
            return target_cls.from_state_dict(state)
        raise NotImplementedError

    def save(self, filename: str = "gp") -> None:
        """Save the surrogate state to an ``.npz`` file using ``state_dict()``."""
        if not filename.endswith(".npz"):
            filename += ".npz"
        np.savez(filename, **self.state_dict())

    @classmethod
    def load(cls, filename: str, **kwargs) -> "Surrogate":
        """
        Load a surrogate from an ``.npz`` file saved with ``save()``.

        When called on ``Surrogate``, automatically dispatches to the concrete
        subclass recorded in the file's ``gp_class`` field.
        """
        if not filename.endswith(".npz"):
            filename += ".npz"
        try:
            data = np.load(filename, allow_pickle=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file {filename}")

        state = {}
        for key in data.files:
            value = data[key]
            if isinstance(value, np.ndarray) and value.shape == ():
                state[key] = value.item()
            else:
                state[key] = value
        state.update(kwargs)
        return cls.from_state_dict(state)

    def copy(self) -> "Surrogate":
        """Create an independent deep copy via ``state_dict()``."""
        return self.__class__.from_state_dict(self.state_dict())

    @abstractmethod
    def hyperparams_dict(self) -> Dict[str, Any]:
        """Return human-readable hyperparameters for logging."""
        ...

    @abstractmethod
    def get_hyperparams(self) -> jnp.ndarray:
        """Return current hyperparameters as a flat array (for fit restarts)."""
        ...

    @abstractmethod
    def get_hyperparam_bounds(self) -> jnp.ndarray:
        """Return log-space bounds for all optimisable hyperparameters, shape ``(2, P)``."""
        ...


class ClassifierMixin(ABC):
    """
    Optional abstract mixin for surrogates that embed a classifier.

    BOBE detects classifier support via
    ``isinstance(gp, ClassifierMixin)`` or ``hasattr(gp, 'train_classifier')``.

    Concrete subclasses must set the following instance attributes:

    Attributes
    ----------
    use_clf : bool
    clf_type : str
    clf_data_size : int
    clf_use_size : int
    probability_threshold : float
    """

    @abstractmethod
    def train_classifier(self) -> None:
        """Train or retrain the internal classifier."""
        ...
