"""
Feasibility classifiers for ClassifierGP.

Provides a common ``Classifier`` ABC with auto-registration and three concrete
implementations:

- ``SVMClassifier``: RBF-kernel scikit-learn SVC with pure-JAX prediction.
- ``MLPClassifier``: Equinox MLP trained with Optax (requires ``BOBE[nn]``).
- ``EllipsoidClassifier``: Pure-JAX quadratic/ellipsoidal boundary trained with
  Optax (requires ``optax``).

All classifiers expose ``fit(x, y, **kwargs)``, pure-JAX ``predict_proba(x)``
(supporting both 1D ``(D,)`` and 2D ``(M, D)`` inputs inside ``jax.jit`` /
``jax.vmap`` / ``jax.grad``), and ``state_dict()`` / ``from_state_dict()``.
"""

from abc import ABC, abstractmethod
import io
from typing import Any, Dict, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.svm import SVC

from .utils.log import get_logger
from .utils.seed import get_new_jax_key, get_numpy_rng

log = get_logger("clf")

try:
    import optax
    OPTAX_AVAILABLE = True
except ImportError:
    OPTAX_AVAILABLE = False
    optax = None
    log.debug("optax is not available. MLPClassifier and EllipsoidClassifier require it.")

try:
    import equinox as eqx
    EQX_AVAILABLE = True
except ImportError:
    EQX_AVAILABLE = False
    eqx = None
    log.debug("Equinox is not available. MLPClassifier requires it.")


CLASSIFIER_REGISTRY: Dict[str, type] = {}


class Classifier(ABC):
    """Abstract base class for binary feasibility classifiers used by ``ClassifierGP``."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        CLASSIFIER_REGISTRY[cls.__name__] = cls

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the classifier has been fitted and can make predictions."""
        ...

    @abstractmethod
    def fit(self, x: jnp.ndarray, y: jnp.ndarray, **kwargs) -> Dict[str, Any]:
        """Fit the classifier on unit-cube points ``x`` and binary labels ``y`` (0/1).

        Returns a dictionary of training metrics (also stored in ``self.metrics``).
        """
        ...

    @abstractmethod
    def _predict_proba_single(self, x: jnp.ndarray) -> jnp.ndarray:
        """Pure-JAX feasibility probability for a single point ``x`` of shape ``(D,)``."""
        ...

    def predict_proba(self, x: jnp.ndarray) -> jnp.ndarray:
        """Predict feasibility probability for a single point ``(D,)`` or batch ``(M, D)``."""
        if not self.is_fitted:
            raise RuntimeError(f"{type(self).__name__} has not been fitted yet.")
        x = jnp.asarray(x)
        if x.ndim == 1:
            return self._predict_proba_single(x)
        return jax.vmap(self._predict_proba_single)(x)

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Return a serializable dictionary capturing configuration and fitted state."""
        ...

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "Classifier":
        """Reconstruct a ``Classifier`` from a dict returned by ``state_dict()``.

        When called on ``Classifier``, dispatches to the concrete subclass recorded
        in ``state['classifier_class']``.
        """
        if cls is Classifier:
            clf_class = state.get("classifier_class")
            target_cls = CLASSIFIER_REGISTRY.get(clf_class)
            if target_cls is None:
                raise ValueError(
                    f"Unknown classifier class '{clf_class}'. "
                    f"Registered classifiers: {list(CLASSIFIER_REGISTRY)}."
                )
            return target_cls.from_state_dict(state)
        raise NotImplementedError


# -----------------------------------------------------------------------------
# SVMClassifier (scikit-learn SVC fit + pure-JAX RBF prediction)
# -----------------------------------------------------------------------------


class SVMClassifier(Classifier):
    """RBF-kernel Support Vector Machine classifier with pure-JAX prediction.

    Parameters
    ----------
    kernel : str, optional
        SVC kernel type. Only ``'rbf'`` is supported by the JAX evaluator.
    gamma : str or float, optional
        Kernel coefficient passed to ``sklearn.svm.SVC``. Default is ``'scale'``.
    C : float, optional
        Regularization parameter passed to ``sklearn.svm.SVC``. Default is ``1e7``.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: Union[str, float] = "scale",
        C: float = 1e7,
    ):
        if kernel != "rbf":
            raise ValueError(
                f"SVMClassifier only supports kernel='rbf', got '{kernel}'."
            )
        self.kernel = kernel
        self.gamma = gamma
        self.C = float(C)

        self.support_vectors: Optional[jnp.ndarray] = None
        self.dual_coef: Optional[jnp.ndarray] = None
        self.intercept: Optional[float] = None
        self.gamma_eff: Optional[float] = None
        self.metrics: Dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return self.support_vectors is not None

    def fit(self, x: jnp.ndarray, y: jnp.ndarray, **kwargs) -> Dict[str, Any]:
        x_np = np.asarray(x)
        y_np = np.asarray(y).reshape(-1)

        rng = get_numpy_rng()
        random_state = int(rng.integers(0, 2**31 - 1))
        clf = SVC(
            kernel=self.kernel,
            gamma=self.gamma,
            C=self.C,
            random_state=random_state,
        )
        clf.fit(x_np, y_np)

        self.support_vectors = jnp.asarray(clf.support_vectors_)
        self.dual_coef = jnp.asarray(clf.dual_coef_[0])
        self.intercept = float(clf.intercept_[0])
        self.gamma_eff = float(clf._gamma)

        self.metrics = {
            "n_support_vectors": int(self.support_vectors.shape[0]),
            "gamma": f"{self.gamma_eff:.2e}",
            "C": f"{self.C:.2e}",
            "intercept": f"{self.intercept:.2e}",
        }
        return self.metrics

    def _predict_proba_single(self, x: jnp.ndarray) -> jnp.ndarray:
        diff = self.support_vectors - x
        norm_sq = jnp.sum(diff ** 2, axis=1)
        kernel_vals = jnp.exp(-self.gamma_eff * norm_sq)
        decision = jnp.sum(self.dual_coef * kernel_vals) + self.intercept
        return jnp.where(decision >= 0, 1.0, 0.0)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "classifier_class": type(self).__name__,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "C": self.C,
            "support_vectors": (
                np.array(self.support_vectors)
                if self.support_vectors is not None
                else None
            ),
            "dual_coef": (
                np.array(self.dual_coef) if self.dual_coef is not None else None
            ),
            "intercept": self.intercept,
            "gamma_eff": self.gamma_eff,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "SVMClassifier":
        clf = cls(
            kernel=state.get("kernel", "rbf"),
            gamma=state.get("gamma", "scale"),
            C=state.get("C", 1e7),
        )
        if state.get("support_vectors") is not None:
            clf.support_vectors = jnp.asarray(state["support_vectors"])
            clf.dual_coef = jnp.asarray(state["dual_coef"])
            clf.intercept = float(state["intercept"])
            clf.gamma_eff = float(state["gamma_eff"])
        clf.metrics = dict(state.get("metrics") or {})
        return clf


# -----------------------------------------------------------------------------
# MLPClassifier (Equinox + Optax)
# -----------------------------------------------------------------------------

if EQX_AVAILABLE:
    class _EquinoxMLP(eqx.Module):
        """Internal Equinox MLP module; passing ``key=None`` disables dropout."""

        linear_layers: list
        dropout_rate: float = eqx.field(static=True)

        def __init__(
            self,
            input_dim: int,
            hidden_dims: tuple,
            dropout_rate: float,
            *,
            key: jax.Array,
        ):
            self.dropout_rate = float(dropout_rate)
            dims = [int(input_dim)] + [int(h) for h in hidden_dims] + [1]
            keys = jax.random.split(key, len(dims) - 1)
            self.linear_layers = [
                eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i])
                for i in range(len(dims) - 1)
            ]

        def __call__(self, x: jnp.ndarray, key: Optional[jax.Array] = None) -> jnp.ndarray:
            for linear in self.linear_layers[:-1]:
                x = jax.nn.relu(linear(x))
                if key is not None and self.dropout_rate > 0.0:
                    key, subkey = jax.random.split(key)
                    mask = jax.random.bernoulli(subkey, 1.0 - self.dropout_rate, x.shape)
                    x = jnp.where(mask, x / (1.0 - self.dropout_rate), 0.0)
            return self.linear_layers[-1](x)
else:
    _EquinoxMLP = None


class MLPClassifier(Classifier):
    """Multi-layer perceptron binary classifier built with Equinox and Optax.

    Parameters
    ----------
    hidden_dims : sequence of int, optional
        Hidden layer widths. Default is ``(32, 32)``.
    dropout_rate : float, optional
        Dropout probability applied after hidden ReLU activations during
        training. Default is ``0.1``.
    lr : float, optional
        Learning rate for ``optax.adamw``. Default is ``1e-3``.
    weight_decay : float, optional
        Weight decay for ``optax.adamw``. Default is ``1e-4``.
    n_epochs : int, optional
        Maximum number of training epochs per restart. Default is ``1000``.
    batch_size : int, optional
        Mini-batch size. If ``None``, chooses ``64`` for ``N < 500`` and
        ``128`` otherwise.
    early_stop_patience : int, optional
        Stop a restart if validation loss does not improve for this many
        epochs. Set to ``0`` to disable early stopping. Default is ``50``.
    n_restarts : int, optional
        Number of training restarts; the first warm-starts from previous
        weights when dimensions match. Default is ``2``.
    val_frac : float, optional
        Fraction of samples held out for early stopping. Default is ``0.2``.
    """

    def __init__(
        self,
        hidden_dims: Sequence[int] = (32, 32),
        dropout_rate: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 1000,
        batch_size: Optional[int] = None,
        early_stop_patience: int = 50,
        n_restarts: int = 2,
        val_frac: float = 0.2,
    ):
        if not EQX_AVAILABLE or not OPTAX_AVAILABLE:
            raise ImportError(
                "Equinox and optax are required for MLPClassifier. "
                "Install with: pip install 'BOBE[nn]'"
            )
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}.")
        if not (0.0 <= val_frac < 1.0):
            raise ValueError(f"val_frac must be in [0, 1), got {val_frac}.")
        if n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {n_epochs}.")
        if n_restarts < 1:
            raise ValueError(f"n_restarts must be >= 1, got {n_restarts}.")
        if batch_size is not None and batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.dropout_rate = float(dropout_rate)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.n_epochs = int(n_epochs)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.early_stop_patience = int(early_stop_patience)
        self.n_restarts = int(n_restarts)
        self.val_frac = float(val_frac)

        self._model = None
        self._input_dim: Optional[int] = None
        self.metrics: Dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, x: jnp.ndarray, y: jnp.ndarray, **kwargs) -> Dict[str, Any]:
        x_arr = jnp.asarray(x, dtype=jnp.float64)
        y_arr = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
        n_samples, d = x_arr.shape

        batch_size = (
            self.batch_size
            if self.batch_size is not None
            else (64 if n_samples < 500 else 128)
        )
        rng = get_numpy_rng()

        use_early_stopping = (
            self.early_stop_patience > 0 and self.val_frac > 0.0 and n_samples > 4
        )
        if use_early_stopping:
            n_val = max(1, min(n_samples - 1, int(n_samples * self.val_frac)))
            val_idx = rng.choice(n_samples, size=n_val, replace=False)
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[val_idx] = False
            x_tr, y_tr = x_arr[train_mask], y_arr[train_mask]
            x_val, y_val = x_arr[val_idx], y_arr[val_idx]
        else:
            x_tr, y_tr = x_arr, y_arr
            x_val, y_val = None, None

        n_tr = int(x_tr.shape[0])
        steps = max(1, n_tr // batch_size)
        optimizer = optax.adamw(self.lr, weight_decay=self.weight_decay)

        @eqx.filter_jit
        def loss_fn(model, batch_x, batch_y, key):
            keys = jax.random.split(key, batch_x.shape[0])
            logits = jax.vmap(lambda xi, ki: model(xi, key=ki))(batch_x, keys)
            return optax.sigmoid_binary_cross_entropy(logits.squeeze(-1), batch_y).mean()

        @eqx.filter_jit
        def inference_loss(model, bx, by):
            logits = jax.vmap(model)(bx)
            return optax.sigmoid_binary_cross_entropy(logits.squeeze(-1), by).mean()

        @eqx.filter_jit
        def train_step(model, opt_state, batch_x, batch_y, key):
            loss_val, grads = eqx.filter_value_and_grad(loss_fn)(
                model, batch_x, batch_y, key
            )
            updates, new_opt_state = optimizer.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            return eqx.apply_updates(model, updates), new_opt_state, loss_val

        best_overall_loss = float("inf")
        best_overall_model = None
        best_overall_metrics: Dict[str, Any] = {}

        for restart in range(self.n_restarts):
            if restart == 0 and self._model is not None and self._input_dim == d:
                current_model = self._model
                log.debug(
                    f"[MLP Restart {restart + 1}/{self.n_restarts}] Warm-starting from previous weights"
                )
            else:
                current_model = _EquinoxMLP(
                    input_dim=d,
                    hidden_dims=self.hidden_dims,
                    dropout_rate=self.dropout_rate,
                    key=get_new_jax_key(),
                )
                log.debug(
                    f"[MLP Restart {restart + 1}/{self.n_restarts}] Random initialization"
                )

            opt_state = optimizer.init(eqx.filter(current_model, eqx.is_array))
            key = get_new_jax_key()

            best_val_loss = float("inf")
            best_restart_model = current_model
            patience_counter = 0
            epochs_trained = 0

            for epoch in range(self.n_epochs):
                perm = rng.permutation(n_tr)
                for step in range(steps):
                    idx = perm[step * batch_size : (step + 1) * batch_size]
                    key, subkey = jax.random.split(key)
                    current_model, opt_state, _ = train_step(
                        current_model, opt_state, x_tr[idx], y_tr[idx], subkey
                    )

                epochs_trained = epoch + 1

                if use_early_stopping:
                    val_loss = float(inference_loss(current_model, x_val, y_val))
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_restart_model = current_model
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.early_stop_patience:
                            log.debug(
                                f"Early stopping at epoch {epochs_trained} "
                                f"(patience={self.early_stop_patience})"
                            )
                            break
                else:
                    best_restart_model = current_model

            final_loss = float(inference_loss(best_restart_model, x_arr, y_arr))
            metrics = {
                "train_loss": f"{final_loss:.2e}",
                "epochs": epochs_trained,
            }
            if use_early_stopping:
                metrics["val_loss"] = f"{best_val_loss:.2e}"

            if final_loss < best_overall_loss:
                best_overall_loss = final_loss
                best_overall_model = best_restart_model
                best_overall_metrics = metrics
                log.debug(
                    f"[MLP Restart {restart + 1}/{self.n_restarts}] "
                    f"New best train_loss: {final_loss:.4e}"
                )

        self._model = best_overall_model
        self._input_dim = int(d)
        self.metrics = best_overall_metrics
        return self.metrics

    def _predict_proba_single(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.sigmoid(self._model(x).squeeze(-1))

    def state_dict(self) -> Dict[str, Any]:
        model_bytes = None
        if self._model is not None:
            buf = io.BytesIO()
            eqx.tree_serialise_leaves(buf, self._model)
            model_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8).copy()
        return {
            "classifier_class": type(self).__name__,
            "hidden_dims": list(self.hidden_dims),
            "dropout_rate": self.dropout_rate,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "early_stop_patience": self.early_stop_patience,
            "n_restarts": self.n_restarts,
            "val_frac": self.val_frac,
            "input_dim": self._input_dim,
            "model_bytes": model_bytes,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "MLPClassifier":
        clf = cls(
            hidden_dims=state["hidden_dims"],
            dropout_rate=state["dropout_rate"],
            lr=state["lr"],
            weight_decay=state["weight_decay"],
            n_epochs=state["n_epochs"],
            batch_size=state.get("batch_size"),
            early_stop_patience=state["early_stop_patience"],
            n_restarts=state["n_restarts"],
            val_frac=state["val_frac"],
        )
        model_bytes = state.get("model_bytes")
        if model_bytes is not None and np.asarray(model_bytes).size > 0:
            input_dim = int(state["input_dim"])
            template = _EquinoxMLP(
                input_dim=input_dim,
                hidden_dims=clf.hidden_dims,
                dropout_rate=clf.dropout_rate,
                key=jax.random.PRNGKey(0),
            )
            raw = np.asarray(model_bytes, dtype=np.uint8)
            clf._model = eqx.tree_deserialise_leaves(io.BytesIO(raw.tobytes()), template)
            clf._input_dim = input_dim
        clf.metrics = dict(state.get("metrics") or {})
        return clf


# -----------------------------------------------------------------------------
# EllipsoidClassifier (pure JAX + Optax)
# -----------------------------------------------------------------------------


def _ellipsoid_unpack_L(flat_L: jnp.ndarray, d: int) -> jnp.ndarray:
    """Build lower-triangular Cholesky-like factor ``L`` with positive diagonal."""
    rows, cols = np.tril_indices(d)
    diag_mask = jnp.asarray(rows == cols)
    processed = jnp.where(diag_mask, jax.nn.softplus(flat_L) + 1e-4, flat_L)
    return jnp.zeros((d, d), dtype=flat_L.dtype).at[rows, cols].set(processed)


def _ellipsoid_logit(
    flat_L: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    mu: jnp.ndarray,
    d: int,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the ellipsoid feasibility logit for a single point ``x``."""
    L = _ellipsoid_unpack_L(flat_L, d)
    z = L.T @ (x - mu)
    md2 = jnp.dot(z, z)
    return -alpha * md2 + beta


class EllipsoidClassifier(Classifier):
    """Pure-JAX ellipsoidal feasibility classifier trained with Optax.

    Models feasibility via a quadratic decision boundary centered at ``mu``
    (typically the current best point ``best_pt``):
    ``logit(x) = -alpha * ||L^T (x - mu)||^2 + beta``.

    Parameters
    ----------
    init_scale : float, optional
        Standard deviation for random initialization of the packed lower-triangular
        matrix ``flat_L``. Default is ``0.1``.
    lr : float, optional
        Learning rate for ``optax.adamw``. Default is ``1e-2``.
    weight_decay : float, optional
        Weight decay for ``optax.adamw``. Default is ``1e-4``.
    n_epochs : int, optional
        Number of training epochs per restart. Default is ``1000``.
    batch_size : int, optional
        Mini-batch size. Default is ``64``.
    n_restarts : int, optional
        Number of training restarts; the first warm-starts from previous
        parameters when dimensions match. Default is ``2``.
    """

    def __init__(
        self,
        init_scale: float = 0.1,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        n_epochs: int = 1000,
        batch_size: int = 64,
        n_restarts: int = 2,
    ):
        if not OPTAX_AVAILABLE:
            raise ImportError(
                "optax is required for EllipsoidClassifier. "
                "Install with: pip install 'BOBE[nn]'"
            )
        if n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {n_epochs}.")
        if n_restarts < 1:
            raise ValueError(f"n_restarts must be >= 1, got {n_restarts}.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

        self.init_scale = float(init_scale)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.n_restarts = int(n_restarts)

        self._flat_L: Optional[jnp.ndarray] = None
        self._alpha: Optional[jnp.ndarray] = None
        self._beta: Optional[jnp.ndarray] = None
        self._mu: Optional[jnp.ndarray] = None
        self._ndim: Optional[int] = None
        self.metrics: Dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return self._flat_L is not None

    def fit(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        *,
        best_pt: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        x_arr = jnp.asarray(x, dtype=jnp.float64)
        y_arr = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
        n_samples, d = x_arr.shape
        d = int(d)

        if best_pt is not None:
            mu = jnp.asarray(best_pt, dtype=jnp.float64).reshape(-1)
            if mu.shape != (d,):
                raise ValueError(f"best_pt must have shape ({d},), got {mu.shape}.")
        else:
            mu = 0.5 * jnp.ones(d, dtype=jnp.float64)

        tril = d * (d + 1) // 2
        steps = max(1, n_samples // self.batch_size)
        rng = get_numpy_rng()
        optimizer = optax.adamw(self.lr, weight_decay=self.weight_decay)

        @jax.jit
        def loss_fn(trainable, batch_x, batch_y):
            logits = jax.vmap(
                lambda xi: _ellipsoid_logit(
                    trainable["flat_L"],
                    trainable["alpha"],
                    trainable["beta"],
                    mu,
                    d,
                    xi,
                )
            )(batch_x)
            return optax.sigmoid_binary_cross_entropy(logits, batch_y).mean()

        @jax.jit
        def train_step(trainable, opt_state, bx, by):
            loss_val, grads = jax.value_and_grad(loss_fn)(trainable, bx, by)
            updates, new_opt_state = optimizer.update(grads, opt_state, trainable)
            return optax.apply_updates(trainable, updates), new_opt_state, loss_val

        best_overall_loss = float("inf")
        best_trainable = None
        best_metrics: Dict[str, Any] = {}

        for restart in range(self.n_restarts):
            if restart == 0 and self.is_fitted and self._ndim == d:
                trainable = {
                    "flat_L": self._flat_L,
                    "alpha": self._alpha,
                    "beta": self._beta,
                }
                log.debug(
                    f"[Ellipsoid Restart {restart + 1}/{self.n_restarts}] Warm-starting from previous params"
                )
            else:
                trainable = {
                    "flat_L": jax.random.normal(get_new_jax_key(), (tril,)) * self.init_scale,
                    "alpha": jnp.ones((), dtype=jnp.float64),
                    "beta": jnp.zeros((), dtype=jnp.float64),
                }
                log.debug(
                    f"[Ellipsoid Restart {restart + 1}/{self.n_restarts}] Random initialization"
                )

            opt_state = optimizer.init(trainable)

            for epoch in range(self.n_epochs):
                perm = rng.permutation(n_samples)
                for step in range(steps):
                    idx = perm[step * self.batch_size : (step + 1) * self.batch_size]
                    trainable, opt_state, _ = train_step(
                        trainable, opt_state, x_arr[idx], y_arr[idx]
                    )

            final_loss = float(loss_fn(trainable, x_arr, y_arr))
            metrics = {
                "train_loss": f"{final_loss:.2e}",
                "epochs": self.n_epochs,
            }
            if final_loss < best_overall_loss:
                best_overall_loss = final_loss
                best_trainable = trainable
                best_metrics = metrics
                log.debug(
                    f"[Ellipsoid Restart {restart + 1}/{self.n_restarts}] "
                    f"New best train_loss: {final_loss:.4e}"
                )

        self._flat_L = best_trainable["flat_L"]
        self._alpha = best_trainable["alpha"]
        self._beta = best_trainable["beta"]
        self._mu = mu
        self._ndim = d
        self.metrics = best_metrics
        return self.metrics

    def _predict_proba_single(self, x: jnp.ndarray) -> jnp.ndarray:
        logit = _ellipsoid_logit(
            self._flat_L, self._alpha, self._beta, self._mu, self._ndim, x
        )
        return jax.nn.sigmoid(logit)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "classifier_class": type(self).__name__,
            "init_scale": self.init_scale,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "n_restarts": self.n_restarts,
            "flat_L": np.array(self._flat_L) if self._flat_L is not None else None,
            "alpha": float(self._alpha) if self._alpha is not None else None,
            "beta": float(self._beta) if self._beta is not None else None,
            "mu": np.array(self._mu) if self._mu is not None else None,
            "ndim": self._ndim,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "EllipsoidClassifier":
        clf = cls(
            init_scale=state["init_scale"],
            lr=state["lr"],
            weight_decay=state["weight_decay"],
            n_epochs=state["n_epochs"],
            batch_size=state["batch_size"],
            n_restarts=state["n_restarts"],
        )
        if state.get("flat_L") is not None:
            clf._flat_L = jnp.asarray(state["flat_L"], dtype=jnp.float64)
            clf._alpha = jnp.asarray(state["alpha"], dtype=jnp.float64)
            clf._beta = jnp.asarray(state["beta"], dtype=jnp.float64)
            clf._mu = jnp.asarray(state["mu"], dtype=jnp.float64)
            clf._ndim = int(state["ndim"])
        clf.metrics = dict(state.get("metrics") or {})
        return clf
