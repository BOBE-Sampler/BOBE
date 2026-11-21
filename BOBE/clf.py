# interfaces and routines for some classifiers
# SVM, Neural Networks, Ellipsoidal bound, etc.

import numpy as np
import jax
import jax.numpy as jnp
from sklearn.svm import SVC
from typing import Callable, Dict, Any, Union, List, Optional, Tuple
from functools import partial
from .utils.log import get_logger
from .utils.seed import get_numpy_rng, get_new_jax_key
log = get_logger("clf")

try:
    import optax
    OPTAX_AVAILABLE = True
except ImportError:
    OPTAX_AVAILABLE = False
    optax = None
    log.debug("optax is not available. NN and Ellipsoid classifiers will require it.")

try:
    from flax import linen as nn
    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False
    nn = None
    log.debug("Flax is not available. Only SVM classifier will be available.")


# -----------------------------------------------------------------------------
# Standalone training and prediction functions for classifiers
# -----------------------------------------------------------------------------

def train_svm_classifier(X, Y, settings = {}, init_params=None, **kwargs):
    """Train SVM classifier and return parameters, metrics, and predict function."""
    gamma = settings.get('gamma', 'scale')
    C = settings.get('C', 1e7)
    kernel = settings.get('kernel', 'rbf')
    
    rng = get_numpy_rng()
    random_state = int(rng.integers(0, 2**31 - 1))
    clf = SVC(kernel=kernel, gamma=gamma, C=C, random_state=random_state)
    clf.fit(X, Y)
    support_vectors = clf.support_vectors_
    dual_coef = clf.dual_coef_[0]  # convert to 1D array
    intercept = float(clf.intercept_[0])
    gamma_eff = float(clf._gamma) # note: this is the effective gamma value used by scikit-learn

    # convert to jax arrays
    support_vectors = jnp.array(support_vectors)
    dual_coef = jnp.array(dual_coef)
    metrics = {
        'n_support_vectors': len(support_vectors),
        'gamma': f"{gamma_eff:.2e}",
        'C': f"{C:.2e}",
        'intercept': f"{intercept:.2e}",
    }
    params = {
        'support_vectors': support_vectors,
        'dual_coef': dual_coef,
        'intercept': intercept,
        'gamma_eff': gamma_eff
    }
    
    # Create predict function
    predict_fn = jax.jit(partial(svm_predict_proba, support_vectors=support_vectors, 
                                dual_coef=dual_coef, intercept=intercept, gamma=gamma_eff))
    
    return params, metrics, predict_fn

def get_svm_predict_proba_fn(params):
    """Get prediction function for SVM classifier from parameters (for loading from file)."""
    support_vectors = params['support_vectors']
    dual_coef = params['dual_coef']
    intercept = params['intercept']
    gamma = params['gamma_eff']
    return jax.jit(partial(svm_predict_proba, support_vectors=support_vectors, 
                          dual_coef=dual_coef, intercept=intercept, gamma=gamma))


# -----------------------------------------------------------------------------
# Neural Network Classifier (currently in development)

def train_nn_classifier(X, Y, settings = {}, init_params=None, **kwargs):
    """Train neural network classifier and return parameters, metrics, and predict function."""
    if not FLAX_AVAILABLE or not OPTAX_AVAILABLE:
        raise ImportError("Flax and optax are required for NN classifier. "
            "Install with: pip install 'BOBE[nn]'"
        )
    # Create model with settings
    label_size = X.shape[0]
    if label_size < 500:
        settings.update({'hidden_dims': [32, 32]})
        settings.update({'batch_size': 64})
    else:
        settings.update({'hidden_dims': [32, 32]})
        settings.update({'batch_size': 128})
    model = MLPClassifier(**settings)
    
    # Train with multiple restarts
    params, metrics = train_nn_multiple_restarts(
        model=model,
        x=X, y=Y,
        init_params=init_params
    )
    
    # Create predict function
    def predict_proba_fn(x):
        logits = model.apply(params, x, train=False)
        return jax.nn.sigmoid(logits.squeeze(-1))
    predict_fn = jax.jit(predict_proba_fn)
    
    return params, metrics, predict_fn

def get_nn_predict_proba_fn(params, settings = {}, **kwargs):
    """Get prediction function for NN classifier from parameters (for loading from file)."""
    # Recreate model with same settings to get the apply function
    model = MLPClassifier(**settings)
    
    def predict_proba_fn(x):
        logits = model.apply(params, x, train=False)
        return jax.nn.sigmoid(logits.squeeze(-1))
    return jax.jit(predict_proba_fn)

def train_ellipsoid_classifier(X, Y, settings = {}, init_params=None, **kwargs):
    """Train ellipsoid classifier and return parameters, metrics, and predict function."""
    if not FLAX_AVAILABLE or not OPTAX_AVAILABLE:
        raise ImportError(
            "Flax and optax are required for Ellipsoid classifier. "
            "Install with: pip install 'BOBE[nn]'"
        )
    d = X.shape[1]
    mu = kwargs.get('best_pt', 0.5*jnp.ones(d))
    
    # label_size = X.shape[0]
    # if label_size < 500:
    #     settings.update({'batch_size': 64})
    # else:
    #     settings.update({'batch_size': 128})    

    # Create model with settings
    model = EllipsoidClassifier(d=d, mu=mu, **settings)
    
    # Train with multiple restarts
    params, metrics = train_ellipsoid_multiple_restarts(
        model=model,
        x=X, y=Y,
        init_params=init_params,
    )
    
    def predict_proba_fn(x):
        logits = model.apply(params, x, train=False)
        return jax.nn.sigmoid(logits.squeeze())
    predict_fn = jax.jit(predict_proba_fn)
    
    return params, metrics, predict_fn

def get_ellipsoid_predict_proba_fn(params, settings, d, **kwargs):
    """Get prediction function for ellipsoid classifier from parameters (for loading from file)."""
    mu = kwargs.get('best_pt', 0.5*jnp.ones(d))
    model = EllipsoidClassifier(d=d, mu=mu, **settings)
    
    def predict_proba_fn(x):
        logits = model.apply(params, x, train=False)
        return jax.nn.sigmoid(logits.squeeze())
    return jax.jit(predict_proba_fn)

# Dictionary mapping classifier types to their functions
CLASSIFIER_REGISTRY = {
    'svm': {
        'train_fn': train_svm_classifier,
        'predict_fn': get_svm_predict_proba_fn,
    },
    'nn': {
        'train_fn': train_nn_classifier,
        'predict_fn': get_nn_predict_proba_fn,
    },
    'ellipsoid': {
        'train_fn': train_ellipsoid_classifier,
        'predict_fn': get_ellipsoid_predict_proba_fn,
    }
}

# -----------------------------------------------------------------------------
# SVM prediction functions
# -----------------------------------------------------------------------------

def svm_predict(x: jnp.ndarray, support_vectors: jnp.ndarray, dual_coef: jnp.ndarray, intercept: float, gamma: float):
    """
    Compute the decision function for SVM with RBF kernel.
    
    Arguments:
      x: Input data point, shape (n_features,)
      support_vectors: JAX array of support vectors, shape (n_sv, n_features)
      dual_coef: JAX array of dual coefficients, shape (n_sv,)
      intercept: Scalar bias term.
      gamma: RBF kernel gamma parameter.
      
    Returns:
      Decision function value (scalar). Sign of this value gives the predicted class.
    """
    # Compute squared Euclidean distances between x and each support vector.
    diff = support_vectors - x  # shape (n_sv, n_features)
    norm_sq = jnp.sum(diff ** 2, axis=1)  # shape (n_sv,)
    # Compute RBF kernel values.
    kernel_vals = jnp.exp(-gamma * norm_sq)  # shape (n_sv,)
    # Compute the decision function.
    decision = jnp.sum(dual_coef * kernel_vals) + intercept
    return decision

def svm_predict_proba(x: jnp.ndarray, support_vectors: jnp.ndarray, dual_coef: jnp.ndarray, intercept: float, gamma: float):
    decision = svm_predict(x, support_vectors, dual_coef, intercept, gamma)
    return jnp.where(decision >= 0, 1.0, 0.0)  # Binary classification: 1 if decision >= 0, else 0


# -----------------------------------------------------------------------------
# Neural Network Classifiers
# -----------------------------------------------------------------------------

# Common training utilities
def train_with_restarts(
    train_fn: Callable,
    x: jnp.ndarray,
    y: jnp.ndarray,
    n_restarts: int = 2,
    init_params = None,
    **train_kwargs
) -> Tuple[Dict, Dict]:
    """
    Train model with multiple restarts using the entire dataset.

    Args:
        train_fn: Training function that returns (params, metrics)
        x: (N, d) features
        y: (N,) labels
        n_restarts: number of random restarts
        init_params: initial parameters for first restart
        **train_kwargs: passed to train_fn
    """
    best_train_loss = jnp.inf
    best_params = None
    best_metrics = {}

    for i in range(n_restarts):
        # Use initial params for first restart, None for others
        restart_init_params = init_params if i == 0 else None

        if i == 0 and init_params is not None:
            log.debug(f"[Restart {i+1}/{n_restarts}] Using provided initial parameters")
        elif i > 0:
            log.debug(f"[Restart {i+1}/{n_restarts}] Using random initialization")

        # Use entire dataset for training (train_fn pulls fresh keys internally)
        params, metrics = train_fn(
            x_train=x, y_train=y,
            init_params=restart_init_params,
            **train_kwargs
        )

        train_loss = float(metrics['train_loss'])

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_params = params
            best_metrics = metrics
            log.debug(f"[Restart {i+1}/{n_restarts}] New best train_loss: {train_loss:.4e}")

    log.debug(f"[Training] Best model selected with train_loss = {best_train_loss:.4e}")
    return best_params, best_metrics


# Neural Network Classifier
if FLAX_AVAILABLE:
    class MLPClassifier(nn.Module):
        hidden_dims: list = (32, 32)
        dropout_rate: float = 0.1
        lr: float = 1e-3
        weight_decay: float = 1e-4
        n_epochs: int = 1000
        batch_size: int = 128
        early_stop_patience: int = 50
        n_restarts: int = 2
        val_frac: float = 0.2

        @nn.compact
        def __call__(self, x, train: bool = False):
            for h in self.hidden_dims:
                x = nn.Dense(h)(x)
                x = nn.relu(x)
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
            x = nn.Dense(1)(x)
            return x
else:
    MLPClassifier = None

def train_nn(
    model: MLPClassifier,
    x_train: jnp.ndarray, y_train: jnp.ndarray,
    init_params=None,
    **kwargs
):
    """Simplified NN training using entire dataset"""
    N, d = x_train.shape

    rng_opt = get_numpy_rng()

    # Handle initialization
    if init_params is not None:
        params = init_params
    else:
        params = model.init(get_new_jax_key(), jnp.ones((1, d)), train=True)

    optimizer = optax.adamw(model.lr, weight_decay=model.weight_decay)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, batch_x, batch_y, rng):
        logits = model.apply(params, batch_x, train=True, rngs={"dropout": rng})
        return optax.sigmoid_binary_cross_entropy(logits.squeeze(-1), batch_y).mean()

    @jax.jit
    def train_step(params, opt_state, batch_x, batch_y, rng):
        grads = jax.grad(loss_fn)(params, batch_x, batch_y, rng)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state

    x_np, y_np = np.array(x_train), np.array(y_train)
    steps = max(1, x_train.shape[0] // model.batch_size)

    for epoch in range(model.n_epochs):
        perm_train = rng_opt.permutation(x_train.shape[0])
        for i in range(steps):
            idx = perm_train[i*model.batch_size:(i+1)*model.batch_size]
            bx = jnp.array(x_np[idx])
            by = jnp.array(y_np[idx])
            params, opt_state = train_step(params, opt_state, bx, by, get_new_jax_key())

    # Compute final training loss
    final_train_loss = loss_fn(params, x_train, y_train, get_new_jax_key())
    
    metrics = {
        'train_loss': f"{float(final_train_loss):.2e}",
        'epochs': epoch + 1,
    }

    return params, metrics

def train_nn_multiple_restarts(model: MLPClassifier, x: jnp.ndarray, y: jnp.ndarray, **kwargs):
    """Wrapper for NN training with restarts"""
    return train_with_restarts(partial(train_nn, model), x, y,
                               n_restarts=model.n_restarts, **kwargs)

# Ellipsoid Classifier with center at best fit point
if FLAX_AVAILABLE:
    class EllipsoidClassifier(nn.Module):
        d: int
        mu: jnp.ndarray
        init_scale: float = 0.1
        lr: float = 1e-2
        weight_decay: float = 1e-4
        n_epochs: int = 1000
        batch_size: int = 64
        patience: int = 25
        n_restarts: int = 2
        val_frac: float = 0.1

        def setup(self):
            tril = self.d * (self.d + 1) // 2
            self.flat_L = self.param("flat_L", nn.initializers.normal(self.init_scale), (tril,))
            self.alpha = self.param("alpha", nn.initializers.ones, ())
            self.beta = self.param("beta", nn.initializers.zeros, ())

        def _unpack_L(self):
            L_matrix = jnp.zeros((self.d, self.d))
            tril_indices = jnp.tril_indices(self.d)
            rows, cols = tril_indices
            diagonal_mask = rows == cols
            flat_L_processed = jnp.where(diagonal_mask, nn.softplus(self.flat_L) + 1e-4, self.flat_L)
            return L_matrix.at[tril_indices].set(flat_L_processed)

        @nn.compact
        def __call__(self, x, train: bool = False):
            L = self._unpack_L()
            diff = x - self.mu
            md2 = jnp.einsum("...i,ij,...j->...", diff, L @ L.T, diff)
            logit = -self.alpha * md2 + self.beta
            return logit
else:
    EllipsoidClassifier = None

def train_ellipsoid(
    model: EllipsoidClassifier,
    x_train: jnp.ndarray, y_train: jnp.ndarray,
    init_params=None,
    **kwargs
):
    """Simplified ellipsoid training using entire dataset"""
    rng = get_numpy_rng()

    # Handle initialization
    if init_params is not None:
        params = init_params
    else:
        params = model.init(get_new_jax_key(), x_train)

    optimizer = optax.adamw(model.lr, weight_decay=model.weight_decay)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, batch_x, batch_y):
        logits = model.apply(params, batch_x, train=False)
        return optax.sigmoid_binary_cross_entropy(logits, batch_y).mean()

    @jax.jit
    def train_step(params, opt_state, bx, by):
        grads = jax.grad(loss_fn)(params, bx, by)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state

    x_np, y_np = np.array(x_train), np.array(y_train)
    steps = max(1, x_train.shape[0] // model.batch_size)

    for epoch in range(model.n_epochs):
        perm_train = rng.permutation(x_train.shape[0])
        for i in range(steps):
            idx = perm_train[i*model.batch_size:(i+1)*model.batch_size]
            bx = jnp.array(x_np[idx])
            by = jnp.array(y_np[idx])
            params, opt_state = train_step(params, opt_state, bx, by)

    # Compute final training loss
    final_train_loss = loss_fn(params, x_train, y_train)
    
    metrics = {
        'train_loss': f"{float(final_train_loss):.2e}",
        'epochs': epoch + 1,
    }

    return params, metrics

def train_ellipsoid_multiple_restarts(model: EllipsoidClassifier, x: jnp.ndarray, y: jnp.ndarray, **kwargs):
    """Wrapper for ellipsoid training with restarts"""
    return train_with_restarts(partial(train_ellipsoid, model), x, y,
                               n_restarts=model.n_restarts, **kwargs)