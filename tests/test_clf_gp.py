"""
Tests for standalone classifiers (SVMClassifier, MLPClassifier, EllipsoidClassifier)
and ClassifierGP functionality.

Tests include:
- Standalone classifier fit, predict_proba (1D/2D), JAX jit/vmap/grad, state_dict round-trip, determinism
- ClassifierGP initialization with SVMClassifier, MLPClassifier, and EllipsoidClassifier
- Classifier training and thresholds
- Predictions with classifier filtering
- Update mechanism and warm-starting
- State save/load (state_dict and .npz Surrogate.load) for all three classifiers
- Input validation (kernel & classifier instance requirements)
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from BOBE.clf import (
    Classifier,
    SVMClassifier,
    MLPClassifier,
    EllipsoidClassifier,
)
from BOBE.clf_gp import ClassifierGP
from BOBE.kernels import RBFKernel
from BOBE.surrogate import Surrogate, ClassifierMixin
from BOBE.utils.seed import set_global_seed


def generate_test_data_with_outliers(n_good=30, n_bad=20, d=2, seed=42):
    """Generate test data with good and bad regions for classifier testing."""
    rng = np.random.RandomState(seed)

    # Good region: near [0.5, 0.5]
    X_good = rng.uniform(0.3, 0.7, size=(n_good, d))
    y_good = -np.sum((X_good - 0.5) ** 2, axis=1, keepdims=True)

    # Bad region: corners
    X_bad = rng.uniform(0, 1, size=(n_bad, d))
    X_bad = np.where(X_bad < 0.5, X_bad * 0.4, 0.6 + X_bad * 0.4)
    y_bad = -10 - np.sum((X_bad - 0.5) ** 2, axis=1, keepdims=True)

    X = np.vstack([X_good, X_bad])
    y = np.vstack([y_good, y_bad])

    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Standalone Classifier tests
# ---------------------------------------------------------------------------


def _make_classifier(kind):
    if kind == "svm":
        return SVMClassifier(gamma="scale", C=1e5)
    if kind == "mlp":
        return MLPClassifier(
            hidden_dims=(16, 8),
            dropout_rate=0.1,
            lr=5e-3,
            n_epochs=200,
            batch_size=32,
            early_stop_patience=30,
            n_restarts=2,
        )
    if kind == "ellipsoid":
        return EllipsoidClassifier(n_epochs=80, batch_size=32, n_restarts=2)
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["svm", "mlp", "ellipsoid"])
def test_standalone_classifier_fit_predict_jax_and_roundtrip(kind):
    """Test standalone fit, 1D/2D predict_proba, jit/vmap/grad, and state_dict round-trip."""
    set_global_seed(123)
    X, y = generate_test_data_with_outliers(n_good=35, n_bad=25, d=2)
    labels = (y.flatten() >= y.max() - 5.0).astype(float)
    best_pt = X[np.argmax(y)]

    clf = _make_classifier(kind)
    assert not clf.is_fitted
    with pytest.raises(RuntimeError, match="not been fitted"):
        clf.predict_proba(jnp.array([0.5, 0.5]))

    # Unfitted state_dict round-trip
    unfitted_restored = Classifier.from_state_dict(clf.state_dict())
    assert type(unfitted_restored) is type(clf)
    assert not unfitted_restored.is_fitted

    metrics = clf.fit(X, labels, best_pt=best_pt)
    assert clf.is_fitted
    assert isinstance(metrics, dict) and len(metrics) > 0

    good_pt = jnp.array([0.5, 0.5])
    bad_pt = jnp.array([0.05, 0.05])
    p_good = clf.predict_proba(good_pt)
    p_bad = clf.predict_proba(bad_pt)
    assert p_good.shape == ()
    assert p_bad.shape == ()
    assert 0.0 <= float(p_good) <= 1.0
    assert 0.0 <= float(p_bad) <= 1.0
    assert float(p_good) > float(p_bad)

    # Batch prediction matches single-point prediction
    batch = jnp.array([[0.5, 0.5], [0.05, 0.05], [0.6, 0.4]])
    p_batch = clf.predict_proba(batch)
    assert p_batch.shape == (3,)
    np.testing.assert_allclose(p_batch[0], p_good, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(p_batch[1], p_bad, rtol=1e-10, atol=1e-12)

    # Pure-JAX compatibility: jit, vmap, and grad (needed by NUTS / acquisition)
    p_jit = jax.jit(clf.predict_proba)(batch)
    np.testing.assert_allclose(p_jit, p_batch, rtol=1e-10, atol=1e-12)
    grad_val = jax.jit(jax.grad(clf.predict_proba))(good_pt)
    assert grad_val.shape == (2,)
    assert np.all(np.isfinite(grad_val))

    # Fitted state_dict round-trip (exact prediction agreement)
    restored = Classifier.from_state_dict(clf.state_dict())
    assert type(restored) is type(clf)
    assert restored.is_fitted
    np.testing.assert_allclose(
        restored.predict_proba(batch), p_batch, rtol=1e-12, atol=1e-12
    )
    if kind == "mlp":
        assert restored.hidden_dims == (16, 8)
        assert restored.batch_size == 32


@pytest.mark.parametrize("kind", ["svm", "mlp", "ellipsoid"])
def test_standalone_classifier_determinism(kind):
    """Two fits with the same global seed produce identical predictions."""
    X, y = generate_test_data_with_outliers(n_good=30, n_bad=20, d=2)
    labels = (y.flatten() >= y.max() - 5.0).astype(float)
    best_pt = X[np.argmax(y)]
    probe = jnp.array([[0.5, 0.5], [0.1, 0.1], [0.45, 0.55]])

    set_global_seed(777)
    clf1 = _make_classifier(kind)
    clf1.fit(X, labels, best_pt=best_pt)
    p1 = clf1.predict_proba(probe)

    set_global_seed(777)
    clf2 = _make_classifier(kind)
    clf2.fit(X, labels, best_pt=best_pt)
    p2 = clf2.predict_proba(probe)

    np.testing.assert_allclose(p1, p2, rtol=1e-12, atol=1e-12)


def test_classifier_validation():
    """Invalid classifier hyperparameters raise clear ValueErrors."""
    with pytest.raises(ValueError, match="kernel='rbf'"):
        SVMClassifier(kernel="linear")
    with pytest.raises(ValueError, match="dropout_rate"):
        MLPClassifier(dropout_rate=1.0)
    with pytest.raises(ValueError, match="val_frac"):
        MLPClassifier(val_frac=-0.1)
    with pytest.raises(ValueError, match="n_epochs"):
        MLPClassifier(n_epochs=0)
    with pytest.raises(ValueError, match="n_restarts"):
        EllipsoidClassifier(n_restarts=0)
    with pytest.raises(ValueError, match="Unknown classifier class"):
        Classifier.from_state_dict({"classifier_class": "DoesNotExist"})


def test_mlp_and_ellipsoid_warm_start():
    """Refitting an already-fitted MLP or Ellipsoid classifier warm-starts cleanly."""
    set_global_seed(99)
    X, y = generate_test_data_with_outliers(n_good=30, n_bad=20, d=2)
    labels = (y.flatten() >= y.max() - 5.0).astype(float)
    best_pt = X[np.argmax(y)]

    for clf in [
        MLPClassifier(hidden_dims=(16, 16), n_epochs=40, early_stop_patience=10, n_restarts=1),
        EllipsoidClassifier(n_epochs=40, n_restarts=1),
    ]:
        clf.fit(X, labels, best_pt=best_pt)
        assert clf.is_fitted
        # Refit with updated best_pt / additional points
        clf.fit(X, labels, best_pt=jnp.array([0.52, 0.48]))
        assert clf.is_fitted
        p = clf.predict_proba(jnp.array([0.5, 0.5]))
        assert np.isfinite(float(p))


# ---------------------------------------------------------------------------
# ClassifierGP tests
# ---------------------------------------------------------------------------


def test_clf_gp_initialization_svm():
    """Test ClassifierGP initialization with SVMClassifier."""
    X, y = generate_test_data_with_outliers(n_good=40, n_bad=30, d=3)

    gp_clf = ClassifierGP(
        kernel=RBFKernel(noise=1e-6, lengthscale_prior="DSLP"),
        classifier=SVMClassifier(gamma="scale", C=1e5),
        train_x=X,
        train_y=y,
        clf_use_size=50,
        clf_threshold=5.0,
        gp_threshold=10.0,
    )

    assert gp_clf.clf_type == "SVMClassifier"
    assert isinstance(gp_clf.classifier, SVMClassifier)
    assert gp_clf.clf_data_size == len(X)
    assert gp_clf.npoints == len(X)
    assert gp_clf.use_clf
    assert gp_clf.classifier.is_fitted
    assert "n_support_vectors" in gp_clf.clf_metrics


def test_clf_gp_initialization_nn():
    """Test ClassifierGP initialization with Equinox MLPClassifier."""
    set_global_seed(42)
    X, y = generate_test_data_with_outliers(n_good=35, n_bad=25, d=2)

    gp_clf = ClassifierGP(
        kernel=RBFKernel(noise=1e-6, lengthscale_prior="DSLP"),
        classifier=MLPClassifier(hidden_dims=(32, 32), n_epochs=200, n_restarts=1),
        train_x=X,
        train_y=y,
        clf_use_size=50,
        clf_threshold=5.0,
        gp_threshold=10.0,
    )

    assert gp_clf.clf_type == "MLPClassifier"
    assert isinstance(gp_clf.classifier, MLPClassifier)
    assert gp_clf.use_clf
    assert gp_clf.classifier.is_fitted
    assert "train_loss" in gp_clf.clf_metrics
    assert float(gp_clf.predict_mean(jnp.array([0.5, 0.5]))) > float(
        gp_clf.predict_mean(jnp.array([0.05, 0.05]))
    )


def test_clf_gp_initialization_ellipsoid():
    """Test ClassifierGP initialization with pure-JAX EllipsoidClassifier."""
    set_global_seed(42)
    X, y = generate_test_data_with_outliers(n_good=35, n_bad=25, d=2)

    gp_clf = ClassifierGP(
        kernel=RBFKernel(noise=1e-6, lengthscale_prior="DSLP"),
        classifier=EllipsoidClassifier(n_epochs=200, n_restarts=1),
        train_x=X,
        train_y=y,
        clf_use_size=50,
        clf_threshold=5.0,
        gp_threshold=10.0,
    )

    assert gp_clf.clf_type == "EllipsoidClassifier"
    assert isinstance(gp_clf.classifier, EllipsoidClassifier)
    assert gp_clf.use_clf
    assert gp_clf.classifier.is_fitted
    assert "train_loss" in gp_clf.clf_metrics
    assert float(gp_clf.predict_mean(jnp.array([0.5, 0.5]))) > float(
        gp_clf.predict_mean(jnp.array([0.05, 0.05]))
    )


def test_clf_gp_predictions():
    """Test predictions with classifier filtering."""
    X, y = generate_test_data_with_outliers(n_good=40, n_bad=30, d=2)

    gp_clf = ClassifierGP(
        kernel=RBFKernel(noise=1e-6, lengthscale_prior="DSLP"),
        classifier=SVMClassifier(),
        train_x=X,
        train_y=y,
        clf_use_size=50,
        clf_threshold=5.0,
        gp_threshold=10.0,
        probability_threshold=0.5,
        minus_inf=-1e5,
    )

    good_point = jnp.array([0.5, 0.5])
    bad_point = jnp.array([0.05, 0.05])
    mean_good = gp_clf.predict_mean(good_point)
    mean_bad = gp_clf.predict_mean(bad_point)

    assert gp_clf.use_clf
    assert mean_bad < mean_good

    test_points = jnp.array([[0.5, 0.5], [0.05, 0.05], [0.6, 0.4]])
    means = gp_clf.predict_mean(test_points)
    vars_ = gp_clf.predict_var(test_points)
    means_joint, vars_joint = gp_clf.predict(test_points)

    assert means.shape == (3,)
    assert vars_.shape == (3,)
    np.testing.assert_allclose(means, means_joint, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(vars_, vars_joint, rtol=1e-10, atol=1e-12)
    for i in range(3):
        mi, vi = gp_clf.predict(test_points[i])
        np.testing.assert_allclose(means_joint[i], mi, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(vars_joint[i], vi, rtol=1e-10, atol=1e-12)


def test_clf_gp_update():
    """Test updating ClassifierGP with new points."""
    X, y = generate_test_data_with_outliers(n_good=25, n_bad=15, d=2)

    gp_clf = ClassifierGP(
        kernel=RBFKernel(noise=1e-6, lengthscale_prior="DSLP"),
        classifier=SVMClassifier(),
        train_x=X,
        train_y=y,
        clf_use_size=30,
        clf_threshold=5.0,
        gp_threshold=10.0,
    )

    initial_clf_size = gp_clf.clf_data_size
    initial_gp_size = gp_clf.npoints

    new_X = jnp.array([[0.55, 0.45], [0.48, 0.52]])
    new_y = -jnp.sum((new_X - 0.5) ** 2, axis=1, keepdims=True)
    gp_clf.update(new_X, new_y)

    assert gp_clf.clf_data_size == initial_clf_size + 2
    assert gp_clf.npoints >= initial_gp_size


def test_clf_gp_classifier_training():
    """Test classifier activation once clf_use_size is reached."""
    X_small, y_small = generate_test_data_with_outliers(n_good=15, n_bad=10, d=2)

    gp_clf = ClassifierGP(
        kernel=RBFKernel(noise=1e-6, lengthscale_prior="DSLP"),
        classifier=SVMClassifier(),
        train_x=X_small,
        train_y=y_small,
        clf_use_size=50,
        clf_threshold=5.0,
    )

    assert not gp_clf.use_clf
    assert not gp_clf.classifier.is_fitted

    X_add, y_add = generate_test_data_with_outliers(n_good=25, n_bad=15, d=2, seed=99)
    gp_clf.update(X_add, y_add)
    gp_clf.train_classifier()

    assert gp_clf.clf_data_size >= gp_clf.clf_use_size
    assert gp_clf.use_clf
    assert gp_clf.classifier.is_fitted


def test_clf_gp_random_point():
    """Test random point generation with classifier."""
    X, y = generate_test_data_with_outliers(n_good=40, n_bad=30, d=2)

    gp_clf = ClassifierGP(
        kernel=RBFKernel(noise=1e-6, lengthscale_prior="DSLP"),
        classifier=SVMClassifier(),
        train_x=X,
        train_y=y,
        clf_use_size=50,
        clf_threshold=5.0,
    )

    rng = np.random.default_rng(42)
    points = np.array([gp_clf.get_random_point(rng=rng) for _ in range(5)])
    assert points.shape == (5, 2)
    assert np.all(points >= 0) and np.all(points <= 1)


@pytest.mark.parametrize("kind", ["svm", "mlp", "ellipsoid"])
def test_clf_gp_state_dict_and_disk_roundtrip(kind, tmp_path):
    """Test ClassifierGP state_dict, copy, and .npz Surrogate.load for all classifier types."""
    set_global_seed(123)
    X, y = generate_test_data_with_outliers(n_good=35, n_bad=25, d=2)

    gp_clf1 = ClassifierGP(
        kernel=RBFKernel(noise=1e-6, lengthscale_prior="DSLP"),
        classifier=_make_classifier(kind),
        train_x=X,
        train_y=y,
        clf_use_size=50,
        clf_threshold=5.0,
        gp_threshold=10.0,
    )
    assert gp_clf1.use_clf
    assert gp_clf1.classifier.is_fitted

    probe = jnp.array([[0.5, 0.5], [0.05, 0.05], [0.6, 0.4]])
    mean1, var1 = gp_clf1.predict(probe)

    # In-memory state_dict + Surrogate polymorphic dispatch
    gp_clf2 = Surrogate.from_state_dict(gp_clf1.state_dict())
    assert isinstance(gp_clf2, ClassifierGP)
    assert gp_clf2.clf_type == gp_clf1.clf_type
    assert gp_clf2.clf_data_size == gp_clf1.clf_data_size
    assert gp_clf2.use_clf == gp_clf1.use_clf
    assert jnp.allclose(gp_clf2.train_x_clf, gp_clf1.train_x_clf)
    mean2, var2 = gp_clf2.predict(probe)
    np.testing.assert_allclose(mean2, mean1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(var2, var1, rtol=1e-10, atol=1e-12)

    # On-disk .npz save + Surrogate.load
    path = tmp_path / f"clf_gp_{kind}"
    gp_clf1.save(str(path))
    gp_clf3 = Surrogate.load(str(path))
    assert isinstance(gp_clf3, ClassifierGP)
    assert gp_clf3.clf_type == gp_clf1.clf_type
    mean3, var3 = gp_clf3.predict(probe)
    np.testing.assert_allclose(mean3, mean1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(var3, var1, rtol=1e-10, atol=1e-12)


def test_clf_gp_copy():
    """Test ClassifierGP copy creates an independent instance."""
    X, y = generate_test_data_with_outliers(n_good=30, n_bad=20, d=2)

    gp_clf1 = ClassifierGP(
        kernel=RBFKernel(noise=1e-6, lengthscale_prior="DSLP"),
        classifier=SVMClassifier(),
        train_x=X,
        train_y=y,
        clf_use_size=40,
    )

    gp_clf2 = gp_clf1.copy()
    gp_clf2.update(jnp.array([[0.6, 0.4]]), jnp.array([[-0.02]]))

    assert gp_clf2.clf_data_size == gp_clf1.clf_data_size + 1


def test_clf_gp_requires_kernel_and_classifier_instances():
    """ClassifierGP requires both a Kernel instance and a Classifier instance."""
    with pytest.raises(TypeError):
        ClassifierGP()
    with pytest.raises(TypeError):
        ClassifierGP(kernel=RBFKernel())
    with pytest.raises(TypeError):
        ClassifierGP(kernel="rbf", classifier=SVMClassifier())
    with pytest.raises(TypeError):
        ClassifierGP(kernel=RBFKernel(), classifier="svm")


def test_clf_gp_deferred_initialization_and_interfaces():
    """Test deferred ClassifierGP init and Surrogate/ClassifierMixin conformance."""
    X, y = generate_test_data_with_outliers(n_good=10, n_bad=5, d=2)
    gp_clf = ClassifierGP(
        kernel=RBFKernel(lengthscale_prior="DSLP"),
        classifier=SVMClassifier(),
        clf_use_size=100,
        train_clf_on_init=False,
    )
    assert isinstance(gp_clf, Surrogate)
    assert isinstance(gp_clf, ClassifierMixin)
    assert gp_clf.ndim is None
    assert gp_clf.npoints == 0
    assert gp_clf.clf_data_size == 0

    gp_clf.update(X, y.flatten())
    assert gp_clf.ndim == 2
    assert gp_clf.npoints == 15
    assert gp_clf.clf_data_size == 15
    assert np.isfinite(float(gp_clf.predict_mean(jnp.array([0.5, 0.5]))))

    # Direct initialize(ndim=...) on a deferred instance
    gp_clf_init = ClassifierGP(
        kernel=RBFKernel(lengthscale_prior="DSLP"),
        classifier=SVMClassifier(),
        clf_use_size=100,
    )
    gp_clf_init.initialize(ndim=2)
    assert gp_clf_init.ndim == 2
    assert gp_clf_init.npoints == 0
    assert gp_clf_init.train_x_clf.shape == (0, 2)
    assert gp_clf_init.train_y_clf.shape == (0, 1)
    gp_clf_init.update(X, y.flatten())
    assert gp_clf_init.npoints == 15
