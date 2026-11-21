"""
Tests for kernel-metric rotation (PrincipalAxesTransform) at the kernel,
GP, and ClassifierGP levels.
"""

from math import sqrt
from types import SimpleNamespace

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import BOBE.acquisition as acquisition
from BOBE.bo import BOBE, get_dimension_based_defaults
from BOBE.gp import GP
from BOBE.clf import SVMClassifier
from BOBE.clf_gp import ClassifierGP
from BOBE.kernels import RBFKernel, MaternKernel
from BOBE.transforms import PrincipalAxesTransform


COVARIANCE_2D = jnp.array([
    [0.09, 0.06],
    [0.06, 0.16],
])

COVARIANCE_3D = jnp.array([
    [0.10, 0.04, 0.02],
    [0.04, 0.18, 0.06],
    [0.02, 0.06, 0.14],
])

COVARIANCE_02 = COVARIANCE_3D[jnp.ix_(jnp.array([0, 2]), jnp.array([0, 2]))]

BOUNDS_2D = jnp.array([[0.0, 0.0], [1.0, 1.0]])
BOUNDS_3D = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
BOUNDS_4D = jnp.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])

LENGTHSCALES_2D = jnp.array([0.18, 0.65])
KERNEL_VARIANCE = 1.7

TRAIN_X_3D = jnp.array([
    [0.10, 0.20, 0.30],
    [0.25, 0.70, 0.45],
    [0.45, 0.35, 0.80],
    [0.70, 0.85, 0.20],
    [0.90, 0.55, 0.65],
])

TRAIN_Y_3D = jnp.array([[-1.2], [-0.4], [0.3], [-0.8], [0.7]])


def manual_kernel_matrix(kernel_cls, xa, xb, transform, lengthscales, kernel_variance):
    xa = transform.forward(xa)
    xb = transform.forward(xb)
    diff = (xa[:, None, :] - xb[None, :, :]) / lengthscales
    dsq = jnp.sum(diff ** 2, axis=-1)
    if kernel_cls is RBFKernel:
        return kernel_variance * jnp.exp(-0.5 * dsq)
    d = jnp.sqrt(dsq)
    sqrt5 = sqrt(5.0)
    return kernel_variance * (1.0 + sqrt5 * d + 5.0 * d ** 2 / 3.0) * jnp.exp(-sqrt5 * d)


def make_surrogate(surrogate_cls=GP, kernel_cls=RBFKernel, **transform_kwargs):
    transform = (
        PrincipalAxesTransform(param_bounds=BOUNDS_3D, **transform_kwargs)
        if transform_kwargs
        else None
    )
    kernel = kernel_cls(
        lengthscales=jnp.array([0.20, 0.45, 0.70]),
        kernel_variance=1.3,
        noise=1e-8,
        input_transform=transform,
    )
    if surrogate_cls is ClassifierGP:
        return ClassifierGP(
            kernel=kernel,
            classifier=SVMClassifier(),
            train_x=TRAIN_X_3D,
            train_y=TRAIN_Y_3D,
            clf_use_size=100,
            train_clf_on_init=False,
            gp_threshold=500.0,
        )
    return GP(kernel=kernel, train_x=TRAIN_X_3D, train_y=TRAIN_Y_3D)


# ---------------------------------------------------------------------------
# Kernel-level metric tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel_cls", [RBFKernel, MaternKernel])
def test_rotated_kernel_matches_manual_metric(kernel_cls):
    transform = PrincipalAxesTransform(covariance=COVARIANCE_2D, param_bounds=BOUNDS_2D)
    kernel = kernel_cls(
        lengthscales=LENGTHSCALES_2D,
        kernel_variance=KERNEL_VARIANCE,
        noise=1e-8,
        input_transform=transform,
    )
    xa = jnp.array([[0.10, 0.20], [0.40, 0.75]])
    xb = jnp.array([[0.25, 0.30], [0.80, 0.60], [0.65, 0.15]])

    actual = kernel.covariance(xa, xb, include_noise=False)
    expected = manual_kernel_matrix(
        kernel_cls, xa, xb, transform, LENGTHSCALES_2D, KERNEL_VARIANCE
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("kernel_cls", [RBFKernel, MaternKernel])
def test_rotation_changes_anisotropic_kernel_metric(kernel_cls):
    transform = PrincipalAxesTransform(covariance=COVARIANCE_2D, param_bounds=BOUNDS_2D)
    rotated_kernel = kernel_cls(LENGTHSCALES_2D, KERNEL_VARIANCE, input_transform=transform)
    unrotated_kernel = kernel_cls(LENGTHSCALES_2D, KERNEL_VARIANCE)

    x = jnp.array([[0.10, 0.20], [0.40, 0.75], [0.80, 0.45]])
    assert not np.allclose(
        rotated_kernel.covariance(x, x, include_noise=False),
        unrotated_kernel.covariance(x, x, include_noise=False),
        rtol=1e-5,
        atol=1e-7,
    )


@pytest.mark.parametrize("kernel_cls", [RBFKernel, MaternKernel])
def test_isotropic_kernel_is_rotation_invariant(kernel_cls):
    transform = PrincipalAxesTransform(covariance=COVARIANCE_2D, param_bounds=BOUNDS_2D)
    lengthscales = jnp.array([0.4, 0.4])
    rotated_kernel = kernel_cls(lengthscales, KERNEL_VARIANCE, input_transform=transform)
    unrotated_kernel = kernel_cls(lengthscales, KERNEL_VARIANCE)

    x = jnp.array([[0.10, 0.20], [0.40, 0.75], [0.80, 0.45]])
    np.testing.assert_allclose(
        rotated_kernel.covariance(x, x, include_noise=False),
        unrotated_kernel.covariance(x, x, include_noise=False),
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize("kernel_cls", [RBFKernel, MaternKernel])
def test_partial_rotation_matches_manual_pretransformation(kernel_cls):
    transform = PrincipalAxesTransform(
        covariance=COVARIANCE_2D,
        rotation_dims=[0, 2],
        param_bounds=BOUNDS_4D,
    )
    lengthscales = jnp.array([0.18, 0.30, 0.65, 0.45])
    transformed_kernel = kernel_cls(lengthscales, KERNEL_VARIANCE, input_transform=transform)
    manual_kernel = kernel_cls(lengthscales, KERNEL_VARIANCE)

    x = jnp.array([
        [0.10, 0.20, 0.30, 0.40],
        [0.45, 0.55, 0.75, 0.65],
        [0.80, 0.15, 0.50, 0.90],
    ])
    np.testing.assert_allclose(
        transformed_kernel.covariance(x, x, include_noise=False),
        manual_kernel.covariance(transform.forward(x), transform.forward(x), include_noise=False),
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize("kernel_cls", [RBFKernel, MaternKernel])
def test_rotation_does_not_change_kernel_diagonal(kernel_cls):
    transform = PrincipalAxesTransform(covariance=COVARIANCE_2D, param_bounds=BOUNDS_2D)
    noise = 1e-5
    kernel = kernel_cls(
        LENGTHSCALES_2D, KERNEL_VARIANCE, noise=noise, input_transform=transform
    )
    x = jnp.array([[0.10, 0.20], [0.40, 0.75], [0.80, 0.45]])
    expected = jnp.full(x.shape[0], KERNEL_VARIANCE + noise)
    np.testing.assert_allclose(
        kernel.diagonal(x, include_noise=True), expected, rtol=1e-6, atol=1e-7
    )


# ---------------------------------------------------------------------------
# Surrogate integration tests (GP and ClassifierGP)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("surrogate_cls", [GP, ClassifierGP])
@pytest.mark.parametrize("kernel_cls", [RBFKernel, MaternKernel])
def test_surrogate_attaches_rotation_and_builds_consistent_cholesky(
    surrogate_cls, kernel_cls
):
    gp = make_surrogate(surrogate_cls, kernel_cls=kernel_cls, covariance=COVARIANCE_3D)
    assert gp.kernel.input_transform is not None
    assert gp.kernel.input_transform.rotation.shape == (3, 3)

    K = gp.kernel.covariance(gp.train_x, gp.train_x, include_noise=True)
    np.testing.assert_allclose(gp.cholesky, jnp.linalg.cholesky(K), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("surrogate_cls", [GP, ClassifierGP])
def test_surrogate_without_rotation_has_no_transform(surrogate_cls):
    gp = make_surrogate(surrogate_cls)
    assert gp.kernel.input_transform is None


@pytest.mark.parametrize("surrogate_cls", [GP, ClassifierGP])
def test_surrogate_rotation_affects_kernel(surrogate_cls):
    rotated = make_surrogate(surrogate_cls, covariance=COVARIANCE_3D)
    unrotated = make_surrogate(surrogate_cls)
    assert not np.allclose(
        rotated.kernel.covariance(rotated.train_x, rotated.train_x, include_noise=False),
        unrotated.kernel.covariance(unrotated.train_x, unrotated.train_x, include_noise=False),
        rtol=1e-5,
        atol=1e-7,
    )


def test_gp_partial_rotation_and_full_vs_reduced_covariance():
    reduced_gp = make_surrogate(GP, covariance=COVARIANCE_02, rotation_dims=[0, 2])
    full_gp = make_surrogate(GP, covariance=COVARIANCE_3D, rotation_dims=[0, 2])

    x = jnp.array([[0.15, 0.25, 0.35], [0.65, 0.75, 0.85]])
    transformed = reduced_gp.kernel.input_transform.forward(x)
    np.testing.assert_allclose(transformed[:, 1], x[:, 1], atol=1e-6)

    np.testing.assert_allclose(
        reduced_gp.kernel.covariance(TRAIN_X_3D, TRAIN_X_3D, include_noise=False),
        full_gp.kernel.covariance(TRAIN_X_3D, TRAIN_X_3D, include_noise=False),
        rtol=1e-6,
        atol=1e-7,
    )


def test_gp_rotation_dimension_validation():
    with pytest.raises(ValueError, match="outside the GP input dimensions"):
        make_surrogate(GP, covariance=jnp.eye(2), rotation_dims=[0, 3])

    with pytest.raises(ValueError, match="rotation_dims cannot be empty"):
        make_surrogate(GP, covariance=COVARIANCE_3D, rotation_dims=[])

    with pytest.raises(ValueError, match="incompatible with the GP input dimensions"):
        make_surrogate(GP, covariance=COVARIANCE_2D)


@pytest.mark.parametrize("surrogate_cls", [GP, ClassifierGP])
@pytest.mark.parametrize("kernel_cls", [RBFKernel, MaternKernel])
def test_surrogate_state_dict_stores_final_transform_and_round_trips(
    surrogate_cls, kernel_cls
):
    gp = make_surrogate(surrogate_cls, kernel_cls=kernel_cls, covariance=COVARIANCE_3D)
    state = gp.state_dict()
    transform_state = state["kernel_state"]["input_transform_state"]

    assert transform_state is not None
    assert transform_state["type"] == "PrincipalAxesTransform"
    assert "rotation" in transform_state
    assert "rotation_dims" in transform_state
    assert "covariance" not in transform_state
    assert "samples" not in transform_state
    assert "weights" not in transform_state

    x = jnp.array([[0.30, 0.45, 0.55], [0.75, 0.25, 0.60]])
    mean_before, var_before = gp.predict(x)

    restored = surrogate_cls.from_state_dict(state)
    assert restored.kernel.input_transform is not None

    mean_after, var_after = restored.predict(x)
    np.testing.assert_allclose(mean_after, mean_before, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(var_after, var_before, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("surrogate_cls", [GP, ClassifierGP])
def test_surrogate_save_load_and_copy_preserve_rotation(surrogate_cls, tmp_path):
    gp = make_surrogate(surrogate_cls, covariance=COVARIANCE_3D)
    x = jnp.array([[0.30, 0.45, 0.55], [0.75, 0.25, 0.60]])
    expected_mean, expected_var = gp.predict(x)

    filename = tmp_path / f"rotated_{surrogate_cls.__name__}"
    gp.save(str(filename))
    loaded = surrogate_cls.load(str(filename))
    copied = gp.copy()

    for other in (loaded, copied):
        assert other.kernel.input_transform is not None
        mean, var = other.predict(x)
        np.testing.assert_allclose(mean, expected_mean, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(var, expected_var, rtol=1e-6, atol=1e-7)


def test_gp_update_recomputes_cholesky_with_rotation():
    gp = make_surrogate(GP, covariance=COVARIANCE_3D)
    gp.update(jnp.array([[0.35, 0.90, 0.50]]), jnp.array([[0.15]]))

    K = gp.kernel.covariance(gp.train_x, gp.train_x, include_noise=True)
    np.testing.assert_allclose(gp.cholesky, jnp.linalg.cholesky(K), rtol=1e-6, atol=1e-7)


def test_classifier_gp_keeps_original_classifier_coordinates():
    gp = make_surrogate(ClassifierGP, covariance=COVARIANCE_3D)
    np.testing.assert_allclose(gp.train_x_clf, TRAIN_X_3D, atol=1e-7)


@pytest.mark.parametrize("surrogate_cls", [GP, ClassifierGP])
def test_surrogate_deferred_initialize_physical_rotation(surrogate_cls):
    phys_bounds = jnp.array([[-2.0, -4.0, 0.0], [2.0, 4.0, 10.0]])
    transform = PrincipalAxesTransform(
        covariance=COVARIANCE_02,
        rotation_dims=["x", "z"],
    )
    kernel = RBFKernel(
        lengthscales=jnp.array([0.20, 0.45, 0.70]),
        kernel_variance=1.2,
        lengthscale_prior="DSLP",
        input_transform=transform,
    )
    if surrogate_cls is ClassifierGP:
        gp = ClassifierGP(kernel=kernel, classifier=SVMClassifier(), clf_use_size=100)
    else:
        gp = GP(kernel=kernel)

    assert gp.kernel.input_transform.rotation is None

    gp.initialize(param_bounds=phys_bounds, param_names=["x", "y", "z"])
    assert gp.kernel.input_transform.rotation is not None
    assert gp.kernel.input_transform.rotation.shape == (2, 2)

    gp.update(TRAIN_X_3D, TRAIN_Y_3D)
    assert gp.param_names == ["x", "y", "z"]
    assert gp.npoints == TRAIN_X_3D.shape[0]
    assert np.isfinite(float(gp.predict_mean(jnp.array([0.5, 0.5, 0.5]))))


# ---------------------------------------------------------------------------
# Adaptive kernel rotation integration tests (GP, ClassifierGP, BOBE)
# ---------------------------------------------------------------------------

def _posterior_samples_3d(seed=1, mean=None, n=400):
    rng = np.random.default_rng(seed)
    mean = np.array([0.45, 0.50, 0.55]) if mean is None else np.asarray(mean)
    cov = np.array([
        [0.04, 0.025, 0.010],
        [0.025, 0.09, -0.015],
        [0.010, -0.015, 0.06],
    ])
    return {"x": rng.multivariate_normal(mean, cov, size=n), "method": "MCMC"}


@pytest.mark.parametrize("surrogate_cls", [GP, ClassifierGP])
@pytest.mark.parametrize("kernel_cls", [RBFKernel, MaternKernel])
def test_surrogate_adaptive_rotation_auto_binds_updates_and_round_trips(
    surrogate_cls, kernel_cls, tmp_path
):
    transform = PrincipalAxesTransform(max_updates=3, acq_threshold=0.5, update_step=2)
    kernel = kernel_cls(
        lengthscales=jnp.array([0.20, 0.45, 0.70]),
        kernel_variance=1.3,
        noise=1e-6,
        input_transform=transform,
    )
    if surrogate_cls is ClassifierGP:
        gp = ClassifierGP(
            kernel=kernel,
            classifier=SVMClassifier(),
            train_x=TRAIN_X_3D,
            train_y=TRAIN_Y_3D,
            clf_use_size=100,
            train_clf_on_init=False,
            gp_threshold=500.0,
        )
    else:
        gp = GP(kernel=kernel, train_x=TRAIN_X_3D, train_y=TRAIN_Y_3D)

    assert gp.has_adaptive_kernel_rotation
    np.testing.assert_allclose(gp.kernel_rotation.rotation, np.eye(3))

    samples = _posterior_samples_3d(seed=12)
    assert gp.try_update_kernel_rotation(samples, acq_val=0.4, step=1)
    assert gp.kernel_rotation.update_count == 1
    assert not np.allclose(gp.kernel_rotation.rotation, np.eye(3))

    # Training points and classifier coordinates remain unchanged in [0, 1]^D
    np.testing.assert_allclose(gp.train_x, TRAIN_X_3D)
    if surrogate_cls is ClassifierGP:
        np.testing.assert_allclose(gp.train_x_clf, TRAIN_X_3D)

    # Cholesky factor matches rotated kernel
    K = gp.kernel.covariance(gp.train_x, gp.train_x, include_noise=True)
    np.testing.assert_allclose(gp.cholesky, jnp.linalg.cholesky(K), rtol=1e-6, atol=1e-7)

    # Gradient through predict_mean remains finite (required by NUTS)
    grad = jax.grad(lambda z: gp.predict_mean(z))(jnp.array([0.4, 0.5, 0.6]))
    assert np.all(np.isfinite(np.asarray(grad)))

    # Round-trip via state_dict, copy, and save/load
    probe = jnp.array([[0.30, 0.45, 0.55], [0.75, 0.25, 0.60]])
    expected_mean, expected_var = gp.predict(probe)
    filename = tmp_path / f"adaptive_{surrogate_cls.__name__}_{kernel_cls.__name__}"
    gp.save(str(filename))

    for clone in (
        surrogate_cls.from_state_dict(gp.state_dict()),
        gp.copy(),
        surrogate_cls.load(str(filename)),
    ):
        assert clone.has_adaptive_kernel_rotation
        assert clone.kernel_rotation.update_count == 1
        np.testing.assert_allclose(
            clone.kernel_rotation.rotation, gp.kernel_rotation.rotation
        )
        np.testing.assert_allclose(
            clone.kernel_rotation.reference_covariance,
            gp.kernel_rotation.reference_covariance,
        )
        mean, var = clone.predict(probe)
        np.testing.assert_allclose(mean, expected_mean, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(var, expected_var, rtol=1e-6, atol=1e-7)


def test_adaptive_rotation_waits_for_min_evals():
    transform = PrincipalAxesTransform(
        update_dims=[0, 1, 2], acq_threshold=0.5, max_updates=2
    )
    gp = GP(
        kernel=MaternKernel(input_transform=transform),
        train_x=TRAIN_X_3D,
        train_y=TRAIN_Y_3D,
    )
    samples = _posterior_samples_3d(seed=12)

    with pytest.raises(ValueError, match="current_evals"):
        gp.try_update_kernel_rotation(samples, acq_val=0.4, step=4, min_evals=50)

    assert not gp.try_update_kernel_rotation(
        samples, acq_val=0.4, step=4, current_evals=49, min_evals=50
    )
    assert gp.kernel_rotation.update_count == 0
    np.testing.assert_allclose(gp.kernel_rotation.rotation, np.eye(3))

    assert gp.try_update_kernel_rotation(
        samples, acq_val=0.4, step=5, current_evals=50, min_evals=50
    )
    assert gp.kernel_rotation.update_count == 1


def test_adaptive_rotation_skips_gracefully_on_bad_samples():
    transform = PrincipalAxesTransform(
        update_dims=[0, 1, 2], acq_threshold=0.5, max_updates=2
    )
    gp = GP(
        kernel=RBFKernel(input_transform=transform),
        train_x=TRAIN_X_3D,
        train_y=TRAIN_Y_3D,
    )
    # Too few samples -> caught and skipped
    assert not gp.try_update_kernel_rotation(
        {"x": np.zeros((2, 3))}, acq_val=0.2, step=1
    )
    assert gp.kernel_rotation.update_count == 0
    np.testing.assert_allclose(gp.kernel_rotation.rotation, np.eye(3))


def test_batch_fantasy_gps_preserve_adapted_kernel_metric():
    transform = PrincipalAxesTransform(
        update_dims=[0, 1, 2], acq_threshold=1.0, max_updates=2
    )
    gp = ClassifierGP(
        kernel=MaternKernel(
            lengthscales=jnp.array([0.2, 0.5, 0.8]),
            kernel_variance=1.5,
            input_transform=transform,
        ),
        classifier=SVMClassifier(),
        train_x=TRAIN_X_3D,
        train_y=TRAIN_Y_3D,
        clf_use_size=100,
        train_clf_on_init=False,
        gp_threshold=500.0,
    )
    assert gp.try_update_kernel_rotation(_posterior_samples_3d(seed=3), acq_val=0.5, step=1)
    probe = TRAIN_X_3D[:4]
    expected_K = np.asarray(gp.kernel.covariance(probe, probe, include_noise=False))
    observed = []

    class RecordingAcquisition(acquisition.AcquisitionFunction):
        def get_next_point(self, point_gp, **kwargs):
            K = np.asarray(point_gp.kernel.covariance(probe, probe, include_noise=False))
            observed.append((type(point_gp), K))
            return np.full(3, 0.2 + 0.1 * len(observed)), float(len(observed))

    points, values = RecordingAcquisition().get_next_batch(
        gp, n_batch=3, acq_kwargs={}, verbose=False, rng=np.random.default_rng(1)
    )
    assert points.shape == (3, 3) and values.shape == (3,)
    assert [t for t, _ in observed] == [ClassifierGP, GP, GP]
    for _, K in observed:
        np.testing.assert_allclose(K, expected_K, rtol=1e-10)


@pytest.mark.parametrize(
    "gp_size, expected_restarts, expected_maxiters",
    [(100, 8, 1000), (200, 4, 500), (500, 4, 500), (900, 4, 200)],
)
def test_rotation_refit_uses_shared_gp_fit_policy(
    gp_size, expected_restarts, expected_maxiters
):
    calls = []
    bobe = BOBE.__new__(BOBE)
    bobe.gp = SimpleNamespace(train_x=np.empty((gp_size, 2)))
    bobe.pool = SimpleNamespace(gp_fit=lambda gp, **kwargs: calls.append((gp, kwargs)))
    bobe.fit_n_points = 10
    bobe.n_points_since_last_fit = 3
    bobe.np_rng = np.random.default_rng(7)

    assert bobe._refit_gp_hyperparameters(
        force=True, reason="after adaptive kernel rotation"
    )
    assert len(calls) == 1 and calls[0][0] is bobe.gp
    assert calls[0][1]["n_restarts"] == expected_restarts
    assert calls[0][1]["maxiters"] == expected_maxiters
    assert calls[0][1]["rng"] is bobe.np_rng and calls[0][1]["use_pool"] is True
    assert bobe.n_points_since_last_fit == 0


def test_bobe_adaptive_rotation_rejects_uniform_mc_method(tmp_path):
    def loglike(x):
        return float(-0.5 * np.sum(np.asarray(x) ** 2))

    transform = PrincipalAxesTransform(max_updates=2, acq_threshold=1.0)
    gp = GP(kernel=RBFKernel(input_transform=transform))
    bobe = BOBE(
        loglikelihood=loglike,
        param_list=["x", "y"],
        param_bounds=np.array([[-3.0, -3.0], [3.0, 3.0]]),
        gp=gp,
        n_sobol_init=8,
        save=False,
        save_dir=str(tmp_path),
        seed=42,
    )
    try:
        with pytest.raises(ValueError, match="mc_points_method='uniform'"):
            bobe.run(acq="wipv", mc_points_method="uniform", max_evals=10)
    finally:
        bobe.pool.close()


def test_bobe_end_to_end_adaptive_kernel_rotation(tmp_path):
    # Correlated 2D Gaussian log-likelihood
    cov_true = np.array([[1.0, 0.85], [0.85, 1.0]])
    inv_cov = np.linalg.inv(cov_true)

    def loglike(x):
        x = np.asarray(x)
        return float(-0.5 * x @ inv_cov @ x)

    transform = PrincipalAxesTransform(
        max_updates=2,
        acq_threshold=None,  # test default fallback from get_dimension_based_defaults
        update_step=1,
        kl_threshold=0.0,
    )
    gp = GP(kernel=MaternKernel(input_transform=transform))
    bobe = BOBE(
        loglikelihood=loglike,
        param_list=["x", "y"],
        param_bounds=np.array([[-4.0, -4.0], [4.0, 4.0]]),
        gp=gp,
        n_sobol_init=8,
        save=False,
        save_dir=str(tmp_path),
        seed=42,
    )
    # Give a generous threshold via default check: first verify default gets filled
    expected_default_thresh = get_dimension_based_defaults(2)["transform_acq_threshold"]
    # Use high threshold in a second check or verify default is populated during run
    results = bobe.run(
        acq="wipv",
        min_evals=8,
        max_evals=14,
        batch_size=2,
        logz_threshold=1e-6,
        ns_n_points=100,
        num_hmc_warmup=64,
        num_hmc_samples=128,
        thinning=2,
        num_chains=2,
        mc_points_size=32,
        do_final_ns=False,
    )

    assert gp.kernel_rotation.acq_threshold == pytest.approx(expected_default_thresh)
    assert "kernel_rotation_update_count" in results["results_manager"].gp_info
    assert np.all((gp.train_x >= 0.0) & (gp.train_x <= 1.0))

    # Also verify an update is definitely triggered when acq_threshold is permissive
    transform2 = PrincipalAxesTransform(
        max_updates=2,
        acq_threshold=1e6,
        update_step=1,
        kl_threshold=0.0,
    )
    gp2 = GP(kernel=MaternKernel(input_transform=transform2))
    bobe2 = BOBE(
        loglikelihood=loglike,
        param_list=["x", "y"],
        param_bounds=np.array([[-4.0, -4.0], [4.0, 4.0]]),
        gp=gp2,
        n_sobol_init=8,
        save=False,
        save_dir=str(tmp_path),
        seed=42,
    )
    results2 = bobe2.run(
        acq="wipv",
        min_evals=8,
        max_evals=12,
        batch_size=2,
        logz_threshold=1e-6,
        ns_n_points=100,
        num_hmc_warmup=64,
        num_hmc_samples=128,
        thinning=2,
        num_chains=2,
        mc_points_size=32,
        do_final_ns=False,
    )
    assert gp2.kernel_rotation.update_count >= 1
    assert not np.allclose(gp2.kernel_rotation.rotation, np.eye(2))
    assert results2["results_manager"].gp_info["kernel_rotation_update_count"] >= 1
    assert results2["results_manager"].gp_info["last_kernel_rotation_iteration"] is not None
    assert np.all((gp2.train_x >= 0.0) & (gp2.train_x <= 1.0))

