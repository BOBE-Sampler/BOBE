"""
Tests for kernel-metric rotation (PrincipalAxesTransform) at the kernel,
GP, and ClassifierGP levels.
"""

from math import sqrt

import numpy as np
import pytest
import jax.numpy as jnp

from BOBE.gp import GP
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

ROTATION_SAMPLES_3D = jnp.array([
    [0.10, 0.20, 0.25],
    [0.20, 0.35, 0.40],
    [0.45, 0.30, 0.70],
    [0.65, 0.80, 0.35],
    [0.85, 0.55, 0.75],
    [0.95, 0.70, 0.60],
])


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
            train_x=TRAIN_X_3D,
            train_y=TRAIN_Y_3D,
            clf_type="svm",
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


@pytest.mark.parametrize(
    "sample_kwargs",
    [
        {"samples": ROTATION_SAMPLES_3D},
        {"samples": ROTATION_SAMPLES_3D, "weights": jnp.array([1.0, 2.0, 3.0, 5.0, 4.0, 2.0])},
        {"samples": ROTATION_SAMPLES_3D, "log_weights": jnp.log(jnp.array([1.0, 2.0, 3.0, 5.0, 4.0, 2.0]))},
    ],
)
def test_gp_accepts_sample_based_rotation_inputs(sample_kwargs):
    gp = make_surrogate(GP, **sample_kwargs)
    assert gp.kernel.input_transform is not None
    assert gp.kernel.input_transform.rotation.shape == (3, 3)


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
def test_surrogate_deferred_bind_likelihood_physical_rotation(surrogate_cls):
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
        gp = ClassifierGP(kernel=kernel, clf_type="svm", clf_use_size=100)
    else:
        gp = GP(kernel=kernel)

    assert gp.kernel.input_transform.rotation is None

    gp.bind_likelihood(phys_bounds, param_names=["x", "y", "z"])
    assert gp.kernel.input_transform.rotation is not None
    assert gp.kernel.input_transform.rotation.shape == (2, 2)

    gp.update(TRAIN_X_3D, TRAIN_Y_3D)
    assert gp.param_names == ["x", "y", "z"]
    assert gp.npoints == TRAIN_X_3D.shape[0]
    assert np.isfinite(float(gp.predict_mean(jnp.array([0.5, 0.5, 0.5]))))
