import numpy as np
import pytest
import jax.numpy as jnp

from BOBE.gp import GP


TRAIN_X = jnp.array([
    [0.10, 0.20, 0.30],
    [0.25, 0.70, 0.45],
    [0.45, 0.35, 0.80],
    [0.70, 0.85, 0.20],
    [0.90, 0.55, 0.65],
])

TRAIN_Y = jnp.array([
    [-1.2],
    [-0.4],
    [0.3],
    [-0.8],
    [0.7],
])

COVARIANCE_3D = jnp.array([
    [0.10, 0.04, 0.02],
    [0.04, 0.18, 0.06],
    [0.02, 0.06, 0.14],
])

COVARIANCE_02 = COVARIANCE_3D[
    jnp.ix_(
        jnp.array([0, 2]),
        jnp.array([0, 2]),
    )
]

ROTATION_SAMPLES = jnp.array([
    [0.10, 0.20, 0.25],
    [0.20, 0.35, 0.40],
    [0.45, 0.30, 0.70],
    [0.65, 0.80, 0.35],
    [0.85, 0.55, 0.75],
    [0.95, 0.70, 0.60],
])


def make_gp(kernel="rbf", **rotation_kwargs):
    return GP(
        train_x=TRAIN_X,
        train_y=TRAIN_Y,
        kernel=kernel,
        lengthscales=jnp.array([0.20, 0.45, 0.70]),
        kernel_variance=1.3,
        noise=1e-8,
        **rotation_kwargs,
    )


@pytest.mark.parametrize(
    "kernel",
    ["rbf", "matern"],
)
def test_gp_attaches_rotation_and_builds_consistent_initial_cache(
    kernel,
):
    gp = make_gp(
        kernel=kernel,
        rotation_covariance=COVARIANCE_3D,
    )

    assert gp.kernel.input_transform is not None

    kernel_matrix = gp.kernel.covariance(
        gp.train_x,
        gp.train_x,
        include_noise=True,
    )

    expected_cholesky = jnp.linalg.cholesky(
        kernel_matrix
    )

    np.testing.assert_allclose(
        gp.cholesky,
        expected_cholesky,
        rtol=1e-6,
        atol=1e-7,
    )


def test_gp_without_rotation_has_no_transform():
    gp = make_gp()

    assert gp.kernel.input_transform is None


def test_gp_accepts_partial_rotation():
    gp = make_gp(
        rotation_covariance=COVARIANCE_02,
        rotation_dims=[0, 2],
    )

    transform = gp.kernel.input_transform

    assert transform is not None

    x = jnp.array([
        [0.15, 0.25, 0.35],
        [0.65, 0.75, 0.85],
    ])

    transformed = transform.forward(x)

    np.testing.assert_allclose(
        transformed[:, 1],
        x[:, 1],
        atol=1e-6,
    )


def test_gp_accepts_full_covariance_for_partial_rotation():
    reduced_gp = make_gp(
        rotation_covariance=COVARIANCE_02,
        rotation_dims=[0, 2],
    )

    full_gp = make_gp(
        rotation_covariance=COVARIANCE_3D,
        rotation_dims=[0, 2],
    )

    reduced_kernel = reduced_gp.kernel.covariance(
        TRAIN_X,
        TRAIN_X,
        include_noise=False,
    )

    full_kernel = full_gp.kernel.covariance(
        TRAIN_X,
        TRAIN_X,
        include_noise=False,
    )

    np.testing.assert_allclose(
        reduced_kernel,
        full_kernel,
        rtol=1e-6,
        atol=1e-7,
    )


def test_gp_accepts_rotation_samples():
    gp = make_gp(
        rotation_samples=ROTATION_SAMPLES
    )

    assert gp.kernel.input_transform is not None
    assert gp.kernel.input_transform.rotation.shape == (3, 3)


def test_gp_accepts_weighted_rotation_samples():
    weights = jnp.array([
        1.0,
        2.0,
        3.0,
        5.0,
        4.0,
        2.0,
    ])

    gp = make_gp(
        rotation_samples=ROTATION_SAMPLES,
        rotation_weights=weights,
    )

    assert gp.kernel.input_transform is not None


def test_gp_accepts_log_weighted_rotation_samples():
    weights = jnp.array([
        1.0,
        2.0,
        3.0,
        5.0,
        4.0,
        2.0,
    ])

    gp = make_gp(
        rotation_samples=ROTATION_SAMPLES,
        rotation_log_weights=jnp.log(weights),
    )

    assert gp.kernel.input_transform is not None


def test_gp_rejects_rotation_dimension_outside_input_space():
    with pytest.raises(
        ValueError,
        match="outside the GP input dimensions",
    ):
        make_gp(
            rotation_covariance=jnp.eye(2),
            rotation_dims=[0, 3],
        )


def test_gp_empty_rotation_dims_uses_transform_validation():
    with pytest.raises(
        ValueError,
        match="rotation_dims cannot be empty",
    ):
        make_gp(
            rotation_covariance=COVARIANCE_3D,
            rotation_dims=[],
        )


def test_gp_detects_rotation_with_wrong_dimensionality():
    covariance = jnp.array([
        [0.10, 0.03],
        [0.03, 0.15],
    ])

    with pytest.raises(
        ValueError,
        match="incompatible with the GP input dimensions",
    ):
        make_gp(
            rotation_covariance=covariance
        )


@pytest.mark.parametrize(
    "kernel",
    ["rbf", "matern"],
)
def test_gp_predictions_survive_state_dict_round_trip(kernel):
    gp = make_gp(
        kernel=kernel,
        rotation_covariance=COVARIANCE_3D,
    )

    x = jnp.array([
        [0.30, 0.45, 0.55],
        [0.75, 0.25, 0.60],
    ])

    mean_before = gp.predict_mean_batched(x)
    var_before = gp.predict_var_batched(x)

    restored = GP.from_state_dict(
        gp.state_dict()
    )

    assert restored.kernel.input_transform is not None

    mean_after = restored.predict_mean_batched(x)
    var_after = restored.predict_var_batched(x)

    np.testing.assert_allclose(
        mean_after,
        mean_before,
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        var_after,
        var_before,
        rtol=1e-6,
        atol=1e-7,
    )


def test_gp_state_dict_stores_final_transform_not_source_data():
    gp = make_gp(
        rotation_covariance=COVARIANCE_3D
    )

    state = gp.state_dict()
    transform_state = state["input_transform_state"]

    assert transform_state["type"] == "PrincipalAxesTransform"
    assert "rotation" in transform_state
    assert "rotation_dims" in transform_state

    assert "covariance" not in transform_state
    assert "samples" not in transform_state
    assert "weights" not in transform_state


def test_gp_save_load_round_trip(tmp_path):
    gp = make_gp(
        rotation_covariance=COVARIANCE_3D
    )

    x = jnp.array([
        [0.30, 0.45, 0.55],
        [0.75, 0.25, 0.60],
    ])

    expected_mean = gp.predict_mean_batched(x)
    expected_var = gp.predict_var_batched(x)

    filename = tmp_path / "rotated_gp"

    gp.save(str(filename))

    loaded = GP.load(str(filename))

    assert loaded.kernel.input_transform is not None

    np.testing.assert_allclose(
        loaded.predict_mean_batched(x),
        expected_mean,
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        loaded.predict_var_batched(x),
        expected_var,
        rtol=1e-6,
        atol=1e-7,
    )


def test_gp_copy_preserves_rotation():
    gp = make_gp(
        rotation_covariance=COVARIANCE_3D
    )

    copied = gp.copy()

    assert copied.kernel.input_transform is not None

    np.testing.assert_allclose(
        copied.kernel.covariance(
            TRAIN_X,
            TRAIN_X,
            include_noise=False,
        ),
        gp.kernel.covariance(
            TRAIN_X,
            TRAIN_X,
            include_noise=False,
        ),
        rtol=1e-6,
        atol=1e-7,
    )


def test_gp_update_recomputes_cache_with_rotation():
    gp = make_gp(
        rotation_covariance=COVARIANCE_3D
    )

    new_x = jnp.array([
        [0.35, 0.90, 0.50]
    ])

    new_y = jnp.array([
        [0.15]
    ])

    gp.update(new_x, new_y)

    kernel_matrix = gp.kernel.covariance(
        gp.train_x,
        gp.train_x,
        include_noise=True,
    )

    expected_cholesky = jnp.linalg.cholesky(
        kernel_matrix
    )

    np.testing.assert_allclose(
        gp.cholesky,
        expected_cholesky,
        rtol=1e-6,
        atol=1e-7,
    )