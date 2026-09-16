import numpy as np
import pytest
import jax.numpy as jnp

from BOBE.transforms import PrincipalAxesTransform


COVARIANCE_2D = jnp.array([
    [0.09, 0.06],
    [0.06, 0.16],
])

FULL_COVARIANCE_4D = jnp.array([
    [0.09, 0.00, 0.06, 0.00],
    [0.00, 0.04, 0.00, 0.00],
    [0.06, 0.00, 0.16, 0.00],
    [0.00, 0.00, 0.00, 0.25],
])

SAMPLES_2D = jnp.array([
    [0.10, 0.20],
    [0.20, 0.35],
    [0.40, 0.30],
    [0.65, 0.75],
    [0.90, 0.60],
])


def weighted_covariance(samples, weights):
    weights = weights / jnp.sum(weights)
    mean = jnp.sum(weights[:, None] * samples, axis=0)
    centred = samples - mean
    return (centred * weights[:, None]).T @ centred


def squared_transformed_difference(transform, xa, xb):
    za = transform.forward(xa)
    zb = transform.forward(xb)
    return jnp.square(za - zb)


def test_covariance_rotation_diagonalises_covariance():
    transform = PrincipalAxesTransform(
        covariance=COVARIANCE_2D
    )

    rotated_covariance = (
        transform.rotation.T
        @ COVARIANCE_2D
        @ transform.rotation
    )

    np.testing.assert_allclose(
        rotated_covariance
        - jnp.diag(jnp.diag(rotated_covariance)),
        0.0,
        atol=1e-6,
    )

    assert rotated_covariance[0, 0] >= rotated_covariance[1, 1]


def test_forward_inverse_round_trip():
    transform = PrincipalAxesTransform(
        covariance=COVARIANCE_2D
    )

    x = jnp.array([
        [0.10, 0.25],
        [0.40, 0.70],
        [0.85, 0.30],
    ])

    restored = transform.inverse(transform.forward(x))

    np.testing.assert_allclose(
        restored,
        x,
        atol=1e-6,
    )


def test_partial_rotation_leaves_unselected_dimensions_unchanged():
    transform = PrincipalAxesTransform(
        covariance=COVARIANCE_2D,
        rotation_dims=[0, 2],
    )

    x = jnp.array([
        [0.10, 0.20, 0.30, 0.40],
        [0.50, 0.60, 0.70, 0.80],
    ])

    transformed = transform.forward(x)

    np.testing.assert_allclose(
        transformed[:, [1, 3]],
        x[:, [1, 3]],
        atol=1e-6,
    )


def test_partial_rotation_forward_inverse_round_trip():
    transform = PrincipalAxesTransform(
        covariance=COVARIANCE_2D,
        rotation_dims=[0, 2],
    )

    x = jnp.array([
        [0.10, 0.20, 0.30, 0.40],
        [0.50, 0.60, 0.70, 0.80],
    ])

    restored = transform.inverse(transform.forward(x))

    np.testing.assert_allclose(
        restored,
        x,
        atol=1e-6,
    )


def test_full_and_reduced_covariance_give_same_metric():
    full_transform = PrincipalAxesTransform(
        covariance=FULL_COVARIANCE_4D,
        rotation_dims=[0, 2],
    )

    reduced_transform = PrincipalAxesTransform(
        covariance=COVARIANCE_2D,
        rotation_dims=[0, 2],
    )

    xa = jnp.array([0.15, 0.20, 0.35, 0.40])
    xb = jnp.array([0.75, 0.60, 0.55, 0.80])

    full_difference = squared_transformed_difference(
        full_transform,
        xa,
        xb,
    )

    reduced_difference = squared_transformed_difference(
        reduced_transform,
        xa,
        xb,
    )

    np.testing.assert_allclose(
        full_difference,
        reduced_difference,
        atol=1e-6,
    )


def test_samples_and_covariance_give_same_metric():
    covariance = weighted_covariance(
        SAMPLES_2D,
        jnp.ones(SAMPLES_2D.shape[0]),
    )

    sample_transform = PrincipalAxesTransform(
        samples=SAMPLES_2D
    )

    covariance_transform = PrincipalAxesTransform(
        covariance=covariance
    )

    xa = jnp.array([0.15, 0.25])
    xb = jnp.array([0.80, 0.65])

    np.testing.assert_allclose(
        squared_transformed_difference(
            sample_transform,
            xa,
            xb,
        ),
        squared_transformed_difference(
            covariance_transform,
            xa,
            xb,
        ),
        atol=1e-6,
    )


def test_weighted_samples_and_covariance_give_same_metric():
    weights = jnp.array([1.0, 2.0, 4.0, 3.0, 5.0])

    covariance = weighted_covariance(
        SAMPLES_2D,
        weights,
    )

    sample_transform = PrincipalAxesTransform(
        samples=SAMPLES_2D,
        weights=weights,
    )

    covariance_transform = PrincipalAxesTransform(
        covariance=covariance
    )

    xa = jnp.array([0.20, 0.30])
    xb = jnp.array([0.70, 0.85])

    np.testing.assert_allclose(
        squared_transformed_difference(
            sample_transform,
            xa,
            xb,
        ),
        squared_transformed_difference(
            covariance_transform,
            xa,
            xb,
        ),
        atol=1e-6,
    )


def test_log_weights_and_weights_give_same_metric():
    weights = jnp.array([1.0, 2.0, 4.0, 3.0, 5.0])
    log_weights = jnp.log(weights)

    weighted_transform = PrincipalAxesTransform(
        samples=SAMPLES_2D,
        weights=weights,
    )

    log_weighted_transform = PrincipalAxesTransform(
        samples=SAMPLES_2D,
        log_weights=log_weights,
    )

    xa = jnp.array([0.20, 0.30])
    xb = jnp.array([0.70, 0.85])

    np.testing.assert_allclose(
        squared_transformed_difference(
            weighted_transform,
            xa,
            xb,
        ),
        squared_transformed_difference(
            log_weighted_transform,
            xa,
            xb,
        ),
        atol=1e-6,
    )


def test_samples_can_be_supplied_as_reduced_chain():
    full_samples = jnp.column_stack([
        SAMPLES_2D[:, 0],
        jnp.linspace(0.1, 0.9, SAMPLES_2D.shape[0]),
        SAMPLES_2D[:, 1],
        jnp.linspace(0.9, 0.1, SAMPLES_2D.shape[0]),
    ])

    full_transform = PrincipalAxesTransform(
        samples=full_samples,
        rotation_dims=[0, 2],
    )

    reduced_transform = PrincipalAxesTransform(
        samples=SAMPLES_2D,
        rotation_dims=[0, 2],
    )

    xa = jnp.array([0.15, 0.20, 0.35, 0.40])
    xb = jnp.array([0.75, 0.60, 0.55, 0.80])

    np.testing.assert_allclose(
        squared_transformed_difference(
            full_transform,
            xa,
            xb,
        ),
        squared_transformed_difference(
            reduced_transform,
            xa,
            xb,
        ),
        atol=1e-6,
    )


def test_state_dict_round_trip():
    transform = PrincipalAxesTransform(
        covariance=COVARIANCE_2D,
        rotation_dims=[0, 2],
    )

    restored = PrincipalAxesTransform.from_state_dict(
        transform.state_dict()
    )

    x = jnp.array([
        [0.10, 0.20, 0.30, 0.40],
        [0.50, 0.60, 0.70, 0.80],
    ])

    np.testing.assert_allclose(
        restored.forward(x),
        transform.forward(x),
        atol=1e-6,
    )

    np.testing.assert_array_equal(
        restored.rotation_dims,
        transform.rotation_dims,
    )


def test_requires_exactly_one_rotation_source():
    with pytest.raises(
        ValueError,
        match="Provide exactly one of covariance or samples",
    ):
        PrincipalAxesTransform()

    with pytest.raises(
        ValueError,
        match="Provide exactly one of covariance or samples",
    ):
        PrincipalAxesTransform(
            covariance=COVARIANCE_2D,
            samples=SAMPLES_2D,
        )


def test_weights_and_log_weights_are_mutually_exclusive():
    weights = jnp.ones(SAMPLES_2D.shape[0])

    with pytest.raises(
        ValueError,
        match="Provide either weights or log_weights",
    ):
        PrincipalAxesTransform(
            samples=SAMPLES_2D,
            weights=weights,
            log_weights=jnp.log(weights),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weights": jnp.ones(2)},
        {"log_weights": jnp.zeros(2)},
    ],
)
def test_weights_require_samples(kwargs):
    with pytest.raises(
        ValueError,
        match="can only be used with samples",
    ):
        PrincipalAxesTransform(
            covariance=COVARIANCE_2D,
            **kwargs,
        )


def test_covariance_must_be_square():
    with pytest.raises(
        ValueError,
        match="Covariance matrix must be square",
    ):
        PrincipalAxesTransform(
            covariance=jnp.ones((2, 3))
        )


def test_samples_must_be_two_dimensional():
    with pytest.raises(
        ValueError,
        match=r"samples must have shape \(N, D\)",
    ):
        PrincipalAxesTransform(
            samples=jnp.ones(5)
        )


@pytest.mark.parametrize(
    "rotation_dims, message",
    [
        ([], "rotation_dims cannot be empty"),
        ([-1, 0], "non-negative"),
        ([0, 0], "duplicate"),
    ],
)
def test_invalid_rotation_dims(rotation_dims, message):
    with pytest.raises(ValueError, match=message):
        PrincipalAxesTransform(
            covariance=COVARIANCE_2D,
            rotation_dims=rotation_dims,
        )


def test_rotation_dims_must_be_compatible_with_covariance():
    with pytest.raises(
        ValueError,
        match="incompatible",
    ):
        PrincipalAxesTransform(
            covariance=COVARIANCE_2D,
            rotation_dims=[0, 1, 3],
        )


def test_rotation_dims_must_be_compatible_with_samples():
    with pytest.raises(
        ValueError,
        match="incompatible",
    ):
        PrincipalAxesTransform(
            samples=SAMPLES_2D,
            rotation_dims=[0, 1, 3],
        )


def test_sample_weights_must_have_correct_shape():
    with pytest.raises(
        ValueError,
        match=r"weights must have shape \(N,\)",
    ):
        PrincipalAxesTransform(
            samples=SAMPLES_2D,
            weights=jnp.ones(3),
        )


def test_log_weights_must_have_correct_shape():
    with pytest.raises(
        ValueError,
        match=r"log_weights must have shape \(N,\)",
    ):
        PrincipalAxesTransform(
            samples=SAMPLES_2D,
            log_weights=jnp.ones(3),
        )


def test_sample_weights_must_be_non_negative():
    weights = jnp.array([1.0, 1.0, -1.0, 1.0, 1.0])

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        PrincipalAxesTransform(
            samples=SAMPLES_2D,
            weights=weights,
        )


def test_sample_weights_must_have_positive_total():
    weights = jnp.zeros(SAMPLES_2D.shape[0])

    with pytest.raises(
        ValueError,
        match="positive total weight",
    ):
        PrincipalAxesTransform(
            samples=SAMPLES_2D,
            weights=weights,
        )