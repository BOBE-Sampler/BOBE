import numpy as np
import pytest
import jax.numpy as jnp

from BOBE.transforms import PrincipalAxesTransform, scale_covariance_to_unit


COVARIANCE_2D = jnp.array([
    [0.09, 0.06],
    [0.06, 0.16],
])

UNIT_BOUNDS_2D = jnp.array([
    [0.0, 0.0],
    [1.0, 1.0],
])

UNIT_BOUNDS_4D = jnp.array([
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0],
])

FULL_COVARIANCE_4D = jnp.array([
    [0.09, 0.00, 0.06, 0.00],
    [0.00, 0.04, 0.00, 0.00],
    [0.06, 0.00, 0.16, 0.00],
    [0.00, 0.00, 0.00, 0.25],
])

PARAM_BOUNDS_4D = jnp.array([
    [10.0, -5.0, 100.0, 0.0],
    [20.0,  5.0, 300.0, 4.0],
])

PARAM_WIDTHS_4D = PARAM_BOUNDS_4D[1] - PARAM_BOUNDS_4D[0]

PHYSICAL_COVARIANCE_4D = (
    FULL_COVARIANCE_4D * jnp.outer(PARAM_WIDTHS_4D, PARAM_WIDTHS_4D)
)

DIMS_02 = jnp.array([0, 2])

PHYSICAL_COVARIANCE_02 = (
    COVARIANCE_2D * jnp.outer(PARAM_WIDTHS_4D[DIMS_02], PARAM_WIDTHS_4D[DIMS_02])
)


def squared_transformed_difference(transform, xa, xb):
    return jnp.square(transform.forward(xa) - transform.forward(xb))


def test_covariance_rotation_diagonalises_covariance():
    transform = PrincipalAxesTransform(
        covariance=COVARIANCE_2D,
        param_bounds=UNIT_BOUNDS_2D,
    )
    rotated_covariance = transform.rotation.T @ COVARIANCE_2D @ transform.rotation
    np.testing.assert_allclose(
        rotated_covariance - jnp.diag(jnp.diag(rotated_covariance)), 0.0, atol=1e-6
    )
    assert rotated_covariance[0, 0] >= rotated_covariance[1, 1]


def test_forward_inverse_round_trip():
    transform = PrincipalAxesTransform(
        covariance=COVARIANCE_2D,
        param_bounds=UNIT_BOUNDS_2D,
    )
    x = jnp.array([[0.10, 0.25], [0.40, 0.70], [0.85, 0.30]])
    np.testing.assert_allclose(transform.inverse(transform.forward(x)), x, atol=1e-6)


def test_partial_rotation_leaves_unselected_dimensions_unchanged():
    transform = PrincipalAxesTransform(
        covariance=COVARIANCE_2D,
        rotation_dims=[0, 2],
        param_bounds=UNIT_BOUNDS_4D,
    )
    x = jnp.array([[0.10, 0.20, 0.30, 0.40], [0.50, 0.60, 0.70, 0.80]])
    transformed = transform.forward(x)
    np.testing.assert_allclose(transformed[:, [1, 3]], x[:, [1, 3]], atol=1e-6)
    np.testing.assert_allclose(transform.inverse(transformed), x, atol=1e-6)


def test_full_and_reduced_covariance_give_same_metric():
    full_transform = PrincipalAxesTransform(
        covariance=FULL_COVARIANCE_4D,
        rotation_dims=[0, 2],
        param_bounds=UNIT_BOUNDS_4D,
    )
    reduced_transform = PrincipalAxesTransform(
        covariance=COVARIANCE_2D,
        rotation_dims=[0, 2],
        param_bounds=UNIT_BOUNDS_4D,
    )
    xa = jnp.array([0.15, 0.20, 0.35, 0.40])
    xb = jnp.array([0.75, 0.60, 0.55, 0.80])
    np.testing.assert_allclose(
        squared_transformed_difference(full_transform, xa, xb),
        squared_transformed_difference(reduced_transform, xa, xb),
        atol=1e-6,
    )


def test_state_dict_round_trip():
    transform = PrincipalAxesTransform(
        covariance=COVARIANCE_2D,
        rotation_dims=[0, 2],
        param_bounds=UNIT_BOUNDS_4D,
    )
    restored = PrincipalAxesTransform.from_state_dict(transform.state_dict())
    x = jnp.array([[0.10, 0.20, 0.30, 0.40], [0.50, 0.60, 0.70, 0.80]])
    np.testing.assert_allclose(restored.forward(x), transform.forward(x), atol=1e-6)
    np.testing.assert_array_equal(restored.rotation_dims, transform.rotation_dims)


def test_static_transform_requires_covariance():
    with pytest.raises(ValueError, match="Provide covariance"):
        PrincipalAxesTransform()


def test_covariance_must_be_square():
    with pytest.raises(ValueError, match="Covariance matrix must be square"):
        PrincipalAxesTransform(
            covariance=jnp.ones((2, 3)),
            param_bounds=UNIT_BOUNDS_2D,
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
            param_bounds=UNIT_BOUNDS_2D,
        )


def test_rotation_dims_must_be_compatible_with_covariance():
    with pytest.raises(ValueError, match="incompatible"):
        PrincipalAxesTransform(
            covariance=COVARIANCE_2D,
            rotation_dims=[0, 1, 3],
            param_bounds=UNIT_BOUNDS_4D,
        )


def test_scale_physical_covariance_to_unit_space():
    cov = scale_covariance_to_unit(PHYSICAL_COVARIANCE_4D, PARAM_BOUNDS_4D)
    np.testing.assert_allclose(cov, FULL_COVARIANCE_4D, atol=1e-6)

    sub_from_full = scale_covariance_to_unit(
        PHYSICAL_COVARIANCE_4D, PARAM_BOUNDS_4D, dims=[0, 2]
    )
    sub_from_reduced = scale_covariance_to_unit(
        PHYSICAL_COVARIANCE_02, PARAM_BOUNDS_4D, dims=[0, 2]
    )
    np.testing.assert_allclose(sub_from_full, COVARIANCE_2D, atol=1e-6)
    np.testing.assert_allclose(sub_from_reduced, COVARIANCE_2D, atol=1e-6)


def test_full_and_reduced_physical_covariance_give_same_rotation():
    full_transform = PrincipalAxesTransform(
        covariance=PHYSICAL_COVARIANCE_4D,
        rotation_dims=[0, 2],
        param_bounds=PARAM_BOUNDS_4D,
    )
    reduced_transform = PrincipalAxesTransform(
        covariance=PHYSICAL_COVARIANCE_02,
        rotation_dims=[0, 2],
        param_bounds=PARAM_BOUNDS_4D,
    )
    xa = jnp.array([0.15, 0.20, 0.35, 0.40])
    xb = jnp.array([0.75, 0.60, 0.55, 0.80])
    np.testing.assert_allclose(
        squared_transformed_difference(full_transform, xa, xb),
        squared_transformed_difference(reduced_transform, xa, xb),
        atol=1e-6,
    )


def test_physical_space_transform_deferred_initialize_and_named_dims():
    transform = PrincipalAxesTransform(
        covariance=PHYSICAL_COVARIANCE_02,
        rotation_dims=["a", "c"],
    )
    assert transform.rotation is None
    with pytest.raises(RuntimeError, match="not initialized"):
        transform.forward(jnp.ones((2, 4)))
    with pytest.raises(RuntimeError, match="not initialized"):
        transform.inverse(jnp.ones((2, 4)))

    transform.initialize(param_bounds=PARAM_BOUNDS_4D, param_names=["a", "b", "c", "d"])
    assert transform.rotation is not None
    np.testing.assert_array_equal(transform.rotation_dims, jnp.array([0, 2]))

    reference = PrincipalAxesTransform(
        covariance=COVARIANCE_2D,
        rotation_dims=[0, 2],
        param_bounds=UNIT_BOUNDS_4D,
    )
    xa = jnp.array([0.15, 0.20, 0.35, 0.40])
    xb = jnp.array([0.75, 0.60, 0.55, 0.80])
    np.testing.assert_allclose(
        squared_transformed_difference(transform, xa, xb),
        squared_transformed_difference(reference, xa, xb),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Adaptive PrincipalAxesTransform tests
# ---------------------------------------------------------------------------

def _gaussian_mc_samples(seed=1, mean=None, scale=1.0, n=500):
    rng = np.random.default_rng(seed)
    mean = np.array([0.4, 0.5, 0.6]) if mean is None else np.asarray(mean)
    base = np.array([
        [0.04, 0.025, 0.010],
        [0.025, 0.09, -0.015],
        [0.010, -0.015, 0.06],
    ])
    return {"x": rng.multivariate_normal(mean, scale * base, size=n), "method": "MCMC"}


def test_default_is_static_and_try_update_is_noop():
    transform = PrincipalAxesTransform(
        covariance=COVARIANCE_2D,
        param_bounds=UNIT_BOUNDS_2D,
    )
    assert not transform.adaptive
    assert transform.max_updates == 0
    assert not transform.try_update(
        {"x": np.random.default_rng(0).normal(0.5, 0.1, size=(100, 2))},
        acq_val=0.01,
        step=1,
        acq_threshold=1.0,
    )
    assert transform.update_count == 0


def test_adaptive_without_initial_covariance_starts_as_identity():
    transform = PrincipalAxesTransform(
        update_dims=[0, 1, 2],
        max_updates=3,
        acq_threshold=0.5,
    )
    assert transform.adaptive
    assert transform.rotation is not None
    np.testing.assert_allclose(transform.rotation, np.eye(3))

    deferred = PrincipalAxesTransform(max_updates=2, acq_threshold=0.5)
    assert deferred.rotation is None
    deferred.initialize(ndim=3)
    np.testing.assert_array_equal(deferred.update_dims, jnp.array([0, 1, 2]))
    np.testing.assert_allclose(deferred.rotation, np.eye(3))

    named = PrincipalAxesTransform(update_dims=["a", "c"], max_updates=2)
    assert named.rotation is None
    named.initialize(param_bounds=PARAM_BOUNDS_4D, param_names=["a", "b", "c", "d"])
    np.testing.assert_array_equal(named.update_dims, jnp.array([0, 2]))
    np.testing.assert_allclose(named.rotation, np.eye(2))


def test_subspace_covariance_is_embedded_in_larger_update_space():
    transform = PrincipalAxesTransform(
        covariance=PHYSICAL_COVARIANCE_02,
        initial_dims=[0, 2],
        update_dims=[0, 2, 3],
        param_bounds=PARAM_BOUNDS_4D,
        max_updates=2,
        acq_threshold=0.5,
    )
    assert transform.rotation.shape == (3, 3)
    np.testing.assert_allclose(transform.rotation[2], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(transform.rotation[:, 2], [0.0, 0.0, 1.0])

    ref = PrincipalAxesTransform(
        covariance=PHYSICAL_COVARIANCE_02,
        rotation_dims=[0, 2],
        param_bounds=PARAM_BOUNDS_4D,
    )
    xa = jnp.array([0.15, 0.20, 0.35, 0.40])
    xb = jnp.array([0.75, 0.60, 0.55, 0.80])
    np.testing.assert_allclose(
        squared_transformed_difference(transform, xa, xb),
        squared_transformed_difference(ref, xa, xb),
        atol=1e-6,
    )
    transformed = transform.forward(xa[None, :])
    np.testing.assert_allclose(transformed[:, 1], xa[1])
    np.testing.assert_allclose(transformed[:, 3], xa[3])


def test_deferred_named_initial_and_update_dims_with_physical_covariance():
    transform = PrincipalAxesTransform(
        covariance=PHYSICAL_COVARIANCE_02,
        initial_dims=["a", "c"],
        update_dims=["a", "c", "d"],
        max_updates=3,
        acq_threshold=0.5,
    )
    assert transform.rotation is None
    transform.initialize(param_bounds=PARAM_BOUNDS_4D, param_names=["a", "b", "c", "d"])
    np.testing.assert_array_equal(transform.initial_dims, jnp.array([0, 2]))
    np.testing.assert_array_equal(transform.update_dims, jnp.array([0, 2, 3]))
    assert transform.rotation.shape == (3, 3)
    np.testing.assert_allclose(transform.rotation[2], [0.0, 0.0, 1.0])


def test_subset_initial_covariance_adapts_over_all_parameters():
    transform = PrincipalAxesTransform(
        covariance=PHYSICAL_COVARIANCE_02,
        initial_dims=["a", "c"],
        max_updates=3,
        acq_threshold=0.5,
    )
    assert transform.rotation is None
    transform.initialize(param_bounds=PARAM_BOUNDS_4D, param_names=["a", "b", "c", "d"])

    np.testing.assert_array_equal(transform.initial_dims, jnp.array([0, 2]))
    np.testing.assert_array_equal(transform.update_dims, jnp.array([0, 1, 2, 3]))
    assert transform.rotation.shape == (4, 4)
    # Uninitialized dims 1 and 3 start as identity axes
    np.testing.assert_allclose(transform.rotation[1], [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(transform.rotation[3], [0.0, 0.0, 0.0, 1.0])
    # Embedded [0, 2] block matches standalone 2D rotation
    ref = PrincipalAxesTransform(
        covariance=PHYSICAL_COVARIANCE_02,
        rotation_dims=[0, 2],
        param_bounds=PARAM_BOUNDS_4D,
    )
    np.testing.assert_allclose(
        transform.rotation[np.ix_([0, 2], [0, 2])], ref.rotation, atol=1e-6
    )

    # Adaptive update replaces the embedded rotation with a full 4D rotation
    rng = np.random.default_rng(13)
    cov_4d = np.array([
        [0.04, 0.02, 0.01, -0.01],
        [0.02, 0.05, 0.015, 0.02],
        [0.01, 0.015, 0.06, -0.015],
        [-0.01, 0.02, -0.015, 0.03],
    ])
    mc = {"x": rng.multivariate_normal(np.full(4, 0.5), cov_4d, size=300)}
    assert transform.try_update(mc, acq_val=0.4, step=1)
    assert transform.update_count == 1
    assert transform.reference_covariance.shape == (4, 4)
    assert not np.allclose(transform.rotation[1], [0.0, 1.0, 0.0, 0.0])
    assert not np.allclose(transform.rotation[3], [0.0, 0.0, 0.0, 1.0])


def test_first_adaptive_update_is_gated_only_by_acquisition():
    transform = PrincipalAxesTransform(
        covariance=np.eye(2),
        initial_dims=[0, 1],
        update_dims=[0, 1, 2],
        param_bounds=jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        acq_threshold=0.5,
        update_step=100,
        max_updates=3,
        kl_threshold=1e6,
    )
    assert not transform.try_update(_gaussian_mc_samples(), acq_val=0.6, step=1)
    assert transform.try_update(_gaussian_mc_samples(), acq_val=0.4, step=1)
    assert transform.update_count == 1
    assert transform.reference_covariance.shape == (3, 3)
    assert transform.last_update_kl is None
    assert transform.last_update_ii == 1
    assert transform.last_update_acq_val == pytest.approx(0.4)


def test_later_adaptive_updates_apply_gap_improvement_kl_and_max_updates_gates():
    transform = PrincipalAxesTransform(
        update_dims=[0, 1, 2],
        acq_threshold=0.5,
        update_step=5,
        max_updates=2,
        kl_threshold=0.1,
    )
    baseline = _gaussian_mc_samples(seed=2)
    assert transform.try_update(baseline, acq_val=0.4, step=1)

    changed = _gaussian_mc_samples(seed=3, mean=[0.9, 0.5, 0.6])
    assert not transform.try_update(changed, acq_val=0.3, step=4)   # step gap
    assert not transform.try_update(changed, acq_val=0.6, step=6)   # above threshold
    assert not transform.try_update(changed, acq_val=0.45, step=6)  # no acq improvement
    assert not transform.try_update(baseline, acq_val=0.3, step=6)  # KL too small
    assert transform.try_update(changed, acq_val=0.3, step=6)
    assert transform.update_count == 2
    assert transform.last_update_kl >= transform.kl_threshold

    further = _gaussian_mc_samples(seed=4, mean=[0.1, 0.1, 0.1])
    assert not transform.try_update(further, acq_val=0.1, step=20)
    assert transform.update_count == 2


def test_kl_accepts_scale_change_even_when_eigenvectors_do_not_change():
    transform = PrincipalAxesTransform(
        update_dims=[0, 1, 2],
        acq_threshold=0.5,
        update_step=1,
        max_updates=2,
        kl_threshold=0.1,
    )
    baseline = _gaussian_mc_samples(seed=8)
    assert transform.try_update(baseline, acq_val=0.4, step=1)
    samples = baseline["x"]
    scaled = {"x": samples.mean(axis=0) + 2.0 * (samples - samples.mean(axis=0))}
    assert transform.try_update(scaled, acq_val=0.3, step=2)
    assert transform.last_update_kl > transform.kl_threshold


def test_update_step_none_allows_only_first_update():
    transform = PrincipalAxesTransform(
        update_dims=[0, 1, 2],
        acq_threshold=0.5,
        update_step=None,
        max_updates=5,
        kl_threshold=0.0,
    )
    assert transform.try_update(_gaussian_mc_samples(seed=1), acq_val=0.4, step=1)
    assert not transform.try_update(
        _gaussian_mc_samples(seed=2, mean=[0.9, 0.1, 0.2]), acq_val=0.1, step=100
    )
    assert transform.update_count == 1


def test_hungarian_axis_alignment_preserves_axis_order_and_signs():
    rng = np.random.default_rng(11)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    permuted = q[:, [2, 0, 1]] * np.array([-1.0, 1.0, -1.0])
    aligned = PrincipalAxesTransform._align_axes(q, permuted)
    np.testing.assert_allclose(aligned, q, atol=1e-12)


def test_adaptive_update_handles_weights_nans_and_round_trip():
    mc = _gaussian_mc_samples(seed=5, n=200)
    mc["x"][0, 1] = np.nan
    weights = np.ones(200)
    weights[1] = np.nan
    mc["weights"] = weights

    transform = PrincipalAxesTransform(
        update_dims=[0, 2],
        acq_threshold=1.0,
        max_updates=2,
    )
    assert transform.try_update(mc, acq_val=0.5, step=2)
    x_np = np.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])
    forward = transform.forward(x_np)
    np.testing.assert_allclose(forward[:, 1], x_np[:, 1], atol=1e-12)
    np.testing.assert_allclose(transform.inverse(forward), x_np, atol=1e-12)


def test_adaptive_state_dict_round_trip():
    transform = PrincipalAxesTransform(
        covariance=PHYSICAL_COVARIANCE_02,
        initial_dims=[0, 2],
        update_dims=[0, 2, 3],
        param_bounds=PARAM_BOUNDS_4D,
        acq_threshold=0.5,
        update_step=4,
        max_updates=3,
        kl_threshold=0.2,
        regularize_eps=1e-8,
    )
    rng = np.random.default_rng(7)
    mc = {"x": rng.multivariate_normal(np.full(4, 0.5), 0.02 * np.eye(4), size=200)}
    assert transform.try_update(mc, acq_val=0.3, step=5)

    restored = PrincipalAxesTransform.from_state_dict(transform.state_dict())
    assert restored.adaptive is True
    assert restored.update_count == 1
    assert restored.last_update_ii == 5
    assert restored.last_update_acq_val == pytest.approx(0.3)
    assert restored.last_update_kl is None
    assert restored.acq_threshold == pytest.approx(0.5)
    assert restored.update_step == 4
    assert restored.max_updates == 3
    assert restored.kl_threshold == pytest.approx(0.2)
    assert restored.regularize_eps == pytest.approx(1e-8)
    np.testing.assert_array_equal(restored.initial_dims, transform.initial_dims)
    np.testing.assert_array_equal(restored.update_dims, transform.update_dims)
    np.testing.assert_allclose(restored.rotation, transform.rotation)
    np.testing.assert_allclose(restored.reference_mean, transform.reference_mean)
    np.testing.assert_allclose(
        restored.reference_covariance, transform.reference_covariance
    )


def test_adaptive_invalid_arguments_are_rejected():
    with pytest.raises(ValueError, match="max_updates"):
        PrincipalAxesTransform(covariance=COVARIANCE_2D, max_updates=-1)
    with pytest.raises(ValueError, match="update_step"):
        PrincipalAxesTransform(update_dims=[0, 1], max_updates=2, update_step=0)
    with pytest.raises(ValueError, match="kl_threshold"):
        PrincipalAxesTransform(update_dims=[0, 1], max_updates=2, kl_threshold=-0.1)
    with pytest.raises(ValueError, match="regularize_eps"):
        PrincipalAxesTransform(update_dims=[0, 1], max_updates=2, regularize_eps=-1e-6)
    with pytest.raises(ValueError, match="not both"):
        PrincipalAxesTransform(
            rotation_dims=[0, 1], update_dims=[0, 1], max_updates=2
        )
    with pytest.raises(ValueError, match="subset"):
        PrincipalAxesTransform(
            covariance=COVARIANCE_2D,
            initial_dims=[0, 3],
            update_dims=[0, 1, 2],
            max_updates=2,
        )
    transform = PrincipalAxesTransform(update_dims=[0, 1], max_updates=2)
    with pytest.raises(ValueError, match="acq_threshold"):
        transform.try_update(_gaussian_mc_samples(), acq_val=0.2, step=1)

    transform = PrincipalAxesTransform(
        update_dims=[0, 1, 2], max_updates=2, acq_threshold=1.0
    )
    with pytest.raises(ValueError, match="finite posterior samples"):
        transform.try_update({"x": np.zeros((3, 3))}, acq_val=0.2, step=1)
