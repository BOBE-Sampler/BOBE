import numpy as np
import jax.numpy as jnp

from BOBE.clf_gp import GPwithClassifier


TRAIN_X = jnp.array([
    [0.10, 0.20],
    [0.25, 0.70],
    [0.45, 0.35],
    [0.70, 0.85],
    [0.90, 0.55],
])

TRAIN_Y = jnp.array([
    [-1.2],
    [-0.4],
    [0.3],
    [-0.8],
    [0.7],
])

COVARIANCE = jnp.array([
    [0.09, 0.06],
    [0.06, 0.16],
])


def make_classifier_gp(**rotation_kwargs):
    return GPwithClassifier(
        train_x=TRAIN_X,
        train_y=TRAIN_Y,
        clf_type="svm",
        clf_use_size=100,
        train_clf_on_init=False,
        lengthscales=jnp.array([0.20, 0.65]),
        kernel_variance=1.2,
        gp_threshold=500.0,
        **rotation_kwargs,
    )


def test_classifier_gp_passes_rotation_to_base_gp():
    gp = make_classifier_gp(
        rotation_covariance=COVARIANCE
    )

    assert gp.kernel.input_transform is not None
    assert gp.kernel.input_transform.rotation.shape == (2, 2)


def test_classifier_gp_rotation_affects_gp_kernel():
    rotated = make_classifier_gp(
        rotation_covariance=COVARIANCE
    )

    unrotated = make_classifier_gp()

    rotated_kernel = rotated.kernel.covariance(
        rotated.train_x,
        rotated.train_x,
        include_noise=False,
    )

    unrotated_kernel = unrotated.kernel.covariance(
        unrotated.train_x,
        unrotated.train_x,
        include_noise=False,
    )

    assert not np.allclose(
        rotated_kernel,
        unrotated_kernel,
        rtol=1e-5,
        atol=1e-7,
    )


def test_classifier_gp_state_dict_contains_transform():
    gp = make_classifier_gp(
        rotation_covariance=COVARIANCE
    )

    state = gp.state_dict()

    assert state["input_transform_state"] is not None
    assert (
        state["input_transform_state"]["type"]
        == "PrincipalAxesTransform"
    )


def test_classifier_gp_state_dict_round_trip():
    gp = make_classifier_gp(
        rotation_covariance=COVARIANCE
    )

    restored = GPwithClassifier.from_state_dict(
        gp.state_dict()
    )

    assert restored.kernel.input_transform is not None

    np.testing.assert_allclose(
        restored.kernel.covariance(
            restored.train_x,
            restored.train_x,
            include_noise=False,
        ),
        gp.kernel.covariance(
            gp.train_x,
            gp.train_x,
            include_noise=False,
        ),
        rtol=1e-6,
        atol=1e-7,
    )


def test_classifier_gp_predictions_survive_state_round_trip():
    gp = make_classifier_gp(
        rotation_covariance=COVARIANCE
    )

    x = jnp.array([
        [0.30, 0.45],
        [0.75, 0.25],
    ])

    expected_mean = gp.predict_mean_batched(x)
    expected_var = gp.predict_var_batched(x)

    restored = GPwithClassifier.from_state_dict(
        gp.state_dict()
    )

    np.testing.assert_allclose(
        restored.predict_mean_batched(x),
        expected_mean,
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        restored.predict_var_batched(x),
        expected_var,
        rtol=1e-6,
        atol=1e-7,
    )


def test_classifier_gp_save_load_round_trip(tmp_path):
    gp = make_classifier_gp(
        rotation_covariance=COVARIANCE
    )

    filename = tmp_path / "rotated_classifier_gp"

    gp.save(str(filename))

    loaded = GPwithClassifier.load(
        str(filename)
    )

    assert loaded.kernel.input_transform is not None

    np.testing.assert_allclose(
        loaded.kernel.covariance(
            loaded.train_x,
            loaded.train_x,
            include_noise=False,
        ),
        gp.kernel.covariance(
            gp.train_x,
            gp.train_x,
            include_noise=False,
        ),
        rtol=1e-6,
        atol=1e-7,
    )


def test_classifier_itself_still_uses_original_coordinates():
    gp = make_classifier_gp(
        rotation_covariance=COVARIANCE
    )

    np.testing.assert_allclose(
        gp.train_x_clf,
        TRAIN_X,
        atol=1e-7,
    )