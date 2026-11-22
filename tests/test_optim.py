"""Tests for BOBE.optim (optimize_scipy, optimize_optax, optimize_optax_vmap)."""

import jax.numpy as jnp
import numpy as np
import pytest

from BOBE.acquisition import EI, WIPStd
from BOBE.optim import (
    _SCIPY_DEFAULT_OPTIONS,
    _copy_options,
    _get_optimizer,
    _setup_bounds,
    optimize_optax,
    optimize_optax_vmap,
    optimize_scipy,
)


def _quadratic(x, target=jnp.array([0.3, -0.4])):
    return jnp.sum((x - target) ** 2)


def _double_well(x):
    # Global minimum at x = 2.0 (f = 0.0), local minimum at x = -2.0 (f = 1.0)
    return jnp.minimum((x[0] - 2.0) ** 2, (x[0] + 2.0) ** 2 + 1.0)


def test_setup_bounds_and_get_optimizer():
    assert _setup_bounds(None, 3) is None

    shared = _setup_bounds([-1.0, 2.0], 3)
    assert shared.shape == (2, 3)
    np.testing.assert_allclose(shared[0], [-1.0, -1.0, -1.0])
    np.testing.assert_allclose(shared[1], [2.0, 2.0, 2.0])

    explicit = _setup_bounds(np.array([[-1.0, 0.0], [1.0, 5.0]]), 2)
    assert explicit.shape == (2, 2)

    with pytest.raises(ValueError, match="incompatible"):
        _setup_bounds(np.zeros((2, 3)), 2)

    # _get_optimizer resolves known optimizers without mutating caller kwargs
    kwargs = {"b1": 0.8}
    opt = _get_optimizer("adam", learning_rate=1e-2, optimizer_kwargs=kwargs)
    assert opt is not None
    assert kwargs == {"b1": 0.8}

    assert _get_optimizer("sgd", learning_rate=1e-2) is not None
    with pytest.raises(ValueError, match="not found in optax"):
        _get_optimizer("nonexistent_optimizer_xyz")


def test_optimize_scipy_unbounded_bounded_and_multistart():
    # Unbounded quadratic
    x0 = jnp.array([[2.0, 2.0], [-1.5, 1.5]])
    best_x, best_f = optimize_scipy(
        _quadratic, num_params=2, x0=x0, maxiter=100, n_restarts=2
    )
    np.testing.assert_allclose(best_x, [0.3, -0.4], atol=1e-5)
    assert best_f == pytest.approx(float(_quadratic(best_x)), abs=1e-10)

    # Bounded quadratic where unconstrained optimum (0.3, -0.4) is clipped to [0.5, 1.0] x [-0.2, 1.0]
    bounds = jnp.array([[0.5, -0.2], [1.0, 1.0]])
    x0_b = jnp.array([[0.8, 0.5]])
    best_x_b, best_f_b = optimize_scipy(
        _quadratic, num_params=2, bounds=bounds, x0=x0_b, maxiter=100, n_restarts=1
    )
    np.testing.assert_allclose(best_x_b, [0.5, -0.2], atol=1e-5)
    assert best_f_b == pytest.approx(float(_quadratic(best_x_b)), abs=1e-10)

    # Multi-restart picks global minimum of double well
    x0_dw = jnp.array([[-2.5], [2.5]])
    best_x_dw, best_f_dw = optimize_scipy(
        _double_well, num_params=1, x0=x0_dw, maxiter=100, n_restarts=2
    )
    np.testing.assert_allclose(best_x_dw, [2.0], atol=1e-4)
    assert best_f_dw == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("opt_fn", [optimize_optax, optimize_optax_vmap])
def test_optimize_optax_unbounded_and_bounded(opt_fn):
    # Unbounded quadratic
    x0 = jnp.array([[1.2, -1.2], [-0.8, 0.9]])
    best_x, best_f = opt_fn(
        _quadratic,
        num_params=2,
        x0=x0,
        optimizer_options={"name": "adam", "lr": 5e-2, "early_stop_patience": 30},
        maxiter=300,
        n_restarts=2,
    )
    np.testing.assert_allclose(best_x, [0.3, -0.4], atol=1e-2)
    assert best_f == pytest.approx(float(_quadratic(best_x)), abs=1e-8)

    # Bounded quadratic (optax takes unit-cube x0 and returns original-space params)
    bounds = jnp.array([[0.5, -0.2], [1.5, 1.0]])
    x0_u = jnp.array([[0.7, 0.7], [0.2, 0.4]])
    best_x_b, best_f_b = opt_fn(
        _quadratic,
        num_params=2,
        bounds=bounds,
        x0=x0_u,
        optimizer_options={"name": "adam", "lr": 5e-2, "early_stop_patience": 30},
        maxiter=300,
        n_restarts=2,
    )
    np.testing.assert_allclose(best_x_b, [0.5, -0.2], atol=1e-2)
    assert best_f_b == pytest.approx(float(_quadratic(best_x_b)), abs=1e-8)


@pytest.mark.parametrize("opt_fn", [optimize_optax, optimize_optax_vmap])
def test_optax_pairs_best_params_and_scores_final_update(opt_fn):
    def bowl(x):
        return jnp.sum(x ** 2)

    # Overshooting SGD (lr=1.2 on x^2 diverges: x_0=1 -> x_1=-1.4 -> x_2=1.96 ...).
    # Must return the best iterate (x_0=1.0, f=1.0), NOT the worse post-step iterate.
    x0 = jnp.array([[1.0]])
    best_x, best_f = opt_fn(
        bowl,
        num_params=1,
        x0=x0,
        optimizer_options={"name": "sgd", "lr": 1.2, "early_stop_patience": 3},
        maxiter=5,
        n_restarts=1,
    )
    np.testing.assert_allclose(best_x, [1.0], atol=1e-12)
    assert best_f == pytest.approx(1.0, abs=1e-12)
    assert best_f == pytest.approx(float(bowl(best_x)), abs=1e-12)

    # Single descending SGD step (lr=0.25 on x^2: x_0=1.0 -> x_1=0.5, f=0.25).
    # Post-loop evaluation must score x_1 and return (0.5, 0.25).
    best_x1, best_f1 = opt_fn(
        bowl,
        num_params=1,
        x0=x0,
        optimizer_options={"name": "sgd", "lr": 0.25, "early_stop_patience": 5},
        maxiter=1,
        n_restarts=1,
    )
    np.testing.assert_allclose(best_x1, [0.5], atol=1e-12)
    assert best_f1 == pytest.approx(0.25, abs=1e-12)
    assert best_f1 == pytest.approx(float(bowl(best_x1)), abs=1e-12)


def test_optimizer_options_not_mutated_across_calls():
    scipy_opts = {"method": "L-BFGS-B", "ftol": 1e-8, "gtol": 1e-8}
    scipy_snapshot = dict(scipy_opts)
    x0 = jnp.array([[1.0, 1.0]])
    for _ in range(2):
        optimize_scipy(_quadratic, num_params=2, x0=x0, optimizer_options=scipy_opts, maxiter=20, n_restarts=1)
        assert scipy_opts == scipy_snapshot

    optax_opts = {"name": "sgd", "lr": 1e-2, "early_stop_patience": 5}
    optax_snapshot = dict(optax_opts)
    for fn in (optimize_optax, optimize_optax_vmap):
        for _ in range(2):
            fn(_quadratic, num_params=2, x0=x0, optimizer_options=optax_opts, maxiter=10, n_restarts=1)
            assert optax_opts == optax_snapshot

    # AcquisitionFunction copies caller optimizer_options rather than aliasing it
    caller_opts = {"method": "L-BFGS-B", "ftol": 1e-7}
    ei = EI(optimizer="scipy", optimizer_options=caller_opts)
    wip = WIPStd(optimizer="scipy", optimizer_options=caller_opts)
    caller_opts["method"] = "Nelder-Mead"
    assert ei.optimizer_options["method"] == "L-BFGS-B"
    assert wip.optimizer_options["method"] == "L-BFGS-B"

    # Empty dict from default AcquisitionFunction/GP still receives scipy defaults
    assert _copy_options(None, default=_SCIPY_DEFAULT_OPTIONS) == _SCIPY_DEFAULT_OPTIONS
    assert _copy_options({}, default=_SCIPY_DEFAULT_OPTIONS) == _SCIPY_DEFAULT_OPTIONS
    assert _copy_options({"method": "TNC"}, default=_SCIPY_DEFAULT_OPTIONS) == {"method": "TNC"}


@pytest.mark.parametrize("opt_fn", [optimize_scipy, optimize_optax, optimize_optax_vmap])
def test_optim_input_validation(opt_fn):
    with pytest.raises(ValueError, match="x0"):
        opt_fn(_quadratic, num_params=2, x0=None, n_restarts=1)

    with pytest.raises(ValueError, match="restarts"):
        opt_fn(_quadratic, num_params=2, x0=jnp.array([[0.5, 0.5]]), n_restarts=3)
