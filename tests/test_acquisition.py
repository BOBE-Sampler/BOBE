"""
Tests for Acquisition Functions.

Tests include:
- Expected Improvement (EI)
- Log Expected Improvement (LogEI)
- Acquisition function optimization
- Batch acquisition
- Integration with GP
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import sys
from scipy.special import erfcx as scipy_erfcx
from BOBE import BOBE, ClassifierGP, SVMClassifier
from BOBE.gp import GP
from BOBE.kernels import RBFKernel, MaternKernel
from BOBE.acquisition import (
    AcquisitionFunction,
    EI,
    LogEI,
    WIPV,
    WIPStd,
    _erfcx_jax,
    _log1mexp_jax,
    get_mc_points,
)


def generate_test_gp(n_samples=30, d=2, seed=42):
    """Generate a simple GP for testing acquisition functions."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, size=(n_samples, d))
    # Simple quadratic function with maximum at [0.7, 0.7]
    y = -np.sum((X - 0.7)**2, axis=1, keepdims=True)

    gp = GP(
        train_x=X,
        train_y=y,
        kernel=RBFKernel(
            lengthscales=jnp.array([0.3] * d),
            kernel_variance=1.0,
            noise=1e-6,
        ),
    )

    return gp


def test_ei_initialization():
    """Test EI acquisition function initialization."""
    print("\n" + "="*80)
    print("TEST: EI Initialization")
    print("="*80)
    
    ei = EI(optimizer="scipy")
    
    assert ei.name == "EI", f"Expected name 'EI', got {ei.name}"
    assert ei.optimizer == "scipy", "Optimizer should be scipy"
    
    print(f"✓ EI initialized with optimizer: {ei.optimizer}")
    print(f"✓ Acquisition function name: {ei.name}")


def test_logei_initialization():
    """Test LogEI acquisition function initialization."""
    print("\n" + "="*80)
    print("TEST: LogEI Initialization")
    print("="*80)
    
    logei = LogEI(optimizer="scipy")
    
    assert logei.name == "LogEI", f"Expected name 'LogEI', got {logei.name}"
    assert logei.optimizer == "scipy", "Optimizer should be scipy"
    
    print(f"✓ LogEI initialized with optimizer: {logei.optimizer}")
    print(f"✓ Acquisition function name: {logei.name}")


def test_ei_evaluation():
    """Test EI acquisition function evaluation."""
    print("\n" + "="*80)
    print("TEST: EI Evaluation")
    print("="*80)
    
    gp = generate_test_gp(n_samples=25, d=2)
    ei = EI()
    
    # Test points
    test_points = [
        jnp.array([0.7, 0.7]),  # Near optimum
        jnp.array([0.1, 0.1]),  # Far from optimum
        jnp.array([0.5, 0.5]),  # Middle
    ]
    
    best_y = jnp.max(gp.train_y)
    zeta = 0.0
    
    print(f"Best observed y: {best_y:.4f}")
    print(f"\nEI evaluations:")
    
    for i, pt in enumerate(test_points):
        ei_val = -ei.fun(pt, gp, best_y, zeta)  # Negative because optimizer minimizes
        print(f"  Point {pt}: EI = {ei_val:.4e}")
        assert ei_val >= 0, f"EI must be non-negative, got {ei_val}"
    
    print(f"\n✓ EI evaluation successful, all values non-negative")


def test_logei_evaluation():
    """Test LogEI acquisition function evaluation."""
    print("\n" + "="*80)
    print("TEST: LogEI Evaluation")
    print("="*80)
    
    gp = generate_test_gp(n_samples=25, d=2)
    logei = LogEI()
    
    test_points = [
        jnp.array([0.7, 0.7]),
        jnp.array([0.1, 0.1]),
        jnp.array([0.5, 0.5]),
    ]
    
    best_y = jnp.max(gp.train_y)
    zeta = 0.0
    
    print(f"Best observed y: {best_y:.4f}")
    print(f"\nLogEI evaluations:")
    
    for i, pt in enumerate(test_points):
        logei_val = -logei.fun(pt, gp, best_y, zeta)  # Negative because optimizer minimizes
        print(f"  Point {pt}: LogEI = {logei_val:.4e}")
    
    print(f"\n✓ LogEI evaluation successful")


def test_ei_optimization():
    """Test EI optimization to find next point."""
    print("\n" + "="*80)
    print("TEST: EI Optimization")
    print("="*80)
    
    gp = generate_test_gp(n_samples=20, d=2, seed=123)
    ei = EI(optimizer="scipy")
    
    best_y = jnp.max(gp.train_y)
    acq_kwargs = {'best_y': float(best_y), 'zeta': 0.0}
    
    print(f"Optimizing EI to find next point...")
    print(f"Best observed y: {best_y:.4f}")
    
    rng = np.random.default_rng(42)
    next_point, ei_val = ei.get_next_point(
        gp=gp,
        acq_kwargs=acq_kwargs,
        maxiter=100,
        n_restarts=5,
        verbose=False,
        rng=rng
    )
    
    print(f"\nNext point: {next_point}")
    print(f"EI value: {ei_val:.4e}")
    
    assert next_point.shape == (2,), f"Expected shape (2,), got {next_point.shape}"
    assert np.all(next_point >= 0) and np.all(next_point <= 1), "Point should be in unit cube"
    assert ei_val >= 0, f"EI value should be non-negative, got {ei_val}"
    
    # Predict at the new point
    pred_mean = gp.predict_mean(next_point)
    print(f"GP prediction at next point: {pred_mean:.4f}")
    
    print(f"\n✓ EI optimization successful")


def test_logei_optimization():
    """Test LogEI optimization to find next point."""
    print("\n" + "="*80)
    print("TEST: LogEI Optimization")
    print("="*80)
    
    gp = generate_test_gp(n_samples=20, d=2, seed=456)
    logei = LogEI(optimizer="scipy")
    
    best_y = jnp.max(gp.train_y)
    acq_kwargs = {'best_y': float(best_y), 'zeta': 0.0}
    
    print(f"Optimizing LogEI to find next point...")
    print(f"Best observed y: {best_y:.4f}")
    
    rng = np.random.default_rng(42)
    next_point, logei_val = logei.get_next_point(
        gp=gp,
        acq_kwargs=acq_kwargs,
        maxiter=100,
        n_restarts=5,
        verbose=False,
        rng=rng
    )
    
    print(f"\nNext point: {next_point}")
    print(f"LogEI value: {logei_val:.4e}")
    
    assert next_point.shape == (2,), f"Expected shape (2,), got {next_point.shape}"
    assert np.all(next_point >= 0) and np.all(next_point <= 1), "Point should be in unit cube"
    
    pred_mean = gp.predict_mean(next_point)
    print(f"GP prediction at next point: {pred_mean:.4f}")
    
    print(f"\n✓ LogEI optimization successful")


def test_batch_acquisition():
    """Test batch acquisition (getting multiple points)."""
    print("\n" + "="*80)
    print("TEST: Batch Acquisition")
    print("="*80)
    
    gp = generate_test_gp(n_samples=25, d=2, seed=789)
    ei = EI(optimizer="scipy")
    
    best_y = jnp.max(gp.train_y)
    acq_kwargs = {'best_y': float(best_y), 'zeta': 0.0}
    
    n_batch = 3
    print(f"Getting batch of {n_batch} points...")
    
    rng = np.random.default_rng(42)
    batch_points, batch_vals = ei.get_next_batch(
        gp=gp,
        n_batch=n_batch,
        acq_kwargs=acq_kwargs,
        maxiter=100,
        n_restarts=3,
        verbose=False,
        rng=rng
    )
    
    print(f"\nBatch points shape: {batch_points.shape}")
    print(f"Batch EI values: {batch_vals}")
    
    assert batch_points.shape == (n_batch, 2), f"Expected shape ({n_batch}, 2), got {batch_points.shape}"
    assert batch_vals.shape == (n_batch,), f"Expected shape ({n_batch},), got {batch_vals.shape}"
    assert np.all(batch_points >= 0) and np.all(batch_points <= 1), "All points should be in unit cube"
    
    for i, pt in enumerate(batch_points):
        print(f"  Point {i+1}: {pt}, EI={batch_vals[i]:.4e}")
    
    print(f"\n✓ Batch acquisition successful")


def test_acquisition_with_different_gp_settings():
    """Test acquisition functions with different GP configurations."""
    print("\n" + "="*80)
    print("TEST: Acquisition with Different GP Settings")
    print("="*80)
    
    # GP with RBF kernel
    gp_rbf = generate_test_gp(n_samples=20, d=2, seed=111)

    # GP with Matern kernel
    rng = np.random.RandomState(222)
    X = rng.uniform(0, 1, size=(20, 2))
    y = -np.sum((X - 0.7)**2, axis=1, keepdims=True)
    gp_matern = GP(train_x=X, train_y=y, kernel=MaternKernel(noise=1e-6))
    
    ei = EI()
    best_y = jnp.max(gp_rbf.train_y)
    
    test_point = jnp.array([0.6, 0.6])
    
    ei_rbf = -ei.fun(test_point, gp_rbf, best_y, 0.0)
    ei_matern = -ei.fun(test_point, gp_matern, best_y, 0.0)
    
    print(f"EI with RBF kernel: {ei_rbf:.4e}")
    print(f"EI with Matern kernel: {ei_matern:.4e}")
    
    assert ei_rbf > 0 and ei_matern > 0, "Both should give positive EI"
    
    print(f"\n✓ Acquisition works with different kernels")


def test_acquisition_optimization_convergence():
    """Test that optimization converges to reasonable points."""
    print("\n" + "="*80)
    print("TEST: Acquisition Optimization Convergence")
    print("="*80)
    
    # Create GP with known optimum near [0.8, 0.8]
    rng = np.random.RandomState(333)
    X = rng.uniform(0, 1, size=(30, 2))
    y = -np.sum((X - 0.8)**2, axis=1, keepdims=True)
    
    gp = GP(train_x=X, train_y=y, kernel=RBFKernel(noise=1e-6))
    ei = EI(optimizer="scipy")
    
    best_y = jnp.max(gp.train_y)
    acq_kwargs = {'best_y': float(best_y), 'zeta': 0.0}
    
    # Run optimization multiple times
    rng_test = np.random.default_rng(42)
    next_points = []
    
    for i in range(3):
        next_pt, _ = ei.get_next_point(
            gp=gp,
            acq_kwargs=acq_kwargs,
            maxiter=100,
            n_restarts=5,
            verbose=False,
            rng=rng_test
        )
        next_points.append(next_pt)
        print(f"  Run {i+1}: {next_pt}")
    
    # Check that points are generally near the optimum [0.8, 0.8]
    next_points = np.array(next_points)
    distances = np.linalg.norm(next_points - np.array([0.8, 0.8]), axis=1)
    avg_distance = np.mean(distances)
    
    print(f"\nAverage distance from true optimum [0.8, 0.8]: {avg_distance:.4f}")
    
    # Should be reasonably close (within half the domain)
    assert avg_distance < 0.5, f"Optimization not converging well, avg distance: {avg_distance}"
    
    print(f"✓ Acquisition optimization converges to reasonable regions")


def test_wip_acquisitions_and_mc_points():
    """Test WIPV/WIPStd with fantasy cache and NaN/size handling in get_mc_points."""
    gp = generate_test_gp(n_samples=20, d=2, seed=123)
    rng = np.random.default_rng(123)
    mc_points = rng.uniform(0.0, 1.0, size=(32, 2))

    wipv = WIPV()
    wipstd = WIPStd()
    pt_v, val_v = wipv.get_next_point(
        gp,
        acq_kwargs={"mc_samples": {"x": mc_points}, "mc_points_size": 16},
        maxiter=20,
        n_restarts=1,
        verbose=False,
        rng=rng,
    )
    pt_s, val_s = wipstd.get_next_point(
        gp,
        acq_kwargs={"mc_samples": {"x": mc_points}, "mc_points_size": 16},
        maxiter=20,
        n_restarts=1,
        verbose=False,
        rng=rng,
    )
    assert pt_v.shape == (2,) and np.isfinite(val_v)
    assert pt_s.shape == (2,) and np.isfinite(val_s)

    mc_with_nans = {"x": np.array([[0.1, 0.2], [np.nan, 0.3], [0.4, 0.5], [0.6, 0.7]])}
    pts = get_mc_points(mc_with_nans, mc_points_size=10, rng=np.random.default_rng(0))
    assert pts.shape == (3, 2)
    assert np.all(np.isfinite(pts))


def test_pure_jax_erfcx_and_log1mexp():
    """Test pure-JAX erfcx and log1mexp fallbacks for accuracy and finite gradients."""
    xs = jnp.array([0.0, 0.5, 2.0, 10.0, 24.9, 25.1, 50.0, 1e3, 1e6])
    vals = _erfcx_jax(xs)
    expected = scipy_erfcx(np.asarray(xs))
    np.testing.assert_allclose(np.asarray(vals), expected, rtol=1e-13, atol=1e-15)
    assert jnp.all(jnp.isfinite(jax.vmap(jax.grad(_erfcx_jax))(xs[1:])))

    ys = jnp.array([-1e-8, -0.1, -0.69, -0.70, -5.0, -50.0])
    lvals = _log1mexp_jax(ys)
    ys_np = np.asarray(ys)
    lexpected = np.where(
        ys_np > -np.log(2.0),
        np.log(-np.expm1(ys_np)),
        np.log1p(-np.exp(ys_np)),
    )
    np.testing.assert_allclose(np.asarray(lvals), lexpected, rtol=1e-13)
    assert float(lvals[-1]) == pytest.approx(-np.exp(-50.0), rel=1e-12)
    assert jnp.all(jnp.isfinite(jax.vmap(jax.grad(_log1mexp_jax))(ys)))


def _make_wip_gp(n=40, d=3, seed=0, noise=1e-6, lengthscale=0.3, classifier=None, **gp_kwargs):
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(n, d))
    y = -np.sum((x - 0.5) ** 2, axis=1) / 0.05
    kernel = RBFKernel(lengthscales=jnp.full(d, lengthscale), noise=noise)
    if classifier is not None:
        return ClassifierGP(
            kernel=kernel,
            classifier=classifier,
            train_x=jnp.asarray(x),
            train_y=jnp.asarray(y),
            **gp_kwargs,
        )
    return GP(kernel=kernel, train_x=jnp.asarray(x), train_y=jnp.asarray(y), **gp_kwargs)


def _make_mc_pool(m=32, d=3, seed=1):
    rng = np.random.default_rng(seed)
    return jnp.asarray(np.clip(0.5 + 0.15 * rng.normal(size=(m, d)), 0.0, 1.0))


@pytest.mark.parametrize("share_mc_points, pool_size", [(False, 200), (True, 48)])
def test_wip_greedy_batch_matches_copy_and_update(share_mc_points, pool_size):
    """
    Discrete (scan-only) regime: greedy cache conditioning selects the same batch
    as the base-class path (GP copy + update per batch point).
    """
    gp = _make_wip_gp(n=40, d=3, noise=1e-4, lengthscale=0.15)
    pool = _make_mc_pool(m=pool_size)
    acq_kwargs = {"mc_samples": {"x": np.asarray(pool)}, "mc_points_size": 48}
    acq = WIPStd(share_mc_points=share_mc_points, optimize_until_size=0)

    new_x, new_vals = acq.get_next_batch(
        gp, n_batch=3, acq_kwargs=acq_kwargs, rng=np.random.default_rng(0)
    )
    old_x, old_vals = AcquisitionFunction.get_next_batch(
        acq, gp, n_batch=3, acq_kwargs=acq_kwargs, rng=np.random.default_rng(0)
    )

    np.testing.assert_allclose(new_x, old_x)
    # First point uses the real GP in both paths; later values differ only by the
    # y_std rescale the copied GP picks up from its fantasised targets.
    np.testing.assert_allclose(new_vals[0], old_vals[0], rtol=1e-9)
    assert len({tuple(np.round(p, 12)) for p in new_x}) == 3


@pytest.mark.parametrize("variant", ["rbf", "classifier"])
@pytest.mark.parametrize("share_mc_points", [False, True])
def test_wip_small_n_batch_with_continuous_refinement(variant, share_mc_points):
    gp = (
        _make_wip_gp()
        if variant == "rbf"
        else _make_wip_gp(classifier=SVMClassifier(), clf_use_size=1000)
    )
    pool = _make_mc_pool(m=96)
    acq_kwargs = {"mc_samples": {"x": np.asarray(pool)}, "mc_points_size": 32}

    x_batch, vals = WIPStd(share_mc_points=share_mc_points).get_next_batch(
        gp,
        n_batch=3,
        acq_kwargs=acq_kwargs,
        maxiter=20,
        n_restarts=1,
        rng=np.random.default_rng(0),
    )

    assert x_batch.shape == (3, 3)
    assert np.all((x_batch >= 0.0) & (x_batch <= 1.0))
    assert np.all(np.isfinite(vals))
    if share_mc_points:
        assert vals[1] <= vals[0] + 1e-12 and vals[2] <= vals[1] + 1e-12


def test_share_mc_points_defaults_and_bo_plumbing(tmp_path):
    assert WIPStd().share_mc_points is False
    assert WIPV(share_mc_points=True).share_mc_points is True

    def loglike(x):
        return float(-np.sum(x**2))

    bobe = BOBE(
        loglikelihood=loglike,
        param_list=["x0", "x1"],
        param_bounds=np.array([[-1.0, -1.0], [1.0, 1.0]]),
        gp=GP(kernel=RBFKernel()),
        n_sobol_init=4,
        save_dir=str(tmp_path),
        save=False,
    )
    bobe.run(
        acq="wipstd",
        min_evals=4,
        max_evals=6,
        batch_size=2,
        mc_points_method="uniform",
        num_hmc_samples=32,
        mc_points_size=16,
        share_mc_points=True,
        do_final_ns=False,
    )
    assert bobe.acquisition.share_mc_points is True
    assert bobe.results_manager.settings["share_mc_points"] is True


def test_wip_optimize_until_size():
    """optimize_until_size controls whether continuous optimization runs after the MC scan."""
    gp = _make_wip_gp(n=20, d=3)
    pool = _make_mc_pool(m=32, d=3)
    acq_kwargs = {"mc_samples": {"x": np.asarray(pool)}, "mc_points_size": 32}

    with pytest.raises(ValueError, match="optimize_until_size"):
        WIPStd(optimize_until_size=-1)

    # optimize_until_size=0 skips continuous refinement and returns an exact MC point
    pt_discrete, val_discrete = WIPStd(optimize_until_size=0).get_next_point(
        gp, acq_kwargs=acq_kwargs, maxiter=20, n_restarts=1, rng=np.random.default_rng(0)
    )
    assert any(np.allclose(np.asarray(pt_discrete), np.asarray(p)) for p in pool)

    # optimize_until_size=None forces continuous refinement (value <= discrete scan min)
    pt_refined, val_refined = WIPStd(optimize_until_size=None).get_next_point(
        gp, acq_kwargs=acq_kwargs, maxiter=30, n_restarts=1, rng=np.random.default_rng(0)
    )
    assert pt_refined.shape == (3,)
    assert val_refined <= val_discrete + 1e-12


def test_acquisition_optimization_settings_consistent():
    """get_next_point and get_next_batch forward consistent settings to acq_optimize."""
    gp = _make_wip_gp(n=20, d=3)
    pool = _make_mc_pool(m=24, d=3)

    for acq, acq_kwargs in (
        (EI(), {}),
        (WIPStd(), {"mc_samples": {"x": np.asarray(pool)}, "mc_points_size": 16}),
    ):
        recorded = []

        def spy_optimize(*, x0, maxiter, n_restarts, **kwargs):
            recorded.append((maxiter, n_restarts, jnp.atleast_2d(x0).shape[0]))
            return jnp.atleast_2d(x0)[0], 0.0

        acq.acq_optimize = spy_optimize
        acq.get_next_point(gp, acq_kwargs=acq_kwargs, rng=np.random.default_rng(0))
        acq.get_next_batch(gp, n_batch=1, acq_kwargs=acq_kwargs, rng=np.random.default_rng(0))
        assert recorded[0] == recorded[1]
        assert recorded[0][2] == recorded[0][1]  # x0 rows match n_restarts

        acq.get_next_batch(
            gp, n_batch=1, acq_kwargs=acq_kwargs, maxiter=17, n_restarts=3, rng=np.random.default_rng(0)
        )
        assert recorded[2] == (17, 3, 3)


def run_all_tests():
    """Run all acquisition function tests."""
    print("\n" + "="*80)
    print("RUNNING ALL ACQUISITION FUNCTION TESTS")
    print("="*80)
    
    tests = [
        test_ei_initialization,
        test_logei_initialization,
        test_ei_evaluation,
        test_logei_evaluation,
        test_ei_optimization,
        test_logei_optimization,
        test_batch_acquisition,
        test_acquisition_with_different_gp_settings,
        test_acquisition_optimization_convergence,
        test_wip_acquisitions_and_mc_points,
        test_pure_jax_erfcx_and_log1mexp,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ TEST FAILED: {test.__name__}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ TEST ERROR: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
    else:
        print(f"\n✗ {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
