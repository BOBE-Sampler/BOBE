"""
Determinism regression tests for BOBE.

Confirms that two BOBE runs with the same seed on the same likelihood produce
bitwise-identical trajectories on CPU / single-process. Locks in the seed-handling
fixes (clf JAX-key plumbing, static MPI dispatch default, get_new_jax_key for NUTS,
sampler rng threading).
"""

import os
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

import numpy as np
import sys
import tempfile
import shutil
from BOBE import BOBE, GP, ClassifierGP, SVMClassifier, RBFKernel


def rosenbrock_loglike(x):
    """Negative Rosenbrock function as log-likelihood."""
    return -((1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2)


def _run_bobe(acq, seed, save_dir, *, use_clf=False, classifier_factory=SVMClassifier):
    """Run a short BOBE optimization and return the result dict."""
    param_bounds = np.array([[-2, 2], [-2, 2]]).T
    param_list = ['x', 'y']

    if use_clf:
        gp = ClassifierGP(
            kernel=RBFKernel(lengthscale_prior="DSLP"),
            classifier=classifier_factory(),
            clf_use_size=10,
            clf_threshold=75.0,
            gp_threshold=150.0,
            minus_inf=-1e10,
        )
    else:
        gp = GP(kernel=RBFKernel())

    bobe = BOBE(
        loglikelihood=rosenbrock_loglike,
        param_list=param_list,
        param_bounds=param_bounds,
        likelihood_name=f"determinism_{acq}_{seed}",
        gp=gp,
        n_sobol_init=4,
        save=False,
        save_dir=save_dir,
        seed=seed,
        verbosity='ERROR',
    )

    if acq == 'ei':
        return bobe.run(
            acq='ei',
            min_evals=10,
            max_evals=20,
            max_gp_size=20,
            ei_goal=1e-6,
            fit_n_points=5,
            batch_size=1,
        )
    return bobe.run(
        acq='wipstd',
        min_evals=15,
        max_evals=25,
        max_gp_size=25,
        logz_threshold=0.5,
        fit_n_points=8,
        ns_n_points=15,
        batch_size=1,
        mc_points_method='uniform',
    )


def _assert_runs_match(result_a, result_b, label):
    """Assert two BOBE result dicts are bitwise-identical on the trajectory fields."""
    assert result_a is not None and result_b is not None, f"{label}: results missing (worker rank?)"

    gp_a, gp_b = result_a['gp'], result_b['gp']
    train_x_a = np.asarray(gp_a.train_x)
    train_x_b = np.asarray(gp_b.train_x)
    train_y_a = np.asarray(gp_a.train_y)
    train_y_b = np.asarray(gp_b.train_y)

    assert train_x_a.shape == train_x_b.shape, (
        f"{label}: train_x shape mismatch {train_x_a.shape} vs {train_x_b.shape}"
    )
    assert np.array_equal(train_x_a, train_x_b), (
        f"{label}: train_x diverged (max abs diff = {np.max(np.abs(train_x_a - train_x_b)):.3e})"
    )
    assert np.array_equal(train_y_a, train_y_b), (
        f"{label}: train_y diverged (max abs diff = {np.max(np.abs(train_y_a - train_y_b)):.3e})"
    )

    best_a = np.asarray(result_a['best_pt'])
    best_b = np.asarray(result_b['best_pt'])
    assert np.array_equal(best_a, best_b), f"{label}: best_pt diverged ({best_a} vs {best_b})"
    assert result_a['best_val'] == result_b['best_val'], (
        f"{label}: best_val diverged ({result_a['best_val']} vs {result_b['best_val']})"
    )

    print(f"  {label}: train_x {train_x_a.shape}, best_val={result_a['best_val']:.6f} — identical")


def _run_pair(acq, seed, **kwargs):
    """Run BOBE twice with the same seed and assert the trajectories match."""
    dir_a = tempfile.mkdtemp()
    dir_b = tempfile.mkdtemp()
    try:
        result_a = _run_bobe(acq, seed, dir_a, **kwargs)
        result_b = _run_bobe(acq, seed, dir_b, **kwargs)
        # Workers return None; this test must run on the main process.
        if result_a is None or result_b is None:
            return None
        return result_a, result_b
    finally:
        shutil.rmtree(dir_a, ignore_errors=True)
        shutil.rmtree(dir_b, ignore_errors=True)


def test_determinism_ei():
    """Two EI runs with the same seed must produce identical trajectories."""
    print("\n" + "=" * 80)
    print("TEST: BOBE EI determinism (seed=42, two runs)")
    print("=" * 80)

    pair = _run_pair('ei', seed=42)
    if pair is None:
        return  # worker rank
    _assert_runs_match(*pair, label='ei seed=42')
    print("\n✓ EI determinism test passed")


def test_determinism_wipstd():
    """Two WIPStd runs with the same seed must produce identical trajectories."""
    print("\n" + "=" * 80)
    print("TEST: BOBE WIPStd determinism (seed=123, two runs)")
    print("=" * 80)

    pair = _run_pair('wipstd', seed=123)
    if pair is None:
        return
    _assert_runs_match(*pair, label='wipstd seed=123')
    print("\n✓ WIPStd determinism test passed")


def test_determinism_with_classifier():
    """Two classifier-augmented runs with the same seed must produce identical trajectories."""
    print("\n" + "=" * 80)
    print("TEST: BOBE WIPStd + SVM classifier determinism (seed=456, two runs)")
    print("=" * 80)

    pair = _run_pair('wipstd', seed=456, use_clf=True)
    if pair is None:
        return
    _assert_runs_match(*pair, label='wipstd+svm seed=456')
    print("\n✓ Classifier determinism test passed")


def test_different_seeds_diverge():
    """Sanity check: two runs with different seeds should NOT be identical."""
    print("\n" + "=" * 80)
    print("TEST: Different seeds produce different trajectories")
    print("=" * 80)

    dir_a = tempfile.mkdtemp()
    dir_b = tempfile.mkdtemp()
    try:
        result_a = _run_bobe('ei', seed=42, save_dir=dir_a)
        result_b = _run_bobe('ei', seed=43, save_dir=dir_b)
        if result_a is None or result_b is None:
            return

        train_x_a = np.asarray(result_a['gp'].train_x)
        train_x_b = np.asarray(result_b['gp'].train_x)
        # Differently seeded runs should diverge somewhere in the trajectory.
        assert not np.array_equal(train_x_a, train_x_b), (
            "Runs with different seeds produced identical trajectories — seed has no effect"
        )
        print("  Trajectories differ as expected.")
        print("\n✓ Seed sensitivity test passed")
    finally:
        shutil.rmtree(dir_a, ignore_errors=True)
        shutil.rmtree(dir_b, ignore_errors=True)


def run_all_tests():
    print("\n" + "=" * 80)
    print("RUNNING BOBE DETERMINISM REGRESSION TESTS")
    print("=" * 80)

    tests = [
        test_determinism_ei,
        test_determinism_wipstd,
        test_determinism_with_classifier,
        test_different_seeds_diverge,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ TEST FAILED: {t.__name__}")
            print(f"  {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ TEST ERROR: {t.__name__}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print("=" * 80)
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
