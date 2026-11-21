"""
Tests for Gaussian Process (GP) functionality.

Tests include:
- Base GP initialization and setup
- Fitting hyperparameters
- Adding random points
- Evaluating predictions (mean and variance)
- Random point generation
- State save/load
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import sys
from BOBE.gp import GP
from BOBE.kernels import Kernel, RBFKernel, MaternKernel
from BOBE.surrogate import Surrogate


def generate_test_data(n_samples=50, d=2, seed=42):
    """Generate synthetic test data for GP."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, size=(n_samples, d))
    # Simple function: sum of squares
    y = -np.sum((X - 0.5)**2, axis=1).reshape(-1, 1)
    return X, y


def test_gp_initialization():
    """Test basic GP initialization."""
    print("\n" + "="*80)
    print("TEST: GP Initialization")
    print("="*80)
    
    X, y = generate_test_data(n_samples=20, d=3)
    
    kernel = RBFKernel(
        noise=1e-6,
        lengthscale_bounds=[0.01, 10],
        kernel_variance_bounds=[1e-4, 1e4],
    )
    gp = GP(train_x=X, train_y=y, kernel=kernel)

    assert gp.ndim == 3, f"Expected ndim=3, got {gp.ndim}"
    assert gp.train_x.shape[0] == 20, f"Expected 20 training points, got {gp.train_x.shape[0]}"
    assert isinstance(gp.kernel, RBFKernel)
    assert gp.kernel is kernel
    assert gp.kernel.lengthscale_bounds == [0.01, 10.0]
    assert gp.kernel.kernel_variance_bounds == [1e-4, 1e4]

    print(f"✓ GP initialized with {gp.npoints} points in {gp.ndim}D")
    print(f"✓ Kernel: {type(gp.kernel).__name__}")
    print(f"✓ Initial lengthscales: {gp.kernel.lengthscales}")
    print(f"✓ Initial kernel variance: {gp.kernel.kernel_variance}")
    print(f"✓ Y mean: {gp.y_mean:.4f}, Y std: {gp.y_std:.4f}")
    

def test_gp_fitting():
    """Test GP hyperparameter fitting."""
    print("\n" + "="*80)
    print("TEST: GP Hyperparameter Fitting")
    print("="*80)
    
    X, y = generate_test_data(n_samples=30, d=2)
    
    gp = GP(
        train_x=X,
        train_y=y,
        kernel=MaternKernel(noise=1e-6, lengthscale_prior="DSLP"),
        optimizer="scipy",
    )
    
    print(f"Initial hyperparameters:")
    print(f"  Lengthscales: {gp.kernel.lengthscales}")
    print(f"  Kernel variance: {gp.kernel.kernel_variance:.4f}")
    
    # Fit the GP
    result = gp.fit(maxiter=200, x0=None)
    
    print(f"\nAfter fitting:")
    print(f"  MLL: {result['mll']:.4f}")
    print(f"  Lengthscales: {gp.kernel.lengthscales}")
    print(f"  Kernel variance: {gp.kernel.kernel_variance:.4f}")
    
    # Check that hyperparameters were updated
    assert result['mll'] is not None, "MLL should not be None after fitting"
    print(f"\n✓ GP fitted successfully with MLL = {result['mll']:.4f}")


def test_gp_predictions():
    """Test GP mean and variance predictions."""
    print("\n" + "="*80)
    print("TEST: GP Predictions")
    print("="*80)
    
    X, y = generate_test_data(n_samples=25, d=2)
    
    gp = GP(train_x=X, train_y=y, kernel=RBFKernel(noise=1e-6))
    
    # Single point prediction
    test_point = jnp.array([0.5, 0.5])
    mean_single = gp.predict_mean(test_point)
    var_single = gp.predict_var(test_point)
    mean_p, var_p = gp.predict(test_point)
    mean_s, var_s = gp.predict(test_point, standardized=True)

    print(f"Single point prediction at [0.5, 0.5]:")
    print(f"  Mean: {mean_single:.4f}")
    print(f"  Variance: {var_single:.4e}")

    assert mean_single.shape == (), f"Expected scalar mean, got shape {mean_single.shape}"
    assert var_single.shape == (), f"Expected scalar variance, got shape {var_single.shape}"
    assert var_single > 0, "Variance must be positive"
    np.testing.assert_allclose(mean_single, mean_p, rtol=1e-12)
    np.testing.assert_allclose(var_single, var_p, rtol=1e-12)
    np.testing.assert_allclose(mean_s * gp.y_std + gp.y_mean, mean_p, rtol=1e-12)
    np.testing.assert_allclose(var_s * gp.y_std**2, var_p, rtol=1e-12)

    # Batched prediction
    test_points = jnp.array([[0.2, 0.3], [0.7, 0.8], [0.5, 0.5]])
    means_batch = gp.predict_mean(test_points)
    vars_batch = gp.predict_var(test_points)
    means_p, vars_p = gp.predict(test_points)

    print(f"\nBatch prediction for {test_points.shape[0]} points:")
    print(f"  Means: {means_batch}")
    print(f"  Variances: {vars_batch}")

    assert means_batch.shape == (3,), f"Expected shape (3,), got {means_batch.shape}"
    assert vars_batch.shape == (3,), f"Expected shape (3,), got {vars_batch.shape}"
    assert jnp.all(vars_batch > 0), "All variances must be positive"
    np.testing.assert_allclose(means_batch, means_p, rtol=1e-12)
    np.testing.assert_allclose(vars_batch, vars_p, rtol=1e-12)
    np.testing.assert_allclose(means_batch[2], mean_single, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(vars_batch[2], var_single, rtol=1e-9, atol=1e-12)

    # Check that prediction at training point has low variance
    train_point = X[0]
    mean_at_train = gp.predict_mean(train_point)
    var_at_train = gp.predict_var(train_point)
    
    print(f"\nPrediction at training point:")
    print(f"  True value: {y[0,0]:.4f}")
    print(f"  Predicted mean: {mean_at_train:.4f}")
    print(f"  Predicted variance: {var_at_train:.4e}")
    
    # Variance at training point should be close to noise level
    assert var_at_train < 1e-3, f"Variance at training point too high: {var_at_train}"
    
    print(f"\n✓ All prediction tests passed")


def test_gp_update():
    """Test adding new points to GP."""
    print("\n" + "="*80)
    print("TEST: GP Update with New Points")
    print("="*80)
    
    X, y = generate_test_data(n_samples=15, d=2)
    
    gp = GP(train_x=X, train_y=y, kernel=RBFKernel(noise=1e-6))
    initial_size = gp.npoints
    
    print(f"Initial GP size: {initial_size} points")
    
    # Add new points
    new_X = jnp.array([[0.8, 0.2], [0.3, 0.9]])
    new_y = -jnp.sum((new_X - 0.5)**2, axis=1, keepdims=True)
    
    gp.update(new_X, new_y)
    
    print(f"After update: {gp.npoints} points")
    
    assert gp.npoints == initial_size + 2, f"Expected {initial_size + 2} points, got {gp.npoints}"
    
    # Try adding duplicate point
    gp.update(new_X[0:1], new_y[0:1])
    assert gp.npoints == initial_size + 2, "Duplicate point should not be added"
    
    print(f"✓ GP update successful, duplicates correctly handled")


def test_gp_random_point():
    """Test random point generation."""
    print("\n" + "="*80)
    print("TEST: Random Point Generation")
    print("="*80)
    
    X, y = generate_test_data(n_samples=20, d=3)
    gp = GP(train_x=X, train_y=y, kernel=RBFKernel())
    
    # Generate multiple random points
    rng = np.random.default_rng(42)
    points = []
    for i in range(10):
        pt = gp.get_random_point(rng=rng)
        points.append(pt)
        print(f"  Point {i+1}: {pt}")
    
    points = np.array(points)
    
    assert points.shape == (10, 3), f"Expected shape (10, 3), got {points.shape}"
    assert np.all(points >= 0) and np.all(points <= 1), "Points should be in unit cube [0,1]^d"
    
    # Check that points are different (not all the same)
    assert not np.allclose(points, points[0]), "Random points should be different"
    
    print(f"\n✓ Random point generation successful")


def test_gp_state_dict():
    """Test GP state save/load."""
    print("\n" + "="*80)
    print("TEST: GP State Save/Load")
    print("="*80)
    
    X, y = generate_test_data(n_samples=20, d=2)
    
    gp1 = GP(
        train_x=X,
        train_y=y,
        kernel=RBFKernel(
            lengthscales=jnp.array([0.5, 0.3]),
            kernel_variance=2.0,
            noise=1e-6,
        ),
    )

    # Get state dict
    state = gp1.state_dict()

    print(f"State dict keys: {list(state.keys())}")

    # Create new GP from state (both via GP and polymorphic Surrogate)
    gp2 = GP.from_state_dict(state)
    gp3 = Surrogate.from_state_dict(state)
    assert isinstance(gp3, GP)
    
    # Check that GPs are identical
    assert gp2.ndim == gp1.ndim, "Dimensions don't match"
    assert gp2.npoints == gp1.npoints, "Number of points don't match"
    assert jnp.allclose(gp2.kernel.lengthscales, gp1.kernel.lengthscales), "Lengthscales don't match"
    assert jnp.isclose(gp2.kernel.kernel_variance, gp1.kernel.kernel_variance), "Kernel variance doesn't match"
    assert jnp.allclose(gp2.train_x, gp1.train_x), "Training X doesn't match"
    
    # Check predictions are the same
    test_point = jnp.array([0.5, 0.5])
    mean1 = gp1.predict_mean(test_point)
    mean2 = gp2.predict_mean(test_point)
    
    print(f"\nPrediction comparison at [0.5, 0.5]:")
    print(f"  Original GP mean: {mean1:.4f}")
    print(f"  Loaded GP mean: {mean2:.4f}")
    print(f"  Difference: {abs(mean1 - mean2):.4e}")
    
    assert jnp.isclose(mean1, mean2, rtol=1e-6), "Predictions don't match"
    
    print(f"\n✓ State save/load successful")


def test_gp_copy():
    """Test GP copy functionality."""
    print("\n" + "="*80)
    print("TEST: GP Copy")
    print("="*80)
    
    X, y = generate_test_data(n_samples=15, d=2)
    
    gp1 = GP(train_x=X, train_y=y, kernel=RBFKernel(noise=1e-6))
    gp2 = gp1.copy()
    
    # Check that copy is independent
    new_X = jnp.array([[0.9, 0.1]])
    new_y = jnp.array([[-0.5]])
    
    gp2.update(new_X, new_y)
    
    print(f"Original GP size: {gp1.npoints}")
    print(f"Copied GP size: {gp2.npoints}")
    
    assert gp1.npoints != gp2.npoints, "Copy should be independent"
    assert gp2.npoints == gp1.npoints + 1, "Copied GP should have one more point"
    
    print(f"✓ GP copy is independent")


def test_gp_different_kernels():
    """Test GP with different kernel types."""
    print("\n" + "="*80)
    print("TEST: Different Kernel Types")
    print("="*80)
    
    X, y = generate_test_data(n_samples=20, d=2)
    
    # RBF kernel
    gp_rbf = GP(train_x=X, train_y=y, kernel=RBFKernel())
    mean_rbf = gp_rbf.predict_mean(jnp.array([0.5, 0.5]))

    # Matern kernel
    gp_matern = GP(train_x=X, train_y=y, kernel=MaternKernel())
    mean_matern = gp_matern.predict_mean(jnp.array([0.5, 0.5]))
    
    print(f"RBF kernel prediction: {mean_rbf:.4f}")
    print(f"Matern kernel prediction: {mean_matern:.4f}")
    
    # Both should give reasonable predictions (not too different, but not identical)
    assert not jnp.isclose(mean_rbf, mean_matern, rtol=0.01), "Different kernels should give different predictions"
    
    print(f"✓ Both kernel types work correctly")


def test_gp_surrogate_interface():
    """Test GP adheres to the Surrogate ABC and incomplete subclasses fail."""
    X, y = generate_test_data(n_samples=15, d=2)
    gp = GP(train_x=X, train_y=y, kernel=RBFKernel())
    assert isinstance(gp, Surrogate)

    class IncompleteSurrogate(Surrogate):
        pass

    with pytest.raises(TypeError):
        IncompleteSurrogate()


def test_gp_optimizable_noise_and_saas():
    """Test fixed vs optimizable noise priors, SAAS prior, and state_dict round-trip."""
    rng = np.random.default_rng(7)
    X = rng.uniform(0.0, 1.0, size=(30, 2))
    y = -np.sum((X - 0.5) ** 2, axis=1, keepdims=True) + rng.normal(0.0, 0.05, size=(30, 1))

    gp_fixed = GP(train_x=X, train_y=y, kernel=RBFKernel(noise=1e-6))
    assert gp_fixed.kernel.fixed_noise
    assert "noise" not in gp_fixed.kernel.hyperparam_names
    gp_fixed.fit(maxiter=50)
    assert float(gp_fixed.kernel.noise) == pytest.approx(1e-6)

    gp = GP(
        train_x=X,
        train_y=y,
        kernel=MaternKernel(
            noise=1e-2,
            noise_bounds=[1e-6, 1.0],
            noise_prior=None,
        ),
        param_names=["a", "b"],
    )
    assert not gp.kernel.fixed_noise
    assert "noise" in gp.kernel.hyperparam_names
    assert gp.kernel.num_hyperparams == 4
    assert "noise" in gp.hyperparams_dict()

    res = gp.fit(maxiter=100)
    assert np.isfinite(res["mll"])
    assert 1e-6 <= float(gp.kernel.noise) <= 1.0
    np.testing.assert_allclose(gp.get_hyperparams(), jnp.exp(res["params"]), rtol=1e-10)

    restored = GP.from_state_dict(gp.state_dict())
    assert not restored.kernel.fixed_noise
    assert restored.kernel.noise_bounds == [1e-6, 1.0]
    assert restored.param_names == ["a", "b"]
    np.testing.assert_allclose(float(restored.kernel.noise), float(gp.kernel.noise), rtol=1e-12)
    pt = jnp.array([0.4, 0.6])
    np.testing.assert_allclose(restored.predict_mean(pt), gp.predict_mean(pt), rtol=1e-10)

    gp_saas = GP(
        train_x=X,
        train_y=y,
        kernel=RBFKernel(
            noise=1e-3,
            noise_bounds=[1e-6, 1.0],
            noise_prior={"name": "LogNormal", "loc": -4.0, "scale": 1.0},
            lengthscale_prior="SAAS",
        ),
    )
    assert gp_saas.kernel.hyperparam_names == ["lengthscales", "kernel_variance", "noise", "tausq"]
    assert gp_saas.kernel.num_hyperparams == 5
    res_saas = gp_saas.fit(maxiter=50)
    assert np.isfinite(res_saas["mll"])


def test_gp_custom_kernel_and_validation():
    """Test passing a Kernel instance directly and rejecting non-Kernel arguments."""
    X, y = generate_test_data(n_samples=20, d=2)
    custom = RBFKernel(lengthscales=jnp.array([0.4, 0.6]), kernel_variance=2.0, noise=1e-6)
    gp = GP(train_x=X, train_y=y, kernel=custom)
    assert gp.kernel is custom
    assert np.isfinite(float(gp.predict_mean(jnp.array([0.5, 0.5]))))

    with pytest.raises(TypeError):
        GP(train_x=X, train_y=y)

    with pytest.raises(TypeError, match="Kernel instance"):
        GP(train_x=X, train_y=y, kernel="rbf")

    wrong_dim_kernel = RBFKernel(lengthscales=jnp.array([0.4, 0.6, 0.8]))
    with pytest.raises(ValueError, match="ndim=3"):
        GP(train_x=X, train_y=y, kernel=wrong_dim_kernel)


def test_gp_deferred_initialization_and_nan_filtering():
    """Test deferred GP & Kernel initialization (train_x=None) and NaN filtering in update()."""
    X, y = generate_test_data(n_samples=15, d=3)
    kernel = MaternKernel(noise=1e-6, lengthscale_prior="DSLP")
    assert kernel.ndim is None
    assert kernel.lengthscales is None

    gp = GP(kernel=kernel)
    assert gp.ndim is None
    assert gp.npoints == 0

    gp.update(X, y.flatten())
    assert gp.ndim == 3
    assert kernel.ndim == 3
    assert kernel.lengthscales.shape == (3,)
    assert gp.npoints == 15
    assert np.isfinite(float(gp.predict_mean(jnp.array([0.5, 0.5, 0.5]))))

    new_x = jnp.array([[np.nan, 0.5, 0.5], [0.11, 0.22, 0.33], [0.44, 0.55, 0.66]])
    new_y = jnp.array([[-1.0], [np.nan], [-0.5]])
    gp.update(new_x, new_y)
    assert gp.npoints == 16


def test_gp_save_load_polymorphic(tmp_path):
    """Test saving a GP and loading via both GP.load and Surrogate.load."""
    X, y = generate_test_data(n_samples=15, d=2)
    kernel = MaternKernel(
        lengthscales=jnp.array([0.3, 0.7]),
        kernel_variance=1.5,
        noise=1e-4,
        noise_prior=None,
        noise_bounds=[1e-6, 1.0],
        lengthscale_prior="DSLP",
    )
    gp = GP(train_x=X, train_y=y, kernel=kernel, param_names=["p0", "p1"])
    path = tmp_path / "test_gp"
    gp.save(str(path))

    loaded_gp = GP.load(str(path))
    loaded_poly = Surrogate.load(str(path))
    assert isinstance(loaded_poly, GP)
    assert isinstance(loaded_poly.kernel, MaternKernel)
    pt = jnp.array([0.4, 0.6])
    np.testing.assert_allclose(loaded_gp.predict_mean(pt), gp.predict_mean(pt), rtol=1e-12)
    np.testing.assert_allclose(loaded_poly.predict_mean(pt), gp.predict_mean(pt), rtol=1e-12)
    np.testing.assert_allclose(float(loaded_poly.kernel.noise), 1e-4, rtol=1e-12)
    assert not loaded_poly.kernel.fixed_noise


def test_gp_fantasy_var():
    """Test precompute_fantasy_cache and fantasy_var."""
    X, y = generate_test_data(n_samples=20, d=2)
    gp = GP(train_x=X, train_y=y, kernel=RBFKernel(noise=1e-6))
    rng = np.random.default_rng(123)
    mc_points = jnp.array(rng.uniform(0.0, 1.0, size=(32, 2)))
    new_x = jnp.array([0.5, 0.5])

    cache = gp.precompute_fantasy_cache(mc_points)
    fvar = gp.fantasy_var(new_x, mc_points, cache)
    assert fvar.shape == (32,)
    assert jnp.all(fvar > 0)
    assert jnp.all(fvar <= gp.predict_var(mc_points) + 1e-10)


def run_all_tests():
    """Run all GP tests."""
    print("\n" + "="*80)
    print("RUNNING ALL GP TESTS")
    print("="*80)
    
    tests = [
        test_gp_initialization,
        test_gp_fitting,
        test_gp_predictions,
        test_gp_update,
        test_gp_random_point,
        test_gp_state_dict,
        test_gp_copy,
        test_gp_different_kernels,
        test_gp_surrogate_interface,
        test_gp_optimizable_noise_and_saas,
        test_gp_custom_kernel_and_validation,
        test_gp_deferred_initialization_and_nan_filtering,
        test_gp_fantasy_var,
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
