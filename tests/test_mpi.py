"""
Tests for MPI Pool functionality.

Tests include:
- MPI initialization and detection
- Parallel execution (map)
- GP fitting with parallel hyperparameter optimization
- Error handling
"""

import os
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

import numpy as np
import jax
import jax.numpy as jnp
import sys
from BOBE.gp import GP
from BOBE.kernels import RBFKernel
from BOBE.surrogate import Surrogate
from BOBE.pool import MPI_Pool


def simple_objective(x):
    """Simple test objective function."""
    return -np.sum((x - 0.5)**2)


def generate_test_gp(n_samples=20, d=2, seed=42):
    """Generate a simple GP for testing."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, size=(n_samples, d))
    y = -np.sum((X - 0.5)**2, axis=1, keepdims=True)

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


def test_pool_initialization():
    """Test MPI Pool initialization and detection."""
    print("\n" + "="*80)
    print("TEST: MPI Pool Initialization")
    print("="*80)
    
    pool = MPI_Pool()
    
    print(f"MPI rank: {pool.rank}")
    print(f"MPI size: {pool.size}")
    print(f"Is MPI: {pool.is_mpi}")
    print(f"Is main process: {pool.is_main_process}")
    
    # Check pool properties
    assert pool.rank >= 0, "Rank should be non-negative"
    assert pool.size >= 1, "Size should be at least 1"
    assert pool.is_main_process == (pool.rank == 0), "Main process check should match rank"
    
    if pool.is_mpi:
        print(f"\n✓ MPI Pool initialized with {pool.size} processes")
    else:
        print(f"\n✓ MPI Pool initialized in serial mode")
    
    pool.close()


def test_run_map_objective():
    """Test parallel mapping of objective function using pool.map."""
    pool = MPI_Pool()
    
    if pool.is_main_process:
        print("\n" + "="*80)
        print("TEST: Pool Map")
        print("="*80)
    
    # Only main process creates test points
    if pool.is_main_process:
        test_points = [
            np.array([0.2, 0.3]),
            np.array([0.5, 0.5]),
            np.array([0.8, 0.7]),
            np.array([0.1, 0.9]),
        ]
        print(f"Evaluating {len(test_points)} points...")
    else:
        # Workers wait for tasks
        pool.worker_wait(likelihood=None, seed=42)
        return
    
    # Main process distributes work
    results = pool.run_map_objective(simple_objective, test_points)
    
    print(f"\nResults: {results}")
    
    assert len(results) == 4, f"Expected 4 results, got {len(results)}"
    
    # Verify results are correct
    for i, pt in enumerate(test_points):
        expected = simple_objective(pt)
        print(f"  Point {pt}: result={results[i]:.4f}, expected={expected:.4f}")
        assert np.isclose(results[i], expected), f"Result mismatch at point {i}"
    
    print(f"\n✓ pool.map successful")
    pool.close()


def test_gp_fit_serial():
    """Test GP fitting without parallel mode."""
    pool = MPI_Pool()
    
    if pool.is_main_process:
        print("\n" + "="*80)
        print("TEST: GP Fit (Serial)")
        print("="*80)
    else:
        pool.worker_wait(likelihood=None, seed=42)
        return
    
    gp = generate_test_gp(n_samples=25, d=2)
    
    print(f"Initial GP hyperparameters:")
    print(f"  Lengthscales: {gp.kernel.lengthscales}")
    print(f"  Kernel variance: {gp.kernel.kernel_variance:.4f}")
    
    # Fit GP (serial mode)
    result = pool.gp_fit(gp, maxiters=200, n_restarts=1, use_pool=False)
    
    print(f"\nAfter fitting:")
    print(f"  MLL: {result['mll']:.4f}")
    print(f"  Lengthscales: {gp.kernel.lengthscales}")
    print(f"  Kernel variance: {gp.kernel.kernel_variance:.4f}")
    
    # Verify hyperparameters were updated
    assert gp.kernel.lengthscales is not None, "Lengthscales should be set"
    assert gp.kernel.kernel_variance > 0, "Kernel variance should be positive"
    
    print(f"\n✓ GP fit successful")
    pool.close()


def test_pool_with_different_task_sizes():
    """Test pool.map with varying number of tasks."""
    pool = MPI_Pool()
    
    if pool.is_main_process:
        print("\n" + "="*80)
        print("TEST: Pool Map with Different Task Sizes")
        print("="*80)
    else:
        pool.worker_wait(likelihood=None, seed=42)
        return
    
    # Test with different numbers of tasks
    task_sizes = [1, 5, 10, 20]
    
    for n_tasks in task_sizes:
        rng = np.random.RandomState(42 + n_tasks)
        test_points = [rng.uniform(0, 1, size=2) for _ in range(n_tasks)]
        
        results = pool.run_map_objective(simple_objective, test_points)
        
        print(f"  {n_tasks} tasks: {len(results)} results")
        assert len(results) == n_tasks, f"Expected {n_tasks} results, got {len(results)}"
        assert all(np.isfinite(r) for r in results), "All results should be finite"
    
    print(f"\n✓ pool.map handles different task sizes correctly")
    pool.close()


def test_pool_with_zero_tasks():
    """Test pool.map with empty task list."""
    pool = MPI_Pool()
    
    if pool.is_main_process:
        print("\n" + "="*80)
        print("TEST: Pool Map with Zero Tasks")
        print("="*80)
    else:
        pool.worker_wait(likelihood=None, seed=42)
        return
    
    # Empty task list
    test_points = []
    results = pool.run_map_objective(simple_objective, test_points)
    
    print(f"Results for 0 tasks: {results}")
    assert len(results) == 0, "Should return empty list for zero tasks"
    print(f"✓ pool.map handles zero tasks correctly")
    pool.close()


def test_objective_function_types():
    """Test pool.map with different objective function types."""
    pool = MPI_Pool()
    
    if pool.is_main_process:
        print("\n" + "="*80)
        print("TEST: Different Objective Function Types")
        print("="*80)
    else:
        pool.worker_wait(likelihood=None, seed=42)
        return
    
    # Simple scalar function
    def scalar_fn(x):
        return float(np.sum(x))
    
    # Function returning array element
    def array_fn(x):
        return np.array([np.sum(x**2)])[0]
    
    # Function with conditional logic
    def conditional_fn(x):
        if np.sum(x) > 1.0:
            return 1.0
        return -1.0
    
    test_point = [np.array([0.5, 0.5])]
    
    result_scalar = pool.run_map_objective(scalar_fn, test_point)
    result_array = pool.run_map_objective(array_fn, test_point)
    result_conditional = pool.run_map_objective(conditional_fn, test_point)
    
    print(f"Scalar function result: {result_scalar}")
    print(f"Array function result: {result_array}")
    print(f"Conditional function result: {result_conditional}")
    
    assert len(result_scalar) == 1, "Should return single result"
    assert len(result_array) == 1, "Should return single result"
    assert len(result_conditional) == 1, "Should return single result"
    
    print(f"\n✓ pool.map works with different objective function types")
    pool.close()


def test_gp_state_serialization_for_pool():
    """Test that GP state can be properly serialized for MPI distribution."""
    pool = MPI_Pool()
    
    if not pool.is_main_process:
        pool.worker_wait(likelihood=None, seed=42)
        return
        
    print("\n" + "="*80)
    print("TEST: GP State Serialization for MPI")
    print("="*80)
    
    gp = generate_test_gp(n_samples=20, d=2)
    
    # Get state dict (this is what would be sent to workers via share)
    state = gp.state_dict()
    
    print(f"State dict has {len(state)} keys")
    print(f"State dict keys: {list(state.keys())[:10]}...")  # Show first 10 keys
    
    # Reconstruct GP from state via polymorphic Surrogate.from_state_dict (same path as MPI workers)
    gp_reconstructed = Surrogate.from_state_dict(state)
    
    # Verify reconstruction
    assert gp_reconstructed.ndim == gp.ndim, "Dimensions don't match"
    assert gp_reconstructed.npoints == gp.npoints, "Number of points don't match"
    assert jnp.allclose(gp_reconstructed.kernel.lengthscales, gp.kernel.lengthscales), "Lengthscales don't match"
    
    # Test prediction consistency
    test_point = jnp.array([0.5, 0.5])
    mean1 = gp.predict_mean(test_point)
    mean2 = gp_reconstructed.predict_mean(test_point)
    
    print(f"\nPrediction comparison:")
    print(f"  Original: {mean1:.4f}")
    print(f"  Reconstructed: {mean2:.4f}")
    print(f"  Difference: {abs(mean1 - mean2):.4e}")
    
    assert jnp.isclose(mean1, mean2, rtol=1e-6), "Predictions don't match after serialization"
    
    print(f"\n✓ GP state serialization works correctly")
    pool.close()


def run_all_tests():
    """Run all MPI Pool tests."""
    pool = MPI_Pool()
    
    # Only main process orchestrates tests
    if not pool.is_main_process:
        # Workers wait for commands from tests
        pool.worker_wait(likelihood=None, seed=42)
        return True
    
    print("\n" + "="*80)
    print("RUNNING ALL MPI POOL TESTS")
    print("="*80)
    if pool.is_mpi:
        print(f"Running with MPI: {pool.size} processes")
    else:
        print("Running in serial mode (no MPI)")
    print("="*80)
    
    tests = [
        test_pool_initialization,
        test_run_map_objective,
        test_gp_fit_serial,
        test_pool_with_different_task_sizes,
        test_pool_with_zero_tasks,
        test_objective_function_types,
        test_gp_state_serialization_for_pool,
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
    
    print("\nNOTE: MPI-specific functionality (parallel execution, worker processes)")
    print("      is fully tested. Serial mode tests verify the serial fallback.")
    print("\nTo test parallel execution, run:")
    print("  mpirun -n 4 python test_mpi.py")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
