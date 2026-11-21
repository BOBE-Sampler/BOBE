"""
Test BOBE on 2D test functions.

This test runs both EI and WIPStd acquisition functions on simple 2D optimization problems
to verify the basic functionality of the Bayesian Optimization loop.
"""

import os
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

import numpy as np
import sys
from BOBE import BOBE, GP, GPwithClassifier, RBFKernel, MaternKernel, PrincipalAxesTransform


def rosenbrock_loglike(x):
    """
    Negative Rosenbrock function as log-likelihood.
    Global minimum at (1, 1) with value 0.
    """
    return -((1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2)


def himmelblau_loglike(x):
    """
    Negative Himmelblau function as log-likelihood.
    Has four identical local minima.
    """
    return -((x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2)


def check_output_files(output_dir, base_name, check_chains=True):
    """
    Helper function to verify expected output files are created.
    
    Parameters
    ----------
    output_dir : str
        Directory where output files should be located
    base_name : str
        Base name of the output files (likelihood name)
    check_chains : bool
        If True, also check for chain files (only relevant for WIPV/WIPStd modes)
    """
    # Files that should always be present
    required_files = [
        f"{base_name}.minimum.txt",    # GetDist minimum table format
        f"{base_name}.minimum",        # GetDist minimum detailed format
        f"{base_name}_results.pkl",    # Main results pickle
        f"{base_name}_timing.json"     # Timing data
    ]
    
    # Chain files and stats only created when samples exist (WIPV/WIPStd with nested sampling)
    chain_files = [
        f"{base_name}.txt",            # GetDist chain file
        f"{base_name}.paramnames",     # GetDist parameter names
        f"{base_name}.ranges",         # GetDist parameter ranges
        f"{base_name}_stats.json",     # Summary statistics (requires samples)
    ]
    
    expected_files = required_files
    if check_chains:
        expected_files += chain_files
    
    print("\nChecking for expected output files:")
    all_exist = True
    for filename in expected_files:
        filepath = os.path.join(output_dir, filename)
        exists = os.path.exists(filepath)
        status = "✓" if exists else "✗"
        print(f"  {status} {filename}")
        if not exists:
            all_exist = False
    
    assert all_exist, "Some expected output files were not created"
    
    # Verify minimum files have basic content
    minimum_txt = os.path.join(output_dir, f"{base_name}.minimum.txt")
    minimum_file = os.path.join(output_dir, f"{base_name}.minimum")
    
    with open(minimum_txt, 'r') as f:
        txt_lines = f.readlines()
    assert len(txt_lines) >= 2, "Minimum .txt file should have at least 2 lines"
    assert txt_lines[0].startswith('#'), "First line should be header"
    
    with open(minimum_file, 'r') as f:
        min_lines = f.readlines()
    assert len(min_lines) >= 4, "Minimum file should have at least 4 lines"
    assert '-log(Like)' in min_lines[0], "Should contain log-likelihood"
    
    print("✓ All expected output files verified")


def test_bobe_ei_2d():
    """Test BOBE with EI acquisition on 2D Rosenbrock function."""
    print("\n" + "="*80)
    print("TEST: BOBE with EI on 2D Rosenbrock")
    print("="*80)
    
    import tempfile
    import shutil
    
    # Create temporary directory for output
    temp_dir = tempfile.mkdtemp()
    
    try:
        param_bounds = np.array([[-2, 2], [-2, 2]]).T
        param_list = ['x', 'y']
        
        bobe = BOBE(
            loglikelihood=rosenbrock_loglike,
            param_list=param_list,
            param_bounds=param_bounds,
            likelihood_name="rosenbrock_ei_test",
            gp=GP(kernel=RBFKernel()),
            n_sobol_init=4,
            save=True,
            save_dir=temp_dir,
            seed=42,
            verbosity='WARNING'
        )
        
        # Run with EI
        results = bobe.run(
            acq='ei',  # lowercase
            min_evals=15,
            max_evals=40,
            max_gp_size=40,
            ei_goal=1e-6,
            fit_n_points=5,
            batch_size=1
        )
        
        # Workers return None
        if results is None:
            return None
        
        # Check results structure
        assert 'gp' in results, "Results missing 'gp' key"
        assert 'likelihood' in results, "Results missing 'likelihood' key"
        assert 'results_manager' in results, "Results missing 'results_manager' key"
        assert 'best_val' in results, "Results missing 'best_val' key"
        assert 'best_pt' in results, "Results missing 'best_pt' key"
        assert 'termination_reason' in results, "Results missing 'termination_reason' key"
        
        # EI doesn't generate samples or logz - should be empty dicts
        assert 'samples' in results, "Results missing 'samples' key"
        assert 'logz' in results, "Results missing 'logz' key"
        assert results['samples'] == {}, "EI should have empty samples dict"
        assert results['logz'] == {}, "EI should have empty logz dict"
        
        # Check GP was trained (should have at least initial points)
        assert results['gp'].train_x.shape[0] >= 10, "GP has fewer training points than expected"
        assert results['gp'].train_x.shape[1] == 2, "GP has wrong dimensionality"
        
        # Check best point is reasonable (should be close to global optimum at [1, 1])
        best_pt = results['best_pt']
        best_val = results['best_val']
        
        print(f"\nBest point found: x={best_pt[0]:.4f}, y={best_pt[1]:.4f}")
        print(f"Best value: {best_val:.4f}")
        print(f"Distance from optimum (1, 1): {np.linalg.norm(best_pt - np.array([1, 1])):.4f}")
        print(f"Termination reason: {results['termination_reason']}")
        print(f"Total evaluations: {results['gp'].train_x.shape[0]}")
        
        # Check that best value is better than initial random samples (should be negative and large)
        assert best_val > -1000, f"Best value {best_val} is unexpectedly poor"
        
        # Check output files (EI mode doesn't create chain files)
        check_output_files(temp_dir, "rosenbrock_ei_test", check_chains=False)
        
        print("\n✓ EI test passed")
        return results
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_bobe_wipstd_2d():
    """Test BOBE with WIPStd acquisition on 2D Himmelblau function."""
    print("\n" + "="*80)
    print("TEST: BOBE with WIPStd on 2D Himmelblau")
    print("="*80)
    
    import tempfile
    import shutil
    
    # Create temporary directory for output
    temp_dir = tempfile.mkdtemp()
    
    try:
        param_bounds = np.array([[-5, 5], [-5, 5]]).T
        param_list = ['x', 'y']
        
        bobe = BOBE(
            loglikelihood=himmelblau_loglike,
            param_list=param_list,
            param_bounds=param_bounds,
            likelihood_name="himmelblau_test",
            gp=GP(kernel=RBFKernel()),
            n_sobol_init=4,
            save=True,
            save_dir=temp_dir,
            seed=123,
            verbosity='WARNING'
        )
        
        # Run with WIPStd
        results = bobe.run(
            acq='wipstd',  # lowercase
            min_evals=25,
            max_evals=60,
            max_gp_size=60,
            logz_threshold=0.5,
            convergence_n_iters=2,
            fit_n_points=8,
            ns_n_points=15,
            batch_size=1,
            mc_points_method='uniform'
        )
        
        # Workers return None
        if results is None:
            return None
        
        # Check results structure
        assert 'gp' in results, "Results missing 'gp' key"
        assert 'likelihood' in results, "Results missing 'likelihood' key"
        assert 'results_manager' in results, "Results missing 'results_manager' key"
        assert 'best_val' in results, "Results missing 'best_val' key"
        assert 'best_pt' in results, "Results missing 'best_pt' key"
        assert 'termination_reason' in results, "Results missing 'termination_reason' key"
        
        # Check GP was trained
        assert results['gp'].train_x.shape[0] >= 30, "GP has fewer than min_evals training points"
        assert results['gp'].train_x.shape[1] == 2, "GP has wrong dimensionality"
        
        # Himmelblau has 4 minima, all with value 0
        # Check best point is reasonable
        best_pt = results['best_pt']
        best_val = results['best_val']
        
        print(f"\nBest point found: x={best_pt[0]:.4f}, y={best_pt[1]:.4f}")
        print(f"Best value: {best_val:.4f}")
        print(f"Termination reason: {results['termination_reason']}")
        print(f"Total evaluations: {results['gp'].train_x.shape[0]}")
        
        # Check that best value is reasonable (negative Himmelblau min is 0, so best should be close to 0)
        assert best_val > -500, f"Best value {best_val} is unexpectedly poor"
        
        # Check samples were generated (for WIPStd)
        if 'samples' in results and results['samples']:
            print(f"Generated {len(results['samples']['x'])} samples")
            assert len(results['samples']['x']) > 0, "No samples generated"
        
        # Check output files (WIPStd mode creates chain files)
        check_output_files(temp_dir, "himmelblau_test", check_chains=True)
        
        print("\n✓ WIPStd test passed")
        return results
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_bobe_with_classifier():
    """Test BOBE with classifier on 2D function."""
    print("\n" + "="*80)
    print("TEST: BOBE with Classifier on 2D Rosenbrock")
    print("="*80)
    
    param_bounds = np.array([[-2, 2], [-2, 2]]).T
    param_list = ['x', 'y']
    
    gp = GPwithClassifier(
        kernel=RBFKernel(lengthscale_prior="DSLP"),
        clf_type='svm',
        clf_use_size=10,
        clf_threshold=75.0,
        gp_threshold=150.0,
        minus_inf=-1e10,
    )
    bobe = BOBE(
        loglikelihood=rosenbrock_loglike,
        param_list=param_list,
        param_bounds=param_bounds,
        likelihood_name="rosenbrock_clf_test",
        gp=gp,
        n_sobol_init=4,
        save=False,
        seed=456,
        verbosity='WARNING'
    )
    
    # Run with WIPStd and classifier
    results = bobe.run(
        acq='wipstd',  # lowercase
        min_evals=20,
        max_evals=50,
        max_gp_size=50,
        logz_threshold=0.5,
        fit_n_points=6,
        ns_n_points=12,
        batch_size=1
    )
    
    # Workers return None
    if results is None:
        return None
    
    # Check results
    assert 'gp' in results
    assert hasattr(results['gp'], 'use_clf'), "GP should be GPwithClassifier"
    
    best_pt = results['best_pt']
    best_val = results['best_val']
    
    print(f"\nBest point found: x={best_pt[0]:.4f}, y={best_pt[1]:.4f}")
    print(f"Best value: {best_val:.4f}")
    print(f"Termination reason: {results['termination_reason']}")
    print(f"Total evaluations: {results['gp'].train_x.shape[0]}")
    
    print("\n✓ Classifier test passed")
    return results


def test_bobe_custom_gp_kernel_and_physical_rotation(tmp_path):
    """Test passing a configured GP + Kernel (with physical rotation) and resume into BOBE."""
    param_bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]]).T
    param_list = ['x', 'y']
    phys_cov = np.array([[0.5, 0.2], [0.2, 0.8]])

    transform = PrincipalAxesTransform(
        covariance=phys_cov,
        rotation_dims=['x', 'y'],
    )
    kernel = MaternKernel(
        noise=1e-6,
        lengthscale_prior='DSLP',
        input_transform=transform,
    )
    gp = GP(kernel=kernel, optimizer='scipy')

    import pytest
    with pytest.raises(ValueError, match="surrogate instance"):
        BOBE(
            loglikelihood=rosenbrock_loglike,
            param_list=param_list,
            param_bounds=param_bounds,
            likelihood_name='missing_gp_test',
            n_sobol_init=4,
            save=False,
            verbosity='WARNING',
        )

    bobe = BOBE(
        loglikelihood=rosenbrock_loglike,
        param_list=param_list,
        param_bounds=param_bounds,
        likelihood_name='custom_gp_test',
        gp=gp,
        n_sobol_init=6,
        save=True,
        save_dir=str(tmp_path),
        seed=7,
        verbosity='WARNING',
    )
    assert bobe.gp is gp
    assert isinstance(bobe.gp.kernel, MaternKernel)
    assert bobe.gp.param_names == ['x', 'y']
    assert bobe.gp.kernel.input_transform.rotation is not None
    assert bobe.gp.kernel.input_transform.rotation.shape == (2, 2)
    assert bobe.gp.npoints == 6
    bobe.pool.close()

    # Resume always restores from resume_file even if a different gp is passed
    resumed = BOBE(
        loglikelihood=rosenbrock_loglike,
        param_list=param_list,
        param_bounds=param_bounds,
        likelihood_name='custom_gp_test',
        gp=GP(kernel=RBFKernel()),
        resume=True,
        resume_file=str(tmp_path / 'custom_gp_test'),
        save=False,
        save_dir=str(tmp_path),
        seed=7,
        verbosity='WARNING',
    )
    assert isinstance(resumed.gp.kernel, MaternKernel)
    assert resumed.gp.kernel.input_transform is not None
    assert resumed.gp.npoints == 6
    resumed.pool.close()

    # Pre-populated GP is rejected — custom initial points must go through BOBE
    pre_x = np.array([[0.2, 0.3], [0.5, 0.5], [0.8, 0.7], [0.3, 0.8]])
    pre_y = np.array([[-1.0], [-0.2], [-0.5], [-0.8]])
    pre_gp = GP(train_x=pre_x, train_y=pre_y, kernel=RBFKernel(noise=1e-6))
    with pytest.raises(ValueError, match="empty surrogate"):
        BOBE(
            loglikelihood=rosenbrock_loglike,
            param_list=param_list,
            param_bounds=param_bounds,
            likelihood_name='prepop_test',
            gp=pre_gp,
            n_sobol_init=0,
            save=False,
            seed=7,
            verbosity='WARNING',
        )

    # Custom physical-space initial points passed through BOBE init work with n_sobol_init=0
    phys_init_x = np.array([[-1.0, 0.0], [0.0, 0.5], [1.0, 1.0], [0.5, -0.5]])
    phys_init_y = np.array([rosenbrock_loglike(pt) for pt in phys_init_x])
    bobe_init = BOBE(
        loglikelihood=rosenbrock_loglike,
        param_list=param_list,
        param_bounds=param_bounds,
        likelihood_name='custom_init_test',
        gp=GP(kernel=RBFKernel(noise=1e-6)),
        n_sobol_init=0,
        init_train_x=phys_init_x,
        init_train_y=phys_init_y,
        save=False,
        seed=7,
        verbosity='WARNING',
    )
    assert bobe_init.gp.npoints == 4
    bobe_init.pool.close()


def run_all_tests():
    """Run all 2D tests."""
    print("\n" + "="*80)
    print("RUNNING ALL BOBE 2D TESTS")
    print("="*80)
    
    try:
        test_bobe_ei_2d()
        test_bobe_wipstd_2d()
        test_bobe_with_classifier()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80 + "\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
