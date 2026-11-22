"""
Consolidated tests for the BOBE Bayesian Optimization loop, convergence,
unified checkpointing/resume, multi-phase runs, and surrogate composition.
"""

import json
import os
import time

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

import numpy as np
import pytest

from BOBE import (
    BOBE,
    ClassifierGP,
    GP,
    MaternKernel,
    PrincipalAxesTransform,
    RBFKernel,
    SVMClassifier,
)
from BOBE.utils.results import BOBEResults


def rosenbrock_loglike(x):
    """Negative Rosenbrock function as log-likelihood (global max 0 at (1, 1))."""
    return -((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)


def himmelblau_loglike(x):
    """Negative Himmelblau function as log-likelihood (four equal maxima of 0)."""
    return -((x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2)


def gauss_2d_loglike(x):
    """Simple isotropic 2D Gaussian log-likelihood."""
    x = np.asarray(x, dtype=float)
    return float(-0.5 * np.sum(x ** 2))


def check_output_files(output_dir, base_name, check_chains=True):
    """Verify expected unified BOBE output files are created and well-formed."""
    required_files = [
        f"{base_name}.minimum.txt",
        f"{base_name}.minimum",
        f"{base_name}.pkl",
        f"{base_name}_summary.json",
    ]
    chain_files = [
        f"{base_name}.txt",
        f"{base_name}.paramnames",
        f"{base_name}.ranges",
    ]

    expected_files = required_files + (chain_files if check_chains else [])
    missing = [f for f in expected_files if not os.path.exists(os.path.join(output_dir, f))]
    assert not missing, f"Missing expected output files: {missing}"

    with open(os.path.join(output_dir, f"{base_name}.minimum.txt"), 'r') as f:
        txt_lines = f.readlines()
    assert len(txt_lines) >= 2 and txt_lines[0].startswith('#')

    with open(os.path.join(output_dir, f"{base_name}.minimum"), 'r') as f:
        min_lines = f.readlines()
    assert len(min_lines) >= 4 and '-log(Like)' in min_lines[0]


@pytest.mark.slow
def test_bobe_ei_2d(tmp_path):
    """Test BOBE with EI acquisition on 2D Rosenbrock function."""
    param_bounds = np.array([[-2, 2], [-2, 2]]).T
    bobe = BOBE(
        loglikelihood=rosenbrock_loglike,
        param_list=['x', 'y'],
        param_bounds=param_bounds,
        likelihood_name="rosenbrock_ei_test",
        gp=GP(kernel=RBFKernel()),
        n_sobol_init=4,
        save=True,
        save_dir=str(tmp_path),
        seed=42,
        verbosity='WARNING',
    )

    results = bobe.run(
        acq='ei',
        min_evals=10,
        max_evals=20,
        max_gp_size=25,
        ei_goal=1e-6,
        fit_n_points=5,
        batch_size=1,
    )
    if results is None:
        return

    for key in ('gp', 'likelihood', 'results_manager', 'best_val', 'best_pt', 'termination_reason', 'samples', 'logz'):
        assert key in results
    assert results['samples'] == {}
    assert results['logz'] == {}
    assert results['gp'].train_x.shape[0] >= 10
    assert results['gp'].train_x.shape[1] == 2
    assert results['best_val'] > -1000
    check_output_files(str(tmp_path), "rosenbrock_ei_test", check_chains=False)


@pytest.mark.slow
def test_bobe_wipstd_2d(tmp_path):
    """Test BOBE with WIPStd acquisition on 2D Himmelblau function."""
    param_bounds = np.array([[-5, 5], [-5, 5]]).T
    bobe = BOBE(
        loglikelihood=himmelblau_loglike,
        param_list=['x', 'y'],
        param_bounds=param_bounds,
        likelihood_name="himmelblau_test",
        gp=GP(kernel=RBFKernel()),
        n_sobol_init=4,
        save=True,
        save_dir=str(tmp_path),
        seed=123,
        verbosity='WARNING',
    )

    results = bobe.run(
        acq='wipstd',
        min_evals=16,
        max_evals=30,
        max_gp_size=40,
        logz_threshold=0.5,
        convergence_n_iters=2,
        fit_n_points=8,
        ns_n_points=12,
        batch_size=2,
        mc_points_size=32,
        mc_points_method='uniform',
    )
    if results is None:
        return

    for key in ('gp', 'likelihood', 'results_manager', 'best_val', 'best_pt', 'termination_reason'):
        assert key in results
    assert results['gp'].train_x.shape[0] >= 16
    assert results['gp'].train_x.shape[1] == 2
    assert results['best_val'] > -500
    assert results.get('samples') and len(results['samples']['x']) > 0
    check_output_files(str(tmp_path), "himmelblau_test", check_chains=True)


@pytest.mark.slow
def test_bobe_with_classifier(tmp_path):
    """Test BOBE with ClassifierGP on 2D Rosenbrock and verify total_objective_evals accuracy."""
    param_bounds = np.array([[-2, 2], [-2, 2]]).T
    gp = ClassifierGP(
        kernel=RBFKernel(lengthscale_prior="DSLP"),
        classifier=SVMClassifier(),
        clf_use_size=10,
        clf_threshold=75.0,
        gp_threshold=150.0,
        minus_inf=-1e10,
    )
    bobe = BOBE(
        loglikelihood=rosenbrock_loglike,
        param_list=['x', 'y'],
        param_bounds=param_bounds,
        likelihood_name="rosenbrock_clf_test",
        gp=gp,
        n_sobol_init=4,
        save=False,
        save_dir=str(tmp_path),
        seed=456,
        verbosity='WARNING',
    )

    results = bobe.run(
        acq='wipstd',
        min_evals=12,
        max_evals=24,
        max_gp_size=30,
        logz_threshold=0.5,
        fit_n_points=6,
        ns_n_points=100,
        batch_size=2,
        num_hmc_warmup=32,
        num_hmc_samples=64,
        thinning=2,
        num_chains=2,
        mc_points_size=32,
        do_final_ns=False,
    )
    if results is None:
        return

    assert isinstance(results['gp'], ClassifierGP)
    # Dedicated objective-eval counter matches classifier data size and exceeds filtered GP size
    assert results['results_manager'].gp_info['total_objective_evals'] == bobe.total_objective_evals
    assert bobe.gp.clf_data_size == bobe.total_objective_evals
    assert bobe.gp.train_x.shape[0] < bobe.total_objective_evals


def test_bobe_custom_gp_kernel_and_physical_rotation(tmp_path):
    """Test passing a configured GP + Kernel (with physical rotation) and custom init points."""
    param_bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]]).T
    param_list = ['x', 'y']
    phys_cov = np.array([[0.5, 0.2], [0.2, 0.8]])

    transform = PrincipalAxesTransform(covariance=phys_cov, rotation_dims=['x', 'y'])
    kernel = MaternKernel(noise=1e-6, lengthscale_prior='DSLP', input_transform=transform)
    gp = GP(kernel=kernel, optimizer='scipy')

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
        n_sobol_init=8,
        save=True,
        save_dir=str(tmp_path),
        seed=7,
        verbosity='WARNING',
    )
    assert bobe.gp is gp
    assert isinstance(bobe.gp.kernel, MaternKernel)
    assert bobe.gp.param_names == ['x', 'y']
    assert bobe.gp.kernel.input_transform.rotation.shape == (2, 2)
    assert bobe.gp.npoints == 8
    bobe.close()

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
    assert resumed.gp.npoints == 8
    resumed.close()

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

    # Custom physical-space initial points work with n_sobol_init=0
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
    bobe_init.close()


def test_dual_logz_and_kl_convergence_gating(tmp_path):
    """Convergence requires both logz_std and symmetric KL below threshold."""
    param_bounds = np.array([[-2.0, -2.0], [2.0, 2.0]])
    bobe = BOBE(
        loglikelihood=gauss_2d_loglike,
        param_list=['x', 'y'],
        param_bounds=param_bounds,
        likelihood_name='conv_gate_test',
        gp=GP(kernel=RBFKernel()),
        n_sobol_init=4,
        save=False,
        save_dir=str(tmp_path),
        seed=7,
        verbosity='ERROR',
    )
    try:
        bobe.logz_threshold = 0.1
        bobe.kl_convergence_threshold = 0.05
        bobe.convergence_n_iters = 1
        bobe.convergence_counter = 0

        rng = np.random.default_rng(99)
        s1 = rng.normal(loc=0.0, scale=0.5, size=(400, 2))
        s2_close = rng.normal(loc=0.0, scale=0.5, size=(400, 2))
        s3_far = rng.normal(loc=1.0, scale=0.5, size=(400, 2))
        logl = np.zeros(400)
        good_logz = {'mean': -1.0, 'std': 0.02, 'upper': -0.98, 'lower': -1.02, 'var': 4e-4, 'dlogz_sampler': 0.01}
        bad_logz = {'mean': -1.0, 'std': 0.25, 'upper': -0.75, 'lower': -1.25, 'var': 0.0625, 'dlogz_sampler': 0.01}

        # Check 1: prev_samples is None -> KL is inf -> cannot converge yet
        assert not bobe.check_convergence_logz(1, good_logz, s1, logl)
        assert bobe.convergence_counter == 0
        assert bobe.results_manager.convergence_history[-1].delta == pytest.approx(0.02)

        # Check 2: good logz_std but large KL -> still not converged
        assert not bobe.check_convergence_logz(2, good_logz, s3_far, logl)
        assert bobe.convergence_counter == 0

        # Check 3: small KL but bad logz_std -> not converged
        bobe.prev_samples = {'x': s1, 'logl': logl}
        assert not bobe.check_convergence_logz(3, bad_logz, s2_close, logl)
        assert bobe.convergence_counter == 0

        # Check 4: both logz_std and KL below threshold -> converges
        bobe.prev_samples = {'x': s1, 'logl': logl}
        assert bobe.check_convergence_logz(4, good_logz, s2_close, logl)
        assert bobe.convergence_counter == 1
    finally:
        bobe.close()


def test_unified_checkpoint_resume_and_load_results(tmp_path):
    """Unified .pkl checkpointing, BOBEResults.load_results, and mid-run resume."""
    param_bounds = np.array([[-2.0, -2.0], [2.0, 2.0]])
    bobe = BOBE(
        loglikelihood=gauss_2d_loglike,
        param_list=['x', 'y'],
        param_bounds=param_bounds,
        likelihood_name='resume_test',
        gp=GP(kernel=RBFKernel()),
        n_sobol_init=4,
        save=True,
        save_dir=str(tmp_path),
        save_step=1,
        seed=42,
        verbosity='ERROR',
    )
    res1 = bobe.run(
        acq='wipstd',
        min_evals=6,
        max_evals=8,
        max_gp_size=20,
        logz_threshold=0.5,
        kl_convergence_threshold=0.5,
        fit_n_points=4,
        ns_n_points=2,
        batch_size=1,
        mc_points_size=32,
        mc_points_method='uniform',
        do_final_ns=False,
    )

    pkl_file = tmp_path / 'resume_test.pkl'
    summary_file = tmp_path / 'resume_test_summary.json'
    assert pkl_file.exists()
    assert summary_file.exists()
    assert not (tmp_path / 'resume_test_gp.npz').exists()
    assert not (tmp_path / 'resume_test_intermediate.json').exists()
    assert not (tmp_path / 'resume_test_results.pkl').exists()

    summary_data = json.loads(summary_file.read_text())
    assert summary_data['gp_info']['total_objective_evals'] == 8
    assert len(res1['results_manager'].gp_noises) > 0

    # BOBEResults.load_results works directly on unified .pkl without double-counting runtime
    loaded_res = BOBEResults.load_results(str(tmp_path / 'resume_test'))
    assert loaded_res.gp_info['total_objective_evals'] == 8
    assert len(loaded_res.gp_noises) == len(res1['results_manager'].gp_noises)
    assert loaded_res.termination_reason == res1['termination_reason']
    orig_runtime = res1['results_manager'].get_timing_summary()['total_runtime']
    loaded_runtime = loaded_res.get_timing_summary()['total_runtime']
    assert loaded_runtime == pytest.approx(orig_runtime, rel=1e-2, abs=1e-2)

    # Resume restores counters, GP, and prev_samples
    resumed = BOBE(
        loglikelihood=gauss_2d_loglike,
        param_list=['x', 'y'],
        param_bounds=param_bounds,
        likelihood_name='resume_test',
        resume=True,
        resume_file=str(tmp_path / 'resume_test'),
        save=True,
        save_dir=str(tmp_path),
        seed=42,
        verbosity='ERROR',
    )
    assert resumed.total_objective_evals == 8
    assert resumed.start_iteration == bobe.current_iteration
    assert resumed.gp.npoints == bobe.gp.npoints
    assert resumed.best_f == pytest.approx(bobe.best_f)
    if bobe.prev_samples is not None:
        assert resumed.prev_samples is not None
        np.testing.assert_allclose(resumed.prev_samples['x'], bobe.prev_samples['x'])

    res2 = resumed.run(
        acq='wipstd',
        min_evals=6,
        max_evals=10,
        max_gp_size=20,
        logz_threshold=0.001,
        kl_convergence_threshold=0.001,
        fit_n_points=4,
        ns_n_points=50,
        batch_size=1,
        mc_points_size=32,
        mc_points_method='uniform',
        do_final_ns=False,
    )
    assert res2['results_manager'].gp_info['total_objective_evals'] == 10
    assert resumed.current_iteration == bobe.current_iteration + 2


def test_already_converged_resume_skips_or_continues(tmp_path):
    """Already-converged resume skips when threshold is not stricter and continues when stricter."""
    param_bounds = np.array([[-2.0, -2.0], [2.0, 2.0]])
    bobe = BOBE(
        loglikelihood=gauss_2d_loglike,
        param_list=['x', 'y'],
        param_bounds=param_bounds,
        likelihood_name='already_conv',
        gp=GP(kernel=RBFKernel()),
        n_sobol_init=4,
        save=True,
        save_dir=str(tmp_path),
        seed=11,
        verbosity='ERROR',
    )
    logz_dict = {'mean': -1.5, 'std': 0.05, 'upper': -1.45, 'lower': -1.55, 'var': 0.0025, 'dlogz_sampler': 0.01}
    bobe.results_manager.update_convergence(
        iteration=5, logz_dict=logz_dict, converged=True, threshold=0.2, delta=0.05
    )
    bobe.converged = True
    bobe.termination_reason = "LogZ converged (periodic NS)"
    bobe.samples_dict = {
        'x': np.zeros((20, 2)),
        'weights': np.ones(20) / 20.0,
        'logl': np.zeros(20),
    }
    bobe.results_dict = {'logz': logz_dict}
    bobe.current_iteration = 5
    bobe.finalise_results()
    bobe._save_checkpoint()
    bobe.close()

    summary_path = tmp_path / 'already_conv_summary.json'
    mtime_before = summary_path.stat().st_mtime
    time.sleep(0.02)

    # Not stricter -> skip BO loop and leave existing files untouched
    r_skip = BOBE(
        loglikelihood=gauss_2d_loglike,
        param_list=['x', 'y'],
        param_bounds=param_bounds,
        likelihood_name='already_conv',
        resume=True,
        resume_file=str(tmp_path / 'already_conv'),
        save=True,
        save_dir=str(tmp_path),
        verbosity='ERROR',
    )
    out_skip = r_skip.run(acq='wipstd', logz_threshold=0.2, max_evals=10, mc_points_method='uniform')
    assert out_skip['termination_reason'] == "Already converged in previous run"
    assert summary_path.stat().st_mtime == mtime_before
    assert len(out_skip['samples']['x']) == 20

    # Stricter threshold (even though prev delta 0.05 < 0.1) -> continue optimization
    r_cont = BOBE(
        loglikelihood=gauss_2d_loglike,
        param_list=['x', 'y'],
        param_bounds=param_bounds,
        likelihood_name='already_conv',
        resume=True,
        resume_file=str(tmp_path / 'already_conv'),
        save=True,
        save_dir=str(tmp_path),
        seed=11,
        verbosity='ERROR',
    )
    out_cont = r_cont.run(
        acq='wipstd',
        logz_threshold=0.1,
        min_evals=100,
        max_evals=6,
        max_gp_size=20,
        batch_size=1,
        mc_points_size=32,
        mc_points_method='uniform',
        do_final_ns=False,
    )
    assert out_cont['termination_reason'] == "Maximum evaluations reached"
    assert r_cont.total_objective_evals == 6


def test_multiphase_run_close_pool_and_no_tuple_acq(tmp_path):
    """Chaining run(..., close_pool=False) across EI -> WIPStd and idempotent close()."""
    param_bounds = np.array([[-2.0, -2.0], [2.0, 2.0]])
    bobe = BOBE(
        loglikelihood=gauss_2d_loglike,
        param_list=['x', 'y'],
        param_bounds=param_bounds,
        likelihood_name='multiphase_test',
        gp=GP(kernel=RBFKernel()),
        n_sobol_init=4,
        save=True,
        save_dir=str(tmp_path),
        seed=5,
        verbosity='ERROR',
    )

    with pytest.raises(TypeError, match="acq must be a string"):
        bobe.run(acq=('ei', 'wipstd'), max_evals=6, close_pool=False)

    bobe.run(
        acq='ei',
        max_evals=6,
        max_gp_size=20,
        ei_goal=1e-12,
        fit_n_points=4,
        close_pool=False,
    )
    assert not bobe.pool._closed
    assert bobe.total_objective_evals == 6
    iter_after_p1 = bobe.current_iteration
    assert iter_after_p1 == 2

    res_p2 = bobe.run(
        acq='wipstd',
        min_evals=20,
        max_evals=8,
        max_gp_size=20,
        batch_size=1,
        mc_points_size=32,
        mc_points_method='uniform',
        do_final_ns=False,
        close_pool=False,
    )
    assert not bobe.pool._closed
    assert bobe.total_objective_evals == 8
    assert bobe.current_iteration == iter_after_p1 + 2
    assert res_p2['results_manager'].gp_info['total_objective_evals'] == 8

    bobe.close()
    assert bobe.pool._closed
    bobe.close()
    assert bobe.pool._closed
