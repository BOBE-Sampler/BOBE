"""Tests for GP surrogate samplers (Dynesty nested sampling and NumPyro NUTS)."""

import numpy as np
import pytest

from BOBE import GP, RBFKernel
from BOBE.acquisition import get_mc_samples
from BOBE.samplers import nested_sampling_Dy, sample_GP_NUTS


def _make_fitted_gp(seed=0, n=12):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, 2))
    Y = np.array([[-0.5 * np.sum(((pt - 0.5) * 4.0) ** 2)] for pt in X])
    return GP(train_x=X, train_y=Y, kernel=RBFKernel(noise=1e-5))


def test_nested_sampling_both_logz_err_methods_and_output_keys():
    gp = _make_fitted_gp(seed=1)
    bounds = np.array([[-2.0, -4.0], [2.0, 4.0]])
    rng1 = np.random.default_rng(10)
    rng2 = np.random.default_rng(10)

    samples_default, logz_default, ok_default = nested_sampling_Dy(
        gp=gp, mode='acq', ndim=2, dlogz=0.5, maxcall=5000,
        print_progress=False, rng=rng1, param_bounds=bounds,
    )
    assert ok_default
    for key in ('x', 'u', 'weights', 'logl', 'eq_u', 'eq_x', 'eq_logl', 'best', 'method'):
        assert key in samples_default, f"Missing key {key} in nested_sampling_Dy output"
    assert samples_default['method'] == 'nested'
    assert np.all(samples_default['u'] >= 0.0) and np.all(samples_default['u'] <= 1.0)
    assert np.all(samples_default['eq_u'] >= 0.0) and np.all(samples_default['eq_u'] <= 1.0)
    # Physical coordinates should span outside [0, 1] for [-4, 4]
    assert np.min(samples_default['x'][:, 1]) < 0.0 or np.max(samples_default['x'][:, 1]) > 1.0
    # Default mean_std sets upper/lower = mean ± std and var = std**2
    assert np.isclose(logz_default['upper'] - logz_default['mean'], logz_default['std'])
    assert np.isclose(logz_default['mean'] - logz_default['lower'], logz_default['std'])
    assert np.isclose(logz_default['var'], logz_default['std'] ** 2)

    # Integral error method remains available
    samples_int, logz_int, ok_int = nested_sampling_Dy(
        gp=gp, mode='acq', ndim=2, dlogz=0.5, maxcall=5000,
        print_progress=False, rng=rng2, param_bounds=bounds,
        logz_err_method='integral',
    )
    assert ok_int
    assert np.isfinite(logz_int['mean'])
    assert np.isfinite(logz_int['std']) and logz_int['std'] >= 0.0
    assert logz_int['upper'] >= logz_int['lower']

    with pytest.raises(ValueError, match="logz_err_method"):
        nested_sampling_Dy(gp=gp, ndim=2, logz_err_method='invalid')


def test_get_mc_samples_ns_returns_equal_weighted_unit_cube_points():
    gp = _make_fitted_gp(seed=3)
    bounds = np.array([[-2.0, -4.0], [2.0, 4.0]])
    mc = get_mc_samples(
        gp=gp, method='NS', np_rng=np.random.default_rng(11), param_bounds=bounds
    )
    assert mc['method'] == 'nested'
    assert np.all(mc['x'] >= 0.0) and np.all(mc['x'] <= 1.0)
    assert np.all(mc['weights'] == 1.0)
    assert 'x_physical' in mc and mc['x_physical'].shape == mc['x'].shape


def test_sample_gp_nuts_flat_and_r_hat():
    gp = _make_fitted_gp(seed=2)
    rng = np.random.default_rng(20)

    # Multi-chain flattened (default)
    res_flat = sample_GP_NUTS(
        gp, np_rng=rng, num_chains=2, warmup_steps=32, num_samples=64, thinning=2, flat=True
    )
    assert res_flat['x'].shape == (2 * 32, 2)
    assert res_flat['logp'].shape == (2 * 32,)
    assert res_flat['best'].shape == (2,)
    assert 'R' in res_flat and np.isfinite(res_flat['R']) and res_flat['R'] > 0.0

    # Multi-chain unflattened
    res_chains = sample_GP_NUTS(
        gp, np_rng=rng, num_chains=2, warmup_steps=32, num_samples=64, thinning=2, flat=False
    )
    assert res_chains['x'].shape == (2, 32, 2)
    assert res_chains['logp'].shape == (2, 32)
    assert res_chains['best'].shape == (2,)
    assert np.isfinite(res_chains['R'])

    # Single chain works even on multi-device hosts and reports R = nan
    res_single = sample_GP_NUTS(
        gp, np_rng=rng, num_chains=1, warmup_steps=16, num_samples=32, thinning=2, flat=False
    )
    assert res_single['x'].shape == (1, 16, 2)
    assert np.isnan(res_single['R'])
