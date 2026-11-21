"""Tests for core utility helpers in BOBE.utils.core."""

import numpy as np
import pytest

from BOBE.utils.core import (
    get_R,
    kl_divergence_gaussian,
    principal_axes,
    regularize_covariance,
    resolve_parameter_dimensions,
    scale_to_unit,
    scale_from_unit,
    weighted_mean_and_covariance,
)


def test_resolve_parameter_dimensions():
    assert resolve_parameter_dimensions(["b", 0], ["a", "b", "c"], ndim=3) == [1, 0]
    with pytest.raises(ValueError, match="Unknown parameter"):
        resolve_parameter_dimensions(["z"], ["a", "b"], ndim=2)
    with pytest.raises(ValueError, match="outside"):
        resolve_parameter_dimensions([3], ["a", "b"], ndim=2)


def test_kl_divergence_gaussian():
    kl = kl_divergence_gaussian(np.zeros(2), np.eye(2), np.zeros(2), np.eye(2))
    assert np.isclose(kl["symmetric"], 0.0, atol=1e-12)

    with pytest.raises(np.linalg.LinAlgError):
        kl_divergence_gaussian(
            np.zeros(2),
            np.array([[1.0, 2.0], [2.0, 1.0]]),
            np.zeros(2),
            np.eye(2),
        )


def test_scale_unit_round_trip():
    bounds = np.array([[-2.0, 10.0], [4.0, 20.0]])
    pts = np.array([[-2.0, 10.0], [1.0, 15.0], [4.0, 20.0]])
    u = scale_to_unit(pts, bounds)
    np.testing.assert_allclose(u, [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    np.testing.assert_allclose(scale_from_unit(u, bounds), pts)


def test_principal_axes_orders_descending_and_diagonalizes():
    cov = np.array([[0.09, 0.06], [0.06, 0.16]])
    evals, evecs = principal_axes(cov)
    assert evals[0] >= evals[1] > 0.0
    np.testing.assert_allclose(evecs.T @ evecs, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(evecs @ np.diag(evals) @ evecs.T, cov, atol=1e-12)


def test_regularize_covariance_validation():
    asym = np.array([[1.0, 0.2], [0.4, 2.0]])
    reg = regularize_covariance(asym, ndim=2, regularize_eps=1e-4)
    np.testing.assert_allclose(reg, reg.T)
    np.testing.assert_allclose(np.diag(reg), [1.0 + 1e-4, 2.0 + 1e-4])

    with pytest.raises(ValueError, match="square"):
        regularize_covariance(np.ones((2, 3)))
    with pytest.raises(ValueError, match="expected"):
        regularize_covariance(np.eye(2), ndim=3)
    with pytest.raises(ValueError, match="finite"):
        regularize_covariance(np.array([[1.0, np.nan], [np.nan, 1.0]]))
    with pytest.raises(ValueError, match="positive definite"):
        regularize_covariance(np.diag([1.0, -0.5]))


def test_weighted_mean_and_covariance():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(100, 3))
    mean, cov = weighted_mean_and_covariance(samples)
    np.testing.assert_allclose(mean, samples.mean(axis=0))
    np.testing.assert_allclose(cov, np.cov(samples, rowvar=False))

    # Integer weights match repeated samples (population covariance)
    base = samples[:20]
    counts = rng.integers(1, 5, size=len(base))
    w_mean, w_cov = weighted_mean_and_covariance(base, weights=counts.astype(float))
    rep = np.repeat(base, counts, axis=0)
    np.testing.assert_allclose(w_mean, rep.mean(axis=0))
    np.testing.assert_allclose(w_cov, np.cov(rep, rowvar=False, ddof=0))

    with pytest.raises(ValueError, match="two-dimensional"):
        weighted_mean_and_covariance(np.ones(5))
    with pytest.raises(ValueError, match="at least two"):
        weighted_mean_and_covariance(np.ones((1, 2)))
    with pytest.raises(ValueError, match="do not match"):
        weighted_mean_and_covariance(base, weights=np.ones(3))
    with pytest.raises(ValueError, match="non-negative"):
        weighted_mean_and_covariance(base, weights=np.full(len(base), -1.0))
    with pytest.raises(ValueError, match="positive total weight"):
        weighted_mean_and_covariance(base, weights=np.zeros(len(base)))


def test_get_R_gelman_rubin():
    rng = np.random.default_rng(123)
    # Identical distributions across 4 chains -> R-hat close to 1
    converged = rng.normal(size=(4, 500, 3))
    r_conv = get_R(converged)
    assert np.isfinite(r_conv)
    assert 0.99 <= r_conv <= 1.05

    # Shifted chains -> R-hat clearly > 1
    diverged = converged.copy()
    diverged[0] += 5.0
    diverged[1] -= 5.0
    r_div = get_R(diverged)
    assert r_div > 1.5

    # 2D input (num_chains, num_samples) supported
    r_2d = get_R(rng.normal(size=(3, 200)))
    assert np.isfinite(r_2d) and 0.99 <= r_2d <= 1.1

    # Single chain or single sample -> NaN
    assert np.isnan(get_R(rng.normal(size=(1, 100, 2))))
    assert np.isnan(get_R(rng.normal(size=(3, 1, 2))))

    # Constant chains -> 1.0
    assert get_R(np.ones((3, 50, 2))) == 1.0

    # Wrong ndim raises
    with pytest.raises(ValueError, match="2D or 3D"):
        get_R(np.ones(10))

