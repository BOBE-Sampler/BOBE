"""Tests for core utility helpers in BOBE.utils.core."""

import numpy as np
import pytest

from BOBE.utils.core import (
    kl_divergence_gaussian,
    resolve_parameter_dimensions,
    scale_to_unit,
    scale_from_unit,
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
