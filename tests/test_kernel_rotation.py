from math import sqrt

import numpy as np
import pytest
import jax.numpy as jnp

from BOBE.kernels import RBFKernel, MaternKernel
from BOBE.transforms import PrincipalAxesTransform


COVARIANCE = jnp.array([
    [0.09, 0.06],
    [0.06, 0.16],
])

LENGTHSCALES = jnp.array([0.18, 0.65])
KERNEL_VARIANCE = 1.7


def manual_kernel_matrix(
    kernel_cls,
    xa,
    xb,
    transform,
    lengthscales,
    kernel_variance,
):
    xa = transform.forward(xa)
    xb = transform.forward(xb)

    difference = (
        xa[:, None, :] - xb[None, :, :]
    ) / lengthscales

    dsq = jnp.sum(difference**2, axis=-1)

    if kernel_cls is RBFKernel:
        return kernel_variance * jnp.exp(-0.5 * dsq)

    d = jnp.sqrt(dsq)
    sqrt5 = sqrt(5.0)

    return (
        kernel_variance
        * (1.0 + sqrt5 * d + 5.0 * d**2 / 3.0)
        * jnp.exp(-sqrt5 * d)
    )


@pytest.mark.parametrize(
    "kernel_cls",
    [RBFKernel, MaternKernel],
)
def test_rotated_kernel_matches_manual_metric(kernel_cls):
    transform = PrincipalAxesTransform(
        covariance=COVARIANCE
    )

    kernel = kernel_cls(
        lengthscales=LENGTHSCALES,
        kernel_variance=KERNEL_VARIANCE,
        noise=1e-8,
    )
    kernel.set_input_transform(transform)

    xa = jnp.array([
        [0.10, 0.20],
        [0.40, 0.75],
    ])

    xb = jnp.array([
        [0.25, 0.30],
        [0.80, 0.60],
        [0.65, 0.15],
    ])

    actual = kernel.covariance(
        xa,
        xb,
        include_noise=False,
    )

    expected = manual_kernel_matrix(
        kernel_cls,
        xa,
        xb,
        transform,
        LENGTHSCALES,
        KERNEL_VARIANCE,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "kernel_cls",
    [RBFKernel, MaternKernel],
)
def test_rotation_changes_anisotropic_kernel_metric(kernel_cls):
    transform = PrincipalAxesTransform(
        covariance=COVARIANCE
    )

    rotated_kernel = kernel_cls(
        LENGTHSCALES,
        KERNEL_VARIANCE,
    )
    rotated_kernel.set_input_transform(transform)

    unrotated_kernel = kernel_cls(
        LENGTHSCALES,
        KERNEL_VARIANCE,
    )

    x = jnp.array([
        [0.10, 0.20],
        [0.40, 0.75],
        [0.80, 0.45],
    ])

    rotated = rotated_kernel.covariance(
        x,
        x,
        include_noise=False,
    )

    unrotated = unrotated_kernel.covariance(
        x,
        x,
        include_noise=False,
    )

    assert not np.allclose(
        rotated,
        unrotated,
        rtol=1e-5,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "kernel_cls",
    [RBFKernel, MaternKernel],
)
def test_isotropic_kernel_is_rotation_invariant(kernel_cls):
    transform = PrincipalAxesTransform(
        covariance=COVARIANCE
    )

    lengthscales = jnp.array([0.4, 0.4])

    rotated_kernel = kernel_cls(
        lengthscales,
        KERNEL_VARIANCE,
    )
    rotated_kernel.set_input_transform(transform)

    unrotated_kernel = kernel_cls(
        lengthscales,
        KERNEL_VARIANCE,
    )

    x = jnp.array([
        [0.10, 0.20],
        [0.40, 0.75],
        [0.80, 0.45],
    ])

    np.testing.assert_allclose(
        rotated_kernel.covariance(
            x,
            x,
            include_noise=False,
        ),
        unrotated_kernel.covariance(
            x,
            x,
            include_noise=False,
        ),
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "kernel_cls",
    [RBFKernel, MaternKernel],
)
def test_partial_rotation_matches_manual_pretransformation(kernel_cls):
    transform = PrincipalAxesTransform(
        covariance=COVARIANCE,
        rotation_dims=[0, 2],
    )

    lengthscales = jnp.array([
        0.18,
        0.30,
        0.65,
        0.45,
    ])

    transformed_kernel = kernel_cls(
        lengthscales,
        KERNEL_VARIANCE,
    )
    transformed_kernel.set_input_transform(transform)

    manual_kernel = kernel_cls(
        lengthscales,
        KERNEL_VARIANCE,
    )

    x = jnp.array([
        [0.10, 0.20, 0.30, 0.40],
        [0.45, 0.55, 0.75, 0.65],
        [0.80, 0.15, 0.50, 0.90],
    ])

    actual = transformed_kernel.covariance(
        x,
        x,
        include_noise=False,
    )

    expected = manual_kernel.covariance(
        transform.forward(x),
        transform.forward(x),
        include_noise=False,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "kernel_cls",
    [RBFKernel, MaternKernel],
)
def test_rotation_does_not_change_kernel_diagonal(kernel_cls):
    transform = PrincipalAxesTransform(
        covariance=COVARIANCE
    )

    noise = 1e-5

    kernel = kernel_cls(
        LENGTHSCALES,
        KERNEL_VARIANCE,
        noise=noise,
    )
    kernel.set_input_transform(transform)

    x = jnp.array([
        [0.10, 0.20],
        [0.40, 0.75],
        [0.80, 0.45],
    ])

    expected = jnp.full(
        x.shape[0],
        KERNEL_VARIANCE + noise,
    )

    np.testing.assert_allclose(
        kernel.diagonal(
            x,
            include_noise=True,
        ),
        expected,
        rtol=1e-6,
        atol=1e-7,
    )