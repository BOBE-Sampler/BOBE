from abc import ABC, abstractmethod
from typing import Sequence

import jax.numpy as jnp

from .utils.core import renormalise_log_weights


def prepare_rotation_inputs(covariance=None, samples=None, param_bounds=None, rotation_dims=None, rotation_space="physical"):

    if rotation_space not in ("physical", "unit"):
        raise ValueError(
            "rotation_space must be either 'physical' or 'unit'"
        )

    if (rotation_space == "unit" or sum(x is not None for x in (covariance, samples)) != 1):
        return covariance, samples


    source = jnp.asarray(covariance if covariance is not None else samples)

    if (source.ndim != 2 or covariance is not None and source.shape[0] != source.shape[1]):
        return covariance, samples

    bounds = jnp.asarray(param_bounds)
    input_ndim = source.shape[-1]

    if input_ndim != bounds.shape[1]:
        if rotation_dims is None or input_ndim != len(rotation_dims):
            return covariance, samples
        bounds = bounds[:, jnp.asarray(rotation_dims)]

    lower, upper = bounds
    widths = upper - lower

    if covariance is not None:
        return source / jnp.outer(widths, widths), None

    return None, (source - lower) / widths


class InputTransform(ABC):
    """
    Base interface for kernel input transforms.
    """

    name: str = "base"

    @abstractmethod
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Transform coordinates used to evaluate the kernel.
        """
        raise NotImplementedError

    @abstractmethod
    def inverse(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Invert the kernel coordinate transform.
        """
        raise NotImplementedError


class PrincipalAxesTransform(InputTransform):
    """
    Apply a static principal-axes rotation to the kernel metric.
    """

    name: str = "principal_axes"

    def __init__(
        self,
        covariance: jnp.ndarray = None,
        samples: jnp.ndarray = None,
        weights: jnp.ndarray = None,
        log_weights: jnp.ndarray = None,
        rotation_dims: Sequence[int] = None,
    ):
        """
        Initialise a static principal-axes rotation of the kernel metric.

        Exactly one of covariance or samples must be provided. All inputs are
        assumed to be expressed in the same unit-hypercube coordinates used
        internally by BOBE.

        Parameters
        ----------
        covariance : jnp.ndarray, optional
            Covariance matrix used to define the static rotation.
        samples : jnp.ndarray, optional
            Samples or chain points used to estimate the covariance matrix.
            This may be, for example, weighted nested-sampling samples or an
            MCMC chain.
        weights : jnp.ndarray, optional
            Weights associated with the supplied samples or chain.
        log_weights : jnp.ndarray, optional
            Log weights associated with the supplied samples, such as
            nested-sampling log weights.
        rotation_dims : sequence of int, optional
            Dimensions of the kernel metric to rotate. If None, all
            dimensions are rotated.
        """
        sources = (
            covariance is not None,
            samples is not None,
        )

        if sum(sources) != 1:
            raise ValueError(
                "Provide exactly one of covariance or samples"
            )

        if weights is not None and log_weights is not None:
            raise ValueError(
                "Provide either weights or log_weights, not both"
            )

        if samples is None and (
            weights is not None or log_weights is not None
        ):
            raise ValueError(
                "Weights and log_weights can only be used with samples"
            )

        self.rotation_dims = self._setup_rotation_dims(rotation_dims)

        if covariance is not None:
            covariance = self._select_matrix(covariance)

        else:
            samples = self._select_samples(samples)
            covariance = self._covariance_from_samples(
                samples,
                weights=weights,
                log_weights=log_weights,
            )

        self.rotation = self._rotation_from_covariance(covariance)

    @staticmethod
    def _setup_rotation_dims(rotation_dims):
        if rotation_dims is None:
            return None

        rotation_dims = tuple(int(i) for i in rotation_dims)

        if len(rotation_dims) == 0:
            raise ValueError("rotation_dims cannot be empty")

        if min(rotation_dims) < 0:
            raise ValueError(
                "rotation_dims must contain non-negative indices"
            )

        if len(set(rotation_dims)) != len(rotation_dims):
            raise ValueError(
                "rotation_dims cannot contain duplicate indices"
            )

        return jnp.asarray(rotation_dims, dtype=jnp.int32)

    def _select_matrix(self, matrix):
        matrix = jnp.asarray(matrix)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                "Covariance matrix must be square"
            )

        if self.rotation_dims is None:
            return matrix

        ndim = len(self.rotation_dims)

        if matrix.shape[0] == ndim:
            return matrix

        if matrix.shape[0] <= int(jnp.max(self.rotation_dims)):
            raise ValueError(
                "rotation_dims are incompatible with the supplied matrix"
            )

        return matrix[jnp.ix_(self.rotation_dims, self.rotation_dims)]

    def _select_samples(self, samples):
        samples = jnp.asarray(samples)

        if samples.ndim != 2:
            raise ValueError(
                "samples must have shape (N, D)"
            )

        if self.rotation_dims is None:
            return samples

        ndim = len(self.rotation_dims)

        if samples.shape[1] == ndim:
            return samples

        if samples.shape[1] <= int(jnp.max(self.rotation_dims)):
            raise ValueError(
                "rotation_dims are incompatible with the supplied samples"
            )

        return samples[:, self.rotation_dims]

    @staticmethod
    def _rotation_from_covariance(covariance):
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
        order = jnp.argsort(eigenvalues)[::-1]

        return eigenvectors[:, order]

    @staticmethod
    def _covariance_from_samples(
        samples,
        weights=None,
        log_weights=None,
    ):
        n_samples = samples.shape[0]

        if log_weights is not None:
            log_weights = jnp.asarray(log_weights)

            if log_weights.shape != (n_samples,):
                raise ValueError(
                    "log_weights must have shape (N,)"
                )

            weights = renormalise_log_weights(log_weights)

        elif weights is not None:
            weights = jnp.asarray(weights)

            if weights.shape != (n_samples,):
                raise ValueError(
                    "weights must have shape (N,)"
                )

            if jnp.any(weights < 0):
                raise ValueError(
                    "weights must be non-negative"
                )

            if jnp.sum(weights) <= 0:
                raise ValueError(
                    "weights must have positive total weight"
                )

            weights = weights / jnp.sum(weights)

        else:
            weights = jnp.ones(n_samples) / n_samples

        mean = jnp.sum(weights[:, None] * samples, axis=0)
        centred = samples - mean

        covariance = (centred * weights[:, None]).T @ centred

        return 0.5 * (covariance + covariance.T)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.rotation_dims is None:
            return x @ self.rotation

        rotated = x[..., self.rotation_dims] @ self.rotation

        return x.at[..., self.rotation_dims].set(rotated)

    def inverse(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.rotation_dims is None:
            return x @ self.rotation.T

        unrotated = x[..., self.rotation_dims] @ self.rotation.T

        return x.at[..., self.rotation_dims].set(unrotated)

    def state_dict(self):
        return {
            "type": "PrincipalAxesTransform",
            "rotation": self.rotation,
            "rotation_dims": self.rotation_dims,
        }

    @classmethod
    def from_state_dict(cls, state):
        transform = cls.__new__(cls)
        transform.rotation = jnp.asarray(state["rotation"])

        rotation_dims = state.get("rotation_dims")
        transform.rotation_dims = (
            None
            if rotation_dims is None
            else jnp.asarray(rotation_dims, dtype=jnp.int32)
        )

        return transform