from abc import ABC, abstractmethod
from typing import Sequence

import jax.numpy as jnp

from .utils.core import renormalise_log_weights, resolve_parameter_dimensions


def prepare_rotation_inputs(covariance=None, samples=None, param_bounds=None, rotation_dims=None):
    """Scale physical-space covariance or samples into unit-hypercube coordinates."""
    if sum(x is not None for x in (covariance, samples)) != 1:
        return covariance, samples

    source = jnp.asarray(covariance if covariance is not None else samples)

    if source.ndim != 2 or (covariance is not None and source.shape[0] != source.shape[1]):
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
        param_bounds: jnp.ndarray = None,
        param_names: Sequence[str] = None,
    ):
        """
        Initialise a static principal-axes rotation of the kernel metric.

        Exactly one of ``covariance`` or ``samples`` must be provided, always
        in **physical** parameter coordinates. Internally the inputs are scaled
        to unit-hypercube coordinates using ``param_bounds`` before computing
        the orthogonal rotation matrix.

        Parameters
        ----------
        covariance : jnp.ndarray, optional
            Covariance matrix in physical coordinates used to define the
            static rotation.
        samples : jnp.ndarray, optional
            Samples or chain points in physical coordinates used to estimate
            the covariance matrix (e.g. weighted nested-sampling samples or an
            MCMC chain).
        weights : jnp.ndarray, optional
            Weights associated with the supplied samples.
        log_weights : jnp.ndarray, optional
            Log weights associated with the supplied samples.
        rotation_dims : sequence of int or str, optional
            Dimensions of the kernel metric to rotate. If ``None``, all
            dimensions are rotated. String parameter names require
            ``param_names`` (either here or later via ``bind_space``).
        param_bounds : jnp.ndarray, optional
            Parameter bounds, shape ``(2, D)``, used to convert the physical
            covariance or samples into unit-cube coordinates. If ``None``,
            finalisation is deferred until ``bind_space(param_bounds)`` is
            called (e.g. automatically by ``BOBE``).
        param_names : sequence of str, optional
            Parameter names used to resolve string entries in ``rotation_dims``.
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

        self._raw_rotation_dims = None
        self._pending_inputs = None

        if rotation_dims is not None and any(isinstance(d, str) for d in rotation_dims):
            if param_names is not None:
                resolved = resolve_parameter_dimensions(
                    rotation_dims, param_names, len(param_names), "rotation_dims"
                )
                self.rotation_dims = self._setup_rotation_dims(resolved)
            else:
                self._raw_rotation_dims = list(rotation_dims)
                self.rotation_dims = None
        else:
            self.rotation_dims = self._setup_rotation_dims(rotation_dims)

        self._pending_inputs = {
            "covariance": covariance,
            "samples": samples,
            "weights": weights,
            "log_weights": log_weights,
        }
        self.rotation = None
        if param_bounds is not None:
            self.bind_space(param_bounds, param_names=param_names)

    def _finalize(self, covariance=None, samples=None, weights=None, log_weights=None):
        """Compute the orthogonal rotation matrix from unit-space inputs."""
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

    def bind_space(self, param_bounds, param_names=None):
        """
        Convert physical-space inputs to unit-cube coordinates using
        ``param_bounds`` (and resolve named ``rotation_dims`` via
        ``param_names``) and compute the orthogonal rotation matrix.

        No-op if already finalised.
        """
        if self._pending_inputs is None:
            return

        param_bounds = jnp.asarray(param_bounds)
        ndim = param_bounds.shape[1]

        if self._raw_rotation_dims is not None:
            resolved = resolve_parameter_dimensions(
                self._raw_rotation_dims, param_names, ndim, "rotation_dims"
            )
            self.rotation_dims = self._setup_rotation_dims(resolved)
            self._raw_rotation_dims = None

        inputs = self._pending_inputs
        covariance, samples = prepare_rotation_inputs(
            covariance=inputs["covariance"],
            samples=inputs["samples"],
            param_bounds=param_bounds,
            rotation_dims=self.rotation_dims,
        )

        self._finalize(
            covariance=covariance,
            samples=samples,
            weights=inputs["weights"],
            log_weights=inputs["log_weights"],
        )
        self._pending_inputs = None

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
        if self.rotation is None:
            raise RuntimeError(
                "PrincipalAxesTransform is not bound yet; call bind_space(param_bounds) first."
            )
        if self.rotation_dims is None:
            return x @ self.rotation

        rotated = x[..., self.rotation_dims] @ self.rotation

        return x.at[..., self.rotation_dims].set(rotated)

    def inverse(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.rotation is None:
            raise RuntimeError(
                "PrincipalAxesTransform is not bound yet; call bind_space(param_bounds) first."
            )
        if self.rotation_dims is None:
            return x @ self.rotation.T

        unrotated = x[..., self.rotation_dims] @ self.rotation.T

        return x.at[..., self.rotation_dims].set(unrotated)

    def state_dict(self):
        if self.rotation is None:
            raise RuntimeError(
                "Cannot serialize an unbound PrincipalAxesTransform; call bind_space(param_bounds) first."
            )
        return {
            "type": "PrincipalAxesTransform",
            "rotation": self.rotation,
            "rotation_dims": self.rotation_dims,
        }

    @classmethod
    def from_state_dict(cls, state):
        transform = cls.__new__(cls)
        transform._pending_inputs = None
        transform._raw_rotation_dims = None
        transform.rotation = jnp.asarray(state["rotation"])

        rotation_dims = state.get("rotation_dims")
        transform.rotation_dims = (
            None
            if rotation_dims is None
            else jnp.asarray(rotation_dims, dtype=jnp.int32)
        )

        return transform