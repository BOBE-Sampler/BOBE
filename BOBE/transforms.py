from abc import ABC, abstractmethod
from typing import Sequence

import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

from .utils.core import (
    kl_divergence_gaussian,
    principal_axes,
    regularize_covariance,
    resolve_parameter_dimensions,
    weighted_mean_and_covariance,
)


def scale_covariance_to_unit(covariance, param_bounds, dims=None):
    """Select (if full) and scale a physical-space covariance into unit-cube units."""
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix must be square")

    bounds = np.asarray(param_bounds, dtype=float)
    ndim = bounds.shape[1]
    if dims is None:
        if cov.shape[0] != ndim:
            raise ValueError(
                f"Covariance shape {cov.shape} is incompatible with the GP input dimensions (ndim={ndim})"
            )
        widths = bounds[1] - bounds[0]
    else:
        idx = np.asarray(dims, dtype=int)
        if int(idx.max()) >= ndim:
            raise ValueError("rotation_dims contains indices outside the GP input dimensions")
        if cov.shape[0] == ndim and len(idx) != ndim:
            cov = cov[np.ix_(idx, idx)]
        elif cov.shape[0] != len(idx):
            raise ValueError("rotation_dims are incompatible with the supplied matrix")
        widths = bounds[1, idx] - bounds[0, idx]

    return cov / np.outer(widths, widths)


def _to_int_array(value):
    return None if value is None else jnp.asarray(value, dtype=jnp.int32)


def _to_np_array(value, dtype=None):
    return None if value is None else np.asarray(value, dtype=dtype)


class InputTransform(ABC):
    """Base interface for kernel input transforms."""

    name: str = "base"

    @abstractmethod
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Transform coordinates used to evaluate the kernel."""

    @abstractmethod
    def inverse(self, x: jnp.ndarray) -> jnp.ndarray:
        """Invert the kernel coordinate transform."""


class PrincipalAxesTransform(InputTransform):
    """Static or adaptive principal-axes rotation of the kernel metric.

    By default ``max_updates=0`` (static rotation from a physical-space
    ``covariance``). When ``max_updates > 0``, the orthogonal rotation adapts
    during optimisation from unit-cube posterior MC samples via ``try_update``.
    An initial ``covariance`` may cover only ``initial_dims`` (a subset of
    ``update_dims``), in which case its eigensystem is embedded in an identity
    rotation over ``update_dims`` until the first accepted update.
    """

    name: str = "principal_axes"

    def __init__(
        self,
        covariance=None,
        rotation_dims: Sequence = None,
        initial_dims: Sequence = None,
        update_dims: Sequence = None,
        param_bounds=None,
        param_names: Sequence[str] = None,
        max_updates: int = 0,
        acq_threshold: float = None,
        update_step: int = 10,
        kl_threshold: float = 0.1,
        regularize_eps: float = 1e-10,
    ):
        self.max_updates = int(max_updates)
        if self.max_updates < 0:
            raise ValueError("max_updates must be non-negative")
        self.acq_threshold = None if acq_threshold is None else float(acq_threshold)
        self.update_step = None if update_step is None else int(update_step)
        if self.update_step is not None and self.update_step < 1:
            raise ValueError("update_step must be positive or None")
        self.kl_threshold = float(kl_threshold)
        if self.kl_threshold < 0:
            raise ValueError("kl_threshold must be non-negative")
        self.regularize_eps = float(regularize_eps)
        if self.regularize_eps < 0:
            raise ValueError("regularize_eps must be non-negative")

        if self.max_updates == 0 and covariance is None:
            raise ValueError(
                "Provide covariance for a static transform, or set max_updates > 0 for adaptive rotation"
            )
        if rotation_dims is not None and update_dims is not None:
            raise ValueError("Specify either rotation_dims or update_dims, not both")

        self._update_dims_arg_name = "update_dims" if update_dims is not None else "rotation_dims"
        raw_update = update_dims if update_dims is not None else rotation_dims

        self.update_dims, self._raw_update_dims = self._parse_dims(
            raw_update, param_names, self._update_dims_arg_name
        )
        self.initial_dims, self._raw_initial_dims = self._parse_dims(
            initial_dims, param_names, "initial_dims"
        )
        self.rotation_dims = self.update_dims
        if self.initial_dims is not None and self.update_dims is not None:
            self._check_subset(self.initial_dims, self.update_dims)

        self._pending_covariance = covariance
        self.rotation = None
        self.reference_mean = None
        self.reference_covariance = None
        self.update_count = 0
        self.last_update_ii = None
        self.last_update_acq_val = None
        self.last_update_kl = None

        if param_bounds is not None:
            self.initialize(param_bounds=param_bounds, param_names=param_names)
        elif (
            self._pending_covariance is None
            and self._raw_update_dims is None
            and self._raw_initial_dims is None
            and self.update_dims is not None
        ):
            if self.initial_dims is None:
                self.initial_dims = self.update_dims
            self.rotation = jnp.eye(len(self.update_dims), dtype=float)

    @property
    def adaptive(self) -> bool:
        return self.max_updates > 0

    @classmethod
    def _parse_dims(cls, dims, param_names, name):
        if dims is None:
            return None, None
        dims_list = list(dims)
        if not dims_list:
            raise ValueError(f"{name} cannot be empty")
        if any(isinstance(d, str) for d in dims_list):
            if param_names is not None:
                resolved = resolve_parameter_dimensions(
                    dims_list, param_names, len(param_names), name
                )
                return cls._validate_int_dims(resolved, name), None
            if len(set(dims_list)) != len(dims_list):
                raise ValueError(f"{name} cannot contain duplicate indices")
            return None, dims_list
        return cls._validate_int_dims(dims_list, name), None

    @staticmethod
    def _validate_int_dims(dims, name):
        dims = tuple(int(i) for i in dims)
        if not dims:
            raise ValueError(f"{name} cannot be empty")
        if min(dims) < 0:
            raise ValueError(f"{name} must contain non-negative indices")
        if len(set(dims)) != len(dims):
            raise ValueError(f"{name} cannot contain duplicate indices")
        return jnp.asarray(dims, dtype=jnp.int32)

    @staticmethod
    def _check_subset(initial_dims, update_dims):
        if not {int(i) for i in initial_dims}.issubset({int(i) for i in update_dims}):
            raise ValueError("initial_dims must be a subset of update_dims")

    @staticmethod
    def _check_range(dims, ndim, name):
        if dims is not None and len(dims) > 0 and int(jnp.max(dims)) >= ndim:
            raise ValueError(f"{name} contains indices outside the GP input dimensions")

    def initialize(self, ndim=None, param_bounds=None, param_names=None):
        """Resolve dimensionality/named dims, scale any physical covariance, and build R."""
        if param_bounds is not None:
            bounds_ndim = int(np.asarray(param_bounds).shape[1])
            if ndim is not None and int(ndim) != bounds_ndim:
                raise ValueError(
                    f"ndim={ndim} does not match param_bounds dimensionality ({bounds_ndim})"
                )
            ndim = bounds_ndim
        elif ndim is not None:
            ndim = int(ndim)
        else:
            raise ValueError("Provide ndim or param_bounds to initialize PrincipalAxesTransform")

        if (
            self.rotation is not None
            and self._pending_covariance is None
            and self._raw_update_dims is None
            and self._raw_initial_dims is None
        ):
            self._check_range(self.rotation_dims, ndim, self._update_dims_arg_name)
            return

        if self._pending_covariance is not None and param_bounds is None:
            return
        if (
            (self._raw_update_dims is not None or self._raw_initial_dims is not None)
            and param_names is None
            and param_bounds is None
        ):
            return

        if self._raw_update_dims is not None:
            resolved = resolve_parameter_dimensions(
                self._raw_update_dims, param_names, ndim, self._update_dims_arg_name
            )
            self.update_dims = self._validate_int_dims(resolved, self._update_dims_arg_name)
            self.rotation_dims = self.update_dims
            self._raw_update_dims = None
        if self._raw_initial_dims is not None:
            resolved = resolve_parameter_dimensions(
                self._raw_initial_dims, param_names, ndim, "initial_dims"
            )
            self.initial_dims = self._validate_int_dims(resolved, "initial_dims")
            self._raw_initial_dims = None

        self._check_range(self.update_dims, ndim, self._update_dims_arg_name)
        self._check_range(self.initial_dims, ndim, "initial_dims")

        if self.update_dims is None:
            if self.adaptive:
                self.update_dims = jnp.arange(ndim, dtype=jnp.int32)
                self.rotation_dims = self.update_dims
            elif self.initial_dims is not None:
                self.update_dims = self.initial_dims
                self.rotation_dims = self.update_dims

        if self.initial_dims is not None and self.update_dims is not None:
            self._check_subset(self.initial_dims, self.update_dims)

        if self._pending_covariance is not None:
            init_dims = self.initial_dims if self.initial_dims is not None else self.update_dims
            unit_cov = scale_covariance_to_unit(
                self._pending_covariance, param_bounds, dims=init_dims
            )
            self._set_initial_rotation(unit_cov, init_dims)
            if self.initial_dims is None and init_dims is not None:
                self.initial_dims = init_dims
            self._pending_covariance = None
        else:
            if self.initial_dims is None:
                self.initial_dims = self.update_dims
            self.rotation = jnp.eye(len(self.update_dims), dtype=float)

    def _set_initial_rotation(self, unit_covariance, init_dims):
        cov = regularize_covariance(unit_covariance, name="Kernel rotation covariance")
        _, eigenvectors = principal_axes(cov)
        if self.update_dims is not None and init_dims is not None:
            update_tuple = tuple(int(i) for i in self.update_dims)
            init_tuple = tuple(int(i) for i in init_dims)
            if update_tuple != init_tuple:
                rotation = np.eye(len(update_tuple), dtype=float)
                slots = np.array([update_tuple.index(i) for i in init_tuple], dtype=int)
                rotation[np.ix_(slots, slots)] = eigenvectors
                self.rotation = jnp.asarray(rotation)
                return
        self.rotation = jnp.asarray(eigenvectors)

    def _posterior_samples(self, mc_samples):
        samples = np.asarray(mc_samples["x"], dtype=float)
        dims = (
            np.arange(samples.shape[1], dtype=int)
            if self.update_dims is None
            else np.asarray(self.update_dims, dtype=int)
        )
        samples = samples[:, dims]
        valid = np.all(np.isfinite(samples), axis=1)
        weights = mc_samples.get("weights")
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if weights.shape != (len(valid),):
                raise ValueError("MC sample weights do not match the samples")
            valid &= np.isfinite(weights) & (weights >= 0.0)
            weights = weights[valid]
        samples = samples[valid]
        if len(samples) < len(dims) + 2:
            raise ValueError(
                f"Need at least {len(dims) + 2} finite posterior samples; got {len(samples)}"
            )
        return samples, weights

    @staticmethod
    def _align_axes(old_rotation, new_rotation):
        """Match and sign-align new axes to the currently installed axes."""
        overlap = np.abs(old_rotation.T @ new_rotation)
        rows, cols = linear_sum_assignment(-overlap)
        assignment = np.empty(len(rows), dtype=int)
        assignment[rows] = cols
        aligned = new_rotation[:, assignment]
        signs = np.sign(np.sum(old_rotation * aligned, axis=0))
        signs[signs == 0.0] = 1.0
        return aligned * signs

    def try_update(self, mc_samples, acq_val, step, acq_threshold=None):
        """Attempt an acquisition- and KL-gated posterior rotation update."""
        if not self.adaptive or self.update_count >= self.max_updates:
            return False
        if self.rotation is None:
            raise RuntimeError(
                "PrincipalAxesTransform is not initialized yet; call initialize(...) first."
            )
        threshold = self.acq_threshold if acq_threshold is None else acq_threshold
        if threshold is None:
            raise ValueError("Adaptive kernel rotation requires acq_threshold")
        if acq_val > threshold:
            return False

        is_first = self.update_count == 0
        if not is_first:
            if self.update_step is None or (step - self.last_update_ii) < self.update_step:
                return False
            if acq_val >= self.last_update_acq_val:
                return False

        samples, weights = self._posterior_samples(mc_samples)
        mean, covariance = weighted_mean_and_covariance(samples, weights)
        covariance = regularize_covariance(
            covariance,
            samples.shape[1],
            regularize_eps=self.regularize_eps,
            name="Posterior covariance",
        )
        kl = None
        if not is_first:
            kl = float(
                kl_divergence_gaussian(
                    self.reference_mean, self.reference_covariance, mean, covariance
                )["symmetric"]
            )
            if kl < self.kl_threshold:
                return False

        _, new_rotation = principal_axes(covariance)
        new_rotation = self._align_axes(np.asarray(self.rotation), new_rotation)
        self.rotation = jnp.asarray(new_rotation)
        self.reference_mean = mean
        self.reference_covariance = covariance
        self.update_count += 1
        self.last_update_ii = int(step)
        self.last_update_acq_val = float(acq_val)
        self.last_update_kl = kl
        return True

    def _apply(self, x: jnp.ndarray, transpose: bool = False) -> jnp.ndarray:
        if self.rotation is None:
            raise RuntimeError(
                "PrincipalAxesTransform is not initialized yet; call initialize(...) first."
            )
        matrix = self.rotation.T if transpose else self.rotation
        x = jnp.asarray(x)
        if self.rotation_dims is None:
            return x @ matrix
        return x.at[..., self.rotation_dims].set(x[..., self.rotation_dims] @ matrix)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._apply(x, transpose=False)

    def inverse(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._apply(x, transpose=True)

    def state_dict(self):
        if self.rotation is None:
            raise RuntimeError(
                "Cannot serialize an uninitialized PrincipalAxesTransform; call initialize(...) first."
            )
        return {
            "type": "PrincipalAxesTransform",
            "rotation": np.asarray(self.rotation),
            "rotation_dims": _to_np_array(self.rotation_dims, dtype=np.int32),
            "initial_dims": _to_np_array(self.initial_dims, dtype=np.int32),
            "update_dims": _to_np_array(self.update_dims, dtype=np.int32),
            "acq_threshold": self.acq_threshold,
            "update_step": self.update_step,
            "max_updates": self.max_updates,
            "kl_threshold": self.kl_threshold,
            "regularize_eps": self.regularize_eps,
            "reference_mean": _to_np_array(self.reference_mean),
            "reference_covariance": _to_np_array(self.reference_covariance),
            "update_count": self.update_count,
            "last_update_ii": self.last_update_ii,
            "last_update_acq_val": self.last_update_acq_val,
            "last_update_kl": self.last_update_kl,
        }

    @classmethod
    def from_state_dict(cls, state):
        transform = cls.__new__(cls)
        transform._pending_covariance = None
        transform._raw_update_dims = None
        transform._raw_initial_dims = None
        transform._update_dims_arg_name = "rotation_dims"
        transform.rotation = jnp.asarray(state["rotation"])

        rotation_dims = state.get("rotation_dims")
        transform.rotation_dims = _to_int_array(rotation_dims)
        transform.update_dims = _to_int_array(state.get("update_dims", rotation_dims))
        transform.initial_dims = _to_int_array(state.get("initial_dims", rotation_dims))
        transform.acq_threshold = state.get("acq_threshold")
        transform.update_step = state.get("update_step", 10)
        transform.max_updates = int(state.get("max_updates", 0))
        transform.kl_threshold = float(state.get("kl_threshold", 0.1))
        transform.regularize_eps = float(state.get("regularize_eps", 1e-10))
        transform.reference_mean = _to_np_array(state.get("reference_mean"))
        transform.reference_covariance = _to_np_array(state.get("reference_covariance"))
        transform.update_count = int(state.get("update_count", 0))
        transform.last_update_ii = state.get("last_update_ii")
        transform.last_update_acq_val = state.get("last_update_acq_val")
        transform.last_update_kl = state.get("last_update_kl")
        return transform
