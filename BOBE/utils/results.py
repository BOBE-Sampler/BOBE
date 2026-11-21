"""
Results management system for BOBE sampler.

This module provides comprehensive result storage, unified checkpointing, and
formatting compatible with nested sampling analysis tools such as GetDist.
"""

import json
import os
import pickle
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from getdist import MCSamples
    HAS_GETDIST = True
except ImportError:
    HAS_GETDIST = False
    warnings.warn("GetDist not available. Some functionality will be limited.")

from .log import get_logger

log = get_logger("results")


def convert_jax_to_json_serializable(obj):
    """
    Convert JAX arrays, NumPy scalars/arrays, and nested containers to
    JSON-serializable Python types.
    """
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif hasattr(obj, 'tolist'):  # JAX arrays and numpy arrays
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_jax_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_jax_to_json_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, '__array__'):  # Other array-like objects
        return np.asarray(obj).tolist()
    else:
        return obj


@dataclass
class ConvergenceInfo:
    """Information about convergence checks and nested sampling runs."""
    iteration: int
    logz_dict: Dict[str, float]
    converged: bool
    delta: float
    threshold: float
    dlogz_sampler: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'iteration': self.iteration,
            'logz_dict': self.logz_dict,
            'converged': bool(self.converged),
            'delta': float(self.delta),
            'threshold': float(self.threshold),
            'dlogz_sampler': float(self.dlogz_sampler),
        }


class BOBEResults:
    """
    Comprehensive results management for BOBE runs.

    Handles tracking run diagnostics, writing/loading unified ``.pkl``
    checkpoints, and outputting final summaries and GetDist chain/minimum files.
    """

    def __init__(
        self,
        param_names: List[str],
        param_labels: List[str],
        param_bounds: np.ndarray,
        output_file: str = 'results',
        save_dir: Optional[str] = './',
        settings: Optional[Dict[str, Any]] = None,
        likelihood_name: str = "unknown",
    ):
        """
        Initialize the results manager.

        Args:
            param_names: List of parameter names
            param_labels: List of parameter LaTeX labels
            param_bounds: Parameter bounds array of shape (2, n_params)
            output_file: Base name for output files
            save_dir: Directory where output files are written
            settings: Dictionary of BOBE settings
            likelihood_name: Name of the likelihood function
        """
        self.output_file = output_file or 'results'
        self.save_dir = save_dir or './'
        self.save_path = os.path.join(self.save_dir, self.output_file)
        self.param_names = list(param_names)
        self.param_labels = list(param_labels)
        self.param_bounds = np.array(param_bounds)
        self.ndim = len(self.param_names)
        self.likelihood_name = likelihood_name

        self.settings = dict(settings) if settings else {}

        self._initialize_fresh()
        log.info(f"Initialized BOBE results manager for {self.ndim}D problem")

    def _initialize_fresh(self):
        """Initialize all tracking variables for a fresh run."""
        self.start_time = time.time()
        self.end_time = None
        self.previous_runtime = 0.0

        self.convergence_history: List[ConvergenceInfo] = []
        self.logz_evolution: List[Dict[str, Any]] = []

        self.phase_times = {
            'GP Training': 0.0,
            'Acquisition Optimization': 0.0,
            'True Objective Evaluations': 0.0,
            'Nested Sampling': 0.0,
            'MCMC Sampling': 0.0,
        }
        if self.settings.get('use_clf'):
            self.phase_times['Classifier Training'] = 0.0

        self._active_timers: Dict[str, float] = {}

        # GP hyperparameter tracking
        self.gp_iterations: List[int] = []
        self.gp_lengthscales: List[Any] = []
        self.gp_kernel_variances: List[float] = []
        self.gp_noises: List[float] = []

        # Best loglikelihood tracking
        self.best_loglike_iterations: List[int] = []
        self.best_loglike_values: List[float] = []

        # Acquisition function tracking
        self.acquisition_iterations: List[int] = []
        self.acquisition_values: List[float] = []
        self.acquisition_functions: List[str] = []

        # KL divergence tracking for convergence analysis
        self.kl_iterations: List[int] = []
        self.kl_divergences: List[Dict[str, float]] = []
        self.successive_kl: List[Dict[str, Any]] = []

        # Final results
        self.final_samples = None
        self.final_weights = None
        self.final_loglikes = None
        self.final_logz_dict = None
        self.converged = False
        self.termination_reason = "Unknown"
        self.gp_info: Dict[str, Any] = {}

        # Best point information (for GetDist minimum files)
        self.best_point = None
        self.best_loglike = None
        self.best_iteration = None

    def restore_from_checkpoint(self, data: Dict[str, Any]):
        """Restore tracking state from a checkpoint dict (loaded from ``.pkl``)."""
        self._initialize_fresh()

        if 'convergence_history' in data:
            self.convergence_history = [
                ConvergenceInfo(
                    iteration=conv_dict['iteration'],
                    logz_dict=conv_dict['logz_dict'],
                    converged=conv_dict['converged'],
                    delta=conv_dict['delta'],
                    threshold=conv_dict['threshold'],
                    dlogz_sampler=conv_dict.get('dlogz_sampler', np.nan),
                )
                for conv_dict in data['convergence_history']
            ]

        if 'logz_evolution' in data:
            self.logz_evolution = list(data['logz_evolution'])
        elif 'logz_history' in data:
            self.logz_evolution = list(data['logz_history'])

        acq = data.get('acquisition_data', {})
        self.acquisition_iterations = list(acq.get('iterations', []))
        self.acquisition_values = list(acq.get('values', []))
        self.acquisition_functions = list(acq.get('functions', []))

        gph = data.get('gp_hyperparams', {})
        self.gp_iterations = list(gph.get('iterations', []))
        self.gp_lengthscales = list(gph.get('lengthscales', []))
        self.gp_kernel_variances = list(gph.get('kernel_variances', []))
        self.gp_noises = list(gph.get('noises', []))

        bll = data.get('best_loglike_data', {})
        self.best_loglike_iterations = list(bll.get('iterations', []))
        self.best_loglike_values = list(bll.get('best_loglike', []))

        kl = data.get('kl_data', {})
        self.kl_iterations = list(kl.get('iterations', []))
        self.kl_divergences = list(kl.get('kl_divergences', []))
        self.successive_kl = list(kl.get('successive_kl', []))

        pt = data.get('phase_times', {})
        if not pt and 'timing' in data and isinstance(data['timing'], dict):
            pt = data['timing'].get('phase_times', {})
        for phase, val in pt.items():
            self.phase_times[phase] = float(val)

        self.previous_runtime = float(data.get('previous_runtime', 0.0))
        self.gp_info = dict(data.get('gp_info', {}))
        self.final_samples = data.get('final_samples')
        self.final_weights = data.get('final_weights')
        self.final_loglikes = data.get('final_loglikes')
        self.final_logz_dict = data.get('final_logz_dict')
        self.converged = bool(data.get('converged', False))
        self.termination_reason = str(data.get('termination_reason', 'Unknown'))
        self.best_point = data.get('best_point', data.get('best_pt'))
        self.best_loglike = data.get('best_loglike', data.get('best_f'))
        self.best_iteration = data.get('best_iteration', data.get('best_pt_iteration'))

        log.info(
            f"Restored results state: {len(self.convergence_history)} convergence entries, "
            f"{len(self.acquisition_values)} acq values"
        )

    def save_checkpoint(self, run_state: Dict[str, Any]):
        """Merge BO run state with results tracking state and write a single ``.pkl`` checkpoint."""
        timing = self.get_timing_summary()
        data = {
            'param_names': self.param_names,
            'param_labels': self.param_labels,
            'param_bounds': self.param_bounds,
            'ndim': self.ndim,
            'likelihood_name': self.likelihood_name,
            'output_file': self.output_file,
            'settings': self.settings,
            'run_info': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
                'runtime_hours': timing['total_runtime'] / 3600.0,
                'likelihood_name': self.likelihood_name,
                'output_file': self.output_file,
                'settings': self.settings,
            },
        }
        data.update(self.get_state_dict())
        data.update(run_state)

        pkl_path = self.save_path + '.pkl'
        parent = os.path.dirname(pkl_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = pkl_path + '.tmp'
        with open(tmp_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, pkl_path)
        log.debug(f"Checkpoint written to {pkl_path}")

    def load_checkpoint(self, resume_file: str) -> Optional[Dict[str, Any]]:
        """
        Load a ``.pkl`` checkpoint, restore results tracking state, and return the raw dict
        so that ``BOBE._handle_resume`` can unpack GP/counter fields.
        Returns ``None`` if the file is missing or unreadable.
        """
        pkl_path = resume_file if resume_file.endswith('.pkl') else resume_file + '.pkl'
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            self.restore_from_checkpoint(data)
            log.info(f"Loaded checkpoint from {pkl_path}")
            return data
        except FileNotFoundError:
            log.warning(f"{pkl_path} not found; starting fresh")
            return None
        except Exception as e:
            log.error(f"Failed to load checkpoint {pkl_path}: {e}; starting fresh")
            return None

    def get_last_convergence_state(self) -> Dict[str, Any]:
        """Return the ``converged``/``delta``/``threshold`` fields from the most recent convergence entry."""
        if self.convergence_history:
            last = self.convergence_history[-1]
            return {'converged': last.converged, 'delta': last.delta, 'threshold': last.threshold}
        return {'converged': False, 'delta': None, 'threshold': None}

    def get_state_dict(self) -> Dict[str, Any]:
        """Return all tracking lists/dicts as a serialisable dict for the checkpoint ``.pkl``."""
        timing = self.get_timing_summary()
        return {
            'convergence_history': [c.to_dict() for c in self.convergence_history],
            'logz_evolution': self.logz_evolution,
            'acquisition_data': {
                'iterations': self.acquisition_iterations,
                'values': self.acquisition_values,
                'functions': self.acquisition_functions,
            },
            'gp_hyperparams': {
                'iterations': self.gp_iterations,
                'lengthscales': convert_jax_to_json_serializable(self.gp_lengthscales),
                'kernel_variances': convert_jax_to_json_serializable(self.gp_kernel_variances),
                'noises': convert_jax_to_json_serializable(self.gp_noises),
            },
            'best_loglike_data': {
                'iterations': self.best_loglike_iterations,
                'best_loglike': self.best_loglike_values,
            },
            'kl_data': {
                'iterations': self.kl_iterations,
                'kl_divergences': convert_jax_to_json_serializable(self.kl_divergences),
                'successive_kl': convert_jax_to_json_serializable(self.successive_kl),
            },
            'phase_times': timing['phase_times'],
            'previous_runtime': timing['total_runtime'],
            'gp_info': self.gp_info,
            'final_samples': self.final_samples,
            'final_weights': self.final_weights,
            'final_loglikes': self.final_loglikes,
            'final_logz_dict': self.final_logz_dict,
            'converged': self.converged,
            'termination_reason': self.termination_reason,
            'best_point': self.best_point,
            'best_loglike': self.best_loglike,
            'best_iteration': self.best_iteration,
        }

    def update_acquisition(self, iteration: int, acquisition_value: float, acquisition_function: str):
        """Track acquisition function values throughout iterations."""
        self.acquisition_iterations.append(iteration)
        self.acquisition_values.append(float(acquisition_value))
        self.acquisition_functions.append(acquisition_function)

    def update_gp_hyperparams(
        self,
        iteration: int,
        lengthscales: list,
        kernel_variance: float,
        noise: Optional[float] = None,
    ):
        """Track GP hyperparameter evolution (lengthscales, variance, and optional noise)."""
        self.gp_iterations.append(iteration)
        self.gp_lengthscales.append(lengthscales)
        self.gp_kernel_variances.append(float(kernel_variance))
        if noise is not None:
            self.gp_noises.append(float(noise))

    def update_best_loglike(self, iteration: int, best_loglike: float):
        """Track best loglikelihood evolution."""
        self.best_loglike_iterations.append(iteration)
        self.best_loglike_values.append(best_loglike)

    def update_convergence(
        self,
        iteration: int,
        logz_dict: Dict[str, float],
        converged: bool,
        threshold: float,
        delta: Optional[float] = None,
    ):
        """
        Update convergence information from a nested sampling check.

        Args:
            iteration: Current iteration number
            logz_dict: Dictionary with logz information
            converged: Whether convergence was achieved
            threshold: Convergence threshold used
            delta: Convergence metric value used for the check.
                   Defaults to ``logz_dict['std']`` if not provided.
        """
        if delta is None:
            delta = logz_dict.get('std', np.nan)

        conv_info = ConvergenceInfo(
            iteration=iteration,
            logz_dict=logz_dict.copy(),
            converged=converged,
            delta=delta,
            threshold=threshold,
            dlogz_sampler=logz_dict.get('dlogz_sampler', np.nan),
        )

        self.convergence_history.append(conv_info)

        self.logz_evolution.append({
            'iteration': iteration,
            'logz': logz_dict.get('mean', np.nan),
            'logz_upper': logz_dict.get('upper', np.nan),
            'logz_lower': logz_dict.get('lower', np.nan),
            'logz_err': delta,
            'logz_var': logz_dict.get('var', np.nan),
            'logz_std': logz_dict.get('std', np.nan),
            'dlogz_sampler': logz_dict.get('dlogz_sampler', np.nan),
        })

    def update_kl_divergences(
        self,
        iteration: int,
        successive_kl: Optional[Dict[str, float]] = None,
    ):
        """Update KL divergence tracking for convergence analysis."""
        self.kl_iterations.append(iteration)

        if successive_kl is not None:
            self.successive_kl.append({
                'iteration': iteration,
                **successive_kl,
            })

    def get_last_iteration(self) -> int:
        """Get the last iteration number from the results history, or 0 if empty."""
        if self.convergence_history:
            return self.convergence_history[-1].iteration
        elif self.acquisition_iterations:
            return max(self.acquisition_iterations)
        elif self.gp_iterations:
            return max(self.gp_iterations)
        elif self.best_loglike_iterations:
            return max(self.best_loglike_iterations)
        else:
            return 0

    def start_timing(self, phase_name: str):
        """Start timing a specific phase."""
        if phase_name in self.phase_times:
            self._active_timers[phase_name] = time.time()

    def end_timing(self, phase_name: str):
        """End timing a specific phase and accumulate the time."""
        if phase_name in self._active_timers:
            elapsed = time.time() - self._active_timers[phase_name]
            self.phase_times[phase_name] += elapsed
            del self._active_timers[phase_name]

    def get_timing_summary(self) -> Dict[str, Any]:
        """Get a summary of timing information."""
        current_session_runtime = (self.end_time or time.time()) - self.start_time
        total_runtime = self.previous_runtime + current_session_runtime

        percentages = {}
        if total_runtime > 0:
            for phase, time_spent in self.phase_times.items():
                percentages[phase] = (time_spent / total_runtime) * 100.0

        return {
            'phase_times': self.phase_times.copy(),
            'percentages': percentages,
            'total_runtime': total_runtime,
            'current_session_runtime': current_session_runtime,
            'previous_runtime': self.previous_runtime,
        }

    def get_gp_data(self) -> Dict[str, list]:
        """Get GP hyperparameter evolution data for plotting."""
        return {
            'iterations': self.gp_iterations,
            'lengthscales': convert_jax_to_json_serializable(self.gp_lengthscales),
            'kernel_variances': convert_jax_to_json_serializable(self.gp_kernel_variances),
            'noises': convert_jax_to_json_serializable(self.gp_noises),
        }

    def get_acquisition_data(self) -> Dict[str, list]:
        """Get acquisition function evolution data for plotting."""
        return {
            'iterations': self.acquisition_iterations,
            'values': self.acquisition_values,
            'functions': self.acquisition_functions,
        }

    def get_best_loglike_data(self) -> Dict[str, list]:
        """Get best loglikelihood evolution data for plotting."""
        return {
            'iterations': self.best_loglike_iterations,
            'best_loglike': self.best_loglike_values,
        }

    def finalize(
        self,
        samples_dict: Dict[str, np.ndarray] = {},
        logz_dict: Optional[Dict[str, float]] = None,
        converged: bool = False,
        termination_reason: str = "Max iterations reached",
        gp_info: Optional[Dict[str, Any]] = None,
        best_point: Optional[np.ndarray] = None,
        best_loglike: Optional[float] = None,
        best_iteration: Optional[int] = None,
    ):
        """Finalize the results with final samples and metadata, and write outputs."""
        self.end_time = time.time()

        self.final_samples = samples_dict.get('x', np.array([]))
        self.final_weights = samples_dict.get('weights', np.array([]))
        self.final_loglikes = samples_dict.get('logl', np.array([]))

        if logz_dict is not None:
            self.final_logz_dict = logz_dict
        elif self.convergence_history:
            self.final_logz_dict = self.convergence_history[-1].logz_dict.copy()
        else:
            self.final_logz_dict = {}

        self.converged = converged
        self.termination_reason = termination_reason
        self.gp_info = gp_info or {}

        self.best_point = best_point
        self.best_loglike = best_loglike
        self.best_iteration = best_iteration

        log.info(f"Finalized BOBE results: converged={converged}, reason={termination_reason}")
        if best_point is not None and best_loglike is not None:
            log.info(f"Best point: logL={best_loglike:.6f} at iteration {best_iteration}")

        self.save_chain_files()
        self.save_summary_json()
        self.save_minimum_files()

    def get_results_dict(self) -> Dict[str, Any]:
        """Get results dictionary with samples, evidence evolution, and diagnostics."""
        if self.final_samples is None:
            raise ValueError("Results not finalized. Call finalize() first.")

        if len(self.final_weights) > 0:
            n_effective = int(np.sum(self.final_weights) ** 2 / np.sum(self.final_weights ** 2))
        else:
            n_effective = 0

        timing_summary = self.get_timing_summary()
        runtime = timing_summary['total_runtime']

        return {
            'samples': self.final_samples,
            'weights': self.final_weights,
            'logl': self.final_loglikes,
            'logwt': np.log(self.final_weights + 1e-300) if len(self.final_weights) > 0 else np.array([]),
            'logz': self.final_logz_dict.get('mean', np.nan),
            'logzerr': self.final_logz_dict.get(
                'std',
                self.final_logz_dict.get('upper', 0) - self.final_logz_dict.get('lower', 0),
            ),
            'dlogz_sampler': float(self.final_logz_dict.get('dlogz_sampler', np.nan)),
            'final_logz_dict': self.final_logz_dict.copy(),
            'logz_history': self.logz_evolution,
            'param_names': self.param_names,
            'param_labels': self.param_labels,
            'param_bounds': self.param_bounds,
            'ndim': self.ndim,
            'n_samples': len(self.final_samples),
            'n_effective': n_effective,
            'converged': self.converged,
            'termination_reason': self.termination_reason,
            'convergence_history': [conv.to_dict() for conv in self.convergence_history],
            'gp_info': self.gp_info,
            'acquisition_data': {
                'iterations': self.acquisition_iterations,
                'values': self.acquisition_values,
                'functions': self.acquisition_functions,
            },
            'gp_hyperparams': {
                'iterations': self.gp_iterations,
                'lengthscales': self.gp_lengthscales,
                'kernel_variances': self.gp_kernel_variances,
                'noises': self.gp_noises,
            },
            'best_loglike_data': {
                'iterations': self.best_loglike_iterations,
                'best_loglike': self.best_loglike_values,
            },
            'kl_data': {
                'iterations': self.kl_iterations,
                'kl_divergences': self.kl_divergences,
                'successive_kl': self.successive_kl,
            },
            'timing': timing_summary,
            'run_info': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
                'runtime_hours': runtime / 3600.0,
                'likelihood_name': self.likelihood_name,
                'output_file': self.output_file,
                'settings': self.settings,
            },
        }

    def save_summary_json(self):
        """Save a human-readable summary JSON at the end of the run."""
        timing = self.get_timing_summary()

        param_stats = {}
        if self.final_samples is not None and len(self.final_samples) > 0:
            for i, name in enumerate(self.param_names):
                values = self.final_samples[:, i]
                weights = self.final_weights
                mean = np.average(values, weights=weights)
                std = np.sqrt(np.average((values - mean) ** 2, weights=weights))
                sorted_idx = np.argsort(values)
                cumsum = np.cumsum(weights[sorted_idx]) / weights.sum()

                def wp(p, _ci=cumsum, _vs=values[sorted_idx]):
                    idx = min(np.searchsorted(_ci, p / 100.0), len(_vs) - 1)
                    return float(_vs[idx])

                param_stats[name] = {
                    'mean': float(mean),
                    'std': float(std),
                    '16%': wp(16),
                    '84%': wp(84),
                    'median': wp(50),
                }

        logz = self.final_logz_dict or {}
        summary = {
            'run_info': {
                'likelihood_name': self.likelihood_name,
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': (
                    datetime.fromtimestamp(self.end_time).isoformat()
                    if self.end_time else None
                ),
                'runtime_hours': timing['total_runtime'] / 3600.0,
            },
            'convergence': {
                'converged': bool(self.converged),
                'termination_reason': str(self.termination_reason),
                'n_iterations': len(self.acquisition_iterations),
            },
            'evidence': {
                'logz': float(logz.get('mean', float('nan'))),
                'logz_err': float(logz.get('std', float('nan'))),
                'logz_lower': float(logz.get('lower', float('nan'))),
                'logz_upper': float(logz.get('upper', float('nan'))),
                'dlogz_sampler': float(logz.get('dlogz_sampler', float('nan'))),
            },
            'timing': timing['phase_times'],
            'gp_info': convert_jax_to_json_serializable(self.gp_info),
            'parameters': param_stats,
        }

        summary_file = f"{self.save_path}_summary.json"
        parent = os.path.dirname(summary_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        log.info(f"Saved summary to {summary_file}")

    def save_chain_files(
        self,
        samples_dict: Optional[Dict[str, np.ndarray]] = None,
        filename: Optional[str] = None,
    ):
        """Save chain files in GetDist format using ``MCSamples.saveAsText``."""
        if not HAS_GETDIST:
            log.warning("GetDist not available, cannot save chain files")
            return

        getdist_samples = self.get_getdist_samples(samples_dict)
        if getdist_samples is None:
            log.warning("Could not create MCSamples object")
            return

        if filename is not None:
            output_file = os.path.join(self.save_dir, filename)
        else:
            output_file = self.save_path

        getdist_samples.saveAsText(root=output_file, make_dirs=True)
        log.info(f"Saved GetDist format files to {output_file}")
        log.info("Created: .txt (chain), .paramnames (parameter info), .ranges (parameter bounds)")

    def save_minimum_files(self):
        """
        Save best point in GetDist minimum format (``.minimum.txt`` and ``.minimum``).
        """
        if self.best_point is None or self.best_loglike is None:
            log.debug("No best point data available, skipping minimum files")
            return

        best_point = np.atleast_1d(self.best_point)

        if len(best_point) != self.ndim:
            log.warning(f"Best point dimension {len(best_point)} != {self.ndim}, skipping minimum files")
            return

        minuslogpost = -self.best_loglike
        chi_sq = 2.0 * minuslogpost

        minimum_txt_file = f"{self.save_path}.minimum.txt"
        try:
            with open(minimum_txt_file, 'w') as f:
                header = "#        weight    minuslogpost"
                for param_name in self.param_names:
                    header += f"  {param_name:>13s}"
                f.write(header + "\n")

                line = f"              1  {minuslogpost:13.7f}"
                for val in best_point:
                    line += f"  {val:13.8e}"
                f.write(line + "\n")

            log.info(f"Saved minimum table to {minimum_txt_file}")
        except Exception as e:
            log.warning(f"Failed to save .minimum.txt file: {e}")

        minimum_file = f"{self.save_path}.minimum"
        try:
            with open(minimum_file, 'w') as f:
                f.write(f" -log(Like) = {minuslogpost:.12f}\n")
                f.write(f"  chi-sq    = {chi_sq:.12f}\n")
                f.write("\n")

                for i, (param_name, param_label, val) in enumerate(
                    zip(self.param_names, self.param_labels, best_point), start=1
                ):
                    f.write(f"{i:>5d}  {val:.9e}   {param_name:40s}  {param_label}\n")

            log.info(f"Saved minimum point details to {minimum_file}")
        except Exception as e:
            log.warning(f"Failed to save .minimum file: {e}")

    def get_getdist_samples(self, samples_dict=None) -> Optional['MCSamples']:
        """Convert results to a GetDist ``MCSamples`` object."""
        if not HAS_GETDIST:
            log.warning("GetDist not available, cannot create MCSamples object")
            return None

        if samples_dict is not None:
            samples = samples_dict['x']
            weights = samples_dict['weights']
            loglikes = samples_dict['logl']
            sampler_method = samples_dict.get('method', 'mcmc')
        else:
            if self.final_samples is None:
                log.warning("No final samples available")
                return None
            samples = self.final_samples
            weights = self.final_weights
            loglikes = self.final_loglikes
            sampler_method = 'nested' if self.final_logz_dict else 'mcmc'

        if len(samples) == 0:
            log.warning("Samples array is empty, cannot create MCSamples object")
            return None

        ranges = {
            name: [self.param_bounds[0, i], self.param_bounds[1, i]]
            for i, name in enumerate(self.param_names)
        }

        return MCSamples(
            samples=samples,
            names=self.param_names,
            labels=self.param_labels,
            ranges=ranges,
            weights=weights,
            loglikes=loglikes,
            label='BOBE',
            sampler=sampler_method,
        )

    @classmethod
    def load_results(cls, output_file: str) -> 'BOBEResults':
        """
        Load results from a saved unified ``.pkl`` checkpoint file.

        Args:
            output_file: Path to the checkpoint file (with or without ``.pkl`` suffix).

        Returns:
            ``BOBEResults`` instance populated with the saved run data.
        """
        pkl_path = output_file if output_file.endswith('.pkl') else f"{output_file}.pkl"
        if not Path(pkl_path).exists():
            raise FileNotFoundError(f"Results file not found: {pkl_path}")

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        run_info = data.get('run_info', {})
        base_path = pkl_path[:-4] if pkl_path.endswith('.pkl') else pkl_path
        save_dir = os.path.dirname(base_path) or './'
        base_name = os.path.basename(base_path)

        results = cls(
            param_names=data['param_names'],
            param_labels=data['param_labels'],
            param_bounds=data['param_bounds'],
            output_file=data.get('output_file', base_name),
            save_dir=save_dir,
            settings=data.get('settings', run_info.get('settings', {})),
            likelihood_name=data.get('likelihood_name', run_info.get('likelihood_name', 'unknown')),
        )

        results.restore_from_checkpoint(data)

        # Preserve original start/end timestamps without double-counting previous_runtime
        start_str = run_info.get('start_time')
        end_str = run_info.get('end_time')
        saved_total_runtime = float(data.get('previous_runtime', 0.0))
        if start_str:
            results.start_time = datetime.fromisoformat(start_str).timestamp()
        if end_str:
            results.end_time = datetime.fromisoformat(end_str).timestamp()
            session_dt = max(0.0, results.end_time - results.start_time)
            results.previous_runtime = max(0.0, saved_total_runtime - session_dt)

        log.info(f"Loaded complete results from {pkl_path}")
        return results
