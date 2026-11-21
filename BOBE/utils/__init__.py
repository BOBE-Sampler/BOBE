"""
Utilities package for BOBE.

This package contains utility modules for results management, plotting, 
logging, timing, and other helper functions.
"""

# Import key utilities for easy access
from .results import BOBEResults, load_bobe_results, create_resumable_results
from .plot import BOBESummaryPlotter, plot_final_samples
from .core import (
    scale_to_unit, 
    scale_from_unit, 
    renormalise_log_weights, 
    resample_equal,
    is_cluster_environment
)
from .log import get_logger, setup_logging
from .seed import get_numpy_rng, get_jax_key, set_global_seed

__all__ = [
    # Results management
    'BOBEResults', 'load_bobe_results',
    # Plotting
    'BOBESummaryPlotter', 'plot_final_samples', 
    # Core utilities
    'scale_to_unit', 'scale_from_unit',
    'renormalise_log_weights', 'resample_equal', 'is_cluster_environment',
    # Logging
    'get_logger', 'setup_logging',
    # Random number generation
    'get_numpy_rng', 'get_jax_key', 'set_global_seed',
]
