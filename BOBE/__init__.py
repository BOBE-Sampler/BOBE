"""
BOBE - Bayesian Optimization for Expensive Likelihoods using JAX

BOBE is a package for performing Bayesian model comparison for expensive 
likelihood functions, developed for applications to cosmology. It uses 
Bayesian Optimization to train Gaussian process surrogates and runs 
nested sampling/MCMC on the surrogate instead of the underlying expensive 
likelihood.

Main Components:
- BOBE: Main Bayesian Optimization class (accepts raw callable or Likelihood)
- GP: Gaussian Process implementation
- GPwithClassifier: GP with SVM/NN/Ellipsoid classifier for filtering
- Likelihood classes: Likelihood, CobayaLikelihood (optional, wrapping handled automatically)
- Acquisition functions: EI, LogEI, WIPV, WIPStd

Quick Start:
    >>> from BOBE import BOBE
    >>> import numpy as np
    >>> 
    >>> def my_loglike(x):
    >>>     return -np.sum(x**2)
    >>> 
    >>> bobe = BOBE(
    >>>     loglikelihood=my_loglike,
    >>>     param_list=['x', 'y'],
    >>>     param_bounds=np.array([[-5, 5], [-5, 5]]).T,
    >>>     max_evals=100,
    >>> )
    >>> results = bobe.run(['wipv'])

MPI parallelization and logging are handled transparently - no explicit setup needed.
"""

__version__ = "0.1.0"

# Enable 64-bit floating-point precision in JAX once at package import
import jax
jax.config.update("jax_enable_x64", True)

# Initialize logging before importing any modules that create loggers
from .utils.log import setup_logging
setup_logging(verbosity='INFO')  # Default verbosity, can be overridden in BOBE.__init__

# Core classes
from .bo import BOBE, get_dimension_based_defaults
from .gp import GP
from .clf_gp import GPwithClassifier
from .kernels import Kernel, RBFKernel, MaternKernel
from .transforms import PrincipalAxesTransform
from .surrogate import Surrogate, ClassifierMixin

# Likelihood interfaces
from .likelihood import Likelihood

try:
    from .likelihood import CobayaLikelihood
    _COBAYA_AVAILABLE = True
except ImportError:
    _COBAYA_AVAILABLE = False
    CobayaLikelihood = None

# Acquisition functions
from .acquisition import EI, LogEI, WIPV, WIPStd

# Utilities (commonly used)
from .utils import (
    BOBEResults,
    BOBESummaryPlotter,
    get_logger,
    setup_logging,
    scale_to_unit,
    scale_from_unit,
)

# Build __all__ dynamically based on available optional dependencies
__all__ = [
    # Version
    "__version__",
    # Core classes
    "BOBE",
    "GP",
    "GPwithClassifier",
    "Kernel",
    "RBFKernel",
    "MaternKernel",
    "PrincipalAxesTransform",
    "Surrogate",
    "ClassifierMixin",
    # Utilities
    "get_dimension_based_defaults",
    # Likelihood classes
    "Likelihood",
    # Acquisition
    "EI",
    "LogEI",
    "WIPV",
    "WIPStd",
    # Utilities
    "BOBEResults",
    "BOBESummaryPlotter",
    "get_logger",
    "setup_logging",
    "scale_to_unit",
    "scale_from_unit",
]

# Add optional dependencies to __all__ if available
if _COBAYA_AVAILABLE:
    __all__.append("CobayaLikelihood")
