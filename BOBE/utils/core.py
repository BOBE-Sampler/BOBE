import os
import numpy as np
from scipy.special import logsumexp, erfc
from scipy.stats import chi2
from .seed import get_numpy_rng


def is_cluster_environment():
    """
    Detect if running in a cluster environment by checking common environment variables.
    Returns True if cluster environment is detected, False otherwise.
    """
    cluster_indicators = [
        'SLURM_JOB_ID',      # SLURM
        'PBS_JOBID',         # PBS/Torque
        'LSB_JOBID',         # LSF
        'SGE_TASK_ID',       # SGE
        'COBALT_JOBID',      # Cobalt
        'MOAB_JOBID',        # Moab
        'OMPI_COMM_WORLD_SIZE',  # MPI environment
        'PMI_RANK',          # Intel MPI
        'HOSTNAME'           # Check if hostname contains cluster-like patterns
    ]

    for var in cluster_indicators:
        if os.getenv(var):
            if var == 'HOSTNAME':
                hostname = os.getenv('HOSTNAME', '').lower()
                cluster_patterns = ['node', 'compute', 'worker', 'cluster', 'hpc']
                if any(pattern in hostname for pattern in cluster_patterns):
                    return True
            else:
                return True

    # Additional check for non-interactive environment
    if not os.isatty(1):  # stdout is not a terminal
        return True

    return False


def renormalise_log_weights(log_weights):
    log_total = logsumexp(log_weights)
    normalized_weights = np.exp(log_weights - log_total)
    return normalized_weights


def resample_equal(samples, aux, weights=None, logwts=None):
    rstate = get_numpy_rng()
    # Resample samples to obtain equal weights. Taken from jaxns
    if logwts is not None:
        wts = renormalise_log_weights(logwts)
    else:
        wts = weights
    weights = wts / wts.sum()
    cumulative_sum = np.cumsum(weights)
    cumulative_sum /= cumulative_sum[-1]
    nsamples = len(weights)
    positions = (rstate.random() + np.arange(nsamples)) / nsamples
    idx = np.zeros(nsamples, dtype=int)
    i, j = 0, 0
    while i < nsamples:
        if positions[i] < cumulative_sum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1
    perm = rstate.permutation(nsamples)
    resampled_samples = samples[idx][perm]
    resampled_aux = aux[idx][perm]
    return resampled_samples, resampled_aux


# ---- Convergence KL ----

def _kl_gaussian_single(mu1, Cov1, mu2, Cov2):
    """
    Computes the KL divergence KL(N1 || N2) between two multivariate Gaussians
    N1 = N(mu1, Cov1) and N2 = N(mu2, Cov2).
    """
    d = mu1.shape[0]

    # Log determinant term
    sign1, logdet_Cov1 = np.linalg.slogdet(Cov1)
    sign2, logdet_Cov2 = np.linalg.slogdet(Cov2)
    if sign1 <= 0 or sign2 <= 0:
        raise np.linalg.LinAlgError("Covariance is not positive definite")
    log_det_term = logdet_Cov2 - logdet_Cov1

    # Trace term
    trace_term = np.trace(np.linalg.solve(Cov2, Cov1))

    # Quadratic term
    diff = mu2 - mu1
    quad_term = np.dot(diff, np.linalg.solve(Cov2, diff))

    # Combine terms
    kl_div = 0.5 * (log_det_term - d + trace_term + quad_term)

    return kl_div


def kl_divergence_gaussian(mu1, Cov1, mu2, Cov2):
    """
    Computes the forward, reverse, and symmetric KL divergence between two
    multivariate Gaussian distributions N1=N(mu1, Cov1) and N2=N(mu2, Cov2).
    """
    kl_forward = _kl_gaussian_single(mu1, Cov1, mu2, Cov2)  # KL(N1 || N2)
    kl_reverse = _kl_gaussian_single(mu2, Cov2, mu1, Cov1)  # KL(N2 || N1)
    kl_symmetric = 0.5 * (kl_forward + kl_reverse)

    return {
        'forward': kl_forward,
        'reverse': kl_reverse,
        'symmetric': kl_symmetric
    }


def resolve_parameter_dimensions(dims, param_names, ndim, name="dimensions"):
    """Resolve parameter names or integer indices to validated indices."""
    if dims is None:
        return None
    names = list(param_names) if param_names is not None else None
    resolved = []
    for dim in dims:
        if isinstance(dim, str):
            if names is None or dim not in names:
                raise ValueError(f"Unknown parameter {dim!r} in {name}")
            dim = names.index(dim)
        dim = int(dim)
        if dim < 0 or dim >= ndim:
            raise ValueError(
                f"{name} contains index {dim} outside the {ndim} dimensions"
            )
        resolved.append(dim)
    return resolved


# ---- Misc ----

def get_threshold_for_nsigma(nsigma, d):
    """
    Difference between peak of Gaussian and logprob level for nsigma (taken from GPry).

    Arguments
    ---------
    nsigma : float
        The number of standard deviations to consider.
    d : int
        The dimensionality of the space.

    Returns
    -------
    float
        The threshold value.
    """
    nstd = np.sqrt(chi2.isf(erfc(nsigma / np.sqrt(2)), d))
    return 0.5 * nstd ** 2


def scale_to_unit(x, param_bounds):
    """
    Project from original domain to unit hypercube, X is N x d shaped, param_bounds are 2 x d
    """
    x = (x - param_bounds[0]) / (param_bounds[1] - param_bounds[0])
    return x


def scale_from_unit(x, param_bounds):
    """
    Project from unit hypercube to original domain, X is N x d shaped, param_bounds are 2 x d
    """
    x = x * (param_bounds[1] - param_bounds[0]) + param_bounds[0]
    return x
