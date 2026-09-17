from math import sqrt,pi
from typing import Any,List
import jax.numpy as jnp
import numpy as np
import jax
from jax.scipy.linalg import cho_solve, solve_triangular
jax.config.update("jax_enable_x64", True)
from functools import partial
from .utils.log import get_logger
log = get_logger("gp")
from .optim import optimize_optax, optimize_scipy
from .utils.seed import get_new_jax_key, get_numpy_rng
import numpyro.distributions as dist
from .kernels import Kernel, RBFKernel, MaternKernel
from .transforms import PrincipalAxesTransform


safe_noise_floor = 1e-12

# Constants for DSLP prior
sqrt2 = sqrt(2.)
sqrt3 = sqrt(3.)

class DummyDistribution:
    """A dummy distribution that always returns log_prob = 0.0"""
    def log_prob(self, x):
        return 0.0

def make_distribution(spec: dict) -> dist.Distribution:
    """
    Turn a dictionary specification into a NumPyro distribution.
    
    Parameters
    ----------
    spec : dict
        Dictionary with 'name' key for distribution type and additional
        keyword arguments for the distribution parameters.
        
    Returns
    -------
    dist.Distribution
        NumPyro distribution object.
        
    Examples
    --------
    >>> spec = {"name": "Normal", "loc": 0.0, "scale": 1.0}
    >>> dist = make_distribution(spec)
    """
    # Ensure distribution exists
    dist_class = getattr(dist, spec["name"], None)
    if dist_class is None:
        raise ValueError(f"Distribution {spec['name']} not found in numpyro.distributions.")
    
    # Remove "name"
    kwargs = {k: v for k, v in spec.items() if k != "name"}
    return dist_class(**kwargs)

def saas_prior_logprob(lengthscales, kernel_variance, tausq):
    """
    Compute SAAS prior log probability.
    
    Parameters
    ----------
    lengthscales : jnp.ndarray
        Lengthscale parameters.
    kernel_variance : float
        Kernel variance parameter.
    tausq : float
        SAAS tausq parameter.
        
    Returns
    -------
    float
        Log probability under SAAS priors.
    """
    logprior = dist.LogNormal(0., 1.).log_prob(kernel_variance)
    logprior += dist.HalfCauchy(0.1).log_prob(tausq)
    inv_lengthscales_sq = 1 / (tausq * lengthscales**2)
    logprior += jnp.sum(dist.HalfCauchy(1.).log_prob(inv_lengthscales_sq))
    return logprior

@jax.jit
def gp_mll(k,train_y,num_points):
    """
    Computes the negative marginal log likelihood of the GP
    """
    L = jnp.linalg.cholesky(k)
    alpha = cho_solve((L,True),train_y)
    mll = -0.5*jnp.einsum("ij,ji",train_y.T,alpha) - jnp.sum(jnp.log(jnp.diag(L))) - 0.5*num_points*jnp.log(2*pi)
    return mll


@jax.jit
def fast_update_cholesky(L: jnp.ndarray, k: jnp.ndarray, k_self: float):
    # solve L v = k  -> v has shape (n,)
    v = solve_triangular(L, k, lower=True)

    # new diagonal entry
    diag = jnp.sqrt(k_self - jnp.dot(v, v))

    # print(f"Shapes L: {L.shape}, k: {k.shape}, k_self: {k_self}, v: {v.shape}, diag: {diag.shape}")

    # build a zero (n+1)x(n+1) and fill blocks
    n = L.shape[0]
    new_L = jnp.zeros((n+1, n+1), dtype=L.dtype)
    new_L = new_L.at[:n, :n].set(L)      # top-left
    new_L = new_L.at[n, :n].set(v)       # bottom-left
    new_L = new_L.at[n, n].set(diag)     # bottom-right
    return new_L

class GP:
    
    def __init__(self,train_x,train_y,noise=1e-8,kernel="rbf",optimizer="scipy",optimizer_options={},
                 kernel_variance_bounds = [1e-4, 1e8],lengthscale_bounds = [0.01,5],lengthscales=None,kernel_variance=None,
                 kernel_variance_prior=None, lengthscale_prior=None, tausq=None, tausq_bounds=[1e-4,1e4], rotation_covariance=None, rotation_samples=None, rotation_weights=None, rotation_log_weights=None, rotation_dims=None, param_names: List[str] = None):
        """
        Initialize the Gaussian Process model.

        Parameters
        ----------
        train_x : jnp.ndarray
            Training inputs, shape (N, D).
        train_y : jnp.ndarray
            Objective function values at training points, shape (N, 1).
        noise : float, optional
            Noise parameter added to the diagonal of the kernel. Default is 1e-8.
        kernel : str, optional
            Kernel to use, either "rbf" or "matern". Default is "rbf".
        optimizer : str, optional
            Optimizer to use for hyperparameter tuning. Default is "scipy".
        optimizer_options : dict, optional
            Keyword arguments for the optimizer. Default is {}.
        kernel_variance_bounds : list, optional
            Bounds for the kernel variance. Default is [1e-4, 1e8].
        lengthscale_bounds : list, optional
            Bounds for the lengthscales. Default is [0.01, 10].
        lengthscales : jnp.ndarray, optional
            Initial lengthscale values. If None, defaults to ones. Default is None.
        kernel_variance : float, optional
            Initial kernel variance. If None, defaults to 1.0. Default is None.
        kernel_variance_prior : dict or str, optional
            Specification for the kernel variance prior. 
            If None, defaults to `{'name': 'LogNormal', 'loc': 0.0, 'scale': 1.0}`.
            If 'fixed', the kernel variance will be fixed to the initial value and not optimized.
            Defaults to None.
        lengthscale_prior : str or dict, optional
            Specification for the lengthscale prior. 
            If 'DSLP' or None, uses the DSLP prior. 
            If 'SAAS', uses the SAAS prior with tausq parameter.
            Otherwise, uses the provided distribution spec. Defaults to None.
        tausq : float, optional
            Initial tausq parameter for SAAS prior. Only used when lengthscale_prior='SAAS'. 
            If None, defaults to 1.0. Defaults to None.
        tausq_bounds : list, optional
            Bounds for the tausq parameter (in log space). Only used when lengthscale_prior='SAAS'.
            Defaults to [-4, 4].
        rotation_covariance : jnp.ndarray, optional
            Covariance matrix in GP unit-hypercube coordinates used to define a static rotation of the kernel metric.
        rotation_samples : jnp.ndarray, optional
            Samples or chain points in GP unit-hypercube coordinates used to estimate the covariance matrix for defining the static rotation.
        rotation_weights : jnp.ndarray, optional
            Weights associated with rotation_samples.
        rotation_log_weights : jnp.ndarray, optional
            Log-weights associated with rotation_samples, such as nested-sampling log weights.
        rotation_dims : tuple of int, optional
            Dimensions of the kernel metric to rotate. If None, all dimensions are rotated.
        """
        # Setup and validate training data
        self._setup_training_data(train_x, train_y)
        self.param_names = param_names if param_names is not None else ['x_'+str(i) for i in range(self.ndim)]

        # Setup kernel and initial hyperparameters
        self.kernel_name = kernel if kernel == "rbf" else "matern"
        self.lengthscales = lengthscales if lengthscales is not None else jnp.ones(self.ndim)
        self.kernel_variance = kernel_variance if kernel_variance is not None else 1.0
        self.noise = noise
        
        # Instantiate kernel object
        kernel_classes = {"rbf": RBFKernel, "matern": MaternKernel}
        self.kernel = kernel_classes[self.kernel_name](self.lengthscales, self.kernel_variance, self.noise)

        # Rotation transform for the kernel metric
        rotation_requested = any(
            value is not None
            for value in (
                rotation_covariance,
                rotation_samples,
                rotation_weights,
                rotation_log_weights,
                rotation_dims,
            )
        )

        if rotation_requested:
            if rotation_dims is not None and len(rotation_dims) > 0:
                if max(rotation_dims) >= self.ndim:
                    raise ValueError("rotation_dims contains indices outside the GP input dimensions")

            transform = PrincipalAxesTransform(
                covariance=rotation_covariance,
                samples=rotation_samples,
                weights=rotation_weights,
                log_weights=rotation_log_weights,
                rotation_dims=rotation_dims
            ) 

            expected_ndim = (
                self.ndim
                if rotation_dims is None
                else len(rotation_dims)
            )

            if transform.rotation.shape != (expected_ndim, expected_ndim):
                raise ValueError(
                    f"Rotation dimensions are incompatible with the GP input dimensions. Expected rotation shape {(expected_ndim, expected_ndim)}, got {transform.rotation.shape}."
                )

            self.kernel.set_input_transform(transform)
        
        # Compute initial kernel matrices
        K = self.kernel.covariance(self.train_x, self.train_x, include_noise=True)
        self.cholesky = jnp.linalg.cholesky(K)
        self.alphas = cho_solve((self.cholesky, True), self.train_y)

        # Setup optimizer
        self.optimizer_method = optimizer
        if optimizer == "scipy":
            self.mll_optimize = optimize_scipy
        else:
            self.mll_optimize = optimize_optax
        self.optimizer_options = optimizer_options
    

        # Store bounds
        self.lengthscale_bounds = lengthscale_bounds
        self.kernel_variance_bounds = kernel_variance_bounds
        # Can store tausq for convenience even though it is only used for SAAS
        self.tausq = tausq if tausq is not None else 1.0
        self.tausq_bounds = tausq_bounds

        # Setup priors and optimization parameters
        self._setup_kernel_variance_prior(kernel_variance_prior)
        self._setup_lengthscale_prior(lengthscale_prior)
        self._setup_optimization_parameters()

    def _setup_training_data(self, train_x, train_y):
        """Setup and validate training data, compute standardization parameters."""
        # Check x and y sizes
        if train_x.shape[0] != train_y.shape[0]:
            raise ValueError("train_x and train_y must have the same number of points")
        if train_y.ndim != 2:
            train_y = train_y.reshape(-1, 1)
        if train_x.ndim != 2:
            raise ValueError("train_x must be 2D")

        self.ndim = train_x.shape[1]
        
        # Compute standardization parameters (and handle the case of 0 initialisation points)
        self.y_mean = jnp.mean(train_y) if train_y.size > 0 else 0 
        self.y_std = jnp.std(train_y) if train_y.size > 0 else 1.0
        
        # Handle edge case where std is zero (all values identical or only 1 point)
        if self.y_std == 0:
            log.warning("Training targets have zero variance. Setting std to 1.0 to avoid division by zero.")
            self.y_std = 1.0

        # Store standardized training data
        self.train_x = jnp.array(train_x)
        self.train_y = (train_y - self.y_mean) / self.y_std
        log.debug(f"GP training size = {self.train_x.shape[0]}")

    def _setup_kernel_variance_prior(self, kernel_variance_prior):
        """Setup kernel variance prior and determine if it should be fixed."""
        self.kernel_variance_prior_spec = kernel_variance_prior
        if self.kernel_variance_prior_spec is None:
            self.kernel_variance_prior_spec = {'name': 'Uniform', 'low': self.kernel_variance_bounds[0], 'high': self.kernel_variance_bounds[1]}
        
        # Check if kernel variance should be fixed
        self.fixed_kernel_variance = (self.kernel_variance_prior_spec == 'fixed')
        if not self.fixed_kernel_variance:
            self.kernel_variance_prior_dist = make_distribution(self.kernel_variance_prior_spec)
        else:
            self.kernel_variance_prior_dist = DummyDistribution()

    def _setup_lengthscale_prior(self, lengthscale_prior):
        """Setup lengthscale prior and determine prior function."""
        self.lengthscale_prior_spec = lengthscale_prior
        if self.lengthscale_prior_spec is None:
            self.lengthscale_prior_spec = {'name': 'Uniform', 'low': self.lengthscale_bounds[0], 'high': self.lengthscale_bounds[1]}

        # Set up lengthscale priors and prior function
        if self.lengthscale_prior_spec == 'DSLP':
            self.lengthscale_prior_dist = dist.LogNormal(loc=sqrt2 + 0.5*jnp.log(self.ndim), scale=sqrt3)
            self.prior_func = self._standard_prior_logprob
        elif self.lengthscale_prior_spec == 'SAAS':
            self.lengthscale_prior_dist = None
            self.prior_func = self._saas_prior_logprob  
        else:
            self.lengthscale_prior_dist = make_distribution(self.lengthscale_prior_spec)
            self.prior_func = self._standard_prior_logprob

    def _setup_optimization_parameters(self):
        """Setup parameter names and bounds for optimization."""
        # Build parameter names and bounds based on what's being optimized
        self.hyperparam_names = ['lengthscales']
        self.hyperparam_bounds = [self.lengthscale_bounds] * self.ndim
        
        if not self.fixed_kernel_variance:
            self.hyperparam_names.append('kernel_variance')
            self.hyperparam_bounds.append(self.kernel_variance_bounds)
            
        if self.lengthscale_prior_spec == 'SAAS':
            self.hyperparam_names.append('tausq')
            self.hyperparam_bounds.append(self.tausq_bounds)

        self.hyperparam_bounds = jnp.log(jnp.array(self.hyperparam_bounds).T)
        self.num_hyperparams = self.hyperparam_bounds.shape[1]
        log.debug(f" Hyperparameter bounds =  {self.hyperparam_bounds}")

    def _standard_prior_logprob(self, lengthscales, kernel_variance, tausq=None):
        """Standard prior log probability for DSLP and custom priors."""
        logprior = self.kernel_variance_prior_dist.log_prob(kernel_variance)
        if self.lengthscale_prior_dist is not None:
            logprior += self.lengthscale_prior_dist.log_prob(lengthscales).sum()
        return logprior
    
    def _saas_prior_logprob(self, lengthscales, kernel_variance, tausq):
        """SAAS prior log probability."""
        return saas_prior_logprob(lengthscales, kernel_variance, tausq)
    
    def _parse_hyperparams(self, log_params):
        """Parse log parameters into lengthscales, kernel_variance, and optionally tausq."""
        hyperparams = jnp.exp(log_params)
        lengthscales = hyperparams[:self.ndim]
        
        if self.fixed_kernel_variance:
            kernel_variance = self.kernel_variance  # Use fixed value
            if 'tausq' in self.hyperparam_names:
                tausq = hyperparams[self.ndim] if len(hyperparams) > self.ndim else self.tausq
            else:
                tausq = self.tausq
        else:
            kernel_variance = hyperparams[self.ndim]
            tausq = hyperparams[self.ndim + 1] if len(hyperparams) > self.ndim + 1 else self.tausq
            
        return lengthscales, kernel_variance, tausq

    def neg_mll(self, log_params):
        """
        Computes the negative log marginal likelihood for the GP with given hyperparameters.
        """
        lengthscales, kernel_variance, tausq = self._parse_hyperparams(log_params)
        
        # Update kernel hyperparameters and compute kernel matrix
        self.kernel.update_hyperparams(lengthscales=lengthscales, kernel_variance=kernel_variance)
        K = self.kernel.covariance(self.train_x, self.train_x, include_noise=True)
        mll = gp_mll(K, self.train_y, self.train_y.shape[0])
        
        # Add prior
        mll += self.prior_func(lengthscales, kernel_variance, tausq)
        
        return -mll

    def fit(self, x0: np.ndarray = None, maxiter: int = 500) -> dict:
        """
        Performs a serial fit for a given batch of starting points (x0).
        This method is called by each MPI process on its assigned chunk.

        Arguments
        ---------
        x0 : np.ndarray
            Array of shape (n_restarts_chunk, n_params) containing starting points for optimization (in log space).
        maxiter : int
            Maximum number of iterations for the optimizer. Defaults to 500.

        Returns
        -------
        result : dict
            Dictionary containing the best 'mll' and corresponding 'params' (log space) found.
        """

        if x0 is None: # set to current hyperparameters
            x0 = jnp.log(self.get_hyperparams())[None, :]

        optimizer_options = self.optimizer_options.copy()

        best_params_log, best_loss = self.mll_optimize(
            fun=self.neg_mll,
            num_params=self.num_hyperparams,
            bounds=self.hyperparam_bounds,
            x0=x0, # Use the chunk of starting points passed in
            maxiter=maxiter,
            n_restarts=x0.shape[0], # The number of restarts is the size of the chunk
            optimizer_options=optimizer_options
        )
                
        # Return the result in the format the pool expects
        return {
            'mll': -best_loss,
            'params': best_params_log # Optionally return the raw params
        }

    def update_hyperparams(self, hyperparams):
        """
        Update the GP hyperparameters and recompute the Cholesky and alphas.
        """
        lengthscales, kernel_variance, tausq = self._parse_hyperparams(hyperparams)
        self.lengthscales = lengthscales
        if not self.fixed_kernel_variance:
            self.kernel_variance = kernel_variance
        self.tausq = tausq
        # Update kernel object
        self.kernel.update_hyperparams(lengthscales=self.lengthscales, kernel_variance=self.kernel_variance)
        self.recompute_cholesky()
    
    def predict_mean_single(self,x):
        """
        Single point prediction of mean
        """
        x = jnp.atleast_2d(x)
        k12 = self.kernel.covariance(self.train_x, x, include_noise=False) # shape (N,1)
        mean = jnp.einsum('ij,ji', k12.T, self.alphas)*self.y_std + self.y_mean
        return mean 
    
    def predict_var_single(self,x):
        x = jnp.atleast_2d(x)
        k12 = self.kernel.covariance(self.train_x, x, include_noise=False) # shape (N,1)
        vv = solve_triangular(self.cholesky, k12, lower=True) # shape (N,1)
        k22 = self.kernel.diagonal(x, include_noise=True) # shape (1,) for x (1,ndim)
        var = k22 - jnp.sum(vv*vv,axis=0) 
        var = jnp.clip(var, safe_noise_floor, None)
        return self.y_std**2 * var.squeeze()
    
    def predict_mean_batched(self,x):
        x = jnp.atleast_2d(x)
        return jax.vmap(self.predict_mean_single, in_axes=0)(x)
    
    def predict_var_batched(self,x):
        x = jnp.atleast_2d(x)
        return jax.vmap(self.predict_var_single, in_axes=0)(x)

    def predict_single(self,x):
        """
        Predicts the mean and variance of the GP at x but does not unstandardize it. To use with EI and the like.
        """
        x = jnp.atleast_2d(x)
        k12 = self.kernel.covariance(self.train_x, x, include_noise=False)
        k22 = self.kernel.diagonal(x, include_noise=True)
        mean = jnp.einsum('ij,ji', k12.T, self.alphas)
        vv = solve_triangular(self.cholesky, k12, lower=True) # shape (N,1)
        var = k22 - jnp.sum(vv*vv,axis=0) 
        # handle nans and negative variances due to numerical issues
        var = jnp.where(jnp.isnan(var),safe_noise_floor,var)
        var = jnp.where(var<safe_noise_floor,safe_noise_floor,var)
        return mean, var
    
    def predict_batched(self,x):
        x = jnp.atleast_2d(x)
        return jax.vmap(self.predict_single, in_axes=0,out_axes=(0,0))(x)

    def update(self,new_x,new_y):
        """
        Updates the GP with new training points and refits the GP if refit is True.

        Arguments
        ---------        
        refit: bool
            Whether to refit the GP hyperparameters. Default is True.
        maxiter: int
            The maximum number of iterations for the optax optimizer. Default is 200.
        n_restarts: int
            The number of restarts for the optax optimizer. Default is 4.
        """
        new_x = jnp.atleast_2d(new_x)
        new_y = jnp.atleast_2d(new_y)

        duplicate = False
        new_pts_to_add = []
        new_vals_to_add = []
        
        # Check for duplicates and collect new points
        for i in range(new_x.shape[0]):
            if jnp.any(jnp.all(jnp.isclose(self.train_x, new_x[i], atol=1e-6, rtol=1e-4), axis=1)):
                log.debug(f"Point {new_x[i]} already exists in the training set, not updating")
            else:
                new_pts_to_add.append(new_x[i])
                new_vals_to_add.append(new_y[i])

        # Add new points if any
        if new_pts_to_add:
            new_pts_to_add = jnp.array(new_pts_to_add)
            new_vals_to_add = jnp.array(new_vals_to_add)
            
            # Add to training data
            self.train_x = jnp.vstack([self.train_x, new_pts_to_add])
            train_y_original = jnp.vstack([self.train_y * self.y_std + self.y_mean, new_vals_to_add])
            
            self.y_mean = jnp.mean(train_y_original)
            self.y_std = jnp.std(train_y_original)
            
            if self.y_std == 0:
                log.warning("Training targets have zero variance. Setting std to 1.0 to avoid division by zero.")
                self.y_std = 1.0
            
            self.train_y = (train_y_original - self.y_mean) / self.y_std

            self.recompute_cholesky()


    def recompute_cholesky(self):
        """
        Recomputes the Cholesky decomposition and alphas. Useful if hyperparameters are changed manually.
        """
        K = self.kernel.covariance(self.train_x, self.train_x, include_noise=True)
        self.cholesky = jnp.linalg.cholesky(K)
        self.alphas = cho_solve((self.cholesky, True), self.train_y)

    def fantasy_var(self,new_x,mc_points,k_train_mc):
        """
        Computes the variance of the GP at the mc_points assuming a single point new_x is added to the training set
        """

        new_x = jnp.atleast_2d(new_x)
        # new_train_x = jnp.concatenate([self.train_x,new_x])
        k = self.kernel.covariance(self.train_x, new_x, include_noise=False).flatten()           # shape (n,)
        k_self = self.kernel.diagonal(new_x, include_noise=True)[0]  # scalar
        k11_cho = fast_update_cholesky(self.cholesky,k,k_self)

        # Compute only the extra row for new_x
        k_new_mc = self.kernel.covariance(new_x, mc_points, include_noise=False)  # shape (1, n_mc)
        k12 = jnp.vstack([k_train_mc,k_new_mc])
        k22 = self.kernel.diagonal(mc_points, include_noise=True) # (N_mc,)
        vv = solve_triangular(k11_cho, k12, lower=True) # shape (N_train,N_mc)
        var = k22 - jnp.sum(vv*vv,axis=0) 
        # handle nans and negative variances due to numerical issues
        var = jnp.where(jnp.isnan(var),safe_noise_floor,var)
        var = jnp.where(var<safe_noise_floor,safe_noise_floor,var)
        return var * self.y_std**2 # return to physical scale for better interpretability

    def get_random_point(self,rng=None,nstd=None):
        """
        Returns a random point in the unit cube.
        """
        log.debug(f"Getting random point in unit cube")
        rng = rng if rng is not None else get_numpy_rng()
        pt = rng.uniform(0, 1, size=self.train_x.shape[1])
        return pt

    def state_dict(self):
        """
        Returns a dictionary containing the complete state of the GP.
        This can be used for saving, loading, or copying the GP.
        
        Returns
        -------
        state: dict
            Dictionary containing all necessary information to reconstruct the GP
        """
        state = {
            # Training data (original, unstandardized)
            'train_x': np.array(self.train_x),
            'train_y': np.array(self.train_y * self.y_std + self.y_mean),  # unstandardize
            
            # Hyperparameters
            'lengthscales': np.array(self.lengthscales),
            'kernel_variance': float(self.kernel_variance),
            'noise': float(self.noise),
            'tausq': float(self.tausq),
            
            # Standardization parameters
            'y_mean': float(self.y_mean),
            'y_std': float(self.y_std),
            
            # Model configuration
            'kernel_name': self.kernel_name,
            'lengthscale_prior_spec': self.lengthscale_prior_spec,
            'kernel_variance_prior_spec': self.kernel_variance_prior_spec,
            'fixed_kernel_variance': self.fixed_kernel_variance,
            'optimizer_method': self.optimizer_method,
            'optimizer_options': self.optimizer_options,
            
            # Bounds
            'lengthscale_bounds': self.lengthscale_bounds,
            'kernel_variance_bounds': self.kernel_variance_bounds,
            'tausq_bounds': self.tausq_bounds,

            # Input transform
            'input_transform_state': (
                self.kernel.input_transform.state_dict()
                if self.kernel.input_transform is not None
                else None
            ),
            
            # Computed state
            'cholesky': np.array(self.cholesky) if hasattr(self, 'cholesky') else None,
            'alphas': np.array(self.alphas) if hasattr(self, 'alphas') else None,
            
            # Dimensions
            'ndim': self.ndim,
            
            # Class identifier
            'gp_class': 'GP'
        }
        
        return state
    
    @classmethod
    def from_state_dict(cls, state):
        """
        Creates a GP instance from a state dictionary.
        
        Arguments
        ---------
        state: dict
            State dictionary returned by state_dict()
            
        Returns
        -------
        gp: GP
            The reconstructed GP object
        """
        # Create GP instance
        gp = cls(
            train_x=state['train_x'],
            train_y=state['train_y'],
            noise=state['noise'],
            kernel=state['kernel_name'],
            optimizer=state['optimizer_method'],
            optimizer_options=state['optimizer_options'],
            lengthscales=state['lengthscales'],
            kernel_variance=state['kernel_variance'],
            lengthscale_bounds=state['lengthscale_bounds'],
            kernel_variance_bounds=state['kernel_variance_bounds'],
            kernel_variance_prior=state.get('kernel_variance_prior_spec'),
            lengthscale_prior=state.get('lengthscale_prior_spec'),
            tausq=state.get('tausq', 1.0),
            tausq_bounds=state.get('tausq_bounds', [-4, 4])
        )

        transform_state = state.get('input_transform_state')
        if transform_state is not None:
            if transform_state['type'] != 'PrincipalAxesTransform':
                raise ValueError(f"Unknown input transform {transform_state['type']}")

            transform = PrincipalAxesTransform.from_state_dict(transform_state)
            gp.kernel.set_input_transform(transform)
            
        
        # Restore computed state if available
        if state['cholesky'] is not None and state['alphas'] is not None:
            gp.cholesky = jnp.array(state['cholesky'])
            gp.alphas = jnp.array(state['alphas'])
        elif gp.kernel.input_transform is not None:
            gp.recompute_cholesky()
        # if state['cholesky'] is not None:
        #     gp.cholesky = jnp.array(state['cholesky'])
        # if state['alphas'] is not None:
        #     gp.alphas = jnp.array(state['alphas'])
        
        return gp
    
    @classmethod
    def load(cls, filename, **kwargs):
        """
        Loads a GP from a file
        
        Arguments
        ---------
        filename: str
            The name of the file to load the GP from (with or without .npz extension)
        **kwargs: 
            Additional keyword arguments to pass to the GP constructor
            
        Returns
        -------
        gp: GP
            The loaded GP object
        """
        if not filename.endswith('.npz'):
            filename += '.npz'
            
        try:
            data = np.load(filename, allow_pickle=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file {filename}")
        
        # Convert arrays back to the expected format
        state = {}
        for key in data.files:
            value = data[key]
            if isinstance(value, np.ndarray) and value.shape == ():
                # Handle scalar arrays
                state[key] = value.item()
            else:
                state[key] = value
        
        # Apply any override kwargs
        state.update(kwargs)
        
        # Use from_state_dict for loading
        gp = cls.from_state_dict(state)
        
        log.info(f"Loaded GP from {filename} with {gp.train_x.shape[0]} training points")
        return gp

    def save(self, filename='gp'):
        """
        Save the GP state to a file using state_dict.
        
        Arguments
        ---------
        filename: str
            The filename to save to (with or without .npz extension). Default is 'gp'.
        """
        if not filename.endswith('.npz'):
            filename += '.npz'
        
        state = self.state_dict()
        np.savez(filename, **state)
        log.info(f"Saved GP state to {filename}")


    def copy(self):
        """
        Creates a deep copy of the GP using state_dict.
        
        Returns
        -------
        gp_copy: GP
            A deep copy of the current GP
        """
        state = self.state_dict()
        return self.__class__.from_state_dict(state)
    
    @property
    def npoints(self):
        return self.train_x.shape[0]
    
    def get_hyperparams(self):
        hp = self.lengthscales
        if not self.fixed_kernel_variance:
            hp = jnp.hstack([hp, self.kernel_variance])
        if self.lengthscale_prior_spec == 'SAAS':
            hp = jnp.hstack([hp, self.tausq])
        return hp
    
    def hyperparams_dict(self):
        ls_str = {name: f"{float(val):.4f}" for name, val in zip(self.param_names, self.lengthscales)}
        param_dict = {
            'lengthscales': ls_str,
            'kernel_variance': f"{float(self.kernel_variance):.4f}",
        }
        if 'tausq' in self.hyperparam_names:
            param_dict['tausq'] = f"{float(self.tausq):.4f}"
        return param_dict