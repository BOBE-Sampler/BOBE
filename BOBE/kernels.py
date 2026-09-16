"""
Kernel implementations for Gaussian Process models.

All kernels inherit from the base Kernel class and implement the covariance() method.
JAX JIT compilation is handled at higher levels (acquisition functions, optimization).
"""

from abc import ABC, abstractmethod
from math import sqrt
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Constants for Matérn kernel
sqrt2 = sqrt(2.)
sqrt3 = sqrt(3.)
sqrt5 = sqrt(5.)


class Kernel(ABC):
    """
    Abstract base class for all kernels in BOBE.
    
    Attributes
    ----------
    lengthscales : jnp.ndarray
        Lengthscale parameters for each dimension, shape (D,)
    kernel_variance : float
        Overall variance/amplitude of the kernel
    noise : float
        Observation noise level
    """
    
    def __init__(self, lengthscales, kernel_variance, noise=1e-8):
        """
        Initialize kernel with hyperparameters.
        
        Parameters
        ----------
        lengthscales : jnp.ndarray
            Lengthscale for each input dimension
        kernel_variance : float
            Kernel variance/amplitude parameter
        noise : float, optional
            Noise level added to diagonal. Default is 1e-8.
        """
        self.lengthscales = jnp.array(lengthscales)
        self.kernel_variance = kernel_variance
        self.noise = noise
        self.input_transform = None

    def set_input_transform(self, transform):
        self.input_transform = transform

    def _transform_inputs(self, x):
        if self.input_transform is None:
            return x
        return self.input_transform.forward(x)
    
    def sq_dist(self, xa, xb):
        """
        Compute squared Euclidean distance between two sets of points.
        
        This utility method is used by many kernel implementations.
        
        Parameters
        ----------
        xa : jnp.ndarray
            First set of points, shape (n1, D)
        xb : jnp.ndarray
            Second set of points, shape (n2, D)
            
        Returns
        -------
        sq_dist : jnp.ndarray
            Squared distances, shape (n1, n2)
        """
        return jnp.sum(jnp.square(xa[:, None, :] - xb[None, :, :]), axis=-1)
    
    @abstractmethod
    def covariance(self, xa, xb, include_noise=True):
        """
        Compute covariance matrix between two sets of points.
        
        Parameters
        ----------
        xa : jnp.ndarray
            First set of points, shape (n1, D)
        xb : jnp.ndarray
            Second set of points, shape (n2, D)
        include_noise : bool, optional
            Whether to add noise to diagonal (only when xa is xb). Default is True.
            
        Returns
        -------
        K : jnp.ndarray
            Covariance matrix of shape (n1, n2)
        """
        pass
    
    def diagonal(self, x, include_noise=True):
        """
        Compute only the diagonal of the kernel matrix K(x,x).
        
        For stationary kernels, the diagonal is constant: kernel_variance (+ noise).
        Override this method if your kernel has a non-constant diagonal.
        
        Parameters
        ----------
        x : jnp.ndarray
            Points at which to compute diagonal, shape (n, D)
        include_noise : bool, optional
            Whether to include noise in diagonal. Default is True.
            
        Returns
        -------
        diag : jnp.ndarray
            Diagonal values, shape (n,)
        """
        diag = self.kernel_variance * jnp.ones(x.shape[0])
        if include_noise:
            diag += self.noise
        return diag
    
    def update_hyperparams(self, lengthscales=None, kernel_variance=None, noise=None):
        """
        Update kernel hyperparameters.
        
        Parameters
        ----------
        lengthscales : jnp.ndarray, optional
            New lengthscale values
        kernel_variance : float, optional
            New kernel variance
        noise : float, optional
            New noise level
        """
        if lengthscales is not None:
            self.lengthscales = jnp.array(lengthscales)
        if kernel_variance is not None:
            self.kernel_variance = kernel_variance
        if noise is not None:
            self.noise = noise
    
    def __call__(self, xa, xb, include_noise=True):
        """Convenience method - same as covariance()"""
        return self.covariance(xa, xb, include_noise=include_noise)


class RBFKernel(Kernel):
    """
    Radial Basis Function (RBF) / Squared Exponential kernel.
    
    k(x, x') = σ² * exp(-0.5 * ||x - x'||²/ℓ²)
    
    where σ² is kernel_variance and ℓ is lengthscale.
    """
    
    def covariance(self, xa, xb, include_noise=True):
        """
        Compute RBF covariance matrix.
        
        Parameters
        ----------
        xa : jnp.ndarray
            First set of input points, shape (n1, d).
        xb : jnp.ndarray
            Second set of input points, shape (n2, d).
        include_noise : bool, optional
            Whether to include noise on diagonal. Default is True.
            
        Returns
        -------
        jnp.ndarray
            Kernel matrix of shape (n1, n2).
        """

        xa = self._transform_inputs(xa)
        xb = self._transform_inputs(xb)

        # Scale inputs by lengthscales
        xa_scaled = xa / self.lengthscales
        xb_scaled = xb / self.lengthscales
        
        # Compute squared distances
        sq_dist = self.sq_dist(xa_scaled, xb_scaled)
        
        # Apply RBF kernel
        K = self.kernel_variance * jnp.exp(-0.5 * sq_dist)
        
        # Add noise to diagonal if needed
        if include_noise:
            K += self.noise * jnp.eye(K.shape[0])
        
        return K


class MaternKernel(Kernel):
    """
    Matérn-5/2 kernel.
    
    k(x, x') = σ² * (1 + √5*d + 5*d²/3) * exp(-√5*d)
    
    where d = ||x - x'||/ℓ, σ² is kernel_variance, and ℓ is lengthscale.
    """
    
    def covariance(self, xa, xb, include_noise=True):
        """
        Compute Matérn-5/2 covariance matrix.
        
        Parameters
        ----------
        xa : jnp.ndarray
            First set of input points, shape (n1, d).
        xb : jnp.ndarray
            Second set of input points, shape (n2, d).
        include_noise : bool, optional
            Whether to include noise on diagonal. Default is True.
            
        Returns
        -------
        jnp.ndarray
            Kernel matrix of shape (n1, n2).
        """
        xa = self._transform_inputs(xa)
        xb = self._transform_inputs(xb)

        # Scale inputs by lengthscales
        xa_scaled = xa / self.lengthscales
        xb_scaled = xb / self.lengthscales
        
        # Compute squared distances
        dsq = self.sq_dist(xa_scaled, xb_scaled)
        
        # Safe sqrt to avoid division by zero
        d = jnp.sqrt(jnp.where(dsq < 1e-30, 1e-30, dsq))
        
        # Matérn-5/2 formula
        exp_term = jnp.exp(-sqrt5 * d)
        poly_term = 1. + d * (sqrt5 + d * 5. / 3.)
        K = self.kernel_variance * poly_term * exp_term
        
        # Add noise to diagonal if needed
        if include_noise:
            K += self.noise * jnp.eye(K.shape[0])
        
        return K
