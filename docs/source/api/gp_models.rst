Gaussian Process Models
=======================

JaxBo provides several Gaussian Process implementations optimized for different scenarios.

Base GP Class
-------------

.. autoclass:: BOBE.gp.GP
   :members:
   :undoc-members:
   :show-inheritance:

Gaussian Process with Classifier
---------------------------------

For handling constraints and invalid regions.

.. autoclass:: BOBE.clf_gp.ClassifierGP
   :members:
   :undoc-members:
   :show-inheritance:

Kernel Functions
----------------

JaxBo uses object-oriented kernel implementations for GP covariance computation.

.. autoclass:: BOBE.kernels.Kernel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: BOBE.kernels.RBFKernel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: BOBE.kernels.MaternKernel
   :members:
   :undoc-members:
   :show-inheritance:

Classifier Module
-----------------

.. automodule:: BOBE.clf
   :members:
   :undoc-members:
   :show-inheritance:


