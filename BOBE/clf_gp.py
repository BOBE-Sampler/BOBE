"""
Classifier-GP module combining Gaussian Process regression with a classifier for selective training.
"""

import time
from typing import List
import numpy as np
import jax
import jax.numpy as jnp
from .gp import GP, safe_noise_floor
from .kernels import Kernel
from .surrogate import ClassifierMixin
from .clf import CLASSIFIER_REGISTRY
from .utils.seed import get_numpy_rng
from .utils.log import get_logger
from .utils.core import get_threshold_for_nsigma
log = get_logger("clf_gp")


class GPwithClassifier(GP, ClassifierMixin):
    def __init__(
        self,
        kernel: Kernel,
        train_x=None,
        train_y=None,
        clf_type: str = 'svm',
        clf_settings: dict = None,
        clf_use_size: int = 10,
        clf_update_step: int = 1,
        probability_threshold: float = 0.5,
        minus_inf: float = -1e5,
        clf_threshold: float = 100.,
        gp_threshold: float = 150.,
        optimizer: str = "scipy",
        optimizer_options: dict = None,
        param_names: List[str] = None,
        train_clf_on_init: bool = True,
    ):
        """
        Generic Classifier-GP class combining a GP with a classifier. The GP is trained on the data points
        that are within the GP threshold of the maximum value of the GP.

        Kernel hyperparameters, priors, bounds, observation noise, and any
        kernel-metric transform are configured directly on the ``kernel``
        instance (e.g. ``RBFKernel(...)`` or ``MaternKernel(...)``).

        Parameters
        ----------
        kernel : Kernel
            Kernel instance to use.
        train_x : array-like, shape (n_samples, n_dim), optional
            Initial training points in unit-cube coordinates. If ``None``,
            dimension-dependent setup is deferred until ``initialize(ndim)``
            or the first call to ``update()``.
        train_y : array-like, shape (n_samples,), optional
            Initial training values.
        clf_type : str, optional
            Type of classifier ('svm', 'nn', 'ellipsoid', etc.). Default is 'svm'.
        clf_settings : dict, optional
            Parameters specific to the chosen classifier.
        clf_use_size : int, optional
            Minimum number of points to start using the classifier. Default is 10.
        clf_update_step : int, optional
            Update classifier every `clf_update_step` points after `clf_use_size` is reached. Default is 1.
        probability_threshold : float, optional
            Threshold for classifier probability/score to consider a point feasible. Default is 0.5.
        minus_inf : float, optional
            Value used for infeasible predictions. Default is -1e5.
        clf_threshold : float, optional
            Threshold for classifier training labels. Default is 100.
        gp_threshold : float, optional
            Threshold for adding points to the GP training set. Default is 150.
        optimizer : str, optional
            Optimizer for GP hyperparameter tuning ('scipy' or 'optax'). Default is 'scipy'.
        optimizer_options : dict, optional
            Keyword arguments forwarded to the GP hyperparameter optimizer.
        param_names : list of str, optional
            Parameter names for logging hyperparameters.
        train_clf_on_init : bool, optional
            Whether to train the classifier during ``__init__`` when enough
            data are present. Default is ``True``.
        """
        if not isinstance(kernel, Kernel):
            raise TypeError(
                f"kernel must be a Kernel instance, got {type(kernel).__name__}"
            )

        # Store Classifier Settings
        self.clf_use_size = clf_use_size
        self.clf_update_step = clf_update_step
        self.clf_type = clf_type.lower()
        self.clf_settings = {} if clf_settings is None else dict(clf_settings)
        self.clf_params = None
        self.clf_metrics = {}
        self.probability_threshold = probability_threshold
        self.minus_inf = minus_inf

        # Store classifier functions and settings
        if self.clf_type not in CLASSIFIER_REGISTRY:
            raise ValueError(f"Unsupported classifier type: {self.clf_type}")

        self.clf_train_fn = CLASSIFIER_REGISTRY[self.clf_type]['train_fn']
        self.clf_predict_fn = CLASSIFIER_REGISTRY[self.clf_type]['predict_fn']

        # Handle Thresholds
        self.clf_threshold = clf_threshold
        self.gp_threshold = gp_threshold

        # Store and validate classifier data, then prepare GP subset
        if train_x is not None:
            if train_y is None:
                raise ValueError("train_y must be provided when train_x is provided")
            train_x_arr = jnp.asarray(train_x)
            train_y_arr = jnp.asarray(train_y)
            if train_x_arr.ndim != 2:
                raise ValueError("train_x must be 2D")
            if train_x_arr.shape[0] != train_y_arr.shape[0]:
                raise ValueError("train_x and train_y must have the same number of points")
            if train_y_arr.ndim != 2:
                train_y_arr = train_y_arr.reshape(-1, 1)

            self.train_x_clf = train_x_arr
            self.train_y_clf = train_y_arr

            if self.train_y_clf.size > 0:
                mask_gp = self.train_y_clf.flatten() > (self.train_y_clf.max() - self.gp_threshold)
                train_x_gp = self.train_x_clf[mask_gp]
                train_y_gp = self.train_y_clf[mask_gp]
            else:
                train_x_gp = self.train_x_clf
                train_y_gp = self.train_y_clf
        else:
            self.train_x_clf = None
            self.train_y_clf = None
            train_x_gp = None
            train_y_gp = None

        super().__init__(
            kernel=kernel,
            train_x=train_x_gp,
            train_y=train_y_gp,
            optimizer=optimizer,
            optimizer_options=optimizer_options,
            param_names=param_names,
        )

        # Initialize Classifier
        self.use_clf = self.clf_data_size >= self.clf_use_size
        self._clf_predict_func = None

        if self.use_clf:
            if train_clf_on_init:
                self.train_classifier()
        else:
            log.debug(f"Not enough data ({self.clf_data_size}) to use classifier (need {self.clf_use_size} points), or classifier type not set.")

    def initialize(self, ndim=None):
        """
        Complete all ndim-dependent initialisation for the GP and classifier.

        Called immediately from ``__init__`` when training data are provided,
        or from ``update()`` on first data arrival when constructed with
        ``train_x=None``.
        """
        if ndim is not None:
            self.train_x_clf = jnp.empty((0, ndim))
            self.train_y_clf = jnp.empty((0, 1))
        super().initialize(ndim=ndim)

    def train_classifier(self):
        """Public method to train/retrain the classifier."""
        # Check if classifier data size has reached the threshold
        if not self.use_clf:
            if self.clf_data_size >= self.clf_use_size:
                log.info(f"Classifier data size ({self.clf_data_size}) reached use size ({self.clf_use_size}). Will start using classifier.")
                self.use_clf = True

        if self.use_clf: 
            self._train_classifier()

    def _train_classifier(self):
        """Trains the classifier based on clf_type."""

        start_time = time.time()

        # Determine labels for classifier training
        labels = np.where(
            self.train_y_clf.flatten() < self.train_y_clf.max() - self.clf_threshold,
            0, 1
        )

        log.debug(f" Number of labels 0: {np.sum(labels == 0)}, 1: {np.sum(labels == 1)}")

        # Add method to handle if only class is present
        if np.all(labels == labels[0]):
            # If all labels are the same, we make sure not to use the classifier
            log.debug("All labels are identical. Not using classifier for the moment")
            self.use_clf = False
            return 

        # Prepare kwargs for training
        kwargs = {}
        best_pt = self.train_x_clf[jnp.argmax(self.train_y_clf)]
        kwargs['best_pt'] = best_pt

        # Train classifier using the registered function
        # This now returns params, metrics, and predict_fn
        self.clf_params, self.clf_metrics, self._clf_predict_func = self.clf_train_fn(
            self.train_x_clf, labels, self.clf_settings, 
            init_params=self.clf_params, **kwargs
        )

        log.debug(f"Trained {self.clf_type.upper()} classifier on {self.clf_data_size} points in {time.time() - start_time:.2f}s")
        log.debug(f"Classifier metrics: {self.clf_metrics}") # Use debug for detailed metrics

    def _clf_probs(self, x):
        """Evaluate classifier feasibility probability for a single point or batch."""
        x = jnp.asarray(x)
        if x.ndim == 1:
            return self._clf_predict_func(x)
        return jax.vmap(self._clf_predict_func)(x)

    def predict_mean(self, x):
        gp_mean = super().predict_mean(x)
        if not self.use_clf or self._clf_predict_func is None:
            return gp_mean

        clf_probs = self._clf_probs(x)
        return jnp.where(clf_probs >= self.probability_threshold, gp_mean, self.minus_inf)

    def predict(self, x, *, standardized: bool = False):
        mean, var = super().predict(x, standardized=standardized)
        if not self.use_clf or self._clf_predict_func is None:
            return mean, var

        clf_probs = self._clf_probs(x)
        feasible = clf_probs >= self.probability_threshold
        mean = jnp.where(feasible, mean, self.minus_inf)
        var = jnp.where(feasible, var, safe_noise_floor)
        return mean, var

    def update(self, new_x, new_y):
        """
        Updates the classifier and GP training sets.
        Retrains classifier/GP based on thresholds and steps.
        """
        new_x = jnp.atleast_2d(new_x)
        new_y = jnp.array(new_y)
        if new_y.ndim != 2:
            new_y = new_y.reshape(-1, 1)

        # Complete deferred initialisation on first data arrival
        if self.ndim is None:
            self.initialize(ndim=new_x.shape[1])

        # Filter NaN inputs / targets
        if jnp.any(jnp.isnan(new_x)):
            log.warning(f"NaN detected in new_x! Filtering out {jnp.sum(jnp.any(jnp.isnan(new_x), axis=1))} bad points.")
            valid_mask = ~jnp.any(jnp.isnan(new_x), axis=1)
            new_x = new_x[valid_mask]
            new_y = new_y[valid_mask]
            if new_x.shape[0] == 0:
                log.warning("All new points contain NaN - skipping update.")
                return
        if jnp.any(jnp.isnan(new_y)):
            log.warning(f"NaN detected in new_y! Filtering out {jnp.sum(jnp.any(jnp.isnan(new_y), axis=1))} bad points.")
            valid_mask = ~jnp.any(jnp.isnan(new_y), axis=1)
            new_x = new_x[valid_mask]
            new_y = new_y[valid_mask]
            if new_x.shape[0] == 0:
                log.warning("All new points contain NaN - skipping update.")
                return

        # Check for duplicates in data
        new_pts_to_add = []
        new_vals_to_add = []
        for i in range(new_x.shape[0]):
            if jnp.any(jnp.all(jnp.isclose(self.train_x_clf, new_x[i], atol=1e-6, rtol=1e-4), axis=1)):
                log.debug(f"Point {new_x[i]} already exists in the training set, not updating")
            else:
                new_pts_to_add.append(new_x[i])
                new_vals_to_add.append(new_y[i])

        if new_pts_to_add:
            new_pts_to_add = jnp.atleast_2d(jnp.array(new_pts_to_add))
            new_vals_to_add = jnp.atleast_2d(jnp.array(new_vals_to_add)).reshape(-1, 1)
            self.train_x_clf = jnp.concatenate([self.train_x_clf, new_pts_to_add], axis=0)
            self.train_y_clf = jnp.concatenate([self.train_y_clf, new_vals_to_add], axis=0)

            mask_gp = self.train_y_clf.flatten() > (self.train_y_clf.max() - self.gp_threshold)
            self._setup_training_data(self.train_x_clf[mask_gp], self.train_y_clf[mask_gp])
            self.recompute_cholesky()

            log.debug(f"Classifier data size: {self.train_y_clf.shape[0]}, GP data size: {self.train_y.shape[0]}")

    def get_random_point(self,rng=None, nstd = None):

        rng = rng if rng is not None else get_numpy_rng()

        if self.use_clf:
            if nstd is not None:
                threshold = get_threshold_for_nsigma(nstd,self.ndim)
            else:
                threshold = self.clf_threshold

            pts_idx = self.train_y_clf.flatten() > self.train_y_clf.max() - threshold

            # Sample a random point from the filtered points
            valid_indices = jnp.where(pts_idx)[0]
    
            chosen_index = rng.choice(valid_indices, size=1)[0]

            pt = self.train_x_clf[chosen_index]
            log.debug(f"Random point sampled with value {self.train_y_clf[chosen_index]}")
        else:
            pt = super().get_random_point(rng=rng, nstd=nstd)

        return pt
    
    def state_dict(self):
        """
        Returns a dictionary containing the complete state of the GPwithClassifier.
        This can be used for saving, loading, or copying the GPwithClassifier.
        
        Returns
        -------
        state: dict
            Dictionary containing all necessary information to reconstruct the GPwithClassifier
        """
        # Start with the base GP state
        state = super().state_dict()
        
        # Add classifier-specific data
        classifier_state = {
            # Classifier training data
            'train_x_clf': np.array(self.train_x_clf) if self.train_x_clf is not None else None,
            'train_y_clf': np.array(self.train_y_clf) if self.train_y_clf is not None else None,
            
            # Classifier configuration
            'clf_type': self.clf_type,
            'clf_settings': self.clf_settings,
            'clf_use_size': self.clf_use_size,
            'clf_update_step': self.clf_update_step,
            'probability_threshold': self.probability_threshold,
            'minus_inf': self.minus_inf,
            'clf_threshold': self.clf_threshold,
            'gp_threshold': self.gp_threshold,
            'use_clf': self.use_clf,
            
            # Classifier state
            'clf_params': self.clf_params,
            'clf_metrics': self.clf_metrics,
            
            # Class identifier
            'gp_class': 'GPwithClassifier'
        }
        
        # Update the state with classifier-specific data
        state.update(classifier_state)
        
        return state
    
    @classmethod
    def from_state_dict(cls, state):
        """
        Create a GPwithClassifier instance from a state dictionary returned by ``state_dict()``.
        """
        kernel = Kernel.from_state_dict(state['kernel_state'])
        gp_clf = cls(
            kernel=kernel,
            train_x=state['train_x_clf'],
            train_y=state['train_y_clf'],
            clf_type=state['clf_type'],
            clf_settings=state['clf_settings'],
            clf_use_size=state['clf_use_size'],
            clf_update_step=state['clf_update_step'],
            probability_threshold=state['probability_threshold'],
            minus_inf=state['minus_inf'],
            clf_threshold=state['clf_threshold'],
            gp_threshold=state['gp_threshold'],
            optimizer=state['optimizer_method'],
            optimizer_options=state['optimizer_options'],
            param_names=state.get('param_names'),
            train_clf_on_init=state.get('train_clf_on_init', False),
        )

        if state.get('cholesky') is not None and state.get('alphas') is not None:
            gp_clf.cholesky = jnp.array(state['cholesky'])
            gp_clf.alphas = jnp.array(state['alphas'])

        # Restore classifier state
        gp_clf.use_clf = state['use_clf']
        gp_clf.clf_params = state.get('clf_params')
        gp_clf.clf_metrics = state.get('clf_metrics', {})

        # Regenerate prediction function if classifier parameters exist
        if gp_clf.clf_params is not None:
            if gp_clf.clf_type == 'svm':
                gp_clf._clf_predict_func = gp_clf.clf_predict_fn(gp_clf.clf_params)
            elif gp_clf.clf_type == 'nn':
                gp_clf._clf_predict_func = gp_clf.clf_predict_fn(gp_clf.clf_params, gp_clf.clf_settings)
            elif gp_clf.clf_type == 'ellipsoid':
                d = gp_clf.train_x_clf.shape[1]
                gp_clf._clf_predict_func = gp_clf.clf_predict_fn(
                    gp_clf.clf_params, gp_clf.clf_settings, d
                )

        return gp_clf

    @property
    def clf_data_size(self):
        """Size of the classifier's training inputs."""
        if self.train_x_clf is None:
            return 0
        return self.train_x_clf.shape[0]
    
    @property
    def npoints(self):
        if self.train_x_clf is None:
            return 0
        return self.train_x_clf.shape[0]