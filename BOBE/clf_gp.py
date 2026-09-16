import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from .gp import GP, safe_noise_floor
from .clf import (
    CLASSIFIER_REGISTRY
)
from .transforms import PrincipalAxesTransform
from .utils.seed import get_new_jax_key, get_numpy_rng
from .utils.log import get_logger
from .utils.core import get_threshold_for_nsigma
log = get_logger("clf_gp")


class GPwithClassifier(GP):
    def __init__(self, train_x=None, train_y=None,
                 clf_type='svm', clf_settings={},
                 clf_use_size=10, clf_update_step=1,
                 probability_threshold=0.5, minus_inf=-1e5,
                 clf_threshold=250., gp_threshold=500.,
                 noise=1e-8, kernel="rbf", 
                 optimizer="scipy", optimizer_options={},
                 kernel_variance_bounds=[1e-4, 1e8], lengthscale_bounds=[0.01, 5.],
                 tausq=None, tausq_bounds=[1e-4, 1e4],
                 kernel_variance_prior=None, lengthscale_prior=None, 
                 lengthscales=None, kernel_variance=1.0,
                 rotation_covariance=None, rotation_samples=None,
                 rotation_weights=None, rotation_log_weights=None,
                 rotation_dims=None,
                 param_names=None,
                 train_clf_on_init=True,  # Prevent retraining on copy
                 ):
        """
        Generic Classifier-GP class combining a GP with a classifier. The GP is trained on the data points
        that are within the GP threshold of the maximum value of the GP.

        Arguments
        ---------
        train_x : array-like, shape (n_samples, n_dim)
            Initial training points.
        train_y : array-like, shape (n_samples,)
            Initial training values.
        clf_type : str, optional
            Type of classifier ('svm', 'nn', 'ellipsoid', etc.). Default is 'svm'.
        clf_params : dict, optional
            Parameters specific to the chosen classifier. Default is None.
        clf_use_size : int, optional
            Minimum number of points to start using the classifier. Default is 300.
        clf_update_step : int, optional
            Update classifier every `clf_update_step` points after `clf_use_size` is reached. Default is 5.
        probability_threshold : float, optional
            Threshold for classifier probability/score to consider a point feasible (important for nn, ellipsoid). Default is 0.5.
        minus_inf : float, optional
            Value used for infeasible predictions. Default is -1e5.
        clf_threshold : float, optional
            Threshold for initial classifier training labels (if used).
            If None, `gp_threshold` might be used or a default calculated.
        gp_threshold : float, optional
            Threshold for adding points to the GP training set. Default is 5000.
        noise, kernel, optimizer, kernel_variance_bounds, lengthscale_bounds, lengthscale_priors, lengthscales, kernel_variance:
            GP parameters (see DSLP_GP/SAAS_GP). Note: bounds are now in actual space, not log10.
        rotation_covariance, rotation_samples, rotation_weights, rotation_log_weights, rotation_dims:
            Static kernel-metric rotation parameters. See GP for details.

        """
        # Store Data and Classifier Settings
        self.train_x_clf = jnp.array(train_x)
        self.train_y_clf = jnp.array(train_y).reshape(-1, 1) # Ensure 2D
        self.clf_use_size = clf_use_size
        self.clf_update_step = clf_update_step
        self.clf_type = clf_type.lower()
        self.clf_settings = clf_settings
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


        # Prepare GP Data
        if self.train_y_clf.size > 0:
            mask_gp = self.train_y_clf.flatten() > (self.train_y_clf.max() - self.gp_threshold)
            train_x_gp = self.train_x_clf[mask_gp]
            train_y_gp = self.train_y_clf[mask_gp] 
        else:
            train_x_gp = self.train_x_clf
            train_y_gp = self.train_y_clf

        # Initialize GP using inheritance
        gp_init_kwargs = {
            'train_x': train_x_gp,
            'train_y': train_y_gp,
            'noise': noise,
            'kernel': kernel,
            'optimizer': optimizer,
            'optimizer_options': optimizer_options,
            'kernel_variance_bounds': kernel_variance_bounds,
            'lengthscale_bounds': lengthscale_bounds,
            'lengthscales': lengthscales,
            'kernel_variance': kernel_variance,
            'lengthscale_prior': lengthscale_prior if lengthscale_prior is not None else "DSLP",
            'kernel_variance_prior': kernel_variance_prior,
            'tausq': tausq,
            'tausq_bounds': tausq_bounds,
            'rotation_covariance': rotation_covariance,
            'rotation_samples': rotation_samples,
            'rotation_weights': rotation_weights,
            'rotation_log_weights': rotation_log_weights,
            'rotation_dims': rotation_dims,
            'param_names': param_names,
        }
                    
        super().__init__(**gp_init_kwargs)

        # Initialize Classifier
        self.use_clf = self.clf_data_size >= self.clf_use_size
        self.clf_model_params = None
        self._clf_predict_func = None

        if self.use_clf:
             if train_clf_on_init:
                 self.train_classifier()
        else:
             log.debug(f"Not enough data ({self.clf_data_size}) to use classifier (need {self.clf_use_size} points), or classifier type not set.")

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

    def predict_mean_single(self,x):
        gp_mean = super().predict_mean_single(x)
        if not self.use_clf or self._clf_predict_func is None:
            return gp_mean

        clf_probs = self._clf_predict_func(x)
        return jnp.where(clf_probs >= self.probability_threshold, gp_mean, self.minus_inf)

    def predict_var_single(self,x):
        var  = super().predict_var_single(x)
        if not self.use_clf or self._clf_predict_func is None:
            return var

        clf_probs = self._clf_predict_func(x)
        return jnp.where(clf_probs >= self.probability_threshold, var, safe_noise_floor)

    def predict_mean_batched(self,x):
        x = jnp.atleast_2d(x)
        return jax.vmap(self.predict_mean_single)(x)

    def predict_var_batched(self,x):
        x = jnp.atleast_2d(x)
        return jax.vmap(self.predict_var_single)(x)

    def predict_single(self,x):
        mean, var = super().predict_single(x)
        if not self.use_clf or self._clf_predict_func is None:
            return mean, var

        clf_probs = self._clf_predict_func(x)
        mean = jnp.where(clf_probs >= self.probability_threshold, mean, self.minus_inf)
        var = jnp.where(clf_probs >= self.probability_threshold, var, safe_noise_floor)
        return mean, var

    def fantasy_var(self, new_x, mc_points,k_train_mc):
        """
        Computes the fantasy variance, see gp.py for more details.
        Classifier logic could potentially be added here if needed.
        """
        return super().fantasy_var(new_x, mc_points,k_train_mc)

    def update(self, new_x, new_y):
        """
        Updates the classifier and GP training sets.
        Retrains classifier/GP based on thresholds and steps.
        """
        new_x = jnp.atleast_2d(new_x)
        new_y = jnp.atleast_2d(new_y)

        # Check for duplicates in data 
        new_pts_to_add = []
        new_vals_to_add = []
        for i in range(new_x.shape[0]):
            if jnp.any(jnp.all(jnp.isclose(self.train_x_clf, new_x[i], atol=1e-6,rtol=1e-4), axis=1)):
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
            self.train_x = self.train_x_clf[mask_gp]
            self.train_y = self.train_y_clf[mask_gp].reshape(-1, 1)
            self.y_std = jnp.std(self.train_y) if self.train_y.shape[0] > 1 else 1.0
            self.y_mean = jnp.mean(self.train_y)
            self.train_y = (self.train_y - self.y_mean) / self.y_std
            self.recompute_cholesky()

            log.debug(f"Classifier data size: {self.train_y_clf.shape[0]},  GP data size: {self.train_y.shape[0]}")

    def kernel(self,x1,x2,lengthscales,kernel_variance,noise,include_noise=True):
        """
        Returns the kernel function used by the GP.
        """
        return super().kernel(x1,x2,lengthscales,kernel_variance,noise,include_noise=include_noise)

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
            'train_x_clf': np.array(self.train_x_clf),
            'train_y_clf': np.array(self.train_y_clf),
            
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
        Creates a GPwithClassifier instance from a state dictionary.
        
        Arguments
        ---------
        state: dict
            State dictionary returned by state_dict()
            
        Returns
        -------
        gp_clf: GPwithClassifier
            The reconstructed GPwithClassifier object
        """
        # Create GPwithClassifier instance
        gp_clf = cls(
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
            noise=state['noise'],
            kernel=state['kernel_name'],
            optimizer=state['optimizer_method'],
            optimizer_options=state['optimizer_options'],
            kernel_variance_bounds=state['kernel_variance_bounds'],
            lengthscale_bounds=state['lengthscale_bounds'],
            lengthscales=state['lengthscales'],
            kernel_variance=state['kernel_variance'],
            kernel_variance_prior=state.get('kernel_variance_prior_spec'),
            lengthscale_prior=state.get('lengthscale_prior_spec'),
            tausq=state.get('tausq', 1.0),
            tausq_bounds=state.get('tausq_bounds', [-4, 4]),
            train_clf_on_init=state.get('train_clf_on_init', True),
        )

        transform_state = state.get('input_transform_state')

        if transform_state is not None:
            if transform_state['type'] != 'PrincipalAxesTransform':
                raise ValueError(
                    f"Unknown input transform: {transform_state['type']}"
                )
            transform = PrincipalAxesTransform.from_state_dict(
                transform_state
            )
            gp_clf.kernel.set_input_transform(transform)
            gp_clf.recompute_cholesky()
        
        # # Restore computed state if available
        # if state.get('cholesky') is not None:
        #     gp_clf.cholesky = jnp.array(state['cholesky'])
        # if state.get('alphas') is not None:
        #     gp_clf.alphas = jnp.array(state['alphas'])
        
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

    def save(self, filename='gp'):
        """
        Save the GPwithClassifier state to a file using state_dict.
        
        Arguments
        ---------
        filename: str
            The filename to save to (with or without .npz extension). Default is 'gp'.
        """
        if not filename.endswith('.npz'):
            filename += '.npz'
        
        state = self.state_dict()
        np.savez(filename, **state)
        log.info(f"Saved GPwithClassifier state to {filename}")

    @classmethod
    def load(cls, filename, **kwargs):
        """
        Loads a GPwithClassifier from a file
        
        Arguments
        ---------
        filename: str
            The name of the file to load the GPwithClassifier from (with or without .npz extension)
        **kwargs: 
            Additional keyword arguments to pass to the GPwithClassifier constructor
            
        Returns
        -------
        gp_clf: GPwithClassifier
            The loaded GPwithClassifier object
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
        gp_clf = cls.from_state_dict(state)
        
        log.info(f"Loaded GPwithClassifier from {filename} with {gp_clf.train_x.shape[0]} training points")
        return gp_clf

    def copy(self):
        """
        Creates a deep copy of the GPwithClassifier using state_dict.
        
        Returns
        -------
        gp_clf_copy: GPwithClassifier
            A deep copy of the current GPwithClassifier
        """
        state = self.state_dict()
        return self.__class__.from_state_dict(state)

    @property
    def clf_data_size(self):
        """Size of the classifier's training inputs."""
        return self.train_x_clf.shape[0]
    
    @property
    def npoints(self):
        return self.train_x_clf.shape[0]