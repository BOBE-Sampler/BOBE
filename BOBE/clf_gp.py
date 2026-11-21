"""
Classifier-GP module combining Gaussian Process regression with a classifier for selective training.
"""

import time
from typing import List
import numpy as np
import jax.numpy as jnp
from .gp import GP, safe_noise_floor
from .kernels import Kernel
from .surrogate import ClassifierMixin
from .clf import Classifier
from .utils.seed import get_numpy_rng
from .utils.log import get_logger
from .utils.core import get_threshold_for_nsigma
log = get_logger("clf_gp")


class ClassifierGP(GP, ClassifierMixin):
    def __init__(
        self,
        kernel: Kernel,
        classifier: Classifier,
        train_x=None,
        train_y=None,
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
        Generic Classifier-GP class combining a GP with a feasibility classifier.
        The GP is trained on the data points that are within ``gp_threshold`` of
        the maximum value observed so far.

        Kernel hyperparameters, priors, bounds, observation noise, and any
        kernel-metric transform are configured directly on ``kernel``
        (e.g. ``RBFKernel(...)`` or ``MaternKernel(...)``). Classifier settings
        are configured directly on ``classifier`` (e.g. ``SVMClassifier(...)``,
        ``MLPClassifier(...)``, or ``EllipsoidClassifier(...)``).

        Parameters
        ----------
        kernel : Kernel
            Kernel instance to use.
        classifier : Classifier
            Classifier instance to use (e.g. ``SVMClassifier()``,
            ``MLPClassifier()``, or ``EllipsoidClassifier()``).
        train_x : array-like, shape (n_samples, n_dim), optional
            Initial training points in unit-cube coordinates. If ``None``,
            dimension-dependent setup is deferred until ``initialize(ndim)``
            or the first call to ``update()``.
        train_y : array-like, shape (n_samples,), optional
            Initial training values.
        clf_use_size : int, optional
            Minimum number of points to start using the classifier. Default is 10.
        clf_update_step : int, optional
            Update classifier every ``clf_update_step`` points after ``clf_use_size`` is reached. Default is 1.
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
        if not isinstance(classifier, Classifier):
            raise TypeError(
                f"classifier must be a Classifier instance, got {type(classifier).__name__}"
            )

        self.classifier = classifier
        self.clf_use_size = clf_use_size
        self.clf_update_step = clf_update_step
        self.probability_threshold = probability_threshold
        self.minus_inf = minus_inf
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

        # Initialize classifier usage flag
        self.use_clf = self.clf_data_size >= self.clf_use_size

        if self.use_clf:
            if train_clf_on_init:
                self.train_classifier()
        else:
            log.debug(
                f"Not enough data ({self.clf_data_size}) to use classifier "
                f"(need {self.clf_use_size} points)."
            )

    @property
    def clf_type(self) -> str:
        """Class name of the composed classifier."""
        return type(self.classifier).__name__

    @property
    def clf_metrics(self) -> dict:
        """Metrics dictionary from the most recent classifier fit."""
        return self.classifier.metrics

    def initialize(self, ndim=None, param_bounds=None, param_names=None):
        """
        Complete deferred initialisation for the GP, kernel transform, and classifier.

        Called immediately from ``__init__`` when training data are provided,
        from ``BOBE`` with likelihood bounds/names, or from ``update()`` on
        first data arrival when constructed with ``train_x=None``.
        """
        resolved_ndim = ndim
        if resolved_ndim is None and param_bounds is not None:
            resolved_ndim = int(np.asarray(param_bounds).shape[1])
        if self.train_x_clf is None and resolved_ndim is not None:
            self.train_x_clf = jnp.empty((0, int(resolved_ndim)))
            self.train_y_clf = jnp.empty((0, 1))
        super().initialize(
            ndim=resolved_ndim, param_bounds=param_bounds, param_names=param_names
        )

    def train_classifier(self):
        """Public method to train/retrain the classifier."""
        if not self.use_clf:
            if self.clf_data_size >= self.clf_use_size:
                log.info(
                    f"Classifier data size ({self.clf_data_size}) reached use size "
                    f"({self.clf_use_size}). Will start using classifier."
                )
                self.use_clf = True

        if self.use_clf:
            self._train_classifier()

    def _train_classifier(self):
        """Train the composed classifier on the current labelled dataset."""
        start_time = time.time()

        labels = np.where(
            self.train_y_clf.flatten() < self.train_y_clf.max() - self.clf_threshold,
            0, 1
        )
        log.debug(f" Number of labels 0: {np.sum(labels == 0)}, 1: {np.sum(labels == 1)}")

        if np.all(labels == labels[0]):
            log.debug("All labels are identical. Not using classifier for the moment")
            self.use_clf = False
            return

        best_pt = self.train_x_clf[jnp.argmax(self.train_y_clf)]
        self.classifier.fit(self.train_x_clf, labels, best_pt=best_pt)

        log.debug(
            f"Trained {self.clf_type} on {self.clf_data_size} points "
            f"in {time.time() - start_time:.2f}s"
        )
        log.debug(f"Classifier metrics: {self.classifier.metrics}")

    def predict_mean(self, x):
        gp_mean = super().predict_mean(x)
        if not self.use_clf or not self.classifier.is_fitted:
            return gp_mean

        clf_probs = self.classifier.predict_proba(x)
        return jnp.where(clf_probs >= self.probability_threshold, gp_mean, self.minus_inf)

    def predict(self, x, *, standardized: bool = False):
        mean, var = super().predict(x, standardized=standardized)
        if not self.use_clf or not self.classifier.is_fitted:
            return mean, var

        clf_probs = self.classifier.predict_proba(x)
        feasible = clf_probs >= self.probability_threshold
        mean = jnp.where(feasible, mean, self.minus_inf)
        var = jnp.where(feasible, var, safe_noise_floor)
        return mean, var

    def update(self, new_x, new_y):
        """
        Updates the classifier and GP training sets.
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

    def get_random_point(self, rng=None, nstd=None):
        rng = rng if rng is not None else get_numpy_rng()

        if self.use_clf:
            if nstd is not None:
                threshold = get_threshold_for_nsigma(nstd, self.ndim)
            else:
                threshold = self.clf_threshold

            pts_idx = self.train_y_clf.flatten() > self.train_y_clf.max() - threshold
            valid_indices = jnp.where(pts_idx)[0]
            chosen_index = rng.choice(valid_indices, size=1)[0]
            pt = self.train_x_clf[chosen_index]
            log.debug(f"Random point sampled with value {self.train_y_clf[chosen_index]}")
        else:
            pt = super().get_random_point(rng=rng, nstd=nstd)

        return pt

    def state_dict(self):
        """
        Return a dictionary containing the complete state of the ``ClassifierGP``.
        """
        state = super().state_dict()
        state.update({
            'train_x_clf': np.array(self.train_x_clf) if self.train_x_clf is not None else None,
            'train_y_clf': np.array(self.train_y_clf) if self.train_y_clf is not None else None,
            'classifier_state': self.classifier.state_dict(),
            'clf_use_size': self.clf_use_size,
            'clf_update_step': self.clf_update_step,
            'probability_threshold': self.probability_threshold,
            'minus_inf': self.minus_inf,
            'clf_threshold': self.clf_threshold,
            'gp_threshold': self.gp_threshold,
            'use_clf': self.use_clf,
            'gp_class': 'ClassifierGP',
        })
        return state

    @classmethod
    def from_state_dict(cls, state):
        """
        Create a ``ClassifierGP`` instance from a state dictionary returned by ``state_dict()``.
        """
        kernel = Kernel.from_state_dict(state['kernel_state'])
        classifier = Classifier.from_state_dict(state['classifier_state'])
        gp_clf = cls(
            kernel=kernel,
            classifier=classifier,
            train_x=state['train_x_clf'],
            train_y=state['train_y_clf'],
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

        gp_clf.use_clf = state['use_clf']
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
