import os
from typing import Any, Callable, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from .acquisition import *
from .likelihood import CobayaLikelihood, Likelihood
from .pool import MPI_Pool
from .samplers import nested_sampling_Dy, sample_GP_NUTS
from .surrogate import ClassifierMixin, Surrogate
from .utils.core import kl_divergence_gaussian, scale_from_unit, scale_to_unit
from .utils.log import get_logger, update_verbosity
from .utils.results import BOBEResults
from .utils.seed import get_new_jax_key, get_numpy_rng, set_global_seed

log = get_logger("bo")
log.info(f'JAX using {jax.device_count()} devices.')

_acq_funcs = {"wipv": WIPV, "ei": EI, "logei": LogEI, 'wipstd': WIPStd}


def _gp_fit_budget(gp_size: int):
    """Return the restart and optimizer iteration budgets for a GP training-set size."""
    if gp_size <= 500:
        return 8, 1000
    return 4, 500


def get_dimension_based_defaults(ndim: int):
    """
    Compute reasonable default values for ``run()`` parameters based on problem dimension.

    Users can override any of these by passing explicit values to ``BOBE.run()``.
    """
    defaults = {
        'min_evals': 8 * ndim,  # scales linearly with dimension
        'max_evals': 200 * ndim,  # more evals for higher dimensions
        'max_gp_size': min(2100, 160 * ndim),  # larger GP for higher dimensions
        'batch_size': 2 if ndim <= 6 else min(8, int(2 * (ndim / 6))),  # 2-8 depending on dimension
        'ns_n_points': min(50, 2 * ndim),  # nested sampling frequency, less for higher dimensions
        'num_hmc_warmup': 256 if ndim <= 6 else 512,  # more warmup for higher dimensions
        'num_hmc_samples': min(5000, max(512, int(4096 * (ndim / 20)))),  # capped at 5000
        'mc_points_size': min(512, 32 * ndim),  # more MC points for higher dimensions
        'num_chains': min(6, max(3, jax.device_count())),  # 3-6 chains depending on available devices
        'fit_n_points': min(50, 2 * ndim),  # refit less often for higher dimensions
        'logz_threshold': 0.01 + 0.01 * (ndim / 6) if ndim <= 6 else min(1.0, 0.1 + 0.1 * (ndim / 6) ** 2),
        'transform_acq_threshold': min(5, 2 * (ndim / 10)),  # default acquisition gate for adaptive kernel rotation
    }
    return defaults


class BOBE:

    def __init__(
        self,
        loglikelihood: Union[Callable, str, Dict[str, Any], Likelihood],
        param_list: List[str] = None,
        param_bounds=None,
        param_labels=None,
        likelihood_name=None,
        confidence_for_unbounded=0.9999995,
        gp: Optional[Surrogate] = None,
        n_cobaya_init: int = 4,
        n_sobol_init: int = 4,
        init_train_x=None,
        init_train_y=None,
        resume: bool = False,
        resume_file: Optional[str] = None,
        save_dir: str = '.',
        save: bool = True,
        save_step: int = 1,
        optimizer: str = 'scipy',
        minus_inf: float = -1e10,
        seed: Optional[int] = None,
        verbosity: str = 'INFO',
        dynamic_dispatch: bool = False,
    ):
        """
        Initialize the BOBE (Bayesian Optimization for Bayesian Evidence) sampler.

        Parameters
        ----------
        loglikelihood : callable, str, dict, or Likelihood
            Log-likelihood specification. Can be:
            - A callable function (requires ``param_list`` and ``param_bounds``)
            - A string path to a Cobaya YAML file (automatically creates ``CobayaLikelihood``)
            - A dict with Cobaya info (automatically creates ``CobayaLikelihood``)
            - A ``Likelihood`` instance (``param_list``, ``param_bounds`` ignored)
        param_list : list of str, optional
            Names of parameters. Required if ``loglikelihood`` is a callable.
        param_bounds : array-like, optional
            Parameter bounds, shape ``(2, ndim)``. Required if ``loglikelihood`` is a callable.
        param_labels : list of str, optional
            LaTeX labels for parameters. Defaults to ``param_list``.
        likelihood_name : str, optional
            Name for the likelihood (used in output files).
        confidence_for_unbounded : float, optional
            Confidence level for unbounded Cobaya priors. Default is 0.9999995.
        gp : Surrogate, optional
            Empty surrogate model instance (e.g. ``GP(kernel=RBFKernel(...))`` or
            ``ClassifierGP(kernel=..., classifier=SVMClassifier())``). Required when
            ``resume=False``; ignored when resuming from ``resume_file``.
        n_cobaya_init : int, optional
            Number of initial points from Cobaya reference distribution. Default is 4.
        n_sobol_init : int, optional
            Number of initial Sobol quasi-random points. Default is 4.
        init_train_x : array-like, optional
            User-provided initial training points in physical parameter space, shape ``(n_points, ndim)``.
        init_train_y : array-like, optional
            User-provided initial training values (log-likelihood), shape ``(n_points, 1)`` or ``(n_points,)``.
        resume : bool, optional
            If True, resume from a previous unified ``.pkl`` checkpoint. Default is False.
        resume_file : str, optional
            Base path (or ``.pkl`` path) to resume from. Default is None.
        save_dir : str, optional
            Directory for saving results. Default is ``'.'``.
        save : bool, optional
            Whether to save ``.pkl`` checkpoints. Default is True.
        save_step : int, optional
            Save checkpoint every ``save_step`` iterations. Default is 1.
        optimizer : str, optional
            Optimizer for acquisition function optimization: ``'scipy'`` or ``'optax'``. Default is ``'scipy'``.
        minus_inf : float, optional
            Value representing negative infinity for failed likelihood evaluations. Default is -1e10.
        seed : int, optional
            Random seed for reproducibility. Default is None.
        verbosity : str, optional
            Logging verbosity level: ``'DEBUG'``, ``'INFO'``, ``'WARNING'``, ``'ERROR'``.
        dynamic_dispatch : bool, optional
            If False (default), use static (round-robin) task distribution across MPI ranks.
            Static dispatch is reproducible across runs at a fixed ``nprocs`` and seed.
            If True, use dynamic dispatch (first-available worker).
        """
        if gp is not None and not isinstance(gp, Surrogate):
            raise TypeError(
                f"gp must be a Surrogate instance or None, got {type(gp).__name__}"
            )
        if not (resume and resume_file is not None) and gp is None:
            raise ValueError(
                "A surrogate instance (gp) must be provided when not resuming from a checkpoint."
            )

        # Update logging verbosity
        update_verbosity(verbosity=verbosity)

        # Initialize MPI pool and synchronize all ranks at startup
        self.pool = MPI_Pool(dynamic_dispatch=dynamic_dispatch)
        self.is_main = self.pool.is_main_process
        self.is_mpi = self.pool.is_mpi
        if self.is_mpi:
            self.pool.comm.Barrier()

        # Convert to Likelihood instance and store for all processes
        self.loglikelihood = self._prepare_likelihood(
            loglikelihood, param_list, param_bounds, param_labels,
            likelihood_name, confidence_for_unbounded, minus_inf,
        )
        self.ndim = len(self.loglikelihood.param_list)

        if not self.is_main:
            # Workers only need likelihood and seed - everything else is handled in worker_wait
            self.pool.worker_wait(likelihood=self.loglikelihood, seed=seed)
            return  # Workers never return from worker_wait until pool.close()

        # MAIN PROCESS FULL SETUP
        self._setup_main_process(
            seed, optimizer, save, save_dir, save_step,
            n_cobaya_init, n_sobol_init, minus_inf, resume, dynamic_dispatch,
        )

        # Initialize mutable run-state defaults before resume so resumed values are preserved
        self.prev_samples = None
        self.convergence_counter = 0
        self.converged = False
        self.termination_reason = ''

        # Handle resume if requested (always loads the surrogate from resume_file)
        if resume and resume_file is not None:
            if gp is not None:
                log.info("Resuming from checkpoint file; ignoring passed gp instance.")
            self._handle_resume(resume_file)

        # Fresh start path - generate and train initial GP (main process only)
        if self.fresh_start:
            self._handle_fresh_start(
                gp=gp,
                n_cobaya_init=n_cobaya_init,
                n_sobol_init=n_sobol_init,
                init_train_x=init_train_x,
                init_train_y=init_train_y,
            )

        # Record resolved surrogate metadata in results manager
        self.results_manager.settings['gp_class'] = type(self.gp).__name__
        self.results_manager.settings['use_clf'] = isinstance(self.gp, ClassifierMixin)
        if isinstance(self.gp, ClassifierMixin) and 'Classifier Training' not in self.results_manager.phase_times:
            self.results_manager.phase_times['Classifier Training'] = 0.0

        # Extract best point from GP
        if self.gp.train_y.size > 0:
            idx_best = jnp.argmax(self.gp.train_y)
            self.best_pt_from_gp = scale_from_unit(
                self.gp.train_x[idx_best], self.loglikelihood.param_bounds
            ).flatten()
            best_f_from_gp = float(self.gp.train_y.max()) * self.gp.y_std + self.gp.y_mean
        else:
            best_f_from_gp = -np.inf
            self.best_pt_from_gp = None

        # Use restored best_f/best_pt if available and at least as good
        if not hasattr(self, 'best_f') or best_f_from_gp > getattr(self, 'best_f', -np.inf):
            self.best_f = best_f_from_gp
            self.best_pt = self.best_pt_from_gp
            if not hasattr(self, 'best_pt_iteration'):
                self.best_pt_iteration = self.start_iteration
        elif getattr(self, 'best_pt', None) is None:
            self.best_pt = self.best_pt_from_gp

        if self.best_pt is not None:
            self.best = {
                name: f"{float(val):.6f}"
                for name, val in zip(self.loglikelihood.param_list, np.asarray(self.best_pt).flatten())
            }
            log.info(f"Initial best point {self.best} with value = {self.best_f:.6f}")

        # Dedicated objective-evaluation counter (set on resume or fresh start; fallback to GP size)
        if not hasattr(self, 'total_objective_evals'):
            self.total_objective_evals = int(
                self.results_manager.gp_info.get('total_objective_evals', self.gp.npoints)
            )

        self.current_iteration = self.start_iteration

        # Save initial unified checkpoint
        self._save_checkpoint()

    # ============================================================================
    # INITIALIZATION HELPER METHODS
    # ============================================================================

    def _prepare_likelihood(
        self,
        loglikelihood,
        param_list,
        param_bounds,
        param_labels,
        likelihood_name,
        confidence_for_unbounded,
        minus_inf,
    ):
        """Convert input to Likelihood instance if needed."""
        if isinstance(loglikelihood, Likelihood):
            return loglikelihood

        if isinstance(loglikelihood, (str, dict)):
            return CobayaLikelihood(
                input_file_dict=loglikelihood,
                confidence_for_unbounded=confidence_for_unbounded,
                minus_inf=minus_inf,
                name=likelihood_name if likelihood_name is not None else 'CobayaLikelihood',
            )

        if callable(loglikelihood):
            return Likelihood(
                loglikelihood=loglikelihood,
                param_list=param_list,
                param_bounds=param_bounds,
                param_labels=param_labels,
                name=likelihood_name,
                minus_inf=minus_inf,
            )

        raise ValueError(
            "loglikelihood must be one of: "
            "callable, string (Cobaya YAML path), dict (Cobaya info), or Likelihood instance"
        )

    def _setup_main_process(
        self,
        seed,
        optimizer,
        save,
        save_dir,
        save_step,
        n_cobaya_init,
        n_sobol_init,
        minus_inf,
        resume,
        dynamic_dispatch=False,
    ):
        """Setup full attributes for main process."""
        set_global_seed(seed)
        self.np_rng = get_numpy_rng()

        self.output_file = self.loglikelihood.name
        self.save = save
        self.save_step = save_step
        self.save_dir = save_dir
        if self.save:
            os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, self.output_file)

        if optimizer.lower() not in ['optax', 'scipy']:
            raise ValueError("optimizer must be either 'optax' or 'scipy'")
        self.optimizer = optimizer
        self.minus_inf = minus_inf

        self.results_manager = BOBEResults(
            output_file=self.output_file,
            save_dir=self.save_dir,
            param_names=self.loglikelihood.param_list,
            param_labels=self.loglikelihood.param_labels,
            param_bounds=self.loglikelihood.param_bounds,
            settings={
                'n_cobaya_init': n_cobaya_init,
                'n_sobol_init': n_sobol_init,
                'minus_inf': minus_inf,
                'seed': seed,
                'dynamic_dispatch': dynamic_dispatch,
            },
            likelihood_name=self.loglikelihood.name,
        )

        self.fresh_start = not resume

    def _save_checkpoint(self):
        """Delegate unified ``.pkl`` checkpoint serialisation to the results manager."""
        if not self.is_main or not self.save:
            return
        self.results_manager.save_checkpoint({
            'gp_state': self.gp.state_dict(),
            'total_objective_evals': int(getattr(self, 'total_objective_evals', self.gp.npoints)),
            'start_iteration': int(getattr(self, 'current_iteration', getattr(self, 'start_iteration', 0))),
            'best_f': float(getattr(self, 'best_f', float('-inf'))),
            'best_pt': getattr(self, 'best_pt', None),
            'best_pt_iteration': int(getattr(self, 'best_pt_iteration', 0)),
            'convergence_counter': int(getattr(self, 'convergence_counter', 0)),
            'converged': bool(getattr(self, 'converged', False)),
            'termination_reason': str(getattr(self, 'termination_reason', '')),
            'prev_samples': getattr(self, 'prev_samples', None),
        })

    def _handle_resume(self, resume_file: str):
        """Load a unified ``.pkl`` checkpoint via the results manager and restore BO state."""
        data = self.results_manager.load_checkpoint(resume_file)
        if data is None:
            self.fresh_start = True
            return

        self.gp = Surrogate.from_state_dict(data['gp_state'])
        if self.gp.npoints > 0:
            _ = self.gp.predict_mean(self.gp.train_x[0])
        log.info(f"Loaded GP ({type(self.gp).__name__}) with {self.gp.train_x.shape[0]} training points")

        self.start_iteration = int(data.get('start_iteration', 0))
        self.best_f = float(data.get('best_f', float('-inf')))
        self.best_pt = data.get('best_pt', None)
        self.best_pt_iteration = int(data.get('best_pt_iteration', 0))
        self.convergence_counter = int(data.get('convergence_counter', 0))
        self.converged = bool(data.get('converged', False))
        self.termination_reason = str(data.get('termination_reason', ''))
        self.prev_samples = data.get('prev_samples', None)
        self.total_objective_evals = int(data.get('total_objective_evals', self.gp.npoints))

        cs = self.results_manager.get_last_convergence_state()
        self.prev_converged = cs['converged']
        self.prev_convergence_delta = cs['delta']
        self.prev_convergence_threshold = cs['threshold']

        log.info(
            f"Resuming from iteration {self.start_iteration}, "
            f"total_objective_evals={self.total_objective_evals}, best logL={self.best_f:.4f}"
        )
        self.fresh_start = False

    def _handle_fresh_start(self, gp, n_cobaya_init, n_sobol_init, init_train_x, init_train_y):
        """Handle fresh start initialization (main process only)."""
        self.start_iteration = 0
        self.best_pt_iteration = 0
        self.prev_converged = False
        self.prev_convergence_delta = None
        self.prev_convergence_threshold = None

        if gp is None:
            raise ValueError(
                "A surrogate instance (gp) must be provided for a fresh start."
            )
        if gp.npoints > 0:
            raise ValueError(
                "Pass an empty surrogate to BOBE; provide custom initial training "
                "points via init_train_x and init_train_y in BOBE.__init__."
            )
        self.gp = gp
        if self.gp.ndim is not None and self.gp.ndim != self.ndim:
            raise ValueError(
                f"Provided gp has ndim={self.gp.ndim}, expected {self.ndim} to match likelihood."
            )

        self.gp.initialize(
            ndim=self.ndim,
            param_bounds=self.loglikelihood.param_bounds,
            param_names=self.loglikelihood.param_list,
        )

        has_init = (
            n_sobol_init > 0
            or (isinstance(self.loglikelihood, CobayaLikelihood) and n_cobaya_init > 0)
            or init_train_x is not None
            or init_train_y is not None
        )
        if not has_init:
            raise ValueError(
                "At least one of n_sobol_init, n_cobaya_init, or init_train_x/init_train_y "
                "must provide initial training points."
            )

        train_x, train_y = self._get_initial_training_data(
            n_cobaya_init=n_cobaya_init,
            n_sobol_init=n_sobol_init,
            init_train_x=init_train_x,
            init_train_y=init_train_y,
        )
        self.total_objective_evals = int(train_x.shape[0])

        self._initialize_gp(train_x=train_x, train_y=train_y)

    def _get_initial_training_data(self, n_cobaya_init, n_sobol_init, init_train_x=None, init_train_y=None):
        """
        Generate and evaluate initial training points for the GP.

        1. Generates Sobol initial points in parallel (if ``n_sobol_init > 0``)
        2. Generates Cobaya initial points in parallel (if applicable)
        3. Adds user-provided initial points (if given)
        4. Removes duplicates
        5. Returns points and values in unit cube space for GP training
        """
        point_chunks = []
        val_chunks = []

        if n_sobol_init > 0:
            sobol_points, sobol_vals = self._generate_sobol_points(n_sobol_init)
            if self.is_main and sobol_points.size > 0:
                point_chunks.append(sobol_points)
                val_chunks.append(sobol_vals)

        if isinstance(self.loglikelihood, CobayaLikelihood) and n_cobaya_init > 0:
            cobaya_points, cobaya_vals = self._generate_cobaya_points(n_cobaya_init)
            if self.is_main and cobaya_points.size > 0:
                point_chunks.append(cobaya_points)
                val_chunks.append(cobaya_vals)

        if not self.is_main:
            return None, None

        if init_train_x is not None and init_train_y is not None:
            init_train_x = np.atleast_2d(init_train_x)
            init_train_y = np.atleast_2d(init_train_y).reshape(-1, 1)

            if init_train_x.shape[0] != init_train_y.shape[0]:
                raise ValueError(
                    f"init_train_x and init_train_y must have same number of points. "
                    f"Got {init_train_x.shape[0]} and {init_train_y.shape[0]}"
                )
            if init_train_x.shape[1] != self.ndim:
                raise ValueError(
                    f"init_train_x must have {self.ndim} dimensions. "
                    f"Got {init_train_x.shape[1]}"
                )

            log.info(f"Adding {len(init_train_x)} user-provided initial points")
            point_chunks.append(init_train_x)
            val_chunks.append(init_train_y)
        elif init_train_x is not None or init_train_y is not None:
            raise ValueError("Both init_train_x and init_train_y must be provided together")

        all_points = np.vstack(point_chunks)
        all_vals = np.vstack(val_chunks)

        unique_points, unique_indices = np.unique(all_points, axis=0, return_index=True)
        if len(unique_points) < len(all_points):
            log.warning(
                f"Found and removed {len(all_points) - len(unique_points)} duplicate points "
                f"from the initial set. Final set size: {len(unique_points)}."
            )
            init_points = all_points[unique_indices]
            init_vals = all_vals[unique_indices]
        else:
            init_points = all_points
            init_vals = all_vals

        if n_sobol_init > 0:
            self.results_manager.end_timing('True Objective Evaluations')

        train_x = jnp.array(scale_to_unit(init_points, self.loglikelihood.param_bounds))
        train_y = jnp.array(init_vals)

        return train_x, train_y

    def _generate_sobol_points(self, n_sobol_init: int):
        """Generate Sobol initial points on main rank and evaluate them in parallel."""
        from scipy.stats import qmc

        n_sobol = max(2, n_sobol_init)

        if self.is_main:
            self.results_manager.start_timing('True Objective Evaluations')
            sobol = qmc.Sobol(d=self.ndim, scramble=True, rng=self.np_rng).random(n_sobol)
            sobol_points = scale_from_unit(sobol, self.loglikelihood.param_bounds)
            log.info(f"Evaluating {len(sobol_points)} Sobol initial points")

            all_vals = self.pool.run_map_objective(self.loglikelihood, sobol_points)
            all_vals = np.atleast_2d(all_vals).reshape(-1, 1)
            return sobol_points, all_vals
        else:
            return np.empty((0, self.ndim)), np.empty((0, 1))

    def _generate_cobaya_points(self, n_cobaya_init: int):
        """Generate Cobaya initial points in parallel using pool."""
        if n_cobaya_init <= 0:
            raise ValueError("n_cobaya_init must be greater than zero to generate Cobaya points.")

        if self.is_main:
            log.info(f"Evaluating {n_cobaya_init} Cobaya initial points")
            results_tuples = self.pool.get_cobaya_initial_points(
                self.loglikelihood, n_cobaya_init, rng=self.np_rng
            )
            all_points = np.array([pt for pt, _ in results_tuples])
            all_vals = np.array([[lp] for _, lp in results_tuples])
            return all_points, all_vals
        else:
            return np.empty((0, self.ndim)), np.empty((0, 1))

    def _initialize_gp(self, train_x=None, train_y=None):
        """Populate the surrogate with initial training data and fit its hyperparameters."""
        if not self.is_main:
            return

        if train_x is not None and train_y is not None:
            self.gp.update(train_x, train_y)

        if isinstance(self.gp, ClassifierMixin):
            self.gp.train_classifier()

        self.results_manager.start_timing('GP Training')
        log.info(f"Hyperparameters before refit: {self.gp.hyperparams_dict()}")

        self._fit_gp_hyperparameters(reason="during initial GP training")

        log.info(f"Hyperparameters after refit: {self.gp.hyperparams_dict()}")
        self.results_manager.end_timing('GP Training')

    # ============================================================================
    # RUN HELPER METHODS
    # ============================================================================

    def _fit_gp_hyperparameters(self, reason: Optional[str] = None) -> bool:
        """Fit GP hyperparameters using the unified size-based restart/iteration budget."""
        gp_size = self.gp.train_x.shape[0]
        n_restarts, maxiter = _gp_fit_budget(gp_size)

        reason_suffix = f" {reason}" if reason else ""
        log.info(
            f"Refitting GP hyperparameters{reason_suffix} with {gp_size} "
            f"training points ({n_restarts} restarts, {maxiter} iterations)"
        )
        self.pool.gp_fit(
            self.gp,
            n_restarts=n_restarts,
            maxiters=maxiter,
            rng=self.np_rng,
            use_pool=True,
        )
        if hasattr(self, 'n_points_since_last_fit'):
            self.n_points_since_last_fit = 0
        return True

    def _refit_gp_hyperparameters(self, force: bool = False, reason: Optional[str] = None) -> bool:
        """Conditionally refit GP hyperparameters using the shared policy."""
        gp_size = self.gp.train_x.shape[0]
        if gp_size < 200:
            refit_threshold = min(4, self.fit_n_points)
        elif gp_size < 800:
            refit_threshold = self.fit_n_points
        else:
            refit_threshold = max(50, self.fit_n_points)

        if not force and self.n_points_since_last_fit < refit_threshold:
            return False

        return self._fit_gp_hyperparameters(reason=reason)

    def update_gp(self, new_pts_u, new_vals, step: int = 0, verbose: bool = True):
        """
        Update the GP with new points and values, refit hyperparameters when
        triggered, and record hyperparameter evolution.
        """
        if not self.is_main:
            return

        self.results_manager.start_timing('GP Training')

        gp_size_before = self.gp.train_x.shape[0]
        self.gp.update(new_pts_u, new_vals)
        gp_size_after = self.gp.train_x.shape[0]
        actual_points_added = gp_size_after - gp_size_before
        log.info(
            f"Added {actual_points_added} new points to GP "
            f"(size before: {gp_size_before}, size after: {gp_size_after})"
        )
        self.n_points_since_last_fit += actual_points_added

        if self._refit_gp_hyperparameters() and hasattr(self.gp, 'kernel'):
            ls_str = "[" + ", ".join(f"{float(l):.4f}" for l in self.gp.kernel.lengthscales) + "]"
            log.info(
                f"Updated GP hyperparameters: "
                f"lengthscales={ls_str}, "
                f"kernel_variance={float(self.gp.kernel.kernel_variance):.4e}, "
                f"noise={float(self.gp.kernel.noise):.4e}"
            )

        self.results_manager.end_timing('GP Training')

        if hasattr(self.gp, 'kernel'):
            lengthscales = list(self.gp.kernel.lengthscales)
            kernel_variance = float(self.gp.kernel.kernel_variance)
            gp_noise = float(self.gp.kernel.noise)
            self.results_manager.update_gp_hyperparams(step, lengthscales, kernel_variance, gp_noise)

        if isinstance(self.gp, ClassifierMixin):
            self.results_manager.start_timing('Classifier Training')
            self.gp.train_classifier()
            self.results_manager.end_timing('Classifier Training')

    def get_next_batch(self, acq_kwargs, n_batch, n_restarts, maxiter, early_stop_patience, step, verbose=True):
        """Get the next batch of points using the acquisition function, and track acquisition values."""
        if not self.is_main:
            return None, None

        self.results_manager.start_timing('Acquisition Optimization')
        log.info(f"Optimizing acquisition function '{self.acquisition.name}' to get next {n_batch} points")
        new_pts_u, acq_vals = self.acquisition.get_next_batch(
            gp=self.gp,
            n_batch=n_batch,
            acq_kwargs=acq_kwargs,
            n_restarts=n_restarts,
            maxiter=maxiter,
            early_stop_patience=early_stop_patience,
        )
        self.results_manager.end_timing('Acquisition Optimization')

        acq_val = float(np.mean(acq_vals))
        if verbose:
            log.info(f"Mean acquisition value {acq_val:.4e} at new points")
        self.results_manager.update_acquisition(step, acq_val, self.acquisition.name)

        return new_pts_u, acq_vals

    def evaluate_likelihood(self, new_pts_u, step, verbose=True):
        """Evaluate the true log-likelihood at unit-cube points ``new_pts_u`` via the pool."""
        if not self.is_main:
            return None

        new_pts_u = jnp.atleast_2d(new_pts_u)
        new_pts = scale_from_unit(new_pts_u, self.loglikelihood.param_bounds)

        self.results_manager.start_timing('True Objective Evaluations')
        new_vals = self.pool.run_map_objective(self.loglikelihood, new_pts)
        new_vals = jnp.atleast_2d(new_vals).reshape(-1, 1)
        self.results_manager.end_timing('True Objective Evaluations')

        best_new_idx = int(np.argmax(new_vals))
        best_new_val = float(np.max(new_vals))
        best_new_pt = new_pts[best_new_idx]
        if best_new_val > self.best_f:
            self.best_f = best_new_val
            self.best_pt = best_new_pt
            self.best = {
                name: f"{float(val):.6f}"
                for name, val in zip(self.loglikelihood.param_list, np.asarray(self.best_pt).flatten())
            }
            self.best_pt_iteration = step

        self.total_objective_evals += len(new_pts)
        log.info(
            f"Evaluated objective at {len(new_pts)} new points "
            f"(total objective evals: {self.total_objective_evals})"
        )
        for k, new_pt in enumerate(new_pts):
            new_pt_vals = {
                name: f"{float(val):.4f}"
                for name, val in zip(self.loglikelihood.param_list, np.asarray(new_pt).flatten())
            }
            log.debug(f"New point {new_pt_vals}, {k + 1}/{len(new_pts)}")
            predicted_val = self.gp.predict_mean(new_pts_u[k])
            log.debug(
                f"Objective function value = {new_vals[k].item():.4f}, "
                f"GP predicted value = {predicted_val.item():.4f}"
            )

        return new_vals

    def check_max_evals_and_gpsize(self, current_evals: int) -> bool:
        """Check if maximum objective evaluations or maximum GP size has been reached."""
        if not self.is_main:
            return False

        if current_evals >= self.max_evals:
            self.termination_reason = "Maximum evaluations reached"
            self.results_dict['termination_reason'] = self.termination_reason
            return True
        if self.gp.train_x.shape[0] >= self.max_gp_size:
            self.termination_reason = "Maximum GP size reached"
            self.results_dict['termination_reason'] = self.termination_reason
            return True

        return False

    def finalise_results(self):
        """Compile metadata and finalize results via ``BOBEResults``."""
        if not self.is_main:
            return

        gp_info = {
            'gp_training_set_size': int(self.gp.train_x.shape[0]),
            'gp_final_best_loglike': float(self.best_f),
            'total_objective_evals': int(self.total_objective_evals),
        }

        if isinstance(self.gp, ClassifierMixin):
            gp_info.update({
                'classifier_used': bool(self.gp.use_clf),
                'classifier_type': str(self.gp.clf_type),
                'classifier_training_set_size': int(self.gp.clf_data_size),
                'classifier_use_threshold': int(self.gp.clf_use_size),
                'classifier_probability_threshold': float(self.gp.probability_threshold),
            })
        else:
            gp_info.update({
                'classifier_used': False,
                'classifier_type': None,
                'classifier_training_set_size': 0,
            })

        kernel_rotation = getattr(self.gp, 'kernel_rotation', None)
        if kernel_rotation is not None:
            gp_info.update({
                'kernel_rotation_update_count': kernel_rotation.update_count,
                'last_kernel_rotation_iteration': kernel_rotation.last_update_ii,
                'last_kernel_rotation_acq_val': kernel_rotation.last_update_acq_val,
                'last_kernel_rotation_kl': kernel_rotation.last_update_kl,
            })
        else:
            gp_info.update({
                'kernel_rotation_update_count': 0,
                'last_kernel_rotation_iteration': None,
                'last_kernel_rotation_acq_val': None,
                'last_kernel_rotation_kl': None,
            })

        samples_dict = self.samples_dict or {}
        log.debug(f"Samples dict keys: {samples_dict.keys()}")
        logz_dict = self.results_dict.get('logz', {})

        if not logz_dict:
            log.warning("No logz information found, nested sampling has not been run yet.")

        self.results_manager.finalize(
            samples_dict=samples_dict,
            logz_dict=logz_dict,
            converged=self.converged,
            termination_reason=self.termination_reason,
            gp_info=gp_info,
            best_point=self.best_pt,
            best_loglike=self.best_f,
            best_iteration=self.best_pt_iteration,
        )

        self.results_dict = {
            'gp': self.gp,
            'likelihood': self.loglikelihood,
            'results_manager': self.results_manager,
            'best_val': self.best_f,
            'best_pt': self.best_pt,
            'logz': logz_dict,
            'termination_reason': self.termination_reason,
            'samples': samples_dict,
        }

    def check_convergence_ei(self, step: int, acq_val) -> bool:
        """Check convergence for EI/LogEI based on the acquisition function value."""
        if not self.is_main:
            return False

        if self.acquisition.name.lower() == 'ei':
            acq_val = np.log(acq_val + 1e-100)

        converged = bool(np.all(acq_val < self.ei_goal_log))

        if converged:
            self.convergence_counter += 1
            if self.convergence_counter >= self.convergence_n_iters:
                log.info(
                    f"Convergence for {self.acquisition.name} achieved after "
                    f"{self.convergence_n_iters} successive iterations"
                )
                return True
            log.info(
                f"{self.acquisition.name} convergence iteration "
                f"{self.convergence_counter}/{self.convergence_n_iters}"
            )
            return False
        else:
            self.convergence_counter = 0
            return False

    def check_convergence_logz(self, step: int, logz_dict: Dict[str, float], equal_samples, equal_logl) -> bool:
        """
        Check convergence after nested sampling using both ``logz_std`` and
        symmetric Gaussian KL divergence between consecutive NS posteriors.

        Both ``logz_dict['std'] < logz_threshold`` and
        ``kl_sym < kl_convergence_threshold`` must hold for
        ``convergence_n_iters`` successive checks.
        """
        if not self.is_main:
            return False

        delta = float(logz_dict['std'])
        equal_samples = np.asarray(equal_samples)

        kl_sym = np.inf
        if self.prev_samples is not None:
            try:
                prev_x = self.prev_samples['x']
                mu1, cov1 = np.mean(prev_x, axis=0), np.atleast_2d(np.cov(prev_x, rowvar=False))
                mu2, cov2 = np.mean(equal_samples, axis=0), np.atleast_2d(np.cov(equal_samples, rowvar=False))
                kl = kl_divergence_gaussian(mu1, cov1, mu2, cov2)
                kl_sym = float(kl.get('symmetric', np.inf))
                self.results_manager.update_kl_divergences(iteration=step, successive_kl=kl)
            except Exception as e:
                log.warning(f"KL computation failed: {e}")

        self.prev_samples = {'x': equal_samples, 'logl': equal_logl}

        log.info(
            f"Convergence check: logz_std={delta:.4e} < thresh={self.logz_threshold:.4e} "
            f"= {delta < self.logz_threshold}, "
            f"KL_sym={kl_sym:.4f} (thresh={self.kl_convergence_threshold:.4f})"
        )

        if delta < self.logz_threshold and kl_sym < self.kl_convergence_threshold:
            self.convergence_counter += 1
        else:
            self.convergence_counter = 0

        converged = self.convergence_counter >= self.convergence_n_iters
        self.results_manager.update_convergence(
            iteration=step,
            logz_dict=logz_dict,
            converged=converged,
            threshold=self.logz_threshold,
            delta=delta,
        )
        if converged:
            log.info(f"Convergence achieved after {self.convergence_n_iters} successive iterations")
        else:
            log.info(f"Convergence iteration {self.convergence_counter}/{self.convergence_n_iters}")
        return converged

    # ============================================================================
    # MAIN RUN METHODS
    # ============================================================================

    def run(
        self,
        acq: str = 'wipstd',
        min_evals: Optional[int] = None,
        max_evals: Optional[int] = None,
        max_gp_size: Optional[int] = None,
        logz_threshold: Optional[float] = None,
        convergence_n_iters: int = 1,
        kl_convergence_threshold: float = 0.05,
        ei_goal: float = 1e-10,
        do_final_ns: bool = False,
        fit_n_points: Optional[int] = None,
        batch_size: Optional[int] = None,
        ns_n_points: Optional[int] = None,
        num_hmc_warmup: Optional[int] = None,
        num_hmc_samples: Optional[int] = None,
        mc_points_size: Optional[int] = None,
        thinning: int = 4,
        num_chains: Optional[int] = None,
        mc_points_method: str = 'NUTS',
        zeta_ei: float = 0.0,
        logz_err_method: str = 'mean_std',
        close_pool: bool = True,
    ):
        """
        Run the Bayesian Optimization loop.

        Parameters
        ----------
        acq : str, optional
            Acquisition function to use: ``'WIPStd'``, ``'WIPV'``, ``'EI'``, or ``'LogEI'``.
            For multi-phase workflows (e.g. EI warm-up followed by WIPStd), call ``run()``
            multiple times with ``close_pool=False`` on all phases except the last.
        min_evals : int, optional
            Minimum number of objective evaluations before checking convergence.
        max_evals : int, optional
            Maximum number of objective evaluations.
        max_gp_size : int, optional
            Maximum number of points stored in the GP.
        logz_threshold : float, optional
            Convergence threshold on ``logz_std`` (WIPV/WIPStd).
        convergence_n_iters : int, optional
            Number of successive checks meeting thresholds for convergence. Default is 1.
        kl_convergence_threshold : float, optional
            Maximum symmetric Gaussian KL divergence between consecutive nested-sampling
            posteriors for convergence. Default is 0.05.
        ei_goal : float, optional
            Goal value for EI/LogEI acquisition convergence. Default is 1e-10.
        do_final_ns : bool, optional
            Whether to run final nested sampling if not yet converged at budget limit. Default is False.
        fit_n_points : int, optional
            Refit GP hyperparameters after adding this many points to the GP.
        batch_size : int, optional
            Batch size for WIPV/WIPStd acquisition.
        ns_n_points : int, optional
            Minimum objective evaluations between periodic nested sampling runs.
        num_hmc_warmup : int, optional
            Number of NUTS warmup steps.
        num_hmc_samples : int, optional
            Number of NUTS samples per chain.
        mc_points_size : int, optional
            Number of MC points subsampled for WIPV/WIPStd evaluation.
        thinning : int, optional
            Thinning factor for NUTS samples. Default is 4.
        num_chains : int, optional
            Number of NUTS chains.
        mc_points_method : str, optional
            Method for generating MC points: ``'NUTS'``, ``'NS'``, or ``'uniform'``. Default is ``'NUTS'``.
        zeta_ei : float, optional
            Exploration parameter for EI/LogEI acquisition. Default is 0.0.
        logz_err_method : {'mean_std', 'integral'}, optional
            GP-induced logZ uncertainty estimator passed to :func:`nested_sampling_Dy`.
            Default is ``'mean_std'``.
        close_pool : bool, optional
            Whether to shut down the MPI worker pool when this ``run()`` finishes
            (default True). Set to False when chaining multiple ``run()`` calls on the
            same ``BOBE`` instance; call ``bobe.close()`` (or leave ``close_pool=True``
            on the final call) to release workers.
        """
        if not self.is_main:
            return None

        if not isinstance(acq, str):
            raise TypeError(
                f"acq must be a string (use sequential run(..., close_pool=False) calls for multi-phase runs), "
                f"got {type(acq).__name__}"
            )
        acq_key = acq.lower()
        acqs_funcs_available = list(_acq_funcs.keys())
        if acq_key not in acqs_funcs_available:
            raise ValueError(f"Invalid acquisition function '{acq}'. Valid options are: {acqs_funcs_available}")

        if logz_err_method not in ('mean_std', 'integral'):
            raise ValueError(
                f"logz_err_method must be 'mean_std' or 'integral', got {logz_err_method!r}"
            )

        dim_defaults = get_dimension_based_defaults(self.ndim)

        min_evals = min_evals if min_evals is not None else dim_defaults['min_evals']
        max_evals = max_evals if max_evals is not None else dim_defaults['max_evals']
        max_gp_size = max_gp_size if max_gp_size is not None else dim_defaults['max_gp_size']
        fit_n_points = fit_n_points if fit_n_points is not None else dim_defaults['fit_n_points']
        batch_size = batch_size if batch_size is not None else dim_defaults['batch_size']
        ns_n_points = ns_n_points if ns_n_points is not None else dim_defaults['ns_n_points']
        num_hmc_warmup = num_hmc_warmup if num_hmc_warmup is not None else dim_defaults['num_hmc_warmup']
        num_hmc_samples = num_hmc_samples if num_hmc_samples is not None else dim_defaults['num_hmc_samples']
        mc_points_size = mc_points_size if mc_points_size is not None else dim_defaults['mc_points_size']
        num_chains = num_chains if num_chains is not None else dim_defaults['num_chains']
        logz_threshold = logz_threshold if logz_threshold is not None else dim_defaults['logz_threshold']

        kernel_rotation = getattr(self.gp, 'kernel_rotation', None)
        if kernel_rotation is not None and kernel_rotation.adaptive:
            if mc_points_method.lower() == 'uniform':
                raise ValueError(
                    "Adaptive kernel rotation requires posterior MC samples; "
                    "mc_points_method='uniform' is not supported"
                )
            if kernel_rotation.acq_threshold is None:
                kernel_rotation.acq_threshold = dim_defaults['transform_acq_threshold']

        self.min_evals = min_evals
        self.max_evals = max_evals
        self.max_gp_size = max_gp_size
        self.logz_threshold = logz_threshold
        self.kl_convergence_threshold = kl_convergence_threshold
        self.logz_err_method = logz_err_method

        self.samples_dict = {}
        self.results_dict = {}

        # Check if already converged when resuming (only skip when new threshold is not stricter)
        if self.prev_converged and self.prev_convergence_threshold is not None:
            if logz_threshold >= self.prev_convergence_threshold:
                log.info(
                    f"Previous run converged with threshold={self.prev_convergence_threshold:.6f}; "
                    f"new threshold={logz_threshold:.6f} is not stricter — skipping BO loop."
                )
                self.converged = True
                self.termination_reason = "Already converged in previous run"

                if self.results_manager.convergence_history:
                    last_conv = self.results_manager.convergence_history[-1]
                    self.results_dict['logz'] = last_conv.logz_dict.copy()

                if self.results_manager.final_samples is not None and len(self.results_manager.final_samples) > 0:
                    self.samples_dict = {
                        'x': self.results_manager.final_samples,
                        'weights': self.results_manager.final_weights,
                        'logl': self.results_manager.final_loglikes,
                    }
                    log.info(f"Restored {len(self.samples_dict['x'])} samples from previous run")
                else:
                    self.samples_dict = {}
                    log.warning("No samples found in previous run")

                # Leave existing chain and summary files untouched.
                self.results_dict = {
                    'gp': self.gp,
                    'likelihood': self.loglikelihood,
                    'results_manager': self.results_manager,
                    'best_val': self.best_f,
                    'best_pt': self.best_pt,
                    'logz': self.results_dict.get('logz', {}),
                    'termination_reason': self.termination_reason,
                    'samples': self.samples_dict,
                }
                if close_pool:
                    self.pool.close()
                return self.results_dict
            else:
                log.info(
                    f"Previous run converged with threshold={self.prev_convergence_threshold:.6f}; "
                    f"new threshold={logz_threshold:.6f} is stricter — continuing optimization."
                )
                self.converged = False
                self.convergence_counter = 0

        # Log run settings
        log.info("Using run settings:")
        log.info(f"min_evals = {min_evals}, max_evals = {max_evals}, max_gp_size = {max_gp_size}")
        if acq_key in ['wipv', 'wipstd']:
            acq_info = (
                f"logz_threshold = {logz_threshold:.4f}, "
                f"kl_convergence_threshold = {kl_convergence_threshold:.4f}, "
                f"mc_points_size = {mc_points_size}"
            )
        else:
            acq_info = f"ei_goal = {ei_goal:.4e}"
        log.info(f"convergence_n_iters = {convergence_n_iters}, acq = {acq}, {acq_info}")
        log.info(f"fit_n_points = {fit_n_points}, batch_size = {batch_size}, ns_n_points = {ns_n_points}")

        if kernel_rotation is not None and kernel_rotation.adaptive:
            log.info(
                "Adaptive kernel rotation enabled: "
                f"dims={np.asarray(kernel_rotation.update_dims).tolist()}, "
                f"acq_threshold={kernel_rotation.acq_threshold}, "
                f"update_step={kernel_rotation.update_step}, "
                f"max_updates={kernel_rotation.max_updates}, "
                f"kl_threshold={kernel_rotation.kl_threshold}"
            )
            if kernel_rotation.update_count > 0:
                log.info(
                    "Resuming kernel rotation state: "
                    f"count={kernel_rotation.update_count}, "
                    f"last_ii={kernel_rotation.last_update_ii}, "
                    f"last_acq_val={kernel_rotation.last_update_acq_val}"
                )

        self.convergence_n_iters = convergence_n_iters
        self.ei_goal_log = np.log(ei_goal)
        self.do_final_ns = do_final_ns

        self.fit_n_points = fit_n_points
        self.ns_n_points = ns_n_points
        self.batch_size = batch_size

        self.n_points_since_last_fit = 0
        self.n_points_since_last_ns = 0
        self.num_hmc_warmup = num_hmc_warmup
        self.num_hmc_samples = num_hmc_samples
        self.mc_points_size = mc_points_size
        self.hmc_thinning = thinning
        self.hmc_num_chains = num_chains
        self.mc_points_method = mc_points_method
        self.zeta_ei = zeta_ei

        # Adjust batch_size for MPI load balancing (Public uses static dispatch by default where all ranks evaluate)
        if self.is_mpi:
            n_processes = self.pool.size
            original_batch = self.batch_size
            if self.batch_size % n_processes != 0:
                self.batch_size = (self.batch_size // n_processes) * n_processes
                if self.batch_size < n_processes:
                    self.batch_size = n_processes
                log.info(
                    f"Adjusted batch_size from {original_batch} to {self.batch_size} "
                    f"(multiple of {n_processes} processes)"
                )

        # Reset convergence state for this run phase; preserve restored convergence_counter
        # unless threshold was tightened or we are switching into an EI phase
        if (
            acq_key not in ('wipv', 'wipstd')
            or (self.prev_convergence_threshold is not None and logz_threshold < self.prev_convergence_threshold)
        ):
            self.convergence_counter = 0
        self.converged = False
        self.prev_converged = False
        self.termination_reason = "Max evaluation budget reached"

        self.results_manager.settings.update({
            'acq': acq,
            'min_evals': min_evals,
            'max_evals': max_evals,
            'max_gp_size': max_gp_size,
            'logz_threshold': logz_threshold,
            'kl_convergence_threshold': kl_convergence_threshold,
            'convergence_n_iters': convergence_n_iters,
            'ei_goal': ei_goal,
            'do_final_ns': do_final_ns,
            'fit_n_points': fit_n_points,
            'batch_size': batch_size,
            'ns_n_points': ns_n_points,
            'num_hmc_warmup': num_hmc_warmup,
            'num_hmc_samples': num_hmc_samples,
            'mc_points_size': mc_points_size,
            'thinning': thinning,
            'num_chains': num_chains,
            'mc_points_method': mc_points_method,
            'zeta_ei': zeta_ei,
            'logz_err_method': logz_err_method,
        })

        self.samples_dict = {}
        self.results_dict = {}

        self.current_iteration = self.start_iteration
        self.acquisition = _acq_funcs[acq_key](optimizer=self.optimizer)
        if acq_key == 'wipv':
            self.run_WIPV(ii=self.current_iteration)
        elif acq_key == 'wipstd':
            self.run_WIPStd(ii=self.current_iteration)
        else:
            self.run_EI(ii=self.current_iteration)

        # Advance start_iteration so sequential run(..., close_pool=False) phases continue cleanly
        self.start_iteration = self.current_iteration

        log.info(
            f"Final best point {self.best} with value = {self.best_f:.6f}, "
            f"found at iteration {self.best_pt_iteration}"
        )
        log.info(f"Sampling stopped: {self.termination_reason}")
        log.info(f"Final GP training set size: {self.gp.train_x.shape[0]}, max size: {self.max_gp_size}")

        self.finalise_results()
        # Persist the final convergence decision and posterior samples in the unified checkpoint
        self._save_checkpoint()

        if close_pool:
            self.pool.close()

        return self.results_dict

    def close(self):
        """
        Shut down the MPI worker pool, signalling all workers to exit.

        Safe to call multiple times and in serial (non-MPI) mode. Use this when
        ``run()`` was called with ``close_pool=False`` and no further phases will run.
        """
        self.pool.close()

    def run_EI(self, ii: int = 0):
        """Run the optimization loop for EI/LogEI acquisition functions."""
        if not self.is_main:
            return

        current_evals = self.total_objective_evals
        log.info(f"Starting iteration {ii}")
        converged = False

        while not converged:
            ii += 1
            verbose = True

            if verbose:
                log.info(
                    f"Iteration {ii} of {self.acquisition.name}, "
                    f"objective evals {current_evals}/{self.max_evals}"
                )

            acq_kwargs = {
                'zeta': self.zeta_ei,
                'best_y': max(self.gp.train_y.flatten()) if self.gp.train_y.size > 0 else 0.0,
            }
            n_batch = 1
            new_pts_u, acq_vals = self.get_next_batch(
                acq_kwargs,
                n_batch=n_batch,
                n_restarts=50,
                maxiter=1000,
                early_stop_patience=50,
                step=ii,
                verbose=verbose,
            )
            new_pts_u = jnp.atleast_2d(new_pts_u)

            new_vals = self.evaluate_likelihood(new_pts_u, ii, verbose=verbose)
            current_evals = self.total_objective_evals

            self.update_gp(new_pts_u, new_vals, step=ii, verbose=verbose)

            self.results_manager.update_best_loglike(ii, self.best_f)
            if verbose:
                log.info(
                    f"Current best point {self.best} with value = {self.best_f:.6f}, "
                    f"found at iteration {self.best_pt_iteration}"
                )

            converged = self.check_convergence_ei(ii, acq_vals)
            self.current_iteration = ii

            if ii % self.save_step == 0:
                self._save_checkpoint()

            if converged:
                self.converged = True
                self.termination_reason = f"{self.acquisition.name.upper()} goal reached"
                self.results_dict['termination_reason'] = self.termination_reason
                break

            self.pool.clear_jax_caches()

            if self.check_max_evals_and_gpsize(current_evals):
                break

        self.current_iteration = ii

    def run_weighted_integrated_posterior(self, acq_func_class, ii: int = 0):
        """
        Run the optimization loop for Weighted Integrated Posterior acquisition
        functions (``WIPV`` or ``WIPStd``).
        """
        if not self.is_main:
            return

        self.acquisition = acq_func_class(optimizer=self.optimizer)
        acq_name = self.acquisition.name

        current_evals = self.total_objective_evals
        self.results_manager.start_timing('MCMC Sampling')
        self.mc_samples = get_mc_samples(
            self.gp,
            warmup_steps=self.num_hmc_warmup,
            num_samples=self.num_hmc_samples,
            thinning=self.hmc_thinning,
            num_chains=self.hmc_num_chains,
            np_rng=self.np_rng,
            rng_key=get_new_jax_key(),
            method=self.mc_points_method,
            param_bounds=self.loglikelihood.param_bounds,
        )
        self.results_manager.end_timing('MCMC Sampling')
        self.ns_samples = None
        logz_keys = ['mean', 'upper', 'lower', 'dlogz_sampler']
        _ns_evals_last = 0

        while not self.converged:
            ii += 1
            verbose = True

            if verbose:
                log.info(f"Iteration {ii} of {acq_name}, objective evals {current_evals}/{self.max_evals}")

            acq_kwargs = {'mc_samples': self.mc_samples, 'mc_points_size': self.mc_points_size}
            new_pts_u, acq_vals = self.get_next_batch(
                acq_kwargs,
                n_batch=self.batch_size,
                n_restarts=1,
                maxiter=100,
                early_stop_patience=10,
                step=ii,
                verbose=verbose,
            )
            mean_acq_val = float(np.mean(acq_vals))
            new_pts_u = jnp.atleast_2d(new_pts_u)
            new_vals = self.evaluate_likelihood(new_pts_u, ii, verbose=verbose)
            current_evals = self.total_objective_evals

            self.update_gp(new_pts_u, new_vals, step=ii)
            self.results_manager.update_best_loglike(ii, self.best_f)

            # Update MC samples every iteration
            self.results_manager.start_timing('MCMC Sampling')
            self.mc_samples = get_mc_samples(
                self.gp,
                warmup_steps=self.num_hmc_warmup,
                num_samples=self.num_hmc_samples,
                thinning=self.hmc_thinning,
                num_chains=self.hmc_num_chains,
                method=self.mc_points_method,
                np_rng=self.np_rng,
                rng_key=get_new_jax_key(),
                param_bounds=self.loglikelihood.param_bounds,
            )
            self.results_manager.end_timing('MCMC Sampling')

            if verbose:
                log.info(
                    f"Current best point {self.best} with value = {self.best_f:.6f}, "
                    f"found at iteration {self.best_pt_iteration}"
                )

            self.current_iteration = ii

            if ii % self.save_step == 0:
                self._save_checkpoint()

            # Adaptive kernel rotation check
            if getattr(self.gp, 'has_adaptive_kernel_rotation', False):
                did_update = self.gp.try_update_kernel_rotation(
                    mc_samples=self.mc_samples,
                    acq_val=mean_acq_val,
                    step=ii,
                    current_evals=current_evals,
                    min_evals=self.min_evals,
                )
                if did_update:
                    self.results_manager.start_timing('GP Training')
                    self._refit_gp_hyperparameters(
                        force=True,
                        reason="after adaptive kernel rotation",
                    )
                    self.results_manager.end_timing('GP Training')
                    self._save_checkpoint()

            # Periodic nested sampling for convergence checking
            if (
                current_evals >= self.min_evals
                and (mean_acq_val <= self.logz_threshold)
                and (current_evals - _ns_evals_last >= self.ns_n_points)
            ):
                _ns_evals_last = current_evals
                log.info(f"Periodic nested sampling triggered at eval {current_evals}")
                self.results_manager.start_timing('Nested Sampling')
                ns_samples, logz_dict, ns_success = nested_sampling_Dy(
                    mode='conv',
                    gp=self.gp,
                    ndim=self.ndim,
                    dlogz=0.025,
                    maxcall=int(2e6),
                    rng=self.np_rng,
                    param_bounds=self.loglikelihood.param_bounds,
                    logz_err_method=self.logz_err_method,
                )
                self.results_manager.end_timing('Nested Sampling')
                if ns_success:
                    self.ns_samples = ns_samples
                    logz_str = ", ".join(f"{k}={logz_dict[k]:.4f}" for k in logz_keys if k in logz_dict)
                    log.info(f"Periodic NS LogZ: {logz_str}")
                    self.results_dict['logz'] = logz_dict
                    # Replace mc_samples with equal-weighted unit-cube NS samples for next acquisition step
                    self.mc_samples = {
                        'x': self.ns_samples['eq_u'],
                        'logp': self.ns_samples['eq_logl'],
                        'method': 'nested',
                    }
                    self.converged = self.check_convergence_logz(
                        ii, logz_dict, self.ns_samples['eq_x'], self.ns_samples['eq_logl']
                    )
                    if self.converged:
                        self.termination_reason = "LogZ converged (periodic NS)"
                        self.results_dict['termination_reason'] = self.termination_reason
                        break

            self.pool.clear_jax_caches()

            if self.check_max_evals_and_gpsize(current_evals):
                break

        self.current_iteration = ii

        # Final nested sampling if not yet converged and do_final_ns is True
        if self.do_final_ns and not self.converged:
            self.results_manager.start_timing('GP Training')
            self._refit_gp_hyperparameters(
                force=True,
                reason="before final nested sampling",
            )
            self.results_manager.end_timing('GP Training')

            log.info("Final Nested Sampling")
            self.results_manager.start_timing('Nested Sampling')
            ns_samples, logz_dict, ns_success = nested_sampling_Dy(
                mode='convergence',
                gp=self.gp,
                ndim=self.ndim,
                maxcall=int(5e6),
                dynamic=True,
                dlogz=0.01,
                rng=self.np_rng,
                param_bounds=self.loglikelihood.param_bounds,
                logz_err_method=self.logz_err_method,
            )
            self.results_manager.end_timing('Nested Sampling')
            logz_str = ", ".join(f"{k}={logz_dict[k]:.4f}" for k in logz_keys if k in logz_dict)
            log.info(f"Final LogZ: {logz_str}")
            if ns_success:
                self.ns_samples = ns_samples
                log.info("Using nested sampling results")
                self.converged = self.check_convergence_logz(
                    ii + 1, logz_dict, self.ns_samples['eq_x'], self.ns_samples['eq_logl']
                )
                self.results_dict['logz'] = logz_dict
                if self.converged:
                    self.termination_reason = "LogZ converged"
                    self.results_dict['termination_reason'] = self.termination_reason
        else:
            log.info("Skipping final nested sampling")

        if self.ns_samples is None:
            log.info(
                "No nested sampling results found or nested sampling unsuccessful, "
                "MC samples from HMC/MCMC will be used instead."
            )
            self.results_manager.start_timing('MCMC Sampling')
            mc_samples = get_mc_samples(
                self.gp,
                warmup_steps=self.num_hmc_warmup,
                num_samples=8 * self.num_hmc_samples,
                thinning=self.hmc_thinning,
                method="NUTS",
                np_rng=self.np_rng,
                rng_key=get_new_jax_key(),
            )
            self.results_manager.end_timing('MCMC Sampling')
            samples = np.asarray(scale_from_unit(mc_samples['x'], self.loglikelihood.param_bounds))
            weights = mc_samples['weights'] if 'weights' in mc_samples else np.ones(mc_samples['x'].shape[0])
            loglikes = mc_samples['logp']
        else:
            log.info("Using nested sampling results for final samples and logZ estimation.")
            samples = self.ns_samples['x']
            weights = self.ns_samples['weights']
            loglikes = self.ns_samples['logl']

        self.samples_dict = {
            'x': samples,
            'weights': weights,
            'logl': loglikes,
        }

    def run_WIPStd(self, ii: int = 0):
        """Run optimization loop for WIPStd acquisition function."""
        return self.run_weighted_integrated_posterior(WIPStd, ii)

    def run_WIPV(self, ii: int = 0):
        """Run optimization loop for WIPV acquisition function."""
        return self.run_weighted_integrated_posterior(WIPV, ii)
