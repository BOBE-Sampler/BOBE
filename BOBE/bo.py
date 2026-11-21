import os
import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Union, Tuple, Dict, Any, Callable, List
from .surrogate import Surrogate, ClassifierMixin
from .likelihood import Likelihood, CobayaLikelihood
from .utils.core import scale_from_unit, scale_to_unit, resample_equal, kl_divergence_gaussian
from .utils.seed import set_global_seed, get_numpy_rng, get_new_jax_key
from .samplers import nested_sampling_Dy, sample_GP_NUTS
from .utils.log import get_logger, update_verbosity
from .utils.results import BOBEResults
from .acquisition import *
from .pool import MPI_Pool

log = get_logger("bo")
log.info(f'JAX using {jax.device_count()} devices.')

_acq_funcs = {"wipv": WIPV, "ei": EI, "logei": LogEI, 'wipstd': WIPStd}

def get_dimension_based_defaults(ndim: int):
    """
    Compute reasonable default values for run() parameters based on problem dimension.
    
    This method provides dimension-scaled defaults for parameters that should adapt
    to the complexity of the problem. Users can override these by providing explicit
    values to the run() method.
    
    Returns
    -------
    dict
        Dictionary of default parameter values keyed by parameter name.
    """
    
    defaults = {
        'min_evals': 8 * ndim,  # scales linearly with dimension
        'max_evals': 200 * ndim,  # more evals for higher dimensions
        'max_gp_size': min(2100, 160 * ndim),  # larger GP for higher dimensions
        'batch_size': 2 if ndim <=6 else min(8,int(2*(ndim/6))),  # 2-8 depending on dimension
        'ns_n_points': min(50, 2*ndim),  # nested sampling frequency, less for higher dimensions
        'num_hmc_warmup': 256 if ndim <= 6 else 512,  # more warmup for higher dimensions
        'num_hmc_samples': min(5000, max(512,int(4096*(ndim/20)))),  # more samples for higher dimensions, capped at 5000
        'mc_points_size': min(512, 32*ndim),  # more MC points for higher dimensions
        'num_chains': min(6, max(3,jax.device_count())),  # 3-6 chains depending on available devices
        'fit_n_points': min(50, 2*ndim),  # refit less often for higher dimensions
        'logz_threshold': 0.01 + 0.01*(ndim/6) if ndim<=6 else min(1.,0.1 + 0.1*(ndim/6)**2)  # looser threshold for higher dimensions
    }
    
    return defaults

class BOBE:

    def __init__(self,
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
                 save_step: int = 5,
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
            - A callable function (requires param_list and param_bounds)
            - A string path to Cobaya YAML file (automatically creates CobayaLikelihood)
            - A dict with Cobaya info (automatically creates CobayaLikelihood)
            - A Likelihood instance (param_list, param_bounds ignored)
        param_list : list of str, optional
            Names of parameters. Required if loglikelihood is a callable.
            Ignored for Cobaya likelihoods (extracted from YAML/dict).
        param_bounds : array-like, optional
            Parameter bounds, shape (2, ndim). Required if loglikelihood is a callable.
            Ignored for Cobaya likelihoods (extracted from priors).
        param_labels : list of str, optional
            LaTeX labels for parameters. If not provided, uses param_list.
            Ignored for Cobaya likelihoods (extracted from YAML/dict).
        likelihood_name : str, optional
            Name for the likelihood (used in output files). If not provided, uses 'likelihood'
            for callables or 'cobaya_model' for Cobaya likelihoods.
        confidence_for_unbounded : float, optional
            Confidence level for unbounded Cobaya priors. Default is 0.9999995.
            Only used when loglikelihood is a Cobaya YAML file or dict.
        gp : Surrogate, optional
            Surrogate model instance (e.g. ``GP(kernel=RBFKernel(...))``).
            Required when ``resume=False``. Ignored when resuming from
            ``resume_file`` (the saved surrogate is loaded instead).
        n_cobaya_init : int, optional
            Number of initial points from Cobaya reference distribution.
            Only used for CobayaLikelihood instances. Default is 4.
        n_sobol_init : int, optional
            Number of initial Sobol quasi-random points. Default is 4.
        init_train_x : array-like, optional
            User-provided initial training points in parameter space, shape (n_points, ndim).
            If provided, these will be added to the initial GP training set. Default is None.
        init_train_y : array-like, optional
            User-provided initial training values (log-likelihood), shape (n_points, 1) or (n_points,).
            Must be provided if init_train_x is given. Default is None.
        resume : bool, optional
            If True, resume from a previous run. Default is False.
        resume_file : str, optional
            Path to resume from (base path of saved GP file). Default is None.
        save_dir : str, optional
            Directory for saving results. Default is '.'.
        save : bool, optional
            Whether to save results periodically. Default is True.
        save_step : int, optional
            Save results every save_step iterations. Default is 5.
        optimizer : str, optional
            Optimizer for acquisition function optimization. Options: 'scipy', 'optax'.
            Default is 'scipy'.
        minus_inf : float, optional
            Value representing negative infinity for failed likelihood evaluations.
            Default is -1e10.
        seed : int, optional
            Random seed for reproducibility. Default is None.
        verbosity : str, optional
            Logging verbosity level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'. Default is 'INFO'.
        dynamic_dispatch : bool, optional
            If False (default), use static (round-robin) task distribution across MPI ranks.
            Static dispatch is fully reproducible across runs at a fixed ``nprocs`` and seed.
            If True, use dynamic dispatch (first-available worker).

        Notes
        -----
        MPI parallelization is handled automatically and transparently. Users do not
        need to manage MPI processes explicitly in their scripts. When running with
        MPI (e.g., `mpirun -n 4 python script.py`), worker processes automatically
        participate in parallel likelihood evaluations and GP hyperparameter optimization
        via the `MPI_Pool` class, while only the main process (rank 0) runs the optimization
        loop and manages results. Worker processes enter a waiting loop after initialization
        and process tasks dispatched by the main process.
        """
        if gp is not None and not isinstance(gp, Surrogate):
            raise TypeError(
                f"gp must be a Surrogate instance or None, got {type(gp).__name__}"
            )
        if not (resume and resume_file is not None) and gp is None:
            raise ValueError(
                "A surrogate instance (gp) must be provided when not resuming from a checkpoint."
            )

        # Update logging verbosity if different from default
        update_verbosity(verbosity=verbosity)

        # Initialize MPI pool
        self.pool = MPI_Pool(dynamic_dispatch=dynamic_dispatch)
        self.is_main = self.pool.is_main_process
        self.is_mpi = self.pool.is_mpi

        # Convert to Likelihood instance and store for all processes
        self.loglikelihood = self._prepare_likelihood(
            loglikelihood, param_list, param_bounds, param_labels,
            likelihood_name, confidence_for_unbounded, minus_inf
        )
        self.ndim = len(self.loglikelihood.param_list)

        if not self.is_main:
            # Workers only need likelihood and seed - everything else is handled in worker_wait
            self.pool.worker_wait(likelihood=self.loglikelihood, seed=seed)
            return  # Workers never return from worker_wait until pool.close()

        # MAIN PROCESS FULL SETUP
        self._setup_main_process(
            seed, optimizer, save, save_dir, save_step,
            n_cobaya_init, n_sobol_init, minus_inf, resume, dynamic_dispatch
        )

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

        # Finalize main process initialization

        # Extract best point from GP
        if self.gp.train_y.size > 0:
            idx_best = jnp.argmax(self.gp.train_y)
            self.best_pt = scale_from_unit(self.gp.train_x[idx_best], self.loglikelihood.param_bounds).flatten()
            best_f_from_gp = float(self.gp.train_y.max()) * self.gp.y_std + self.gp.y_mean
        else:
            best_f_from_gp = -np.inf
            self.best_pt = None

        # Use restored best_f if available and better
        if not hasattr(self, 'best_f') or best_f_from_gp > getattr(self, 'best_f', -np.inf):
            self.best_f = best_f_from_gp
            if not hasattr(self, 'best_pt_iteration'):
                self.best_pt_iteration = self.start_iteration

        if self.best_pt is not None:
            self.best = {name: f"{float(val):.6f}" for name, val in zip(self.loglikelihood.param_list, self.best_pt)}
            log.info(f"Initial best point {self.best} with value = {self.best_f:.6f}")

        # Save initial GP
        if self.save:
            self.gp.save(filename=f"{self.save_path}_gp")
            log.info(f"Saving GP to file {self.save_path}_gp")

        # Initialize for KL divergence tracking
        self.prev_samples = None

    # ============================================================================
    # INITIALIZATION HELPER METHODS
    # ============================================================================
    
    def _prepare_likelihood(self, loglikelihood, param_list, param_bounds, param_labels,
                           likelihood_name, confidence_for_unbounded, minus_inf):
        """Convert input to Likelihood instance if needed."""
        if isinstance(loglikelihood, Likelihood):
            return loglikelihood
        
        if isinstance(loglikelihood, (str, dict)):
            # Cobaya YAML file or info dict
            from .likelihood import CobayaLikelihood
            return CobayaLikelihood(
                input_file_dict=loglikelihood,
                confidence_for_unbounded=confidence_for_unbounded,
                minus_inf=minus_inf,
                name=likelihood_name if likelihood_name is not None else 'CobayaLikelihood',
            )
        
        if callable(loglikelihood):
            # Create Likelihood instance from callable
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
    
    
    def _setup_main_process(self, seed, optimizer, save, save_dir, save_step,
                           n_cobaya_init, n_sobol_init, minus_inf, resume, dynamic_dispatch=False):
        """Setup full attributes for main process."""
        set_global_seed(seed)
        self.np_rng = get_numpy_rng()
        
        # File paths and saving
        self.output_file = self.loglikelihood.name
        self.save = save
        self.save_step = save_step
        self.save_dir = save_dir
        if self.save:
            os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, self.output_file)
        
        # Validate optimizer
        if optimizer.lower() not in ['optax', 'scipy']:
            raise ValueError("optimizer must be either 'optax' or 'scipy'")
        self.optimizer = optimizer
        self.minus_inf = minus_inf
        
        # Initialize results manager (settings will be updated when run() is called)
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
                'dynamic_dispatch': dynamic_dispatch
            },
            likelihood_name=self.loglikelihood.name,
            resume_from_existing=resume
        )
        
        self.fresh_start = not resume
    
    def _handle_resume(self, resume_file):
        """Handle resume from existing run (main process only)."""
        try:
            log.info(f"Attempting to resume from file {resume_file}")
            gp_file = resume_file + '_gp'
            self.gp = Surrogate.load(gp_file)
            
            # Test GP functionality
            _ = self.gp.predict_mean(self.gp.train_x[0])
            log.info(f"Loaded GP with {self.gp.train_x.shape[0]} training points")
            
            # Restore iteration and best point info
            if self.results_manager.is_resuming():
                self.start_iteration = self.results_manager.get_last_iteration()
                log.info(f"Resuming from iteration {self.start_iteration}")
                log.info(f"Previous data: {len(self.results_manager.acquisition_values)} acquisition evaluations")
                
                if self.results_manager.best_loglike_values:
                    self.best_f = max(self.results_manager.best_loglike_values)
                    best_idx = self.results_manager.best_loglike_values.index(self.best_f)
                    self.best_pt_iteration = self.results_manager.best_loglike_iterations[best_idx]
                    log.info(f"Restored best loglikelihood: {self.best_f:.4f} at iteration {self.best_pt_iteration}")
                else:
                    self.start_iteration = 0
                    self.best_pt_iteration = 0
                
                if self.results_manager.converged:
                    self.prev_converged = True
                    self.convergence_counter = 1
                    # Store last convergence info for threshold comparison
                    if self.results_manager.convergence_history:
                        last_conv = self.results_manager.convergence_history[-1]
                        self.prev_convergence_delta = last_conv.delta
                        self.prev_convergence_threshold = last_conv.threshold
                        log.info(f"Previous run had converged with delta={self.prev_convergence_delta:.6f}, threshold={self.prev_convergence_threshold:.6f}")
                    else:
                        self.prev_convergence_delta = None
                        self.prev_convergence_threshold = None
                        log.info("Previous run had converged.")
                else:
                    # Not converged in previous run
                    self.prev_converged = False
                    self.prev_convergence_delta = None
                    self.prev_convergence_threshold = None
            else:
                self.start_iteration = 0
                self.best_pt_iteration = 0
                log.info("Starting fresh optimization")
            
            self.fresh_start = False
            
        except Exception as e:
            log.error(f"Failed to load GP from file {gp_file}: {e}")
            log.info("Starting a fresh run instead.")
            self.fresh_start = True
    
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

        # Bind likelihood metadata (parameter names and any deferred physical-space kernel transform)
        self.gp.bind_likelihood(
            self.loglikelihood.param_bounds,
            param_names=self.loglikelihood.param_list,
        )

        # Generate and evaluate initial training points
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

        # Populate and fit the surrogate
        self._initialize_gp(train_x=train_x, train_y=train_y)


    def _get_initial_training_data(self, n_cobaya_init, n_sobol_init, init_train_x=None, init_train_y=None):
        """
        Generate and evaluate initial training points for the GP.

        This method:
        1. Generates Sobol initial points in parallel (if n_sobol_init > 0)
        2. Generates Cobaya initial points in parallel (if applicable)
        3. Adds user-provided initial points (if given)
        4. Removes duplicates
        5. Returns points and values in unit space for GP training
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

        # Only main process continues with processing
        if not self.is_main:
            return None, None

        # Add user-provided initial training data if available
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

        # Remove duplicates
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

        # Convert to unit space for GP
        train_x = jnp.array(scale_to_unit(init_points, self.loglikelihood.param_bounds))
        train_y = jnp.array(init_vals)

        return train_x, train_y

    def _generate_sobol_points(self, n_sobol_init: int):
        """
        Generate Sobol initial points on main rank and evaluate them in parallel using pool.
        
        Parameters
        ----------
        n_sobol_init : int
            Number of Sobol points to generate.
            
        Returns
        -------
        all_points : np.ndarray
            Sobol points in parameter space, shape (n_points, ndim).
        all_vals : np.ndarray
            Likelihood values, shape (n_points, 1).
        """
        from scipy.stats import qmc
        
        n_sobol = max(2, n_sobol_init)
        
        # Main process generates Sobol points and distributes via pool
        if self.is_main:
            self.results_manager.start_timing('True Objective Evaluations')
            sobol = qmc.Sobol(d=self.ndim, scramble=True, rng=self.np_rng).random(n_sobol)
            sobol_points = scale_from_unit(sobol, self.loglikelihood.param_bounds)
            log.info(f"Evaluating {len(sobol_points)} Sobol initial points")
            
            # Use pool to evaluate points in parallel
            all_vals = self.pool.run_map_objective(self.loglikelihood, sobol_points)
            all_vals = np.atleast_2d(all_vals).reshape(-1, 1)
            return sobol_points, all_vals
        else:
            return np.empty((0, self.ndim)), np.empty((0, 1))

    def _generate_cobaya_points(self, n_cobaya_init: int):
        """
        Generate Cobaya initial points in parallel using pool.
        
        Parameters
        ----------
        n_cobaya_init : int
            Number of Cobaya points to generate.
            
        Returns
        -------
        all_points : np.ndarray
            Cobaya points in parameter space, shape (n_points, ndim).
        all_vals : np.ndarray
            Likelihood values, shape (n_points, 1).
        """
        if n_cobaya_init <= 0:
            raise ValueError("n_cobaya_init must be greater than zero to generate Cobaya points.")

        # Use pool to generate Cobaya points in parallel
        if self.is_main:
            results_tuples = self.pool.get_cobaya_initial_points(
                self.loglikelihood, n_cobaya_init, rng=self.np_rng
            )
            
            # Extract points and values from tuples
            all_points = np.array([pt for pt, _ in results_tuples])
            all_vals = np.array([[lp] for _, lp in results_tuples])
            return all_points, all_vals
        else:
            return np.empty((0, self.ndim)), np.empty((0, 1))
    
    def _initialize_gp(self, train_x=None, train_y=None):
        """
        Populate the surrogate with initial training data and fit its
        hyperparameters (main process only).
        """
        if not self.is_main:
            return

        if train_x is not None and train_y is not None:
            self.gp.update(train_x, train_y)

        if isinstance(self.gp, ClassifierMixin):
            self.gp.train_classifier()

        self.results_manager.start_timing('GP Training')
        log.info(f"Hyperparameters before refit: {self.gp.hyperparams_dict()}")

        # Use pool to fit GP in parallel
        self.pool.gp_fit(self.gp, n_restarts=4, maxiters=500, rng=self.np_rng, use_pool=True)

        log.info(f"Hyperparameters after refit: {self.gp.hyperparams_dict()}")
        self.results_manager.end_timing('GP Training')
    

    # ============================================================================
    # RUN HELPER METHODS
    # ============================================================================

    def update_gp(self, new_pts_u, new_vals, step = 0, verbose=True):
        """
        Update the GP with new points and values, and track hyperparameters.
        
        Uses pool for parallel GP fitting when refitting is needed.
        Refits based on number of points added to GP since last fit.
        """
        # Only main process updates GP
        if not self.is_main:
            return
        
        self.results_manager.start_timing('GP Training')
        
        # Track GP size before update
        gp_size_before = self.gp.train_x.shape[0]
        
        # Update GP with new data
        self.gp.update(new_pts_u, new_vals)
        
        # Track actual points added (accounts for filtering by classifier or other mechanisms)
        gp_size_after = self.gp.train_x.shape[0]
        actual_points_added = gp_size_after - gp_size_before
        self.n_points_since_last_fit += actual_points_added
        
        # Determine refit parameters based on training set size and points added
        if gp_size_after < 200:
            # For small training sets, refit more frequently
            refit_threshold = min(4, self.fit_n_points)
            maxiter = 1000
            n_restarts = 8
        elif 200 < gp_size_after < 800:
            # For moderate size training sets
            refit_threshold = self.fit_n_points
            n_restarts = 4
            maxiter = 500
        else:
            # For large training sets, refit less frequently
            refit_threshold = max(50, self.fit_n_points)
            n_restarts = 4
            maxiter = 200
        
        refit = (self.n_points_since_last_fit >= refit_threshold)
        
        # Use pool for parallel GP fitting if refitting
        if refit:
            log.info(f"Refitting GP hyperparameters with {self.gp.train_x.shape[0]} training points ")
            self.pool.gp_fit(self.gp, n_restarts=n_restarts, maxiters=maxiter, rng=self.np_rng, use_pool=True)
            # Reset counter after successful refit
            self.n_points_since_last_fit = 0
        
        self.results_manager.end_timing('GP Training')

        # Extract GP hyperparameters for tracking
        if hasattr(self.gp, 'kernel'):
            lengthscales = list(self.gp.kernel.lengthscales)
            kernel_variance = float(self.gp.kernel.kernel_variance)
            self.results_manager.update_gp_hyperparams(step, lengthscales, kernel_variance)

        if isinstance(self.gp, ClassifierMixin):
            self.results_manager.start_timing('Classifier Training')
            self.gp.train_classifier()
            self.results_manager.end_timing('Classifier Training')

                

    def get_next_batch(self, acq_kwargs, n_batch, n_restarts, maxiter, early_stop_patience, step, verbose=True):
        """
        Get the next batch of points using the acquisition function, and track acquisition values.
        """
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
        """
        Evaluate the likelihood for new points using pool.
        
        Parameters
        ----------
        new_pts_u : array-like
            Points in unit cube space to evaluate, shape (n_points, ndim).
        step : int
            Current iteration number.
        verbose : bool, optional
            Whether to log detailed information.
            
        Returns
        -------
        new_vals : jax.numpy.ndarray
            Evaluated likelihood values, shape (n_points, 1).
        """
        # Only main process evaluates
        if not self.is_main:
            return None
        
        new_pts_u = jnp.atleast_2d(new_pts_u)
        new_pts = scale_from_unit(new_pts_u, self.loglikelihood.param_bounds)
        
        self.results_manager.start_timing('True Objective Evaluations')
        
        # Use pool to evaluate points in parallel
        new_vals = self.pool.run_map_objective(self.loglikelihood, new_pts)
        new_vals = jnp.atleast_2d(new_vals).reshape(-1, 1)
        
        self.results_manager.end_timing('True Objective Evaluations')

        best_new_idx = np.argmax(new_vals)
        best_new_val = float(np.max(new_vals))
        best_new_pt = new_pts[best_new_idx]
        if float(best_new_val) > self.best_f:
            self.best_f = float(best_new_val)
            self.best_pt = best_new_pt
            self.best = {name: f"{float(val):.6f}" for name, val in zip(self.loglikelihood.param_list, self.best_pt.flatten())}
            self.best_pt_iteration = step

        log.info(f"Evaluated objective at {len(new_pts)} new points")
        for k, new_pt in enumerate(new_pts):
            new_pt_vals = {name: f"{float(val):.4f}" for name, val in zip(self.loglikelihood.param_list, new_pt.flatten())}
            log.debug(f"New point {new_pt_vals}, {k+1}/{len(new_pts)}")
            predicted_val = self.gp.predict_mean(new_pts_u[k])
            log.debug(f"Objective function value = {new_vals[k].item():.4f}, GP predicted value = {predicted_val.item():.4f}")

        return new_vals

    def check_max_evals_and_gpsize(self,current_evals):
        """
        Check if the maximum evaluations or GP size has been reached.

        Args:
            current_evals: Current number of objective evaluations.
        """
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
        # here finalize results
        if not self.is_main:
            return
        
        # Prepare return dictionary

        # Extract GP and classifier information
        gp_info = {
            'gp_training_set_size': self.gp.train_x.shape[0],
            'gp_final_best_loglike': float(self.best_f),  # Best value in true physical space
        }
        
        # Add classifier info if using GPwithClassifier, this can be done at the start since there are no results here, only settings.
        if isinstance(self.gp, ClassifierMixin):
            gp_info.update({
                'classifier_used': bool(self.gp.use_clf),
                'classifier_type': str(self.gp.clf_type),
                'classifier_training_set_size': int(self.gp.clf_data_size),
                'classifier_use_threshold': int(self.gp.clf_use_size),
                'classifier_probability_threshold': float(self.gp.probability_threshold)
            })
        else:
            gp_info.update({
                'classifier_used': False,
                'classifier_type': None,
                'classifier_training_set_size': 0
            })

        # Add evidence info if available
        samples_dict = self.samples_dict or {}
        log.debug(f"Samples dict keys: {samples_dict.keys()}")
        logz_dict = self.results_dict.get('logz', {})

        # if logz_dict is empty, warn user
        if not logz_dict:
            log.warning("No logz information found, nested sampling has not been run yet.")

        # Finalize results with comprehensive data
        self.results_manager.finalize(
            samples_dict=samples_dict,
            logz_dict=logz_dict,
            converged=self.converged,
            termination_reason=self.termination_reason,
            gp_info=gp_info,
            best_point=self.best_pt,
            best_loglike=self.best_f,
            best_iteration=self.best_pt_iteration
        )

        # Create final results dictionary with only the specified keys
        self.results_dict = {
            'gp': self.gp,
            'likelihood': self.loglikelihood,
            'results_manager': self.results_manager,
            'best_val': self.best_f,
            'best_pt': self.best_pt,
            'logz': logz_dict,
            'termination_reason': self.termination_reason,
            'samples': samples_dict
        }

    def check_convergence_ei(self, step, acq_val):
        """
        Check convergence for EI/LogEI based on the acquisition function value.

        Args:
            step: Current iteration number.
            acq_val: Current acquisition function value.

        Returns:
            bool: Whether convergence is achieved based on acquisition value.
        """
        if not self.is_main:
            return False
        
        if self.acquisition.name.lower() == 'ei':
            acq_val = np.log(acq_val + 1e-100)  # Avoid log(0)
        
        converged = acq_val < self.ei_goal_log

        if converged:
            self.convergence_counter += 1
            if self.convergence_counter >= self.convergence_n_iters:
                log.info(f"Convergence for {self.acquisition.name} achieved after {self.convergence_n_iters} successive iterations")
                return True
            else:
                log.info(f"{self.acquisition.name} convergence iteration {self.convergence_counter}/{self.convergence_n_iters}")
                return False
        else:
            self.convergence_counter = 0  # Reset counter if not converged
            return False

    def check_convergence_logz(self, step, logz_dict, equal_samples, equal_logl, verbose=True, save_checkpoint=True):
        """
        Check if the nested sampling has converged and compute KL divergence metrics.
        
        Args:
            step: Current iteration number
            logz_dict: Dictionary with logz bounds and mean
            ns_samples: Nested sampling samples with x, weights, logl
            threshold: LogZ convergence threshold
            
        Returns:
            bool: Whether convergence is achieved based on logz only
        """
        if not self.is_main:
            return False
        
        # Standard logz convergence check
        delta = (logz_dict['upper'] - logz_dict['lower'])/2 
        
        # alternative cross-check using std, not used for convergence
        delta_crosscheck = logz_dict['std']

        converged = delta < self.logz_threshold
        
        # Compute KL divergences if we have nested sampling samples
        successive_kl = None
        
        equal_samples = scale_from_unit(equal_samples, self.loglikelihood.param_bounds)
    

        if self.prev_samples is not None:

            prev_samples_x = self.prev_samples['x']
            mu1 = np.mean(prev_samples_x, axis=0)
            cov1 = np.cov(prev_samples_x, rowvar=False)
            mu2 = np.mean(equal_samples, axis=0)
            cov2 = np.cov(equal_samples, rowvar=False)
            successive_kl = kl_divergence_gaussian(mu1, np.atleast_2d(cov1), mu2, np.atleast_2d(cov2))

            log.info(f"Successive KL: symmetric={successive_kl.get('symmetric', 0):.4f}")
            # Store KL divergences if computed
            self.results_manager.update_kl_divergences(
                iteration=step,
                successive_kl=successive_kl
            )

        # Store current samples for next iteration
        self.prev_samples = {'x': equal_samples, 'logl': equal_logl}

        # Update results manager with convergence info and KL divergences
        self.results_manager.update_convergence(
            iteration=step,
            logz_dict=logz_dict,
            converged=converged,
            threshold=self.logz_threshold
        )
        
        log.info(f"Convergence check: delta = {delta:.4f}, step = {step}, threshold = {self.logz_threshold}")
        
        if converged:
            self.convergence_counter += 1
            if self.convergence_counter >= self.convergence_n_iters:
                log.info(f"Convergence achieved after {self.convergence_n_iters} successive iterations")
                converged = True
            else:
                log.info(f"Convergence iteration {self.convergence_counter}/{self.convergence_n_iters}")
                converged = False
        else:
            self.convergence_counter = 0  # Reset counter if not converged
            converged = False

        # Check if this is the smallest delta seen so far and save checkpoint, also ensure delta is reasonably good
        if (delta < self.min_delta_seen) and (delta_crosscheck < 1.0) and save_checkpoint:
            self.min_delta_seen = delta

            # Create checkpoint filename with suffix
            checkpoint_filename = f"{self.output_file}_checkpoint"

            if not converged:

                # Save intermediate results checkpoint
                self.results_manager.save_intermediate(gp=self.gp, filename=f"{checkpoint_filename}")

                # Save getdist chains
                self.results_manager.save_chain_files(samples_dict=self.ns_samples, filename=f"{checkpoint_filename}")

                if verbose:
                    log.info(f"New minimum delta achieved: {delta:.4f}")
                    log.info("Saving checkpoint results for new minimum delta")
                    log.info(f"Saved GP checkpoint to {checkpoint_filename}_gp.npz")
                    log.info(f"Saved intermediate results checkpoint to {checkpoint_filename}.json")

        return converged
        
    # ============================================================================
    # MAIN RUN METHODS
    # ============================================================================

    def run(self, acq: Union[str, Tuple[str]] = 'wipstd',
            min_evals: Optional[int] = None,
            max_evals: Optional[int] = None,
            max_gp_size: Optional[int] = None,
            logz_threshold: Optional[float] = None,
            convergence_n_iters: int = 1,
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
            zeta_ei: float = 0.0):
        """
        Run the Bayesian Optimization loop.
        
        Parameters
        ----------
        acq : str or tuple of str
            Acquisition function(s) to use: 'WIPV', 'EI', 'LogEI', 'WIPStd'.
        min_evals : int, optional
            Minimum number of likelihood evaluations before checking convergence.
            If None, uses dimension-based default from _get_dimension_based_defaults().
        max_evals : int, optional
            Maximum number of likelihood evaluations.
            If None, uses dimension-based default from _get_dimension_based_defaults().
        max_gp_size : int, optional
            Maximum number of points used to train the GP.
            If None, uses dimension-based default from _get_dimension_based_defaults().
        logz_threshold : float, optional
            Convergence threshold for log evidence change (WIPV/WIPStd). 
            If None, uses dimension-based default from _get_dimension_based_defaults().
        convergence_n_iters : int, optional
            Number of successive iterations meeting threshold for convergence. Default is 1.
        ei_goal : float, optional
            Goal value for EI/LogEI acquisition convergence. Default is 1e-10.
        do_final_ns : bool, optional
            Whether to run final nested sampling at convergence (WIPV/WIPStd). Default is False.
        fit_n_points : int, optional
            Refit GP hyperparameters after adding this many new points to the GP.
            If None, uses dimension-based default from _get_dimension_based_defaults().
        batch_size : int, optional
            Batch size for WIPV/WIPStd acquisition.
            If None, uses dimension-based default from _get_dimension_based_defaults().
        ns_n_points : int, optional
            Run nested sampling after adding this many new points to the GP (for WIPV/WIPStd).
            If None, uses dimension-based default from _get_dimension_based_defaults().
        num_hmc_warmup : int, optional
            Number of HMC warmup steps.
            If None, uses dimension-based default from _get_dimension_based_defaults().
        num_hmc_samples : int, optional
            Number of HMC samples to draw.
            If None, uses dimension-based default from _get_dimension_based_defaults().
        mc_points_size : int, optional
            Number of MC points for WIPV acquisition.
            If None, uses dimension-based default from _get_dimension_based_defaults().
        thinning : int, optional
            Thinning factor for MC samples. Default is 4.
        num_chains : int, optional
            Number of parallel HMC chains.
            If None, uses dimension-based default from _get_dimension_based_defaults().
        mc_points_method : str, optional
            Method for generating MC points: 'NUTS', 'NS', or 'uniform'. Default is 'NUTS'.
        zeta_ei : float, optional
            Exploration parameter for EI acquisition. Default is 0.0.
            
        Returns
        -------
        dict
            Results dictionary containing samples, GP, likelihood, and convergence information. Keys include:
        """
        # Workers don't run the optimization loop
        if not self.is_main:
            return None
        
        # Get dimension-based defaults
        dim_defaults = get_dimension_based_defaults(self.ndim)
        
        # Apply defaults for None values
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
        
        # Store convergence parameters
        self.min_evals = min_evals
        self.max_evals = max_evals
        self.max_gp_size = max_gp_size
        self.logz_threshold = logz_threshold

        # Log run settings
        log.info("Using run settings:")
        log.info(f"min_evals = {min_evals}, max_evals = {max_evals}, max_gp_size = {max_gp_size}")
        if acq.lower() in ['wipv', 'wipstd']:
            acq_info = "logz_threshold = {:.4f}".format(logz_threshold)+f", mc_points_size = {mc_points_size}"
        else:
            acq_info = "ei_goal = {:.4e}".format(ei_goal)
        log.info(f"convergence_n_iters = {convergence_n_iters}, acq = {acq}, {acq_info}")
        log.info(f"fit_n_points = {fit_n_points}, batch_size = {batch_size}, ns_n_points = {ns_n_points}")
        
        # Initialize result containers
        self.samples_dict = {}
        self.results_dict = {}
        
        # Check if already converged with new threshold when resuming
        if self.prev_converged and self.prev_convergence_delta is not None:
            if self.prev_convergence_delta < logz_threshold:
                log.info(f"Previous run already converged with delta={self.prev_convergence_delta:.6f} < new threshold={logz_threshold:.6f}")
                log.info("Skipping BO loop and proceeding to finalization")
                self.converged = True
                self.termination_reason = "Already converged in previous run"
                
                # Restore samples and logz from previous run
                if self.results_manager.convergence_history:
                    last_conv = self.results_manager.convergence_history[-1]
                    self.results_dict['logz'] = last_conv.logz_dict.copy()
                
                # Restore samples from results_manager if available
                if self.results_manager.final_samples is not None and len(self.results_manager.final_samples) > 0:
                    self.samples_dict = {
                        'x': self.results_manager.final_samples,
                        'weights': self.results_manager.final_weights,
                        'logl': self.results_manager.final_loglikes
                    }
                    log.info(f"Restored {len(self.samples_dict['x'])} samples from previous run")
                else:
                    self.samples_dict = {}
                    log.warning("No samples found in previous run")
                
                self.finalise_results()
                self.pool.close()
                return self.results_dict
            else:
                log.info(f"Previous run converged with delta={self.prev_convergence_delta:.6f} >= new threshold={logz_threshold:.6f}")
                log.info("Continuing optimization to meet new convergence threshold")
                self.converged = False
                self.convergence_counter = 0
        
        self.convergence_n_iters = convergence_n_iters
        self.ei_goal_log = np.log(ei_goal)
        self.do_final_ns = do_final_ns
        
        # Store run settings
        self.fit_n_points = fit_n_points
        self.ns_n_points = ns_n_points
        self.batch_size = batch_size
        
        # Initialize point counters for triggering GP refit and NS
        self.n_points_since_last_fit = 0
        self.n_points_since_last_ns = 0
        self.num_hmc_warmup = num_hmc_warmup
        self.num_hmc_samples = num_hmc_samples
        self.mc_points_size = mc_points_size
        self.hmc_thinning = thinning
        self.hmc_num_chains = num_chains
        self.mc_points_method = mc_points_method
        self.zeta_ei = zeta_ei
        
        # Adjust batch_size for MPI load balancing
        if self.is_mpi:
            n_processes = self.pool.size
            original_batch = self.batch_size
            if self.batch_size % n_processes != 0:
                self.batch_size = (self.batch_size // n_processes) * n_processes
                if self.batch_size < n_processes:
                    self.batch_size = n_processes
                log.info(f"Adjusted batch_size from {original_batch} to {self.batch_size} "
                        f"(multiple of {n_processes} processes)")
        
        # Initialize convergence state
        self.converged = False
        self.prev_converged = False
        self.convergence_counter = 0
        self.min_delta_seen = np.inf
        self.termination_reason = "Max evaluation budget reached"
        
        # Update results manager settings with all run parameters
        self.results_manager.settings.update({
            'acq': acq,
            'min_evals': min_evals,
            'max_evals': max_evals,
            'max_gp_size': max_gp_size,
            'logz_threshold': logz_threshold,
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
            'zeta_ei': zeta_ei
        })
        
        acqs_funcs_available = list(_acq_funcs.keys())

        self.samples_dict = {}
        self.results_dict = {}

        if isinstance(acq, str):
            acqs = [acq]

        self.current_iteration = self.start_iteration

        for acq in acqs:
            if acq.lower() not in acqs_funcs_available:
                raise ValueError(f"Invalid acquisition function '{acq}'. Valid options are: {acqs_funcs_available}")
            self.acquisition = _acq_funcs[acq.lower()](optimizer=self.optimizer)  # Set acquisition function
            if acq.lower() == 'wipv':
                self.run_WIPV(ii=self.current_iteration)
            elif acq.lower() == 'wipstd':
                self.run_WIPStd(ii=self.current_iteration)
            else:
                self.run_EI(ii=self.current_iteration)

        log.info(f"Final best point {self.best} with value = {self.best_f:.6f}, found at iteration {self.best_pt_iteration}")


        #-------End of BO loop-------
        log.info(f"Sampling stopped: {self.termination_reason}")
        log.info(f"Final GP training set size: {self.gp.train_x.shape[0]}, max size: {self.max_gp_size}")

        self.finalise_results()
        
        # Close the pool and signal workers to exit
        self.pool.close()

        return self.results_dict

    def run_EI(self, ii = 0, ):
        """
        Run the optimization loop for EI/LogEI acquisition functions.
        """
        if not self.is_main:
            return
        
        current_evals = self.gp.npoints
        log.info(f"Starting iteration {ii}")
        converged=False

        while not converged:
            ii += 1
            verbose = True

            if verbose:
                log.info(f"Iteration {ii} of {self.acquisition.name}, objective evals {current_evals}/{self.max_evals}")

            acq_kwargs = {'zeta': self.zeta_ei, 'best_y': max(self.gp.train_y.flatten()) if self.gp.train_y.size > 0 else 0.}
            n_batch = 1
            new_pts_u, acq_vals = self.get_next_batch(acq_kwargs, n_batch = n_batch, n_restarts = 50, maxiter = 1000, early_stop_patience = 50, step = ii, verbose=verbose)
            new_pts_u = jnp.atleast_2d(new_pts_u)

            new_vals = self.evaluate_likelihood(new_pts_u, ii, verbose=verbose)
            current_evals += n_batch

            self.update_gp(new_pts_u, new_vals, step = ii, verbose=verbose)

            self.results_manager.update_best_loglike(ii, self.best_f)
            if verbose:
                log.info(f"Current best point {self.best} with value = {self.best_f:.6f}, found at iteration {self.best_pt_iteration}")

            # if current_evals >= self.min_evals:
            converged = self.check_convergence_ei(ii,acq_vals)

            # Update results manager with iteration info, also save results and gp if save_step
            if ii % self.save_step == 0:
                self.results_manager.save_intermediate(gp=self.gp)

            if converged:
                self.termination_reason = f"{self.acquisition.name.upper()} goal reached"
                self.results_dict['termination_reason'] = self.termination_reason
                break
            self.pool.clear_jax_caches()

            max_evals_or_gpsize_reached = self.check_max_evals_and_gpsize(current_evals)
            if max_evals_or_gpsize_reached:
                break

        # End EI
        self.current_iteration = ii

    def run_weighted_integrated_posterior(self, acq_func_class, ii=0):
        """
        Run the optimization loop for Weighted Integrated Posterior acquisition functions (WIPV or WIPStd).
        
        Parameters
        ----------
        acq_func_class : class
            The acquisition function class to use (WIPV or WIPStd).
        ii : int, optional
            Starting iteration number. Default is 0.
        """
        if not self.is_main:
            return
        
        # Set acquisition function
        self.acquisition = acq_func_class(optimizer=self.optimizer)
        acq_name = self.acquisition.name
        
        current_evals = self.gp.npoints
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
        )
        self.results_manager.end_timing('MCMC Sampling')
        self.ns_samples = None
        
        #logz keys to print
        logz_keys = ['mean', 'upper', 'lower', 'dlogz_sampler']


        while not self.converged:
            ii += 1
            # Check if we should run nested sampling based on points added
            self.n_points_since_last_ns += self.batch_size
            ns_flag = (self.n_points_since_last_ns >= self.ns_n_points) and current_evals >= self.min_evals
            verbose = True

            if verbose:
                log.info(f"Iteration {ii} of {acq_name}, objective evals {current_evals}/{self.max_evals}")

            acq_kwargs = {'mc_samples': self.mc_samples, 'mc_points_size': self.mc_points_size}
            new_pts_u, acq_vals = self.get_next_batch(acq_kwargs, n_batch = self.batch_size, n_restarts = 1, maxiter = 100, early_stop_patience = 10, step = ii, verbose=verbose)
            new_pts_u = jnp.atleast_2d(new_pts_u)
            new_vals = self.evaluate_likelihood(new_pts_u, ii, verbose=verbose)
            current_evals += self.batch_size

            self.update_gp(new_pts_u, new_vals, step = ii)
            self.results_manager.update_best_loglike(ii, self.best_f)

            # Check convergence and update MCMC samples
            if ns_flag and (acq_vals[-1] <= self.logz_threshold):
                self.results_manager.start_timing('Nested Sampling')
                ns_samples, logz_dict, ns_success = nested_sampling_Dy(mode='convergence',
                    gp=self.gp, ndim=self.ndim, maxcall=int(5e6), dynamic=False, dlogz=0.01, equal_weights=False,
                    rng=self.np_rng
                )
                self.results_manager.end_timing('Nested Sampling')

                logz_str = ", ".join([f"{k}={logz_dict[k]:.4f}" for k in logz_keys if k in logz_dict])
                log.info(f"NS success = {ns_success}, LogZ info: {logz_str}")

                self.ns_samples = ns_samples
                if ns_success:
                    equal_samples, equal_logl = resample_equal(ns_samples['x'], ns_samples['logl'], weights=ns_samples['weights'])
                    self.mc_samples = {
                        'x': equal_samples,
                        'logl': equal_logl,
                        'weights': np.ones(equal_samples.shape[0]),
                        'method': 'NS',
                        'best': ns_samples['best']
                    }
                    self.results_dict['logz'] = logz_dict
                    self.converged = self.check_convergence_logz(ii, logz_dict, equal_samples, equal_logl)
                    if self.converged:
                        self.termination_reason = "LogZ converged"
                        self.results_dict['termination_reason'] = self.termination_reason
                
                # Reset counter after running NS
                self.n_points_since_last_ns = 0
            else:
                self.results_manager.start_timing('MCMC Sampling')
                self.mc_samples = get_mc_samples(
                        self.gp,
                        warmup_steps=self.num_hmc_warmup,
                        num_samples=self.num_hmc_samples,
                        thinning=self.hmc_thinning,
                        num_chains=self.hmc_num_chains,
                        method=self.mc_points_method,
                        np_rng=self.np_rng,
                        rng_key=get_new_jax_key()
                    )
                self.results_manager.end_timing('MCMC Sampling')
            
            if verbose:
                log.info(f"Current best point {self.best} with value = {self.best_f:.6f}, found at iteration {self.best_pt_iteration}")

            # Update results manager with iteration info, also save results and gp if save_step
            if ii % self.save_step == 0:
                self.results_manager.save_intermediate(gp=self.gp)

            if self.converged:
                break
            
            self.pool.clear_jax_caches()

            max_evals_or_gpsize_reached = self.check_max_evals_and_gpsize(current_evals)
            if max_evals_or_gpsize_reached:
                break

        # End of main BO loop
        self.current_iteration = ii

        # Final nested sampling if not yet converged and do_final_ns is True
        if self.do_final_ns and not self.converged:
            
            self.results_manager.start_timing('GP Training')
            self.pool.gp_fit(self.gp, n_restarts=4, maxiters=500, rng=self.np_rng, use_pool=True)
            self.results_manager.end_timing('GP Training')

            log.info("Final Nested Sampling")
            self.results_manager.start_timing('Nested Sampling')
            self.ns_samples, logz_dict, ns_success = nested_sampling_Dy(mode='convergence',
                gp=self.gp, ndim=self.ndim, maxcall=int(5e6), dynamic=True, dlogz=0.01, rng=self.np_rng
            )
            self.results_manager.end_timing('Nested Sampling')
            logz_str = ", ".join([f"{k}={logz_dict[k]:.4f}" for k in logz_keys if k in logz_dict])
            log.info(f"Final LogZ: {logz_str}")
            if ns_success:
                equal_samples, equal_logl = resample_equal(self.ns_samples['x'], self.ns_samples['logl'], weights=self.ns_samples['weights'])
                log.info(f"Using nested sampling results")
                self.check_convergence_logz(ii+1, logz_dict, equal_samples, equal_logl, save_checkpoint=False)
                self.results_dict['logz'] = logz_dict
                if self.converged:
                    self.termination_reason = "LogZ converged"
                    self.results_dict['termination_reason'] = self.termination_reason

        if (self.ns_samples is not None) and ns_success:
            samples = self.ns_samples['x']
            weights = self.ns_samples['weights']
            loglikes = self.ns_samples['logl']
        else:
            log.info("No nested sampling results found or nested sampling unsuccessful, MC samples from HMC/MCMC will be used instead.")
            self.results_manager.start_timing('MCMC Sampling')
            mc_samples = get_mc_samples(
                    self.gp, warmup_steps=self.num_hmc_warmup, num_samples=8*self.num_hmc_samples,
                    thinning=self.hmc_thinning, method="NUTS",
                    np_rng=self.np_rng, rng_key=get_new_jax_key())
            self.results_manager.end_timing('MCMC Sampling')
            samples = mc_samples['x']
            weights = mc_samples['weights'] if 'weights' in mc_samples else np.ones(mc_samples['x'].shape[0])
            loglikes = mc_samples['logp']
                
        samples = scale_from_unit(samples, self.loglikelihood.param_bounds)

        self.samples_dict = {
            'x': samples,
            'weights': weights,
            'logl': loglikes
        }

    def run_WIPStd(self, ii=0):
        """Run optimization loop for WIPStd acquisition function."""
        return self.run_weighted_integrated_posterior(WIPStd, ii)

    def run_WIPV(self, ii=0):
        """Run optimization loop for WIPV acquisition function."""
        return self.run_weighted_integrated_posterior(WIPV, ii)