from typing import Optional, Union, List, Tuple, Callable, Any
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize 
from .utils.core import scale_to_unit, scale_from_unit
from .utils.log import get_logger

try:
    import optax
    OPTAX_AVAILABLE = True
except ImportError:
    OPTAX_AVAILABLE = False
    optax = None

log = get_logger("optim")

_SCIPY_DEFAULT_OPTIONS = {"method": "L-BFGS-B", "ftol": 1e-6, "gtol": 1e-6}


def _copy_options(optimizer_options: Optional[dict], default: Optional[dict] = None) -> dict:
    """
    Return a private copy of ``optimizer_options`` (or of ``default`` when it is None).

    The optimizers pop/insert keys while parsing their options; working on a copy keeps
    the caller's dict (e.g. ``AcquisitionFunction.optimizer_options``) intact across calls.
    """
    if optimizer_options is None:
        return dict(default) if default is not None else {}
    return dict(optimizer_options)


def _get_optimizer(optimizer_name: str, learning_rate: float = 1e-3, optimizer_kwargs: Optional[dict] = None):
    """Get the optax optimizer."""
    if not OPTAX_AVAILABLE:
        raise ImportError(
            "optax is required for this optimizer. "
            "Install it with: pip install 'BOBE[nn]' or pip install optax"
        )
    optimizer_name = optimizer_name.lower()
    optimizer_kwargs = _copy_options(optimizer_kwargs)
    optimizer_kwargs.setdefault('learning_rate', learning_rate)

    if optimizer_name == "adam":
        return optax.adam(**optimizer_kwargs)
    elif optimizer_name == "sgd":
        return optax.sgd(**optimizer_kwargs)
    elif optimizer_name == "lbfgs":
        return optax.lbfgs(**optimizer_kwargs)
    else:
        try:
            optimizer_fn = getattr(optax, optimizer_name)
            return optimizer_fn(**optimizer_kwargs)
        except AttributeError:
            raise ValueError(f"Optimizer '{optimizer_name}' not found in optax library")

def _setup_bounds(bounds: Optional[Union[List, Tuple, jnp.ndarray]], num_params: int) -> Optional[jnp.ndarray]:
    """
    Setup parameter bounds.

    Parameters
    ----------
    bounds : None, list, tuple, or jnp.ndarray
        If None, returns None (unbounded optimization).
        If array-like, must be shape (2,) for shared bounds or (2, num_params).
    num_params : int
        Number of parameters.

    Returns
    -------
    Optional[jnp.ndarray]
        Bounds array with shape (2, num_params), or None if unbounded.
    """
    if bounds is None:
        return None

    bounds = jnp.array(bounds)

    if bounds.shape == (2,):  # same bounds for all dimensions
        bounds = jnp.tile(bounds.reshape(1, 2), (num_params, 1)).T
    elif bounds.shape != (2, num_params):
        raise ValueError(f"Bounds shape {bounds.shape} incompatible with {num_params} parameters")

    return bounds

def optimize_optax(
    fun: Callable,
    fun_args: Optional[Tuple] = (),
    fun_kwargs: Optional[dict] = None,
    num_params: int = 1,
    bounds: Optional[Union[List, Tuple, jnp.ndarray]] = None,
    x0: jnp.ndarray = None,
    optimizer_options: Optional[dict] = None,
    maxiter: int = 200,
    n_restarts: int = 1,
    verbose: bool = False,
    ) -> Tuple[jnp.ndarray, float]:
    """
    SEQUENTIAL OPTIMIZER: Minimize a function using JAX + optax,
    iterating through restarts with a Python for-loop.

    ``optimizer_options`` may contain ``name`` (default ``'adam'``), ``lr``
    (default ``1e-3``), ``early_stop_patience`` (default ``25``) and any extra
    optax keyword arguments. The caller's dict is never modified.

    Requires optax to be installed. Install with: pip install 'BOBE[nn]' or pip install optax
    """
    if not OPTAX_AVAILABLE:
        raise ImportError(
            "optax is required for optimize_optax. "
            "Install it with: pip install 'BOBE[nn]' or pip install optax"
        )

    fun_args = () if fun_args is None else fun_args
    fun_kwargs = {} if fun_kwargs is None else fun_kwargs
    optimizer_options = _copy_options(optimizer_options)

    if x0 is None:
        raise ValueError("x0 must be provided (shape: (n_restarts, num_params))")

    x0 = jnp.atleast_2d(x0)
    if x0.shape[0] < n_restarts:
        raise ValueError(f"x0 provided with {x0.shape[0]} restarts but n_restarts={n_restarts}")
    x0 = x0[:n_restarts]

    bounds_arr = _setup_bounds(bounds, num_params)
    if bounds_arr is not None:
        scaled_func = lambda x: fun(scale_from_unit(x, bounds_arr), *fun_args, **fun_kwargs) 
    else:
        scaled_func = lambda x: fun(x, *fun_args, **fun_kwargs)

    early_stop_patience = optimizer_options.pop("early_stop_patience", 25)
    lr = optimizer_options.pop("lr", 1e-3)
    optimizer_name = optimizer_options.pop("name", "adam")
    optimizer = _get_optimizer(optimizer_name, lr, optimizer_options)

    log.debug(f"Starting optax optimization with {optimizer_name}, restarts: {n_restarts}, maxiter: {maxiter}")

    @jax.jit
    def step(params, opt_state):
        val, grad = jax.value_and_grad(scaled_func)(params)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        if bounds_arr is not None:
            params = jnp.clip(params, 0.0, 1.0)
        return params, opt_state, val

    global_best_f = np.inf
    global_best_params = None

    for i, x_init in enumerate(x0):
        try:
            val = scaled_func(x_init)
            if np.isfinite(val) and val < global_best_f:
                global_best_f = val
                global_best_params = x_init
                log.debug(f"  Initial point {i+1}/{n_restarts}: New best found -> {val:.4e}")
        except Exception as e:
            log.warning(f"  Initial point {i+1}/{n_restarts}: Failed with an error: {e}")

    for restart_idx in range(n_restarts):
        log.debug(f"Starting restart {restart_idx + 1}/{n_restarts}")
        current_params = x0[restart_idx]
        opt_state = optimizer.init(current_params)
        best_f_for_restart = float("inf")
        best_params_for_restart = current_params
        patience_counter = early_stop_patience

        for iter_idx in range(maxiter):
            # ``step`` returns the objective at the params it was given, i.e. before
            # the update, so the best value must be paired with ``params_before``.
            params_before = current_params
            current_params, opt_state, current_value = step(current_params, opt_state)
            if current_value < best_f_for_restart:
                best_f_for_restart = current_value
                best_params_for_restart = params_before
                patience_counter = early_stop_patience
            else:
                patience_counter -= 1
                if patience_counter <= 0:
                    if verbose: log.debug(f"Early stopping at iter {iter_idx} (restart {restart_idx+1})")
                    break
            if verbose and iter_idx % 10 == 0: log.debug(f"Restart {restart_idx+1}, iter {iter_idx}, best={float(best_f_for_restart):.4e}")

        # The last update is never scored inside the loop; score it once, reusing
        # the compiled ``step`` (its update is discarded).
        _, _, final_value = step(current_params, opt_state)
        if final_value < best_f_for_restart:
            best_f_for_restart = final_value
            best_params_for_restart = current_params

        if best_f_for_restart < global_best_f:
            global_best_f = best_f_for_restart
            global_best_params = best_params_for_restart
        if verbose: log.debug(f"Restart {restart_idx+1} done. Best={float(best_f_for_restart):.4e}")

    best_params_original = scale_from_unit(global_best_params, bounds_arr) if bounds_arr is not None else global_best_params
    return jnp.array(best_params_original), float(global_best_f)


def optimize_optax_vmap(
    fun: Callable,
    fun_args: Optional[Tuple] = (),
    fun_kwargs: Optional[dict] = None,
    num_params: int = 1,
    bounds: Optional[Union[List, Tuple, jnp.ndarray]] = None,
    x0: jnp.ndarray = None,
    optimizer_options: Optional[dict] = None,
    maxiter: int = 200,
    n_restarts: int = 1,
    verbose: bool = False,
) -> Tuple[jnp.ndarray, float]:
    """
    VECTORIZED OPTIMIZER: Minimize a function using JAX + optax,
    vectorizing over restarts with jax.vmap for parallel execution. Only to be used with EI.

    ``optimizer_options`` is parsed as in :func:`optimize_optax`; the caller's
    dict is never modified.

    Requires optax to be installed. Install with: pip install 'BOBE[nn]' or pip install optax
    """
    if not OPTAX_AVAILABLE:
        raise ImportError(
            "optax is required for optimize_optax_vmap. "
            "Install it with: pip install 'BOBE[nn]' or pip install optax"
        )

    fun_args = () if fun_args is None else fun_args
    fun_kwargs = {} if fun_kwargs is None else fun_kwargs
    optimizer_options = _copy_options(optimizer_options)

    if x0 is None:
        raise ValueError("x0 must be provided (shape: (n_restarts, num_params))")

    x0 = jnp.atleast_2d(x0)
    if x0.shape[0] != n_restarts:
        raise ValueError(f"x0 has {x0.shape[0]} restarts but n_restarts was set to {n_restarts}")

    bounds_arr = _setup_bounds(bounds, num_params)
    if bounds_arr is not None:
        scaled_func = lambda x: fun(scale_from_unit(x, bounds_arr), *fun_args, **fun_kwargs) 
    else:
        scaled_func = lambda x: fun(x, *fun_args, **fun_kwargs)

    early_stop_patience = optimizer_options.pop("early_stop_patience", 25)
    lr = optimizer_options.pop("lr", 1e-3)
    optimizer_name = optimizer_options.pop("name", "adam")
    optimizer = _get_optimizer(optimizer_name, lr, optimizer_options)

    log.debug(f"Starting vectorized optax optimization with {optimizer_name}, restarts: {n_restarts}, maxiter: {maxiter}")

    def step(params, opt_state):
        val, grad = jax.value_and_grad(scaled_func)(params)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        if bounds_arr is not None:
            params = jnp.clip(params, 0.0, 1.0)
        return params, opt_state, val

    v_step = jax.jit(jax.vmap(step, in_axes=(0, 0), out_axes=(0, 0, 0)))

    params = x0
    opt_state = jax.vmap(optimizer.init)(params)
    best_vals = jnp.full(n_restarts, jnp.inf)
    best_params = jnp.zeros_like(params)
    patience_counters = jnp.full(n_restarts, early_stop_patience, dtype=jnp.int32)
    
    for iter_idx in range(maxiter):
        # ``v_step`` returns the objective at the params it was given, i.e. before
        # the update, so improvements must record ``params_before``.
        params_before = params
        params, opt_state, current_vals = v_step(params, opt_state)
        improved_mask = current_vals < best_vals
        best_vals = jnp.where(improved_mask, current_vals, best_vals)
        best_params = jnp.where(improved_mask[:, None], params_before, best_params)
        patience_counters = jnp.where(improved_mask, early_stop_patience, patience_counters - 1)

        if jnp.all(patience_counters <= 0):
            if verbose: log.debug(f"All restarts stopped early at iteration {iter_idx}")
            break
        if verbose and iter_idx % 10 == 0: log.debug(f"Iter {iter_idx}, best overall={float(jnp.min(best_vals)):.4e}")

    # The last update of each restart is never scored inside the loop; score it
    # once, reusing the compiled ``v_step`` (its update is discarded).
    _, _, final_vals = v_step(params, opt_state)
    improved_mask = final_vals < best_vals
    best_vals = jnp.where(improved_mask, final_vals, best_vals)
    best_params = jnp.where(improved_mask[:, None], params, best_params)

    best_restart_idx = jnp.argmin(best_vals)
    global_best_f = best_vals[best_restart_idx]
    global_best_params_scaled = best_params[best_restart_idx]

    if bounds_arr is not None:
        best_params_original = scale_from_unit(global_best_params_scaled, bounds_arr)
    else:
        best_params_original = global_best_params_scaled
    return jnp.array(best_params_original), float(global_best_f)


def optimize_scipy(
    fun: Callable,
    fun_args: Optional[Tuple] = (),
    fun_kwargs: Optional[dict] = None,
    num_params: int = 1,
    bounds: Optional[Union[List, Tuple, jnp.ndarray]] = None,
    x0: Optional[jnp.ndarray] = None,
    optimizer_options: Optional[dict] = None,
    maxiter: int = 200,
    n_restarts: int = 4,
    verbose: bool = False,
) -> Tuple[jnp.ndarray, float]:
    """
    Standalone method to minimize a function using scipy.optimize.minimize.

    Arguments
    ---------
    fun : Callable
        The objective function to minimize
    fun_args : tuple, optional
        Additional arguments to pass to the function
    fun_kwargs : dict, optional
        Additional keyword arguments to pass to the function
    num_params: int
        Number of parameters.
    bounds: Optional[Union[List, Tuple, jnp.ndarray]]
        Parameter bounds in shape (2, num_params).
    x0: Optional[jnp.ndarray]
        Initial guess for the parameters in the same space as ``bounds``,
        shape ``(n_restarts, num_params)`` or ``(num_params,)``.
    optimizer_options: Optional[dict]
        ``method`` (default ``'L-BFGS-B'``) plus keyword options forwarded to
        ``scipy.optimize.minimize``. If ``None``, uses
        ``{'method': 'L-BFGS-B', 'ftol': 1e-6, 'gtol': 1e-6}``. The caller's
        dict is never modified.
    maxiter: int
        Maximum number of iterations.
    n_restarts: int
        Number of restarts for the optimization.
    verbose: bool
        Whether to print progress messages.
    Returns
    -------
    Tuple[jnp.ndarray, float]
        The best parameters found and the corresponding function value.
    """

    fun_args = () if fun_args is None else fun_args
    fun_kwargs = {} if fun_kwargs is None else fun_kwargs
    optimizer_options = _copy_options(optimizer_options, default=_SCIPY_DEFAULT_OPTIONS)
    optimizer_options.update({'maxiter': maxiter})

    method = optimizer_options.pop("method", "L-BFGS-B")

    log.debug(f"Starting scipy optimization with method: {method}, restarts: {n_restarts}, maxiter: {maxiter}")

    bounds_arr = _setup_bounds(bounds, num_params)
    log.debug(f"Function bounds: {bounds_arr}")
    # Create scipy bounds format: list of (min, max) tuples
    if bounds_arr is None:
        scipy_bounds = None
    else:   
        scipy_bounds = [(float(bounds_arr[0, i]), float(bounds_arr[1, i])) for i in range(num_params)]

    # JIT-compiled function that computes both value and gradient
    @jax.jit
    def value_and_grad_func(x):
        return jax.value_and_grad(fun)(x, *fun_args, **fun_kwargs)
    
    if x0 is None:
        raise ValueError("x0 must be provided (shape: (n_restarts, num_params) or (num_params,))")

    x0 = jnp.atleast_2d(x0)
    if x0.shape[0] < n_restarts:
        raise ValueError(f"x0 provided with {x0.shape[0]} restarts but n_restarts={n_restarts}")
    elif x0.shape[0] > n_restarts:
        x0 = x0[:n_restarts]

    # Global best across all restarts
    global_best_f = np.inf
    global_best_params = None

    # check values at initial points
    for i, x_init in enumerate(x0):
        try:
            val, _ = value_and_grad_func(x_init)
            if np.isfinite(val) and val < global_best_f:
                global_best_f = val
                global_best_params = x_init
                log.debug(f"  Initial point {i+1}/{n_restarts}: New best found -> {val:.4e}")
        except Exception as e:
            log.warning(f"  Initial point {i+1}/{n_restarts}: Failed with an error: {e}")

    for i, x_init in enumerate(x0):
        try:
            result = minimize(value_and_grad_func, x_init, method=method, jac=True, bounds=scipy_bounds, options=optimizer_options)

            # Check if the result is acceptable and an improvement
            is_acceptable = result.success or "ITERATIONS REACHED LIMIT" in result.message.upper()
            is_valid = np.isfinite(result.fun)

            if is_acceptable and is_valid and result.fun < global_best_f:
                global_best_f = result.fun
                global_best_params = result.x
                status = "success" if result.success else "max iterations"
                log.debug(f"  Restart {i+1}/{n_restarts}: New best found ({status}) -> {result.fun:.4e}")
            else:
                log.debug(f"  Restart {i+1}/{n_restarts}: Finished but not improved. Status: {result.message}")

        except Exception as e:
            if verbose:
                log.warning(f"  Restart {i+1}/{n_restarts}: Failed with an error: {e}")
            continue
    
    log.debug(f"Scipy optimization ({method}) completed with {n_restarts} restarts: "
            f"Final best_f = {float(global_best_f):.4e}")
    
    return jnp.array(global_best_params), float(global_best_f)