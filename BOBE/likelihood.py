# Class for implementing external loglikelihoods
import numpy as np
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
from .utils.log import get_logger

log = get_logger("likelihood")


class Likelihood:
    """
    Base class for log-likelihoods with common evaluation logic.
    
    Parameters
    ----------
    loglikelihood : callable
        Log-likelihood function that takes parameter array and returns float.
    param_list : list of str
        List of parameter names.
    param_labels : list of str, optional
        LaTeX labels for parameters. Default is None (uses param_list).
    param_bounds : array-like, optional
        Parameter bounds, shape (2, ndim). Default is None (unit cube).
    name : str, optional
        Name for this likelihood. Default is "loglikelihood".
    minus_inf : float, optional
        Value to return for failed evaluations. Default is -1e5.
    """

    def __init__(self,
                 loglikelihood: Callable,
                 param_list: Optional[List[str]],
                 param_labels: Optional[List[str]] = None,
                 param_bounds: Optional[Union[List, np.ndarray]] = None,
                 name: Optional[str] = None,
                 minus_inf: float = -1e10):
        

        self.logl = loglikelihood

        # check param list in correct format
        if not all(isinstance(p, str) for p in param_list):
            raise ValueError("All elements of param_list must be strings corresponding to parameter names.")


        self.param_list = param_list
        self.ndim = len(self.param_list)
        self.param_labels = param_labels if param_labels is not None else param_list
        if param_bounds is None:
            self.param_bounds = np.array(self.ndim * [[0, 1]]).T
            log.warning("No param_bounds provided. Assuming unit cube [0,1] for all parameters.")
        else:
            # validate param_bounds shape
            param_bounds = np.array(param_bounds)
            if param_bounds.shape != (2, self.ndim):
                raise ValueError(f"param_bounds must have shape (2, {self.ndim}), but got {param_bounds.shape}.")
            if not np.all(param_bounds[1] > param_bounds[0]):
                raise ValueError("Upper parameter bounds must be strictly greater than lower bounds.")
            self.param_bounds = param_bounds
        
        self.name = name or "loglikelihood"
        self.minus_inf = minus_inf
        self.logprior_vol = float(np.sum(np.log(self.param_bounds[1] - self.param_bounds[0])))

        log.info(f"Initialized {self.name} with {self.ndim} params")
        log.info(f"Param list: {self.param_list}")
        bounds_str = ", ".join([f"'{name}': [{self.param_bounds[0, i]:.6g}, {self.param_bounds[1, i]:.6g}]" 
                                for i, name in enumerate(self.param_list)])
        log.info(f"Param bounds: {{{bounds_str}}}")
        log.info(f"Logprior volume = {self.logprior_vol:.4f}")

    def _safe_eval(self, x: np.ndarray) -> float:
        """
        Safely evaluate log-likelihood at a single point.
        
        Parameters
        ----------
        x : np.ndarray
            Parameter vector to evaluate.
            
        Returns
        -------
        float
            Log-likelihood value, or minus_inf if evaluation fails.
        """
        try:
            val = float(self.logl(x))
        except Exception:
            log.debug(f"Log-likelihood evaluation failed at point {x}", exc_info=True)
            return self.minus_inf

        if np.isnan(val) or np.isinf(val) or val < self.minus_inf:
            return self.minus_inf
        return val

    def __call__(self, X: Union[np.ndarray, List[float]]) -> float:
        """
        Evaluate the likelihood function at a single point.
        
        This method is designed to be called by pool.run_map_objective, which
        handles distributing tasks across MPI processes in parallel mode
        or iterating in serial mode.
        
        Parameters
        ----------
        X : array-like, shape (ndim,) or (1, ndim)
            Single input parameter vector.
        
        Returns
        -------
        val : float
            Log-likelihood value at the input point.
        """
        X = np.atleast_1d(X)
        if X.ndim > 1:
            if X.shape[0] != 1:
                raise ValueError(
                    "__call__ expects a single point. "
                    "Use pool.run_map_objective for batch evaluations."
                )
            X = X.flatten()
        
        if X.shape[0] != self.ndim:
            raise ValueError(f"Input shape {X.shape} does not match ndim {self.ndim}")
        
        return self._safe_eval(X)


class CobayaLikelihood(Likelihood):
    """Likelihood wrapper for Cobaya models.

    Parameters
    ----------
    input_file_dict : str or dict
        Cobaya input YAML file path or input dictionary.
    confidence_for_unbounded : float, optional
        Confidence level for unbounded priors. Default is 0.9999995.
    minus_inf : float, optional
        Value to return for failed/minus infinity evaluations. Default is -1e10.
    name : str, optional
        Name for this likelihood. Default is "CobayaLikelihood".
    """

    def __init__(self,
                 input_file_dict: Union[str, Dict[str, Any]],
                 confidence_for_unbounded: float = 0.9999995,
                 minus_inf: float = -1e10,
                 name: str = "CobayaLikelihood"):
        
        
        try:
            from cobaya.yaml import yaml_load
            from cobaya.model import get_model
        except ImportError:
            log.error("Cobaya is not installed.",exc_info=True)
            raise

        if isinstance(input_file_dict, str):
            info = yaml_load(input_file_dict)
        else:
            info = input_file_dict

        cobaya_model = get_model(info)

        param_list = list(cobaya_model.parameterization.sampled_params())
        param_bounds = np.array(
            cobaya_model.prior.bounds(confidence_for_unbounded=confidence_for_unbounded)
        ).T
        param_labels = [cobaya_model.parameterization.labels()[k] for k in param_list]

        def cobaya_logp(x):
            return cobaya_model.logpost(x, make_finite=False)

        super().__init__(loglikelihood=cobaya_logp,
                         param_list=param_list,
                         param_labels=param_labels,
                         param_bounds=param_bounds,
                         name=name,
                         minus_inf=minus_inf)

        self.cobaya_model = cobaya_model


    def __call__(self, X) -> float:
        """Evaluate Cobaya likelihood with logprior volume correction. This is added to match Cobaya behaviour."""
        val = super().__call__(X) 
        if val <= self.minus_inf:
            val = self.minus_inf
        return val + self.logprior_vol

    def _get_single_valid_point(self, rng: np.random.Generator) -> tuple:
        """
        WORKER METHOD: Gets a single valid point from the Cobaya model.
        This is the task that will be executed in parallel by each worker.
        """
        pt, res = self.cobaya_model.get_valid_point(
            max_tries=1000, 
            ignore_fixed_ref=False,
            logposterior_as_dict=True, 
            random_state=rng
        )
        
        lp = res["logpost"]
        if lp < self.minus_inf:
            lp = self.minus_inf
        logpost = lp + self.logprior_vol
        
        return (pt, logpost)