"""
Summary plotting module for BOBE runtime visualization.

This module provides comprehensive plotting capabilities for analyzing BOBE runs,
including evidence evolution, GP hyperparameters, timing information, and convergence diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings
from pathlib import Path
import json
from .log import get_logger

log = get_logger("plot")

# try:
#     import seaborn as sns
#     HAS_SEABORN = True
# except ImportError:
#     HAS_SEABORN = False
#     warnings.warn("Seaborn not available. Using matplotlib defaults.")

try:
    from getdist import plots, MCSamples, loadMCSamples
    HAS_GETDIST = True
except ImportError:
    HAS_GETDIST = False
    warnings.warn("GetDist not available. Triangle plots will be limited.")

from .results import BOBEResults
from .core import scale_from_unit, scale_to_unit
from .log import get_logger

log = get_logger("plots")

# Set default plotting style
plt.style.use('default')

# Enable LaTeX rendering for mathematical expressions
plt.rcParams['text.usetex'] = False  # Use mathtext instead of full LaTeX for compatibility
plt.rcParams['font.family'] = 'serif'



def plot_final_samples(gp, samples_dict, param_list, param_labels, plot_params=None, param_bounds=None,
                       reference_samples=None,
                       reference_file=None, reference_ignore_rows=0., reference_label='MCMC',
                       scatter_points=False, markers=None, output_file='output', output_dir='./', **kwargs):
    """
    Plot the final samples from the Bayesian optimization process.

    Arguments
    ----------
    gp : GP object
        The Gaussian process object used for the optimization.
    samples_dict : dict
        The samples from the nested sampling or MCMC process.
    param_list : list
        The list of parameter names.
    param_labels : list
        The list of parameter labels for plotting.
    plot_params : list, optional
        The list of parameters to plot. If None, all parameters will be plotted.
    param_bounds : np.ndarray, optional
        The bounds of the parameters. If None, assumed to be [0,1] for all parameters.
    reference_samples : MCSamples, optional
        The reference getdist MCsamples from the MCMC/Nested Sampling to compare against. 
        If None, will be loaded from the reference_file.
    reference_file : str, optional
        The getdist file root containing the reference samples. If None, will be loaded from the reference_samples.
        If both are None, no reference samples will be plotted.
    reference_ignore_rows : float, optional
        The fraction of rows to ignore in the reference file. Default is 0.0.
    reference_label : str, optional
        The label for the reference samples. Default is 'MCMC'.
    scatter_points : bool, optional
        If True, scatter the training points on the plot. Default is False.
    output_file : str, optional
        The output file name for the plot. Default is 'output'.
    """
    if not HAS_GETDIST:
        log.warning("GetDist not available. Cannot create triangle plots.")
        return

    if plot_params is None:
        plot_params = param_list
    ranges = dict(zip(param_list, param_bounds.T))

    samples = samples_dict['x']

    if param_bounds is None:
        param_bounds = np.array([[0, 1]] * len(param_list)).T
    # samples = scale_from_unit(samples,param_bounds)
    weights = samples_dict['weights']
    gd_samples = MCSamples(samples=samples, names=param_list, labels=param_labels, 
                           ranges=ranges, weights=weights)
    

    plot_samples = [gd_samples]

    if reference_file is not None:
        ref_samples = loadMCSamples(reference_file, settings={'ignore_rows': reference_ignore_rows})
        plot_samples.append(ref_samples)
    elif reference_samples is not None:
        plot_samples.append(reference_samples)

    labels = ['GP', reference_label]

    for label, s in zip(labels, plot_samples):
        log.info(f"Parameter limits from {label}")
        for key in plot_params:
            log.info(s.getInlineLatex(key, limit=1))
    
    ndim = len(plot_params)

    g = plots.get_subplot_plotter(subplot_size=2.5, subplot_size_ratio=1)
    g.settings.legend_fontsize = 22
    g.settings.axes_fontsize = 20
    g.settings.axes_labelsize = 20
    g.settings.title_limit_fontsize = 14   
    g.triangle_plot(plot_samples, params=plot_params, filled=[True, False],
                    contour_colors=['#006FED', 'black'], contour_lws=[1, 1.5],
                    legend_labels=['GP', f'{reference_label}'],
                    markers=markers, marker_args={'lw': 1, 'ls': ':'}) 
    if scatter_points:
        points = scale_from_unit(gp.train_x, param_bounds)
        for i in range(ndim):
            # ax = g.subplots[i,i]
            for j in range(i+1, ndim):
                ax = g.subplots[j, i]
                ax.scatter(points[:, i], points[:, j], alpha=0.5, color='forestgreen', s=5)

    g.export(output_dir + output_file + '_param_posteriors.pdf')




class BOBESummaryPlotter:
    """
    Comprehensive plotting class for BOBE run analysis and diagnostics.
    """
    
    def __init__(self, results: Union[BOBEResults, str], figsize_scale: float = 1.0):
        """
        Initialize the plotter with BOBE results.
        
        Args:
            results: BOBEResults object or path to results file
            figsize_scale: Scale factor for figure sizes (default: 1.0)
        """
        if isinstance(results, str):
            self.results = BOBEResults.load_results(results)
            self.output_file = results
        else:
            self.results = results
            self.output_file = results.output_file
            
        self.figsize_scale = figsize_scale
        self.param_names = self.results.param_names
        self.param_labels = self.results.param_labels
        self.ndim = self.results.ndim
        
        log.info(f"Initialized summary plotter for {self.ndim}D problem: {self.output_file}")
    
    def _format_latex_label(self, label: str) -> str:
        """
        Format a parameter label for proper LaTeX rendering in matplotlib.
        
        Args:
            label: Raw parameter label
            
        Returns:
            Formatted label with proper LaTeX delimiters
        """
        # Simply wrap in math mode if not already wrapped
        if not label.startswith('$') and not label.endswith('$'):
            label = f'${label}$'
        return label
    
    def plot_evidence_evolution(self, logz_data: Optional[Dict] = None,
                               ax: Optional[plt.Axes] = None, 
                               show_convergence: bool = True) -> plt.Axes:
        """
        Plot the evolution of log evidence (logZ) with error bounds.
        
        Args:
            logz_data: Dictionary containing logZ evolution data (uses results if None)
            ax: Matplotlib axes to plot on (creates new if None)
            show_convergence: Whether to mark convergence points
            
        Returns:
            The matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8*self.figsize_scale, 5*self.figsize_scale))
        
        # Use provided data or fall back to results
        if logz_data is not None:
            logz_evolution = logz_data.get('logz_evolution', [])
        else:
            logz_evolution = self.results.logz_evolution
        
        if not logz_evolution:
            ax.text(0.5, 0.5, 'No logZ evolution data available', 
                   transform=ax.transAxes, ha='center', va='center')
            return ax
        
        # Extract data
        data = [(entry['iteration'], entry['logz'], entry['logz_var'], entry['logz_std'],
                entry.get('logz_upper', entry['logz'] + entry.get('logz_err', 0)),
                entry.get('logz_lower', entry['logz'] - entry.get('logz_err', 0)))
                for entry in logz_evolution]
        iterations, logz_values, logz_var, logz_std, logz_upper, logz_lower = map(np.array, zip(*data))

        # Plot evolution with uncertainty
        ax.plot(iterations, logz_values, 'b-', linewidth=2, label='Mean log Z', alpha=0.9)
        # ax.fill_between(iterations, logz_values - logz_std, logz_values + logz_std, 
        #                alpha=0.2, color='red', label='1$\sigma$ region')
        ax.plot(iterations, logz_upper, 'r--', linewidth=1.5, alpha=0.7, label='Upper bound')
        ax.plot(iterations, logz_lower, 'g--', linewidth=1.5, alpha=0.7, label='Lower bound')
        ax.fill_between(iterations, logz_lower, logz_upper, 
                       alpha=0.2, color='blue', label='Uncertainty region')
        
        # Mark convergence points
        if show_convergence:
            # Use provided convergence data or fall back to results
            convergence_history = (logz_data.get('convergence_history', []) if logz_data is not None 
                                 else self.results.convergence_history)
            if convergence_history:
                conv_iterations = [conv.iteration for conv in convergence_history if conv.converged]
                for i, conv_iter in enumerate(conv_iterations):
                    idx = np.searchsorted(iterations, conv_iter)
                    if idx < len(logz_values):
                        ax.axvline(conv_iter, color='red', linestyle='--', alpha=0.7)
                        ax.scatter(conv_iter, logz_values[idx], color='red', s=50, 
                                 marker='o', zorder=5, label='Convergence' if i == 0 else "")
        
        # Final logZ
        final_logz_dict = (logz_data.get('final_logz_dict', {}) if logz_data is not None 
                          else self.results.final_logz_dict)
        if final_logz_dict:
            final_logz = final_logz_dict.get('mean', np.nan)
            if not np.isnan(final_logz):
                ax.axhline(final_logz, color='green', linestyle='-', alpha=0.7, 
                          linewidth=2, label=f'Final log Z = {final_logz:.3f}')
                # limit plot range to +-25 of final logz
                # ax.set_ylim(final_logz - 25, final_logz + 25)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(r'$ \log Z$')
        ax.set_title('Evidence Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Force integer ticks on x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        return ax
    
    def plot_gp_lengthscales(self, gp_data: Optional[Dict] = None, 
                            ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot evolution of GP lengthscales only.
        
        Args:
            gp_data: Dictionary containing GP hyperparameter evolution data
            ax: Matplotlib axes to plot on (creates new if None)
            
        Returns:
            The matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10*self.figsize_scale, 6*self.figsize_scale))
        
        if gp_data is None:
            ax.text(0.5, 0.5, 'No GP lengthscale data provided\n'
                   'Pass gp_data dictionary with evolution info', 
                   transform=ax.transAxes, ha='center', va='center')
            return ax
        
        # Extract hyperparameter evolution
        if 'iterations' not in gp_data or 'lengthscales' not in gp_data:
            ax.text(0.5, 0.5, 'Invalid GP data format\n'
                   'Need "iterations" and "lengthscales" keys', 
                   transform=ax.transAxes, ha='center', va='center')
            return ax
        
        iterations = np.array(gp_data['iterations'])
        lengthscales = np.array(gp_data['lengthscales'])  # Shape: [n_iterations, n_params]

        # Check if we have any data and if lengthscales has the right dimensions
        if len(iterations) == 0 or len(lengthscales) == 0:
            ax.text(0.5, 0.5, 'No GP lengthscale data available', 
                   transform=ax.transAxes, ha='center', va='center')
            return ax
        
        # Ensure lengthscales is 2D
        if lengthscales.ndim == 1:
            ax.text(0.5, 0.5, 'GP lengthscale data has incorrect shape\n'
                   'Expected 2D array [n_iterations, n_params]', 
                   transform=ax.transAxes, ha='center', va='center')
            return ax

        log.debug(f"shape {lengthscales.shape}")

        # Plot lengthscales for each parameter
        colors = plt.cm.Set1(np.linspace(0, 1, self.ndim))
        for i in range(self.ndim):
            if i < lengthscales.shape[1]:
                # Format parameter label for LaTeX rendering
                label = self._format_latex_label(self.param_labels[i])
                
                ax.plot(iterations, lengthscales[:, i], 
                       color=colors[i], linewidth=2, 
                       label=label)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Lengthscale')
        ax.set_title('GP Lengthscale Evolution')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Force integer ticks on x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        return ax
    
    def plot_gp_kernel_variance(self, gp_data: Optional[Dict] = None, 
                           ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot evolution of GP kernel variance only.
        
        Args:
            gp_data: Dictionary containing GP hyperparameter evolution data
            ax: Matplotlib axes to plot on (creates new if None)
            
        Returns:
            The matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10*self.figsize_scale, 6*self.figsize_scale))
        
        if gp_data is None:
            ax.text(0.5, 0.5, 'No GP kernel variance data provided\n'
                   'Pass gp_data dictionary with evolution info', 
                   transform=ax.transAxes, ha='center', va='center')
            return ax
        
        # Extract kernel variance evolution
        if 'iterations' not in gp_data or 'kernel_variances' not in gp_data:
            # Check for backward compatibility with old 'outputscales' key
            if 'outputscales' in gp_data:
                kernel_variances_key = 'outputscales'
                display_name = 'Kernel Variance (from outputscales)'
            else:
                ax.text(0.5, 0.5, 'Invalid GP data format\n'
                       'Need "iterations" and "kernel_variances" keys', 
                       transform=ax.transAxes, ha='center', va='center')
                return ax
        else:
            kernel_variances_key = 'kernel_variances'
            display_name = 'Kernel Variance'
        
        iterations = np.array(gp_data['iterations'])
        kernel_variances = np.array(gp_data[kernel_variances_key])
        
        # Plot kernel variance
        ax.plot(iterations, kernel_variances, 'purple', linewidth=2, 
               label='Kernel Variance', alpha=0.8)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Kernel Variance')
        ax.set_title('GP Kernel Variance Evolution')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Force integer ticks on x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        return ax
    
    def plot_gp_hyperparameters(self, gp_data: Optional[Dict] = None, 
                               ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot evolution of GP hyperparameters (backward compatibility - now plots lengthscales only).
        
        Args:
            gp_data: Dictionary containing GP hyperparameter evolution data
            ax: Matplotlib axes to plot on (creates new if None)
            
        Returns:
            The matplotlib axes object
        """
        # For backward compatibility, this now calls the lengthscales plot
        return self.plot_gp_lengthscales(gp_data=gp_data, ax=ax)
    
    def plot_best_loglike_evolution(self, best_loglike_data: Optional[Dict] = None,
                                   ax: Optional[plt.Axes] = None, scatter_improvements = False) -> plt.Axes:
        """
        Plot evolution of the best log-likelihood found so far.
        
        Args:
            best_loglike_data: Dictionary with 'iterations' and 'best_loglike' keys
            ax: Matplotlib axes to plot on (creates new if None)
            
        Returns:
            The matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8*self.figsize_scale, 5*self.figsize_scale))
        
        if best_loglike_data is None:
            ax.text(0.5, 0.5, 'No best log-likelihood data provided\n'
                   'Pass best_loglike_data dictionary', 
                   transform=ax.transAxes, ha='center', va='center')
            return ax
        
        if 'iterations' not in best_loglike_data or 'best_loglike' not in best_loglike_data:
            ax.text(0.5, 0.5, 'Invalid best log-likelihood data format\n'
                   'Need "iterations" and "best_loglike" keys', 
                   transform=ax.transAxes, ha='center', va='center')
            return ax
        
        iterations = np.array(best_loglike_data['iterations'])
        best_loglike = np.array(best_loglike_data['best_loglike'])
        
        # Plot evolution
        ax.plot(iterations, best_loglike, 'g-', linewidth=2, 
               label='Best log-likelihood', alpha=0.8)
        
        # Mark improvements

        if scatter_improvements:
            improvements = np.diff(best_loglike) > 0
            if np.any(improvements):
                improve_iter = iterations[1:][improvements]
                improve_vals = best_loglike[1:][improvements]
                ax.scatter(improve_iter, improve_vals, color='red', s=10, 
                      marker='o', alpha=0.5, label='Improvements')
        
        # Final value
        if len(best_loglike) > 0:
            final_val = best_loglike[-1]
            ax.axhline(final_val, color='green', linestyle='--', alpha=0.5,
                      label=f'Final best = {final_val:.3f}')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best log-likelihood')
        ax.set_title('Best Log-likelihood Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Force integer ticks on x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        return ax
    
    def plot_acquisition_evolution(self, acquisition_data: Optional[Dict] = None,
                                  ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot the evolution of acquisition function values throughout iterations.
        
        Args:
            acquisition_data: Dictionary with acquisition data (gets from results if None)
            ax: Matplotlib axes to plot on (creates new if None)
            
        Returns:
            The matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10*self.figsize_scale, 6*self.figsize_scale))
        
        # Get acquisition data
        if acquisition_data is None:
            acquisition_data = self.results.get_acquisition_data()
        
        if not acquisition_data or 'iterations' not in acquisition_data or len(acquisition_data['iterations']) == 0:
            ax.text(0.5, 0.5, 'No acquisition function data available', 
                   transform=ax.transAxes, ha='center', va='center')
            return ax
        
        iterations = np.array(acquisition_data['iterations'])
        values = np.array(acquisition_data['values'])
        functions = acquisition_data['functions']
        
        # Transform and plot each function type
        unique_functions = list(set(functions))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_functions)))
        
        y_values = []
        for i, func_name in enumerate(unique_functions):
            mask = np.array(functions) == func_name
            func_iterations, func_values = iterations[mask], values[mask]
            
            # Transform values based on function type
            if func_name in ["WIPV", "EI"]:
                #plot log10 of values of EI and WIPV
                func_values = np.log10(func_values + 1e-100) # avoid log(0)
            else:
                func_values = func_values / np.log(10) # convert to log10 from log for LogEI

            label = func_name

            y_values.extend(func_values)
            
            # Plot with lines connecting points
            ax.plot(func_iterations, func_values, 'o-', color=colors[i], 
                   label=label, alpha=0.7, markersize=4, linewidth=1)
        
        # Mark function switches
        if len(unique_functions) > 1:
            switch_points = [iterations[i] for i in range(1, len(functions)) 
                           if functions[i] != functions[i-1]]
            for i, switch_iter in enumerate(switch_points):
                ax.axvline(switch_iter, color='red', linestyle='--', alpha=0.5,
                          label='Function switch' if i == 0 else '')
        
        # Set reasonable y-limits
        if y_values:
            y_min, y_max = np.min(y_values), np.max(y_values)
            ax.set_ylim(max(y_min-1, -20.), min(y_max+1, 5.))
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Acquisition Function Value (log 10)')
        ax.set_title('Acquisition Function Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_timing_breakdown(self, timing_data: Optional[Dict] = None,
                             ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot timing breakdown of different phases of the algorithm.
        
        Args:
            timing_data: Dictionary with timing information for different phases
            ax: Matplotlib axes to plot on (creates new if None)
            
        Returns:
            The matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8*self.figsize_scale, 6*self.figsize_scale))
        
        # Prepare timing data
        if timing_data is None:
            # Use proper timing summary that accounts for resumed runs
            timing_summary = self.results.get_timing_summary()
            timing_data = timing_summary['phase_times'].copy()
            timing_data['Total Runtime'] = timing_summary['total_runtime']
        elif 'phase_times' in timing_data:
            timing_data = {phase: time for phase, time in timing_data['phase_times'].items() if time > 0}
        
        phases, times = list(timing_data.keys()), list(timing_data.values())
        
        # Create bar plot
        colors = plt.cm.Set3(np.linspace(0, 1, len(phases))) if len(phases) > 1 else ['skyblue']
        bars = ax.bar(range(len(phases)), times, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                   f'{time_val:.1f}s', ha='center', va='bottom')
        
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, rotation=45 if len(phases) > 1 else 0, ha='right' if len(phases) > 1 else 'center')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Runtime Breakdown' if len(phases) > 1 else 'Runtime Information')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        return ax
    
    def plot_convergence_diagnostics(self, convergence_data: Optional[Dict] = None,
                                     ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot convergence diagnostics including thresholds and delta evolution.
        
        Args:
            convergence_data: Dictionary containing convergence history data (uses results if None)
            ax: Matplotlib axes to plot on (creates new if None)
            
        Returns:
            The matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8*self.figsize_scale, 5*self.figsize_scale))
        
        # Use provided data or fall back to results
        if convergence_data is not None:
            convergence_history = convergence_data.get('convergence_history', [])
        else:
            convergence_history = self.results.convergence_history
        
        if not convergence_history:
            ax.text(0.5, 0.5, 'No convergence history available', 
                   transform=ax.transAxes, ha='center', va='center')
            return ax
        
        # Extract and plot convergence data
        conv_data = [(conv.iteration, conv.delta, conv.threshold, conv.converged) 
                     for conv in convergence_history]
        iterations, deltas, thresholds, converged_flags = zip(*conv_data)

        ax.plot(iterations, deltas, 'b-', linewidth=2, label=r'$\Delta \log Z$', alpha=0.8)
        ax.plot(iterations, thresholds, 'r--', linewidth=2, label='Threshold', alpha=0.7)
        
        # Mark convergence points
        conv_points = [(it, delta) for it, delta, conv in zip(iterations, deltas, converged_flags) if conv]
        if conv_points:
            conv_its, conv_deltas = zip(*conv_points)
            ax.scatter(conv_its, conv_deltas, color='green', s=50, 
                      marker='o', zorder=5, label='Converged points')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel(r'$\Delta \log Z$')
        ax.set_title('Convergence Diagnostics')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Force integer ticks on x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        return ax
    
    def plot_kl_divergences(self, kl_data: Optional[Dict] = None,
                           ax: Optional[plt.Axes] = None, annotate=False) -> plt.Axes:
        """
        Plot successive KL divergences between NS iterations (reverse, forward, symmetric).
        
        Args:
            kl_data: Dictionary containing KL divergence data (uses results if None)
            ax: Matplotlib axes to plot on (creates new if None)
            
        Returns:
            The matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8*self.figsize_scale, 5*self.figsize_scale))
        
        def handle_invalid_kl_value(val, large_value=1e3):
            """Replace invalid/infinite KL values with a large number for consistent plotting."""
            if np.isfinite(val) and val >= 0:
                return min(val, large_value)  # Cap at large_value but keep valid values
            else:
                return large_value  # Replace invalid values with large_value
        
        # Define successive KL data sources and labels
        successive_kl_sources = [
            ('reverse', 'Reverse KL', '#d62728'),
            ('forward', 'Forward KL', '#1f77b4'), 
            ('symmetric', 'Symmetric KL', '#2ca02c')
        ]
        
        plot_count = 0
        
        # Use provided data or fall back to results
        successive_kl = (kl_data.get('successive_kl', []) if kl_data is not None 
                        else getattr(self.results, 'successive_kl', []))
        
        if successive_kl:
            for value_key, label, color in successive_kl_sources:
                iterations, values = [], []
                
                for entry in successive_kl:
                    iterations.append(entry.get('iteration', 0))
                    values.append(handle_invalid_kl_value(entry.get(value_key, np.nan)))
                
                # Plot all data points (no filtering)
                if iterations and values:
                    ax.plot(iterations, values, 'o-', color=color, label=label, 
                           linewidth=2, markersize=6, alpha=0.8)
                    plot_count += 1
        
        # Handle empty plot
        if plot_count == 0:
            ax.text(0.5, 0.5, 'No successive KL divergence data available', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title('Successive KL Divergences')
            return ax
        
        # Configure plot
        ax.set_xlabel('Iteration')
        ax.set_ylabel('KL Divergence')
        ax.set_title('Successive KL Divergences')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Force integer ticks on x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add annotation
        if annotate:
            ax.text(0.02, 0.98, 'KL divergences between successive NS iterations\n'
                            'Invalid values shown as 1000', 
               transform=ax.transAxes, va='top', ha='left', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7), fontsize=10)
        
        return ax
    
    def plot_parameter_evolution(self, param_evolution_data: Optional[Dict] = None,
                                max_params: int = 4) -> plt.Figure:
        """
        Plot evolution of parameter values during optimization.
        
        Args:
            param_evolution_data: Dictionary with parameter evolution data
            max_params: Maximum number of parameters to plot
            
        Returns:
            The matplotlib figure object
        """
        n_plot = min(self.ndim, max_params)
        fig, axes = plt.subplots(n_plot, 1, figsize=(8*self.figsize_scale, 3*n_plot*self.figsize_scale))
        axes = [axes] if n_plot == 1 else axes
        
        if param_evolution_data is None:
            for i, ax in enumerate(axes):
                ax.text(0.5, 0.5, f'No evolution data for {self.param_names[i]}', 
                       transform=ax.transAxes, ha='center', va='center')
            return fig
        
        for i in range(n_plot):
            param_name = self.param_names[i]
            data = param_evolution_data.get(param_name, {})
            iterations, values = data.get('iterations', []), data.get('values', [])
            
            if iterations and values:
                axes[i].plot(iterations, values, 'o-', linewidth=1, markersize=3, alpha=0.7)
                axes[i].set_ylabel(self._format_latex_label(self.param_labels[i]))
                axes[i].grid(True, alpha=0.3)
                
                # Add parameter bounds
                if hasattr(self.results, 'param_bounds') and i < len(self.results.param_bounds[0]):
                    lower, upper = self.results.param_bounds[0, i], self.results.param_bounds[1, i]
                    axes[i].axhline(lower, color='red', linestyle=':', alpha=0.5, label='Bounds')
                    axes[i].axhline(upper, color='red', linestyle=':', alpha=0.5)
                    axes[i].legend()
            else:
                axes[i].text(0.5, 0.5, f'No data for {param_name}', 
                           transform=axes[i].transAxes, ha='center', va='center')
        
        axes[-1].set_xlabel('Iteration')
        plt.tight_layout()
        return fig
    
    def create_summary_dashboard(self, 
                                logz_data: Optional[Dict] = None,
                                convergence_data: Optional[Dict] = None,
                                kl_data: Optional[Dict] = None,
                                gp_data: Optional[Dict] = None,
                                best_loglike_data: Optional[Dict] = None,
                                acquisition_data: Optional[Dict] = None,
                                timing_data: Optional[Dict] = None,
                                save_path: Optional[str] = None,
                                title: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive summary dashboard with all diagnostic plots.
        
        Args:
            logz_data: Log evidence evolution data
            convergence_data: Convergence diagnostics data
            kl_data: KL divergence data
            gp_data: GP hyperparameter evolution data
            best_loglike_data: Best log-likelihood evolution data
            acquisition_data: Acquisition function evolution data
            timing_data: Timing breakdown data
            save_path: Path to save the figure (optional)
            
        Returns:
            The matplotlib figure object
        """
        # Create figure with subplots (3x3 grid to include KL divergences)
        fig = plt.figure(figsize=(18*self.figsize_scale, 18*self.figsize_scale))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Top row: Evidence, GP lengthscales, GP kernel variance
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_evidence_evolution(logz_data=logz_data, ax=ax1)
        
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_gp_lengthscales(gp_data=gp_data, ax=ax2)
        
        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_gp_kernel_variance(gp_data=gp_data, ax=ax3)

        # Middle row: Convergence, Best log-likelihood, Acquisition
        ax4 = fig.add_subplot(gs[1, 0])
        self.plot_convergence_diagnostics(convergence_data=convergence_data, ax=ax4)

        ax5 = fig.add_subplot(gs[1, 1])
        self.plot_best_loglike_evolution(best_loglike_data=best_loglike_data, ax=ax5)

        ax6 = fig.add_subplot(gs[1, 2])
        self.plot_acquisition_evolution(acquisition_data=acquisition_data, ax=ax6)
        
        # Bottom row: KL divergences, Timing breakdown, Summary stats
        ax7 = fig.add_subplot(gs[2, 0])
        self.plot_kl_divergences(kl_data=kl_data, ax=ax7)
        
        ax8 = fig.add_subplot(gs[2, 1])
        self.plot_timing_breakdown(timing_data=timing_data, ax=ax8)
        
        ax9 = fig.add_subplot(gs[2, 2])
        self.plot_summary_stats(ax=ax9)
        
        # Add overall title
        title_str = title if title is not None else self.output_file
        fig.suptitle(f'BOBE Summary: {title_str}', 
                    fontsize=18*self.figsize_scale, y=0.95)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            log.info(f"Saved summary plot to {save_path}")
        
        return fig
    
    def plot_summary_stats(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot key summary statistics as text.
        
        Args:
            ax: Matplotlib axes to plot on (creates new if None)
            
        Returns:
            The matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6*self.figsize_scale, 4*self.figsize_scale))
        
        # Basic run info
        stats_lines = [
            f"Dimensions: {self.ndim}D",
            f"Likelihood: {self.results.likelihood_name}"
        ]
        
        # GP size (number of training points)
        gp_size = self.results.gp_info.get("gp_training_set_size", "N/A")
        stats_lines.append(f"GP size: {gp_size}")

        # Classifier info
        classifier_used = self.results.gp_info.get("classifier_used", False)
        classifier_type = self.results.gp_info.get("classifier_type", "N/A")
        if classifier_used:
            stats_lines.append(f"Classifier: {classifier_type}")
            total_evals = self.results.gp_info.get("classifier_training_set_size", "N/A")
        else:
            stats_lines.append("Classifier: No")
            total_evals = gp_size  # Total evaluations = GP size if no classifier

        # Total evaluations
        stats_lines.append(f"Total evaluations: {total_evals}")

        # Evidence estimate
        if self.results.final_logz_dict:
            logz_data = self.results.final_logz_dict
            logz = logz_data.get('mean', np.nan)
            # Use std first, then fall back to upper-lower bounds calculation  
            logz_err = logz_data.get('std', np.nan)
            if np.isnan(logz_err) and 'upper' in logz_data and 'lower' in logz_data:
                logz_err = (logz_data['upper'] - logz_data['lower']) / 2.0
            
            if not np.isnan(logz):
                if not np.isnan(logz_err):
                    stats_lines.append(f"log Z = {logz:.4f} ± {logz_err:.4f}")
                else:
                    stats_lines.append(f"log Z = {logz:.4f}")

        # Runtime and convergence - use timing summary for proper total runtime including resumed runs
        timing_summary = self.results.get_timing_summary()
        total_runtime = timing_summary['total_runtime']
        if total_runtime > 0:
            runtime_str = f"{total_runtime/3600:.2f} hours" if total_runtime > 3600 else f"{total_runtime:.1f} seconds"
            stats_lines.append(f"Runtime: {runtime_str}")
                    
        stats_lines.extend([
            f"Converged: {'Yes' if self.results.converged else 'No'}",
            f"Termination: {self.results.termination_reason}"
        ])
        
        # Display formatted text
        ax.text(0.1, 0.95, '\n'.join(stats_lines), transform=ax.transAxes, 
               fontsize=12*self.figsize_scale, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Run summary',loc='center')
        
        return ax
    
    def save_all_plots(self, output_dir: Optional[str] = None, **data_kwargs):
        """
        Save all individual plots and the summary dashboard.
        
        Args:
            output_dir: Directory to save plots (uses output_file base if None)
            **data_kwargs: Data dictionaries for different plot types
        """
        if output_dir is None:
            output_dir = Path(self.output_file).parent
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        base_name = Path(self.output_file).stem
        
        # Individual plots
        plots_to_save = [
            ('evidence_evolution', self.plot_evidence_evolution),
            ('convergence_diagnostics', self.plot_convergence_diagnostics),
            ('kl_divergences', self.plot_kl_divergences),
        ]
        
        # Optional plots with data
        if 'gp_data' in data_kwargs:
            plots_to_save.append(('gp_lengthscales', 
                                lambda ax: self.plot_gp_lengthscales(data_kwargs['gp_data'], ax=ax)))
            plots_to_save.append(('gp_kernel_variance', 
                                lambda ax: self.plot_gp_kernel_variance(data_kwargs['gp_data'], ax=ax)))
        
        if 'best_loglike_data' in data_kwargs:
            plots_to_save.append(('best_loglike_evolution', 
                                lambda ax: self.plot_best_loglike_evolution(data_kwargs['best_loglike_data'], ax=ax)))
        
        if 'timing_data' in data_kwargs:
            plots_to_save.append(('timing_breakdown', 
                                lambda ax: self.plot_timing_breakdown(data_kwargs['timing_data'], ax=ax)))
        
        # Save individual plots
        for plot_name, plot_func in plots_to_save:
            fig, ax = plt.subplots(1, 1, figsize=(8*self.figsize_scale, 6*self.figsize_scale))
            if plot_name == 'evidence_evolution':
                plot_func(logz_data=data_kwargs.get('logz_data'), ax=ax)
            elif plot_name == 'convergence_diagnostics':
                plot_func(convergence_data=data_kwargs.get('convergence_data'), ax=ax)
            elif plot_name == 'kl_divergences':
                plot_func(kl_data=data_kwargs.get('kl_data'), ax=ax)
            else:
                plot_func(ax=ax)
            plt.tight_layout()
            save_path = output_dir / f"{base_name}_{plot_name}.pdf"
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
            log.info(f"Saved {plot_name} to {save_path}")
        
        # Save summary dashboard
        dashboard_fig = self.create_summary_dashboard(**data_kwargs)
        dashboard_path = output_dir / f"{base_name}_summary_dashboard.pdf"
        dashboard_fig.savefig(dashboard_path, bbox_inches='tight')
        plt.close(dashboard_fig)
        log.info(f"Saved summary dashboard to {dashboard_path}")


def create_summary_plots(results_file: str, 
                        gp_data: Optional[Dict] = None,
                        best_loglike_data: Optional[Dict] = None,
                        timing_data: Optional[Dict] = None,
                        param_evolution_data: Optional[Dict] = None,
                        output_dir: Optional[str] = None,
                        figsize_scale: float = 1.0) -> BOBESummaryPlotter:
    """
    Convenience function to create all summary plots for a BOBE run.
    
    Args:
        results_file: Path to BOBE results file (without extension)
        gp_data: GP hyperparameter evolution data
        best_loglike_data: Best log-likelihood evolution data  
        timing_data: Timing breakdown data
        param_evolution_data: Parameter evolution data
        output_dir: Directory to save plots
        figsize_scale: Scale factor for figure sizes
        
    Returns:
        BOBESummaryPlotter object
    """
    plotter = BOBESummaryPlotter(results_file, figsize_scale=figsize_scale)
    
    # Create and save all plots
    plotter.save_all_plots(
        output_dir=output_dir,
        gp_data=gp_data,
        best_loglike_data=best_loglike_data,
        timing_data=timing_data,
        param_evolution_data=param_evolution_data
    )
    
    return plotter


# Data format documentation for users
def get_data_format_examples() -> Dict[str, Dict]:
    """
    Return example data formats for the plotting functions.
    
    Returns:
        Dictionary with example data structures
    """
    examples = {
        'gp_data': {
            'iterations': [10, 20, 30, 40, 50],
            'lengthscales': [[1.0, 0.5], [0.8, 0.6], [0.7, 0.7], [0.6, 0.8], [0.5, 0.9]],
            'kernel_variances': [2.0, 1.8, 1.6, 1.4, 1.2]
        },
        'best_loglike_data': {
            'iterations': [1, 5, 10, 15, 20],
            'best_loglike': [-10.0, -8.5, -7.2, -6.8, -6.5]
        },
        'timing_data': {
            'GP Training': 45.2,
            'Nested Sampling': 120.8,
            'Optimization': 30.1,
            'I/O Operations': 5.3
        },
    }
    
    return examples