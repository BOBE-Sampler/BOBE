import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    os.cpu_count()
)
from BOBE import BOBE, GP, RBFKernel
from BOBE.utils.core import renormalise_log_weights, scale_from_unit
import time
import matplotlib.pyplot as plt
import seaborn as sns # optional for improved plot aesthetics
from getdist import MCSamples, plots
import numpy as np
from dynesty import DynamicNestedSampler

def loglike(X, slow=True):
    logpdf = -0.25 * (5 * (0.2 - X[0])) ** 2 - (20 * (X[1] / 4 - X[0] ** 4)) ** 2
    if slow: # artificial delay to simulate slow likelihood
        time.sleep(2)
    return logpdf

def prior_transform(x):
    x[0] = x[0]*2 - 1 
    x[1] = x[1]*3 - 1 
    return x

def main():
    # Problem setup
    ndim = 2
    param_list = ['x1', 'x2']
    param_labels = ['x_1', 'x_2']
    param_bounds = np.array([[-1, 1], [-1, 2]]).T
    likelihood_name = f'banana'
    
    start = time.time()
    print("Starting BOBE run...")

    # Initialize surrogate and BOBE instance
    kernel = RBFKernel()
    gp = GP(kernel=kernel, optimizer='scipy')
    bobe = BOBE(
        loglikelihood=loglike,
        param_list=param_list,
        param_bounds=param_bounds,
        param_labels=param_labels,
        likelihood_name=likelihood_name,
        gp=gp,
        verbosity='INFO',
        n_sobol_init=8,
        optimizer='scipy',
        seed=42,
        save_dir='./results/',
        save=True,
    )
    
    # Run optimization with convergence and run settings
    results = bobe.run(
        acq='wipstd',
        min_evals=25,
        max_evals=250,
        max_gp_size=250,
        logz_threshold=5e-2,
        do_final_ns=True,
        fit_n_points=1,
        batch_size=1,
        ns_n_points=2,
        num_hmc_warmup=256,
        num_hmc_samples=2048,
        mc_points_size=512,
        num_chains=4,
        convergence_n_iters=2,
    )

    end = time.time()

    if results is not None:  # when running in MPI mode, only rank 0 returns results, rest return None

        gp = results['gp']
        logz_dict = results.get('logz', {})
        likelihood = results['likelihood']
        results_manager = results['results_manager']
        samples = results['samples']

        manual_timing = end - start

        print("\n" + "="*60)
        print("RUN COMPLETED")
        print(f"Final LogZ: {logz_dict.get('mean', 'N/A'):.4f}")
        if 'upper' in logz_dict and 'lower' in logz_dict:
            print(f"LogZ uncertainty: ±{(logz_dict['upper'] - logz_dict['lower'])/2:.4f}")

        print("="*60)
        print(f"Manual timing: {manual_timing:.2f} seconds ({manual_timing/60:.2f} minutes)")


        # Create Dynesty samples to compare against
        dns_sampler =  DynamicNestedSampler(loglike,prior_transform,ndim=ndim,
                                               sample='rwalk',logl_kwargs={'slow': False})

        dns_sampler.run_nested(print_progress=True,dlogz_init=0.01) 
        res = dns_sampler.results  
        mean = res['logz'][-1]
        logz_err = res['logzerr'][-1]
        print(f"Mean logz from dynesty = {mean:.4f} +/- {logz_err:.4f}")

        dns_samples = res['samples']
        weights = renormalise_log_weights(res['logwt'])

        reference_samples = MCSamples(samples=dns_samples, names=param_list, labels=param_labels,
                                    weights=weights, 
                                    ranges= dict(zip(param_list,param_bounds.T)))  


        # Create MCSamples from BOBE results
        sample_array = samples['x']
        weights_array = samples['weights']
        BOBE_Samples = MCSamples(samples=sample_array, names=param_list, labels=param_labels,
                                    weights=weights_array, 
                                    ranges= dict(zip(param_list,param_bounds.T)))
        
        # Create parameter samples plot
        print("Creating parameter samples plot...")
        sns.set_theme('notebook', 'ticks', palette='husl')
        plt.rcParams['text.usetex'] = True # optional for LaTeX-style text rendering
        plt.rcParams['font.family'] = 'serif'

        g = plots.get_subplot_plotter(subplot_size=2.5, subplot_size_ratio=1)
        g.settings.legend_fontsize = 16
        g.settings.axes_fontsize = 16
        g.settings.axes_labelsize = 16
        g.triangle_plot([BOBE_Samples,reference_samples], filled=[True, False],
                    contour_colors=['#006FED', 'black'], contour_lws=[1, 1.],
                    legend_labels=['BOBE', 'Nested Sampler']) 
        # add scatter points for gp training data
        points = scale_from_unit(gp.train_x, param_bounds)
        for i in range(ndim):
            # ax = g.subplots[i,i]
            for j in range(i+1, ndim):
                ax = g.subplots[j, i]
                ax.scatter(points[:, i], points[:, j], alpha=0.75, color='red', s=4)
        g.export(f'./results/{likelihood.name}_samples.pdf')

        # Print timing analysis
        print("DETAILED TIMING ANALYSIS")

        timing_data = results_manager.get_timing_summary()

        print(f"Automatic timing: {timing_data['total_runtime']:.2f} seconds ({timing_data['total_runtime']/60:.2f} minutes)")
        print("Phase Breakdown:")
        print("-" * 40)  
        for phase, time_spent in timing_data['phase_times'].items():
            if time_spent > 0:
                percentage = timing_data['percentages'].get(phase, 0)
                print(f"{phase:25s}: {time_spent:8.2f}s ({percentage:5.1f}%)")


        # Plot acquisition data
        acquisition_data = results_manager.get_acquisition_data()
        iterations = np.array(acquisition_data['iterations'])
        values = np.array(acquisition_data['values'])
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(iterations, values,  linestyle='-')
        ax.set_yscale('log')
        ax.set_xlabel(r'Iteration')
        ax.set_ylabel(r'Acquisition Value')
        plt.savefig(f"./results/{likelihood.name}_acquisition.pdf", bbox_inches='tight')

if __name__ == "__main__":
    main()