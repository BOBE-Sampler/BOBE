import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    os.cpu_count()
)
from BOBE import BOBE, ClassifierGP, SVMClassifier, RBFKernel
from BOBE.utils.core import renormalise_log_weights, scale_from_unit
import time
import matplotlib.pyplot as plt
import seaborn as sns # optional for improved plot aesthetics
from getdist import MCSamples, plots, loadMCSamples
import numpy as np

def main():

    # Set up the cosmological likelihood
    cobaya_input_file = 'cosmo_input/LCDM_lite.yaml'
    likelihood_name = f'Planck_lite_LCDM_z0025'

    start = time.time()
    print("Starting BOBE run...")

    # Pass Cobaya YAML file path directly to BOBE
    kernel = RBFKernel()
    gp = ClassifierGP(kernel=kernel, classifier=SVMClassifier(), minus_inf=-1e5)
    bobe = BOBE(
        loglikelihood=cobaya_input_file,
        likelihood_name=likelihood_name,
        confidence_for_unbounded=0.9999995,
        gp=gp,
        resume=False,
        resume_file=f'{likelihood_name}',
        save_dir='./results/LCDM_Lite/',
        save=True,
        verbosity='INFO',
        n_cobaya_init=4, 
        n_sobol_init=8,
        minus_inf=-1e5,
        seed=10,
    )
    
    results = bobe.run(
        acq='wipstd',
        min_evals=50, 
        max_evals=250,
        max_gp_size=200,
        fit_n_points=4, 
        ns_n_points=4,
        batch_size=4,
        num_hmc_warmup=256,
        num_hmc_samples=2048, 
        mc_points_size=512,
        logz_threshold=0.025,
        do_final_ns=False,
    )

    end = time.time()

    if results is not None:  # when running in MPI mode, only rank 0 returns results, rest return None

        gp = results['gp']
        logz_dict = results.get('logz', {})
        likelihood = results['likelihood']
        results_manager = results['results_manager']
        samples = results['samples']
        param_bounds = likelihood.param_bounds
        param_list = likelihood.param_list
        param_labels = likelihood.param_labels
        ndim = len(param_list)

        manual_timing = end - start

        print("\n" + "="*60)
        print("RUN COMPLETED")
        print(f"Final LogZ: {logz_dict.get('mean', 'N/A'):.4f}")
        if 'upper' in logz_dict and 'lower' in logz_dict:
            print(f"LogZ uncertainty: ±{(logz_dict['upper'] - logz_dict['lower'])/2:.4f}")

        print("="*60)
        print(f"Manual timing: {manual_timing:.2f} seconds ({manual_timing/60:.2f} minutes)")

        reference_samples = loadMCSamples(
            './cosmo_input/chains/Planck_lite_pchord',
            # settings={'ignore_rows': 0.3, 'label': 'MCMC'}
        )

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
        g.triangle_plot([BOBE_Samples,reference_samples], params=['ombh2','omch2','H0','ns','logA','tau'],
                        filled=[True, False],
                    contour_colors=['#006FED', 'black'], contour_lws=[1, 1.5],
                    legend_labels=['BOBE', 'Nested Sampling'],) 
        # add scatter points for gp training data
        points = scale_from_unit(gp.train_x, param_bounds)
        for i in range(ndim):
            for j in range(i+1, ndim):
                ax = g.subplots[j, i]
                ax.scatter(points[:, i], points[:, j], alpha=0.75, color='red', s=4)
        g.export(f'./results/LCDM_Lite/{likelihood.name}_samples.pdf')

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
        plt.savefig(f"./results/LCDM_Lite/{likelihood.name}_acquisition.pdf", bbox_inches='tight')

if __name__ == "__main__":
    main()