# BOBE Copilot Instructions

## Project Overview

BOBE is a Bayesian Optimization package for expensive likelihood functions, primarily targeting cosmology applications. It trains Gaussian Process (GP) surrogates for expensive likelihoods and runs nested sampling/MCMC on the surrogate instead of the true likelihood to compute Bayesian Evidence.

**Core workflow**: Initialize points → Train GP surrogate → Acquire new points via acquisition functions → Update GP → Run nested sampling → Check convergence → Repeat

## Architecture

### Main Components

1. **BOBE (`BOBE/bo.py`)**: Primary orchestrator class that manages the entire BO loop
2. **GP Models (`BOBE/gp.py`, `BOBE/clf_gp.py`)**: 
   - `GP`: Base Gaussian Process with JAX-based kernels (RBF, Matérn) and priors (DSLP, SAAS)
   - `ClassifierGP`: Extended GP with SVM/NN/Ellipsoid classifier to filter low-likelihood regions
3. **Likelihood Interfaces (`BOBE/likelihood.py`)**:
   - `Likelihood`: Base class for any likelihood function
   - `CobayaLikelihood`: Interfaces with Cobaya for cosmological likelihoods (optional dependency)
4. **Acquisition Functions (`BOBE/acquisition.py`)**: EI, LogEI, WIPV (Weighted Integrated Posterior Variance), WIPStd
5. **Nested Sampler (`BOBE/nested_sampler.py`)**: Dynesty wrapper for evidence computation on GP surrogate
6. **MPI Pool (`BOBE/utils/pool.py`)**: Custom MPI parallelization for likelihood evaluations and GP fitting

### Data Flow

```
User → run_bobe() → BOBE.run()
  ↓
Initial points (Sobol/Cobaya) → Evaluate on true likelihood
  ↓
Train GP surrogate → Optimize acquisition function → Evaluate candidates → Update GP
  ↓
Run nested sampling on GP → Check convergence (KL divergence + logZ threshold)
  ↓
Results saved to disk (GP state, samples, logZ, timing)
```

## Critical Conventions

### JAX Configuration
Always enable 64-bit precision at module top:
```python
import jax
jax.config.update("jax_enable_x64", True)
from numpyro.util import enable_x64
enable_x64()
```

### Parameter Space Convention
- **Internal**: All GP/acquisition operations work in unit cube [0,1]^D
- **External**: User-facing parameters use physical bounds
- Use `scale_to_unit()` and `scale_from_unit()` from `BOBE/utils/core.py` for conversions
- Training data stored in both spaces: `train_x` (unit), `train_x_clf` (unit for classifier GP)

### Classifier-GP Thresholds
Two separate thresholds in `ClassifierGP`:
- `clf_threshold`: For training classifier labels (e.g., 250 logL units below max)
- `gp_threshold`: For selecting which points to train GP on (e.g., 500 logL units below max)
- Points filtered by classifier before GP prediction to avoid wasted computation

### MPI Pattern
Worker processes call `pool.worker_wait(likelihood, seed)` and loop indefinitely until TASK_EXIT. Master distributes:
- `TASK_OBJECTIVE_EVAL`: Likelihood evaluations at candidate points
- `TASK_GP_FIT`: GP hyperparameter optimization with multiple restarts
- `TASK_COBAYA_INIT`: Sample initial points from Cobaya reference distribution
- Check `pool.is_master` before I/O operations; workers are silent

### Results Management
`BOBEResults` class (`BOBE/utils/results.py`) handles all saving:
- Periodic saves every `save_step` iterations
- GP state saved as pickle (`.gp` extension)
- Samples saved as `.npz` (x, weights, logl, logp)
- Summary JSON with logZ, timing, convergence info
- Resume from `resume_file` by loading GP state and samples

## Development Workflows

### Running Examples
Serial execution:
```bash
python examples/Planck_DESI_LCDM.py 8 svm 42
# Args: num_devices (for JAX), clf_type, seed
```

MPI execution (for expensive likelihoods >1s):
```bash
mpirun -n 4 python examples/your_script.py
# Automatically detected by MPI_Pool()
```

### Testing
No formal test suite in `tests/` directory. `test_methods/` contains development scripts and benchmarks (e.g., `test_bo_gpjax.py`, `test_dynesty_vs_harmonic.py`)

### Environment Setup
Primary method (exact versions):
```bash
conda env create -f environment.yml
conda activate BOBE
pip install -e .
```

Optional dependencies (see `pyproject.toml`):
- `[nn]`: Flax + Optax for neural network classifiers
- `[cobaya]`: Cobaya for cosmological likelihoods
- `[mpi]`: mpi4py for parallelization
- `[all]`: Everything

### Documentation
Hosted on ReadTheDocs (see `docs/`):
```bash
cd docs
./build_docs.sh  # Builds HTML docs
```

## Key Implementation Details

### GP Hyperparameter Optimization
Two optimizers available (`BOBE/optim.py`):
- `scipy`: L-BFGS-B (default, more robust)
- `optax`: Adam/SGD with JAX autodiff (faster but requires tuning)

Multiple restarts distributed across MPI processes if available. Uses log-space bounds for lengthscales/variance.

### Acquisition Function Optimization
WIPV batch mode: Optimizes multiple points simultaneously via:
1. Sobol initialization (scaled to unit cube)
2. L-BFGS-B optimization with classifier filtering
3. Returns best `batch_size` points ranked by acquisition value

### Convergence Criteria
BOBE terminates when BOTH conditions met for `convergence_n_iters` consecutive iterations:
1. KL divergence between consecutive posterior samples < threshold (default 0.5)
2. Change in logZ < `logz_threshold` (default 0.01)

### Nested Sampling on GP
Dynesty runs on GP surrogate (not true likelihood):
- Prior transform: identity (already in unit cube)
- Dynamic nested sampling mode preferred (`dynamic=True`)
- `dlogz=0.1` for evidence convergence
- Returns samples, weights, logZ with error estimate

### Classifier Training Schedule
In `ClassifierGP`, classifier trains when:
- Total points >= `clf_use_size` (default 10)
- Every `clf_update_step` new points after threshold reached
- Labels: 1 if within `clf_threshold` of max, 0 otherwise
- SVM uses `C=1e7`, `gamma='scale'` by default

### Cobaya Integration
`CobayaLikelihood` wraps Cobaya YAML files:
- Extracts sampled parameters from `params` block
- Handles unbounded priors with `confidence_for_unbounded` (e.g., 0.9999995)
- Uses Cobaya's `get_valid_point()` for initialization if `n_cobaya_init > 0`
- Example YAML: `examples/cosmo_input/LCDM_Planck_DESIDr2.yaml`

## Common Pitfalls

1. **Forgetting unit cube scaling**: Always check if function expects unit or physical space
2. **MPI worker hangs**: Ensure master sends TASK_EXIT to all workers before terminating
3. **Classifier overhead**: Disable with `use_clf=False` for cheap likelihoods (<0.1s)
4. **JAX device count**: Set `XLA_FLAGS` environment variable for multi-device parallelism (examples set this via `sys.argv[1]`)
5. **Resume failures**: Ensure `resume_file` points to directory with `.gp` file, not the file itself

## File Naming Patterns

- GP states: `{likelihood_name}_gp.pkl`
- Samples: `{likelihood_name}_samples.npz`
- Summary: `{likelihood_name}_summary.json`
- Plots: `{likelihood_name}_cosmo.pdf`, `{likelihood_name}_full.pdf`

## Useful Utilities

- `BOBESummaryPlotter` (`BOBE/utils/plot.py`): Automated convergence plots
- `plot_final_samples()`: Corner plots with GetDist integration
- `get_logger()`: Consistent logging with rank-aware MPI output
- `set_global_seed()`: Synchronize NumPy/JAX RNG across processes
