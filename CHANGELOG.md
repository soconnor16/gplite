# Changelog

Notable changes to this project will be documented in this file.

## [3.0.0] - 2026-06-09

### Added
- **Custom Loss and Selection Functions** Added support for custom loss functions
for optimization and selection functions for active learning. Usage examples can
be found in the examples folder, documentation on expected function signatures
can be found in the [ActiveLearning](src/gplite/ActiveLearning/) and 
[GaussianProcess](src/gplite/GaussianProcess/) module-level README files.

### Fixed

- **"Forgetful" Optimizer Bug** Fixed an initialization bug in both GP and AL 
hybrid optimizers where Phase 2 (local refinement) incorrectly initialized the 
known best loss value to infinity. It now correctly inherits the best loss from 
Phase 1, preventing the model from discarding valid global screening results 
if local refinement fails.
- **LinAlgError Optimization Bug** Improved L-BFGS-B optimizer stability by 
catching linear algebra errors directly inside the objective function wrappers. 
It now returns infinity (and dummy gradients when gradients are used) to deflect 
the optimizer away from non-positive-definite covariance matrices instead of 
crashing the optimization loop.
- **Divide-by-Zero in Acquisition Functions** Fixed numerical instability in 
`expected_improvement_max` and `expected_improvement_min` where predictions with 
near-zero variance caused exploding Z-scores. Variance masks now strictly check 
against `EPSILON` (1e-8) instead of `0`.
- **NoneType Optimization Error** Fixed an `AttributeError` in Active Learning 
hyperparameter optimization where explicitly passing `None` as the objective 
function would crash the script when evaluating `.lower()`.
- **Variable Names validation Bug** Fixed a bug in the name validation that 
takes place when saving a `GaussianProcess` instance for a string. The function 
will now accept single strings more intuitively and give more helpful error 
messages if invalid values are passed accidentally.

### Changed

- **Formatting and Documentation** Ensured sure files were formatted consistently 
with ruff, update documentation.
- **Cleanup** Refactor the `learn` method for Active Learners to reduce 
repeated code by introducing a unified exit block and extracting data 
acquisition into a private helper method.
- **Optional Final Optimization** The `final_optimization_method` argument in 
the Active Learner `learn` method now accepts "None", allowing users to train a 
full model with Active Learning without any implicit or explicit hyperparameter 
optimization.
- **Example Files** Expanded example files and and divided into appropriately 
named folders.
- **Benchmarking Suite** Removed the benchmarking suite.

### Breaking Changes

- **Active Learning API** Arguments `max_points`, `rmse_threshold`, and 
`optimize_interval` were moved from the initialization of an Active Learning 
instance to the `learn` method. This should provide more flexible learning 
behavior such as multi-stage learning with different objective functions in 
different stages.
- **Active Learning Logging**
  - **stdout logging**: The `update` and `update_interval` arguments were 
  removed from the `learn` method along with the default final printing with the 
  final RMSE and number of points used. This is now an optional behavior that 
  has been moved to a logger and can be enabled via the INFO level in the 
  logging config. 
  - **file logging**: The `log` boolean argument has been removed and replaced 
  with an optional `log_file` argument which specifies the filepath to write log 
  info to during training. The `log_update_interval` argument has also been 
  removed and replaced with a `log_interval` argument which controls the log 
  interval for both optional stdout logging and optional file logging. 

## [2.1.3] - 2026-05-04

### Changed
- **Package Name** Changed 'gpy' to 'gplite' as 'gpy' was taken on pypi

## [2.1.2] - 2026-03-07

### Added

- **Benchmarking** [Benchmark suite](benchmarks/) for testing compute-heavy 
sections of this package

## [2.1.1] - 2026-03-07

### Added

- **Performance** Minor changes to computation utilities and optimization 
algorithms for increased performance

### Fixed

- v2.1.0 changelog now correctly reflects the date it was released

## [2.1.0] - 2026-03-05

### Added

- **Matérn kernel** (`MaternKernel`) with support for ν=1.5 (once differentiable) 
and ν=2.5 (twice differentiable), including isotropic and anisotropic variants
- **Expected Improvement** selection strategies for active learning: `ei_max` 
(maximize) and `ei_min` (minimize) for Bayesian optimization use cases
- **Model save/load** `gp.save(filepath)` and `GaussianProcess.load(filepath)` 
using pickle serialization
- Efficient `_compute_diag` method on all kernels for O(n) predictive variance 
computation instead of O(n²)
- [Example files for usage reference](examples/)
- **Logging capability** to give ActiveLearner updates in a file rather than 
(or along with) stdout

### Fixed

- Predictive variance now correctly computed for non-unit-diagonal kernels 
(e.g., `ConstantKernel * RBFKernel`); previously hardcoded `k(x,x) = 1`
- Active learning optimizer now refits the model after setting final 
hyperparameters, matching GP optimizer behavior
- Active learning optimizer fallback path now refits when all screening runs fail
- Missing `raise` in `validate_variable_names` — wrong number of variable names 
was silently accepted
- Active learning `max_points` budget now correctly accounts for initial 
training points and clamps `batch_size` to prevent overshooting
- Installation instructions now have the correct link for installing the package 
with uv and standalone pip

### Changed

- Refactored Cholesky decomposition into two clear phases 
(exponential noise retry, then eigenvalue fallback), fixing incorrect noise 
return value after eigenvalue correction
- Renamed `SMALL_EPSILON` to `EPSILON` in `_utils/_constants.py`
- `_validate_input_data` in base `Kernel` class changed from abstract to 
concrete method, reducing boilerplate in concrete kernel subclasses
- Default `max_points` in `ActiveLearner` now uses `len(y_full)` instead of 
`np.floor(len(y_full))`, returning an int instead of float
- Removed unused `self.x_test` and `self.y_test` attributes from `GaussianProcess`
- Simplified `fit()` method by removing redundant `_fit_without_optimization()` 
call in the optimize branch

## [2.0.0] - 2026-02-15

### Features

- Gaussian Process regression with automatic hyperparameter optimization via log marginal likelihood
- Kernels RBF, Periodic, Constant, with additive and product composition (`+` and `*`)
- Anisotropic (ARD) kernel variants with per-dimension hyperparameters when applicable 
- Active learning with uncertainty, max absolute error, and random selection strategies
- Input and target normalization
- String export of fitted GP expressions for integration with external tools (e.g., OpenMM)
