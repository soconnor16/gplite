# Changelog

Notable changes to this project will be documented in this file.

## [2.1.3] - 2026-05-04

### Changed
- **Package Name** Changed 'gpy' to 'gplite' as 'gpy' was taken on pypi

## [2.1.2] - 2026-03-07

### Added
- **Benchmarking** [Benchmark suite](benchmarks/) for testing compute-heavy sections of this package

## [2.1.1] - 2026-03-07

### Added

- **Performance** Minor changes to computation utilities and optimization algorithms for increased performance

### Fixed

- v2.1.0 changelog now correctly reflects the date it was released

## [2.1.0] - 2026-03-05

### Added

- **Matérn kernel** (`MaternKernel`) with support for ν=1.5 (once differentiable) and ν=2.5 (twice differentiable), including isotropic and anisotropic variants
- **Expected Improvement** selection strategies for active learning: `ei_max` (maximize) and `ei_min` (minimize) for Bayesian optimization use cases
- **Model save/load**: `gp.save(filepath)` and `GaussianProcess.load(filepath)` using pickle serialization
- Efficient `_compute_diag` method on all kernels for O(n) predictive variance computation instead of O(n²)
- [Example files for usage reference](examples/)
- **Logging capability** to give ActiveLearner updates in a file rather than (or along with) stdout

### Fixed

- Predictive variance now correctly computed for non-unit-diagonal kernels (e.g., `ConstantKernel * RBFKernel`); previously hardcoded `k(x,x) = 1`
- Active learning optimizer now refits the model after setting final hyperparameters, matching GP optimizer behavior
- Active learning optimizer fallback path now refits when all screening runs fail
- Missing `raise` in `validate_variable_names` — wrong number of variable names was silently accepted
- Active learning `max_points` budget now correctly accounts for initial training points and clamps `batch_size` to prevent overshooting
- Installation instructions now have the correct link for installing the package with uv and standalone pip

### Changed

- Refactored Cholesky decomposition into two clear phases (exponential noise retry, then eigenvalue fallback), fixing incorrect noise return value after eigenvalue correction
- Renamed `SMALL_EPSILON` to `EPSILON` in `_utils/_constants.py`
- `_validate_input_data` in base `Kernel` class changed from abstract to concrete method, reducing boilerplate in concrete kernel subclasses
- Default `max_points` in `ActiveLearner` now uses `len(y_full)` instead of `np.floor(len(y_full))`, returning an int instead of float
- Removed unused `self.x_test` and `self.y_test` attributes from `GaussianProcess`
- Simplified `fit()` method by removing redundant `_fit_without_optimization()` call in the optimize branch

## [2.0.0] - 2026-02-15


### Features

- Gaussian Process regression with automatic hyperparameter optimization via log marginal likelihood
- Kernels: RBF, Periodic, Constant, with additive and product composition (`+` and `*`)
- Anisotropic (ARD) kernel variants with per-dimension hyperparameters when applicable 
- Active learning with uncertainty, max absolute error, and random selection strategies
- Input and target normalization
- String export of fitted GP expressions for integration with external tools (e.g., OpenMM)
