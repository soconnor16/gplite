# GaussianProcess Module

Core Gaussian Process regression implementation with automatic hyperparameter optimization.

## Overview

A Gaussian Process defines a distribution over functions:

```
f(x) ~ GP(m(x), k(x, x'))
```

where `m(x)` is the mean function (assumed zero) and `k(x, x')` is the covariance/kernel function.

### Prediction Equations

Given training data `(X, y)`, predictions at new points `X*` are:

```
Mean:     μ* = K(X*, X) @ α,  where α = K(X, X)⁻¹ @ y
Variance: σ²* = K(X*, X*) - K(X*, X) @ K(X, X)⁻¹ @ K(X, X*)
```

## Usage

### Basic Fitting and Prediction

```python
from gplite import GaussianProcess, RBFKernel
import numpy as np

# training data
X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([1.0, 2.1, 2.9, 4.2])

# create GP with RBF kernel
kernel = RBFKernel(length_scale=1.0)
gp = GaussianProcess(kernel, normalize_x=True)

# fit without optimization
gp.fit(X, y)

# fit with hyperparameter optimization
gp.fit(X, y, optimize=True, objective="lml") # or "log_marginal_likelihood" 

# predict
X_test = np.array([[1.5], [2.5], [3.5]])
y_mean = gp.predict(X_test)
y_mean, y_std = gp.predict(X_test, return_std=True)
y_mean, y_cov = gp.predict(X_test, return_cov=True)
```

### Normalization

By default, input features are normalized to zero mean and unit variance:

```python
# automatic normalization (default)
gp = GaussianProcess(kernel, normalize_x=True)

# disable if you normalize manually
gp = GaussianProcess(kernel, normalize_x=False)
```

Target values are always normalized internally.

### Save and Load

Save a fitted model to disk and load it back for later use:

```python
# save
gp.save("model.pkl")

# load
gp_loaded = GaussianProcess.load("model.pkl")
y_pred = gp_loaded.predict(X_test)
```


### String Export

Generate a mathematical expression for the fitted GP:

```python
# for 2D input with variables named 'x' and 'y'
expression = gp.to_str(variable_names=["x", "y"])
```

This is useful for exporting to external tools like OpenMM custom forces.

## Class Reference

### `GaussianProcess(kernel, normalize_x=True)`

**Parameters:**
- `kernel` (Kernel): Kernel instance defining the covariance function
- `normalize_x` (bool): Whether to normalize input features. Default: `True`

**Methods:**
- `fit(x, y, optimize=False, objective="lml")`: Fit the GP to training data
- `predict(x, return_std=False, return_cov=False)`: Make predictions
- `optimize_hyperparameters(objective, num_restarts=5)`: Optimize kernel hyperparameters
- `save(filepath)`: Save model to file
- `load(filepath)` *(classmethod)*: Load model from file
- `to_str(variable_names)`: Generate string representation

**Attributes:**
- `kernel`: The kernel instance
- `x_train`, `y_train`: Normalized training data
- `alpha`: Weights used for prediction

## Hyperparameter Optimization

Hyperparameters are optimized by maximizing the log marginal likelihood (LML):

```
log p(y|X,θ) = -0.5 * y.T @ K⁻¹ @ y - 0.5 * log|K| - n/2 * log(2π)
                    (data fit)      (complexity)    (constant)
```

LML naturally balances data fit against model complexity - the complexity term penalizes overly flexible models, preventing overfitting without requiring a separate validation set.

```python
# during fit
gp.fit(X, y, optimize=True)

# or separately
gp.fit(X, y)
gp.optimize_hyperparameters(num_restarts=5)
```

Uses a two-phase hybrid approach: global screening from multiple starting points, then local refinement of top candidates.
