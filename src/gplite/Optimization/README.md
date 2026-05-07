# Optimization Module

Hyperparameter optimization for Gaussian Process models.

## Overview

GP hyperparameters (kernel parameters and noise) significantly affect model performance. This module provides optimization routines to find good values automatically.

## Optimization Strategy

Uses a **two-phase hybrid approach**:

### Phase 1: Global Screening
- Generate multiple starting points via Latin Hypercube Sampling in log-space
- Run quick L-BFGS-B optimization (30 iterations) from each
- Identify promising basins

### Phase 2: Local Refinement
- Take top 2 candidates from screening
- Run thorough L-BFGS-B optimization (300 iterations) on each
- Return best result

This is more efficient than full optimization from all starting points, as most random starts land in poor basins.

## GP Optimization

### Log Marginal Likelihood

The primary objective for GP hyperparameter optimization:

```
log p(y|X,θ) = -0.5 * y.T @ K⁻¹ @ y    (data fit)
              - 0.5 * log|K|           (complexity penalty)
              - n/2 * log(2π)          (constant)
```

The data fit term rewards explaining the data well. The complexity penalty prevents overfitting by penalizing complex models.

### Usage

```python
from gplite import GaussianProcess, RBFKernel

gp = GaussianProcess(RBFKernel(length_scale=1.0))

# Optimize during fit
gp.fit(X, y, optimize=True, objective="lml")

# Or optimize separately
gp.fit(X, y)
gp.optimize_hyperparameters(objective="lml", num_restarts=5)
```


## Active Learning Optimization

After active learning completes, a final optimization step tunes hyperparameters for prediction accuracy.

### Available Objectives

| Objective | Description |
|-----------|-------------|
| `"rmse"` | Root mean squared error on full dataset |
| `"mae"` | Mean absolute error on full dataset |
| `"none"` | Skip optimization |

### Usage

```python
learner.learn(
    learning_strategy="uncertainty",
    final_optimization_method="rmse"  # Optimize for RMSE at the end
)
```

## Module Structure

```
Optimization/
├── gaussian_process/
│   ├── optimization.py    # GP hyperparameter optimization
│   └── loss_functions.py  # LML and gradients
└── active_learning/
    ├── optimization.py    # AL final optimization
    └── loss_functions.py  # RMSE, MAE
```


## What Gets Optimized

- All kernel hyperparameters (length scales, periods, constants)
- Noise variance (jitter added to diagonal for numerical stability)

Bounds are defined by each kernel and typically span several orders of magnitude.
