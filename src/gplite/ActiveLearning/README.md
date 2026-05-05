# ActiveLearning Module

Intelligent data sampling for efficient model training.

## Overview

Pool-Based Active learning reduces the amount of labeled data needed by strategically selecting the most informative points from an unlabeled pool.

### The Active Learning Loop

```
1. Train GP on current labeled set
2. Compute acquisition scores for unlabeled points
3. Select and label highest-scoring point(s)
4. Repeat until stopping criterion met
```

## Usage

### Basic Active Learning

```python
from gplite import ActiveLearner, PeriodicKernel
import numpy as np

# full dataset 
X_full = np.linspace(0, 2*np.pi, 200).reshape(-1, 1)
y_full = np.sin(X_full).ravel()  

# create active learner
kernel = PeriodicKernel(length_scale=1.0, period=2*np.pi)
learner = ActiveLearner(
    kernel=kernel,
    x_full=X_full,
    y_full=y_full,
    rmse_threshold=0.05,  # stop when RMSE < 0.05
    max_points=50,        # or when 50 points used
    optimize_interval=10  # re-optimize every 10 iterations
)

# run active learning
learner.learn(
    learning_strategy="uncertainty",
    batch_size=1,
    final_optimization_method="rmse",
    update=True,          # print progress
    update_interval=10    # print progress every 10 iterations
)

# access trained model
predictions = learner.gp.predict(X_full)
print(f"Points used: {len(learner.y_train)}")
```

## Selection Strategies

### Uncertainty Sampling (`"uncertainty"`)

Selects points where the model is most uncertain (highest predictive variance).

```python
learner.learn(learning_strategy="uncertainty")
```

**Best for:** Exploration, when you want broad coverage of the input space.

### Maximum Absolute Error (`"mae"`)

Selects points where the model makes the largest errors.

```python
learner.learn(learning_strategy="mae")
```

**Best for:** Exploitation, when you want to fix specific problem areas.

### Expected Improvement — Maximize (`"ei_max"`)

Selects points with the highest expected improvement over the current best (maximum) observed value. Balances exploitation (high predicted mean) with exploration (high uncertainty).

```
EI(x) = (μ(x) - f_best) * Φ(Z) + σ(x) * φ(Z)
where Z = (μ(x) - f_best) / σ(x), f_best = max(y_train)
```

```python
learner.learn(learning_strategy="ei_max")
```

**Best for:** Bayesian optimization when searching for the maximum of a function.

### Expected Improvement — Minimize (`"ei_min"`)

Selects points with the highest expected improvement below the current best (minimum) observed value.

```
EI(x) = (f_best - μ(x)) * Φ(Z) + σ(x) * φ(Z)
where Z = (f_best - μ(x)) / σ(x), f_best = min(y_train)
```

```python
learner.learn(learning_strategy="ei_min")
```

**Best for:** Bayesian optimization when searching for the minimum of a function (e.g., energy minimization).

### Random (`"random"`)

Baseline strategy with uniform random selection.

```python
learner.learn(learning_strategy="random")
```

**Best for:** Comparison baseline, or when domain knowledge suggests uniform sampling.

## Class Reference

### `ActiveLearner(kernel, x_full, y_full, ...)`

**Parameters:**
- `kernel` (Kernel): Kernel for the internal GP model
- `x_full` (array): Complete pool of input features
- `y_full` (array): Complete pool of target values
- `max_points` (int, optional): Maximum training points to use. Defaults to full dataset if not passed
- `rmse_threshold` (float): Target RMSE for stopping. Default: `0.5`
- `optimize_interval` (int, optional): Iterations between hyperparameter optimization. Defaults to 1 (optimize each iteration)

**Methods:**
- `learn(learning_strategy, batch_size=1, ...)`: Run the active learning loop
- `select_next_point(selection_function, n_points=1)`: Select next point(s) to add

**Attributes:**
- `gp`: The underlying GaussianProcess model
- `x_train`, `y_train`: Current training data
- `x_full`, `y_full`: Complete data pool
- `remaining_indices`: Indices of unlabeled points

## Stopping Criteria

The learning loop stops when any of these conditions is met:

1. **RMSE threshold reached**: Model achieves target accuracy
2. **Max points reached**: Budget exhausted
3. **Pool exhausted**: All points have been labeled

## Batch Active Learning

Select multiple points per iteration:

```python
learner.learn(
    learning_strategy="uncertainty",
    batch_size=5  # add 5 points per iteration
)
```

Useful when labeling has high fixed cost but low marginal cost.
