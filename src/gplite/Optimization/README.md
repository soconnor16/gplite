# **Optimization Module**

Internal hyperparameter optimization routines for Gaussian Process and Active Learning models.

## **Overview**

The `Optimization` module provides the backend routines for tuning kernel 
hyperparameters. It is primarily accessed indirectly when a user calls 
`gp.fit(optimize=True)`, `gp.optimize_hyperparameters()`, or when an 
`ActiveLearner` concludes its loop.

Because hyperparameter loss landscapes are often non-convex and contain multiple 
local minima, this module implements a "Filtered Multi-Start" approach to 
identify the optimal hyperparameter configuration while minimizing unnecessary 
computational overhead. It supports two distinct optimization paradigms: 
standard Gaussian Process fitting and Active Learning refinement.

---

## **Optimization Methodology**

The optimizer relies on a two-phase hybrid approach that combines global 
exploration with gradient-based local refinement. Initial starting points are 
generated using Latin Hypercube Sampling (LHS) in log-space to ensure broad 
coverage of the parameter bounds.

### Phase 1: Global Screening (The Filter)
The optimizer evaluates the objective function across the LHS-generated 
starting points. 
* This phase acts as a broad sweep to identify promising regions in the parameter 
space.
* It efficiently filters out mathematically poor regions of the search space 
(e.g., length scales that are significantly misaligned with the data) without 
wasting expensive CPU cycles on full gradient descent.

### Phase 2: Local Refinement (Multi-Start L-BFGS-B)
The algorithm takes a predefined subset of the best configurations found 
during the screening phase and uses them as starting vectors for local 
optimization.
* It utilizes SciPy's `L-BFGS-B` algorithm, which strictly respects the 
mathematical boundaries of the hyperparameters.
* After running to convergence for all candidates in the subset, the optimizer 
returns the hyperparameter state from the specific run that achieved the lowest 
overall loss.

---

## **The Objective Functions**

The module is split into two distinct sub-packages, each utilizing a different 
mathematical objective to evaluate model performance.

### 1. Gaussian Process Optimization (Log Marginal Likelihood)
By default, standard GP optimization minimizes the **Negative Log Marginal 
Likelihood (LML)**. 

LML is the standard objective for Gaussian Processes because it naturally 
penalizes overly complex models without requiring a separate validation dataset 
and while being less prone to overfitting. The optimizer utilizes exact 
analytical gradients provided by the kernel (`compute_with_gradient`) to 
efficiently traverse the LML landscape. The equation evaluates two primary 
components:
* **Data Fit**: Evaluates how well the hyperparameters allow the model to 
explain the training data.
* **Complexity Penalty**: Penalizes the model if the hyperparameters define a 
covariance structure that is too flexible or erratic.

### 2. Active Learning Optimization (Prediction Error)
Unlike standard GP fitting, Active Learning optimization directly minimizes 
global prediction error, typically **Root Mean Squared Error (RMSE)** or **Max 
Absolute Error (MAE)**. This is possible because the Active Learning 
optimization, when enabled, takes place only once at the end of the learning 
process, whereas the Gaussian Process optimization takes place continuously 
throughout it. Where optimizing with a loss function directly tied to error 
would cause overfitting if used as the dominant strategy, this risk is highly 
minimized when it is used as a final refinement method.

This is executed across the full target dataset. While standard GP optimization 
focuses on generalizing from the training subset, the Active Learning 
optimization ensures the final model is specifically tuned for maximum 
predictive accuracy on the known data distribution once the point-acquisition 
loop concludes.

---

## **Module Interfaces and Extensibility**

This module is designed to operate independently of specific kernel definitions. 
It communicates with the rest of the library via a standardized interface.

### Kernel Interface Requirements
To be compatible with the optimizer, a kernel must implement:
* `get_params()` and `set_params()`: To retrieve and update the internal 
hyperparameter state as a flat 1D array.
* `bounds`: To provide the `(min, max)` constraints for the Phase 1 sampling 
and Phase 2 L-BFGS-B algorithm.
* `compute_with_gradient(X, X)`: To supply the GP loss function with both the 
covariance matrix and its exact gradients in a single pass.

### Custom Objective Functions
The optimizer accepts any Python Callable as the objective function. Users can 
pass a custom function to the `objective` parameter in `gp.fit()`, or the `
final_optimization_method` in the `ActiveLearner.learn()` method. Documentation 
on the expected signatures for custom functions can be found in the respective 
module-level README files for `GaussianProcess` and `ActiveLearner`.
