# **ActiveLearning Module**

Intelligent data sampling for efficient model training.

## **Overview**

Quality data is the foundation of any useful machine learning model, but 
determining which data points to include in a training set is a non-trivial 
problem. Random sampling often captures redundant information, while training on 
exhaustive datasets introduces severe computational bottlenecks. The `ActiveLearning`
module provides an automated framework to address this selection problem. By 
evaluating a candidate pool and selecting samples that offer the highest 
information gain, it reduces the total number of data points required for the 
model to reach convergence. The selection strategy can be tailored on a 
per-project basis, providing a flexible and efficient protocol for model 
development.

---

## **Core Concepts and Architecture**

### The Active Learning Loop
The `ActiveLearner` class manages the active learning lifecycle. Calling 
`.learn()` executes the following loop:


1. Fit: Trains the underlying Gaussian Process (GP) on the current labeled 
dataset
2. Evaluate: Computes the  acquisition scores for all remaining unlabeled 
points using your chosen strategy.
3. Acquire: Select and label highest-scoring point(s) and moves them from
the training pool to the training set.
4. **Repeat**: Continues until a stopping criterion is met (e.g., a target RMSE
is reached, or a specific budget of points is exhausted)


### Built-in Selection Strategies
- **Uncertainty**("`uncertainty`"): Selects points where the the GP's predicted
variance (σ^2) is highest. This is a pure exploration strategy, focusing on
regions the model knows least about.
- **Maximum Absolute Error**("`mae`"): Selects points where the predicted mean
is furthest from the true value. This is useful for refining a model to get rid
of potential outliers.
- **Expected Improvement**("`ei_min`"/"`ei_max`"): Balances exploration (high
uncertainty) and exploitation (high predicted mean). Ideal for optimization tasks
where you are actively searching for the global maximum or minimum of a function.
- **Random**("`random`"): A uniform random sampling strategy, primarily used as
a baseline to compare the efficacy of other learning strategies.
- **Custom Functions**: Pass any Python Callable to define custom acquisition 
logic.

### Custom Plugins
The `ActiveLearner` class is designed to be highly extensible. If the built-in 
strategies do not fit your specific use-case, you can inject custom Python 
functions directly into the learning loop. The built-in optimization and point
selection strategies should work well for most use cases, however, I find it
important that this package remain customizable. Should you find yourself needing
a custom selection-strategy or final optimization method, reference for the 
required function signatures can be found below:

**1. Custom Selection Strategy:**
```python
def custom_selection_function(
    learner: "ActiveLearner", # the function must accept an active learning instance
    n_points: int # the function must accept a variable to pass the batch size of selection points
) -> np.ndarray:
    # 1. get the data points that haven't been used get
    pool_x = learner.x_full[learner.remaining_indices]

    # 2. custom scoring
    custom_scores = ...
    selected_indices = ...

    # 3. CRITICAL: map local indices back to global learner indices
    return learner.remaining_indices[selected_indices]
```
**2. Custom Loss Function**
```python
def custom_loss_function(learner: "ActiveLearner") -> float:
    # ... custom loss function logic ...
    return float(loss_value)
```
---

## **Usage Example**
A high-level overview of initializing a learner and executing a learning loop

```python
from gplite.ActiveLearning import ActiveLearner
from gplite.Kernels import RBFKernel

# this can be any kernel of choice
kernel = RBFKernel(length_scale=1.5)

# 1. Initialize the Learner
learner = ActiveLearner(
    kernel=kernel, 
    x_full=X_full, # your full input dataset
    y_full=y_full  # your full output dataset
)

# 2. Execute the Learning Loop
# the loop will automatically track RMSE, optimize hyperparameters, 
# and stop when it hits 0.05 RMSE or runs out of points.
learner.learn(
    learning_strategy="uncertainty",
    rmse_threshold=0.05,
    optimize_interval=5,  # re-optimize hyperparameters every 5 iterations
    batch_size=1          # select 1 point per iteration
)

# 3. Access the optimized model for prediction
final_predictions = learner.gp.predict(X_test)
```

---

## **Class Reference**

The `ActiveLearner` class manages the lifecycle of your active learning loop, 
tracking which data points have been selected and interfacing directly with the 
underlying Gaussian Process model.

### 1. Initialization
```python
ActiveLearner(kernel, x_full, y_full)
```
Prepares the learner with the complete candidate pool and the chosen covariance 
function.
* **`kernel`** (*Kernel*): An initialized covariance kernel function.
* **`x_full`** (*NumPy Array*): The complete matrix of candidate input features.
* **`y_full`** (*NumPy Array*): The complete vector of candidate target values.

### 2. Core Methods

#### `.learn(...)`
```python
# all .learn() arguments and their default values
.learn(
    learning_strategy="uncertainty", 
    max_points=None, 
    rmse_threshold=0.5, 
    optimize_interval=1, 
    batch_size=1, 
    final_optimization_method=None,
    log_file=None,
    log_interval=5
)
```
Executes the automated active learning loop. It iteratively scores the 
remaining data pool, extracts the most informative points, updates the 
training set, and checks stopping criteria.
* **`learning_strategy`** (*str or Callable*): The acquisition function used 
to score candidates. Pass a built-in string (`"uncertainty"`, `"mae"`, `"ei_max"`, 
`"ei_min"`, `"random"`) or a custom function.
* **`max_points`** (*int, optional*): The maximum size allowed for the training 
set before the loop terminates.
* **`rmse_threshold`** (*float*): The target accuracy. The loop terminates early 
if the model's global RMSE falls below this value.
* **`optimize_interval`** (*int, optional*): How often (in iterations) to 
re-optimize the GP hyperparameters. Set to `None` to disable intermediate tuning.
* **`batch_size`** (*int*): The number of data points to acquire and add per iteration.
* **`final_optimization_method`** (*str or Callable, optional*): A loss function 
(`"rmse"`, `"mae"`, or a custom function) to run a final hyperparameter tuning 
pass once the loop ends.
* **`log_file`** (*str or Path, optional*): A file path to log details about the
learning process into.
* **`log_interval`** (*int, optional*): How often (in iterations) to write log
data to the log file if one is provided.

#### `.select_next_point(...)`
```python
.select_next_point(selection_function, n_points=1)
```
Evaluates the remaining pool using a specific scoring function and returns the indices of the top candidates. This is a lower-level method useful if you want to write a completely manual training loop.
* **`selection_function`** (*Callable*): The mathematical function used to score 
the pool.
* **`n_points`** (*int*): The number of indices to return.

### 3. Exposed Attributes
You can access these attributes at any point during or after execution to inspect 
the current state of the learner:
* **`learner.gp`**: The underlying `GaussianProcess` instance.
* **`learner.x_train` / `learner.y_train`**: NumPy arrays containing the data 
points selected for training so far.
* **`learner.remaining_indices`**: A NumPy array containing the indices of the 
points in the candidate pool that have *not* been selected yet.
