# **GaussianProcess Module**

Core Gaussian Process regression implementation with automatic hyperparameter optimization.

## **Overview**

A Gaussian Process (GP) is a non-parametric approach to machine learning that
defines a distribution over possible functions. GPs are probabilistic models,
which means that their predictions come with information on their
**uncertainty**. Unlike traditional models that output a single deterministic guess, a
GP outputs a full probability distribution. It tells you not only *what* it
thinks the answer is, but exactly how *confident* it is in that prediction.

Because they are deeply grounded in Bayesian probability, GPs are exceptionally
data-efficient. They can model complex, non-linear relationships using a fraction
of the training data typically required by deep neural networks. This combination
of high data efficiency and built-in uncertainty quantification makes GPs the
standard tool for domains where data is expensive or slow to gather, such as
active learning and Bayesian optimization.

---

## **Core Concepts and Architecture**

### Prediction Equations

Given training data `(X, y)`, predictions at new points `X*` are computed
as such:

```text
Mean:     μ* = K(X*, X) @ α,  where α = K(X, X)^(-1) @ y
Variance: σ²* = K(X*, X*) - K(X*, X) @ K(X, X)^(-1) @ K(X, X*)
```

### Hyperparameter Optimization

Hyperparameters (like kernel length scales and noise) heavily dictate model
performance. By default, the GP optimizes these by maximizing the Log Marginal
Likelihood (LML).

LML is the go-to method for optimizing GP models, balancing the data fit
against a complexity penalty to prevent overfitting without requiring a
separate validation set or regularization term. The optimization uses a Two-Phase
hybrid approach (Global Screening -> Local Refinement) to efficiently navigate
the hyperparameter space and avoid poor local minima.

### Custom Plugins

The `GaussianProcess` class is designed to be highly extensible. If the built-in
strategies do not fit your specific use-case, you can inject custom Python
functions directly into the optimization method. The built-in optimization
strategy (LML) should work well for most use cases, however, I find it important
that this package remain customizable. Should you find yourself needing a custom
optimization method, reference for the required function signature can be found
below:

#### Custom Loss Function

```python
def custom_loss_function(gp: "GaussianProcess") -> float:
    # ... custom loss function logic ...
    return float(loss_value)
```

---

## **Usage Example**

A high-level overview of initializing, optimizing, and predicting with a GP model.

```python
import numpy as np
from gplite.GaussianProcess import GaussianProcess
from gplite.Kernels import RBFKernel

# 1. Initialize GP with a chosen kernel
# by default, inputs (X) are automatically standardized to 0-mean, 1-variance
gp = GaussianProcess(RBFKernel(length_scale=1.0), standardize_inputs=True)

# 2. Fit the model to training data
# setting optimize=True automatically tunes the kernel hyperparameters
gp.fit(X_train, y_train, optimize=True, objective="lml")

# 3. Predict on new data
y_preds = gp.predict(X_test)
```

---

## **Class Reference**

### 1. Initialization

```python
GaussianProcess(kernel, standardize_inputs=True)
```

Prepares the GP model with the specified covariance function.

* **`kernel`** (*Kernel*): An initialized covariance kernel function.
* **`standardize_inputs`** (*bool*): Whether to automatically standardize input
features to zero mean and unit variance. Target values are always standardized
internally. Default is `True`.

### 2. Core Methods

#### `.fit(...)`

```python
.fit(x, y, optimize=False, objective="lml")
```

Fits the Gaussian Process model to the training data, optionally tuning
hyperparameters.

* **`x`** (*NumPy Array*): Input features.
* **`y`** (*NumPy Array*): Target values.
* **`optimize`** (*bool*): Whether to run hyperparameter optimization.
* **`objective`** (*str or Callable*): The loss function to minimize if
`optimize=True`. Pass `"lml"` or a custom function. Defaults to `"lml"`.

#### `.predict(...)`

```python
.predict(x, return_std=False, return_cov=False)
```

Generates predictions for new input data.

* **`x`** (*NumPy Array*): New input features to evaluate.
* **`return_std`** (*bool*): If `True`, returns `(mean, standard_deviation)`.
* **`return_cov`** (*bool*): If `True`, returns `(mean, full_covariance_matrix)`.

#### `.optimize_hyperparameters(...)`

```python
.optimize_hyperparameters(objective="lml", num_restarts=10)
```

Manually triggers hyperparameter optimization loop on an already-fitted model.

#### `.to_str(...)`

```python
.to_str(variable_names=["x", "y"])
```

Generates a raw mathematical string representation of the fitted GP, translating
the internal weights and kernel logic into an explicit equation. This package
was originally designed to interface with OpenMM, so all outputted strings are
formatted to be compatible with the mathematical functions they support.

#### `.save(...)` / `.load(...)`

```python
gp.save(filepath="model.pkl")
loaded_gp = GaussianProcess.load(filepath="model.pkl")
```

Saves the fitted model to disk, or loads a previously saved instance.

### 3. Exposed Attributes

You can access these internal properties after fitting:

* **`gp.kernel`**: The kernel instance (containing the current optimized hyperparameters).
* **`gp.x_train` / `gp.y_train`**: The standardized data the model was trained on.
* **`gp.alpha`**: The internal weights `(K^(-1) @ y)` computed during fitting,
used directly for downstream predictions.
