# **Kernels Module**

Covariance function definitions for Gaussian Process regression.

## **Overview**

Kernels define the covariance between function values at different input points:

```text
Cov(f(x), f(x')) = k(x, x')
```

The kernel determines the structural properties of the functions a Gaussian Process can represent, such as smoothness, periodicity, or trends. Choosing the right kernel is the primary way to inject domain knowledge into the model. All valid kernels implemented in this module are symmetric and positive semi-definite (their covariance matrices are guaranteed to have non-negative eigenvalues).

---

## **Core Concepts and Architecture**

### Isotropic vs. Anisotropic (ARD)
Except for the `ConstantKernel`, all built-in kernels support both **Isotropic** and **Anisotropic** configurations via the `isotropic` parameter. This dictates how the kernel handles multi-dimensional input data:
* **Isotropic (`isotropic=True`)**: Uses a single hyperparameter value shared across all input dimensions. This assumes all features exist on a similar scale and contribute equally to the covariance.
* **Anisotropic (`isotropic=False`)**: Uses separate hyperparameter values for each dimension (often called Automatic Relevance Determination or ARD). This allows the model to learn the relative importance of each feature independently. This is best for data whose trends are dimensionally-dependent.

### Kernel Composition
Kernels can be combined using standard Python operators `+` (addition) and `*` (multiplication) to model complex, multi-scale behaviors:
* **Additive Kernels (`+`)**: Sum of independent components (e.g., trend + 
seasonality).
* **Product Kernels (`*`)**: Product of independent components (e.g., 
amplitude-varying periodic).

---

## **Usage Example**
A high-level overview of initializing and composing a kernel for use with a
GaussianProcess.
```python
from gplite.Kernels import RBFKernel, PeriodicKernel
from gplite.GaussianProcess import GaussianProcess

# an RBF Kernel to model a smooth trend
trend = RBFKernel(length_scale=50.0)

# a Periodic Kernel to model a seasonal trend
seasonality = PeriodicKernel(length_scale=1.5, period=12.0)

# add them together so the GP models both behaviors simultaneously
composite_kernel = trend + seasonality

# pass the composite kernel to the GaussianProcess for regression
gp = GaussianProcess(kernel=composite_kernel)
```

---

## **Class Reference**

### 1. Available Kernels
* **`RBFKernel(length_scale, isotropic=True)`**: Radial Basis Function 
(Squared Exponential). Produces infinitely differentiable (very smooth) 
functions. It is the standard choice for most interpolation tasks.
* **`MaternKernel(length_scale, nu=2.5, isotropic=True)`**: A flexible family 
with controllable smoothness (`nu=1.5` or `nu=2.5`). Less restrictive than RBF, 
making it ideal for physical processes with finite differentiability.
* **`PeriodicKernel(length_scale, period, isotropic=True)`**: Designed for 
modeling repeating patterns, such as seasonal data.
* **`ConstantKernel(constant)`**: Returns a constant covariance regardless of 
inputs. Used primarily as a bias or a global scaling amplitude when multiplied 
with other kernels.

### 2. Core Methods
All kernel instances provide the following public methods:

#### `.compute(...)`
```python
.compute(x1, x2) # or simply kernel(x1, x2)
```
Computes the covariance matrix between two sets of input points.
* **`x1`, `x2`** (*NumPy Array*): Input feature matrices of shape `(n_samples, n_features)`.
* **Returns**: A covariance matrix of shape `(n_samples_1, n_samples_2)`.

#### `.gradient(...)`
```python
.gradient(x1, x2)
```
Computes the gradients of the kernel matrix with respect to its hyperparameters.
* **Returns**: A tuple of gradient arrays (one for each hyperparameter).

#### `.compute_with_gradient(...)`
```python
.compute_with_gradient(x1, x2)
```
Am optimized method that computes both the kernel matrix and its gradients in a 
single pass. Used heavily by the GP optimizer to prevent 
redundant distance calculations.

#### `.get_params()` / `.set_params(...)`
```python
params = kernel.get_params()
kernel.set_params(new_params)
```
Interfaces for retrieving or updating the kernel's state (length scales, 
periods, constants) as a flat 1D array.

### 3. Exposed Attributes
* **`kernel.hyperparameters`**: A tuple of strings defining the names of the 
kernel's parameters.
* **`kernel.bounds`**: A list of `(min, max)` tuples dictating the allowed 
search space during optimization.
