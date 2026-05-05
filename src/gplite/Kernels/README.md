# Kernels Module

Kernel (covariance) functions for Gaussian Process regression.

## Overview

Kernels define the covariance between function values at different input points:

```
Cov(f(x), f(x')) = k(x, x')
```

The kernel determines the properties of functions the GP can represent (smoothness, periodicity, etc.).

### Kernel Properties

All valid kernels are:
- **Symmetric**: `k(x, x') = k(x', x)`
- **Positive semi-definite**: Kernel matrices have non-negative eigenvalues

## Available Kernels

### RBF (Radial Basis Function) Kernel

Also known as Squared Exponential or Gaussian kernel. Produces infinitely differentiable (very smooth) functions.

```
K(x, x') = exp(-0.5 * ||x - x'||² / l²)
```

```python
from gplite import RBFKernel

# isotropic (single length scale for all dimensions)
kernel = RBFKernel(length_scale=1.0, isotropic=True)

# anisotropic (separate length scale per dimension)
kernel = RBFKernel(length_scale=[1.0, 2.0, 0.5], isotropic=False)
```

### Matérn Kernel

A flexible kernel family with controllable smoothness. Less restrictive than RBF for physical processes with finite differentiability.

```
ν = 3/2:  K(x, x') = (1 + √3 r) exp(-√3 r)
ν = 5/2:  K(x, x') = (1 + √5 r + 5r²/3) exp(-√5 r)
```

where `r = ||x - x'|| / l`.

```python
from gplite import MaternKernel

# Matérn 5/2 — twice differentiable (default)
kernel = MaternKernel(length_scale=1.0, nu=2.5)

# Matérn 3/2 — once differentiable (rougher functions)
kernel = MaternKernel(length_scale=1.0, nu=1.5)

# anisotropic
kernel = MaternKernel(length_scale=[1.0, 2.0], nu=2.5, isotropic=False)
```

**When to use:** Physical processes where infinite smoothness (RBF) is unrealistic. Matérn 5/2 is a popular default in many GP applications.

### Periodic Kernel

For modeling repeating patterns (e.g., torsion angles, seasonal data).

```
K(x, x') = exp(-2 * Σᵢ sin²(π|xᵢ - x'ᵢ| / pᵢ) / lᵢ²)
```

```python
from gplite import PeriodicKernel
import numpy as np

# isotropic
kernel = PeriodicKernel(length_scale=1.0, period=2*np.pi, isotropic=True)

# anisotropic
kernel = PeriodicKernel(
    length_scale=[1.0, 2.0],
    period=[2*np.pi, np.pi],
    isotropic=False
)
```

### Constant Kernel

Returns a constant covariance regardless of inputs. Used as bias or scaling factor.

```
K(x, x') = c
```

```python
from gplite import ConstantKernel

kernel = ConstantKernel(constant=2.0)
```

## Combining Kernels

Kernels can be combined using `+` (addition) and `*` (multiplication):

### Additive Kernels

Sum of independent components:

```python
# smooth trend + periodic pattern
kernel = RBFKernel(length_scale=5.0) + PeriodicKernel(length_scale=1.0, period=2*np.pi)
```

### Product Kernels

Multiplicative interactions (one pattern modulates another):

```python
# scaled periodic kernel
kernel = ConstantKernel(constant=2.0) * PeriodicKernel(length_scale=1.0, period=2*np.pi)

# amplitude-varying periodic
kernel = RBFKernel(length_scale=10.0) * PeriodicKernel(length_scale=1.0, period=2*np.pi)
```

### Complex Combinations

```python
# trend + seasonal + noise
kernel = (
    RBFKernel(length_scale=10.0) +  # long-term trend
    PeriodicKernel(length_scale=1.0, period=1.0) +  # seasonal
    ConstantKernel(constant=0.1)  # baseline
)
```

## Kernel Methods

All kernels implement:

```python
# compute kernel matrix
K = kernel.compute(X1, X2)  # shape: (n, m)

# compute gradients w.r.t. hyperparameters
grads = kernel.gradient(X1, X2)  # tuple of gradient tensors

# get/set hyperparameters
params = kernel.get_params()  # flat array
kernel.set_params(new_params)

# direct call syntax
K = kernel(X1, X2)  # same as kernel.compute(X1, X2)
```

## Isotropic vs Anisotropic

**Isotropic**: Single hyperparameter value shared across all input dimensions
- Fewer parameters to optimize
- Assumes all features have similar scales

**Anisotropic** (ARD - Automatic Relevance Determination): Separate values per dimension
- More flexible, can learn feature importance
- Length scale → ∞ means feature is irrelevant
- More parameters, may need more data

```python
# 3D input, isotropic (1 length scale)
kernel = RBFKernel(length_scale=1.0, isotropic=True)

# 3D input, anisotropic (3 length scales)
kernel = RBFKernel(length_scale=[1.0, 2.0, 0.5], isotropic=False)
```
