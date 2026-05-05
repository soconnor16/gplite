# gplite

A lightweight Gaussian Process regression library built on NumPy and SciPy, originally designed for delta machine learning in computational chemistry.

## Features

- Gaussian Process regression with automatic hyperparameter optimization
- Multiple kernels: RBF, Matérn, Periodic, Constant (combinable via `+` and `*`)
- Active learning with multiple selection strategies (uncertainty, MAE, expected improvement)
- Model saving and loading with pickle
- Anisotropic kernels (support for per-dimension hyperparameters)
- String export for integration with external tools

## Installation

```bash
pip install gplite
```
or for optional dependencies to run the example files

```bash
pip install "gplite[examples]"
```


## Quick Start (more detailed examples can be found [here](examples/))
### Gaussian Process Regression

```python
import numpy as np
from gplite import GaussianProcess, RBFKernel

X_train = np.linspace(0, 10, 20).reshape(-1, 1)
y_train = np.sin(X_train).ravel() + 0.1 * np.random.randn(20)

kernel = RBFKernel(length_scale=1.0)
gp = GaussianProcess(kernel)
gp.fit(X_train, y_train, optimize=True)

X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_mean, y_std = gp.predict(X_test, return_std=True)
```

### Matérn Kernel

```python
from gplite import MaternKernel

# Matérn 5/2 (twice differentiable, default)
kernel = MaternKernel(length_scale=1.0, nu=2.5)

# Matérn 3/2 (once differentiable, rougher functions)
kernel = MaternKernel(length_scale=1.0, nu=1.5)
```

### Combining Kernels

```python
from gplite import RBFKernel, PeriodicKernel, ConstantKernel
import numpy as np

# additive: smooth trend + periodic pattern
kernel = RBFKernel(length_scale=2.0) + PeriodicKernel(length_scale=1.0, period=2*np.pi)

# product: scaled periodic
kernel = ConstantKernel(constant=2.0) * PeriodicKernel(length_scale=1.0, period=2*np.pi)
```

### Anisotropic Kernels

```python
from gplite import RBFKernel

# separate length scale per input dimension
kernel = RBFKernel(length_scale=[1.0, 5.0, 0.5], isotropic=False)
```

### Active Learning

```python
from gplite import ActiveLearner, PeriodicKernel
import numpy as np

X_full = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)
y_full = np.sin(X_full).ravel()

kernel = PeriodicKernel(length_scale=1.0, period=2*np.pi)
learner = ActiveLearner(
    kernel=kernel,
    x_full=X_full,
    y_full=y_full,
    rmse_threshold=0.1,
    max_points=50
)

learner.learn(learning_strategy="uncertainty")
y_pred = learner.gp.predict(X_full)
```

### Save and Load

```python
# save a fitted model
gp.save("model.pkl")

# load it back
from gplite import GaussianProcess
gp_loaded = GaussianProcess.load("model.pkl")
y_pred = gp_loaded.predict(X_test)
```

### String Export

```python
expression = gp.to_str(variable_names=["x"])
```

## Modules

- [GaussianProcess](src/gplite/GaussianProcess/) - core GP regression
- [Kernels](src/gplite/Kernels/) - RBF, Matérn, Periodic, Constant, composites
- [ActiveLearning](src/gplite/ActiveLearning/) - data sampling strategies
- [Optimization](src/gplite/Optimization/) - hyperparameter optimization

## Requirements

- Python >= 3.10
- NumPy >= 2.2
- SciPy >= 1.15

## License

GPLv3
