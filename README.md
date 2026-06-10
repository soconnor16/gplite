# **gplite**

A lightweight Gaussian Process regression library built on NumPy and SciPy,
originally designed for Δ-Machine Learning.

## **Features**

- Gaussian Process regression with automatic hyperparameter optimization
- Multiple kernels: RBF, Matérn, Periodic, Constant (combinable via `+` and `*`)
- Active learning with multiple selection strategies (uncertainty, maximum absolute
error, expected improvement)
- Model saving and loading with pickle
- Anisotropic kernels (support for per-dimension hyperparameters)
- String export for integration with external tools

## **Installation**

```bash
pip install gplite
```
or for optional dependencies to run the example files

```bash
pip install "gplite[examples]"
```

## **Quick Start**

The core workflow of `gplite` is designed to be highly intuitive. More detailed, end-to-end examples can be found in the [examples](examples/) directory.

### 1. Standard Regression & Composable Kernels
```python
import numpy as np
from gplite import GaussianProcess, RBFKernel, PeriodicKernel

X_train, y_train = ... # load your data

# easily combine kernels for complex data
kernel = (
    RBFKernel(length_scale=2.0) + 
    PeriodicKernel(length_scale=1.0, period=2*np.pi)
)

# fit your model and use it for predictions
gp = GaussianProcess(kernel)
gp.fit(X_train, y_train, optimize=True)

y_mean, y_std = gp.predict(X_test, return_std=True)
```

### 2. Automated Active Learning 
```python
from gplite import ActiveLearner

learner = ActiveLearner(kernel=kernel, x_full=X_full, y_full=y_full)

# automatically train the model where it is most uncertain
learner.learn(
    learning_strategy="uncertainty", 
    rmse_threshold=0.1, 
    max_points=50
)

y_pred = learner.gp.predict(X_test)
```

## **Modules**

- [GaussianProcess](src/gplite/GaussianProcess/) - core GP regression
- [Kernels](src/gplite/Kernels/) - RBF, Matérn, Periodic, Constant, composites
- [ActiveLearning](src/gplite/ActiveLearning/) - automated model training
- [Optimization](src/gplite/Optimization/) - hyperparameter optimization

## **Requirements**

- Python >= 3.10
- NumPy >= 2.2.6
- SciPy >= 1.15.3

## **License**

GPLv3
