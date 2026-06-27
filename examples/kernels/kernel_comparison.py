"""Compares RBF, Matérn, and Periodic kernels on the same data to demonstrate the
importance of proper Kernel selection during training.
"""

import matplotlib.pyplot as plt
import numpy as np
from gplite import GaussianProcess, MaternKernel, PeriodicKernel, RBFKernel
from numpy.typing import NDArray


def target_function(x: NDArray) -> NDArray:
    """Synthetic target function to fit to."""
    return 0.5 * np.sin(3 * x) + 0.8 * np.exp(-0.1 * (x - 10) ** 2)


# generate training data
rng = np.random.default_rng(0)
x_train = np.sort(rng.uniform(-5, 5, 25)).reshape(-1, 1)
y_train = target_function(x_train).ravel() + 0.05 * rng.standard_normal(25)

# test points for plotting
x_test = np.linspace(-5, 5, 300).reshape(-1, 1)
y_true = target_function(x_test).ravel()

kernels = [
    ("RBF", RBFKernel(length_scale=1.0)),
    ("Matérn 5/2", MaternKernel(length_scale=1.0, nu=2.5)),
    ("Periodic", PeriodicKernel(length_scale=1.0, period=2 * np.pi)),
]

results = []
for name, kernel in kernels:
    gp = GaussianProcess(kernel, standardize_inputs=True)
    gp.fit(x_train, y_train, optimize=True)
    y_pred, y_std = gp.predict(x_test, return_std=True, return_cov=False)
    results.append((name, y_pred, y_std))

# plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, (name, y_pred, y_std) in zip(axes, results):
    ax.fill_between(
        x_test.ravel(),
        y_pred - 2 * y_std,
        y_pred + 2 * y_std,
        alpha=0.2,
        color="steelblue",
        label="±2σ",
    )
    # plot 95% confidence interval
    ax.plot(x_test, y_pred, color="steelblue", linewidth=2, label="Prediction")
    ax.plot(
        x_test,
        y_true,
        color="tomato",
        linestyle="--",
        linewidth=2,
        label="True",
    )
    ax.scatter(
        x_train,
        y_train,
        c="black",
        s=30,
        zorder=3,
        label=f"Training ({len(x_train)})",
    )
    ax.set_title(f"{name} Kernel", fontsize=14)
    ax.set_xlabel("x", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("y", fontsize=12)
fig.suptitle(
    "Kernel Comparison — Same Data, Different Assumptions", fontsize=16
)
fig.tight_layout()
plt.show()
