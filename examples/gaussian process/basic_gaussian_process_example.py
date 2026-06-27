"""Intro Gaussian Process example. This should be sufficient for most regular usage
and exposes the core API and most commonly used settings.
"""

import matplotlib.pyplot as plt
import numpy as np
from gplite import GaussianProcess, RBFKernel
from numpy.typing import NDArray


def target_function(x: NDArray) -> NDArray:
    """Synthetic target function we will be fitting the model to."""
    return np.sin(x) * np.cos(0.5 * x) + 0.2 * np.sin(3 * x)


def main() -> None:
    # generate training data
    rng = np.random.default_rng(0)
    x_train = np.linspace(0, 12, 30).reshape(-1, 1)
    y_train = target_function(x_train).ravel() + 0.1 * rng.standard_normal(30)

    # test points to predict on
    x_test = np.linspace(0, 12, 500).reshape(-1, 1)
    # true target data
    y_true = target_function(x_test).ravel()

    # initialize kernel and GP
    kernel = RBFKernel(length_scale=0.2)
    gp = GaussianProcess(kernel=kernel)
    # fit GP to data with optimization enabled
    gp.fit(x_train, y_train, optimize=True)

    # use GP to make predictions on test data
    y_pred, y_std = gp.predict(x=x_test, return_std=True)

    plt.rcParams["figure.figsize"] = (12, 8)
    plt.plot(x_test, y_true, color="blue", linewidth=2, label="True")
    plt.plot(
        x_test,
        y_pred,
        color="red",
        linewidth=2,
        linestyle="--",
        label="Prediction",
    )

    # plot the 95% confidence interval of the model at each prediction point
    plt.fill_between(
        x_test.ravel(),
        y_pred - 2 * y_std,
        y_pred + 2 * y_std,
        alpha=0.2,
        color="steelblue",
        label="±2σ",
    )

    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Basic Gaussian Process Demo")
    plt.show()


if __name__ == "__main__":
    main()
