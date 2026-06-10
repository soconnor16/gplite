"""
Demonstrates how to combine multiple kernels to model more complex data patterns
that a single kernel might struggle with.
"""

import matplotlib.pyplot as plt
import numpy as np
from gplite import GaussianProcess, PeriodicKernel, RBFKernel


def target_function(x):
    """
    Synthetic data for demonstration.
    """
    trend = 0.05 * x**2
    periodicity = 2 * np.sin(x)

    return trend * periodicity


def main():
    rng = np.random.default_rng(0)

    # random training data
    x_train = rng.uniform(0, 10, 40)
    y_train = target_function(x_train) + 0.05 * rng.standard_normal(40)

    # test data extends farther than train data so we can see how well the model
    # extrapolates
    x_test = np.linspace(0, 12, 400).reshape(-1, 1)
    y_true = target_function(x_test).ravel()

    # initialize an rbf kernel to try to capture our trend
    trend_kernel = RBFKernel(1.0)

    # and a periodic kernel to try to capture the periodicity
    periodic_kernel = PeriodicKernel(1.0, 3.1415)

    # create our composite kernel
    kernel = trend_kernel * periodic_kernel

    # fit and predict
    gp = GaussianProcess(kernel)
    gp.fit(x_train, y_train, optimize=True)

    # extract our predictions and uncertainty
    y_pred, y_std = gp.predict(x_test, return_std=True)

    plt.rcParams["figure.figsize"] = (12, 6)

    # highlight the extrapolation region
    plt.axvspan(10, 12, color="lightgray", alpha=0.3, label="Extrapolation")

    plt.plot(x_test, y_true, color="blue", linewidth=2, label="True Pattern")
    plt.plot(
        x_test,
        y_pred,
        color="red",
        linewidth=2,
        linestyle="--",
        label="GP Prediction",
    )
    plt.fill_between(
        x_test.ravel(),
        y_pred - 2 * y_std,
        y_pred + 2 * y_std,
        alpha=0.2,
        color="steelblue",
        label="±2σ",
    )

    plt.scatter(
        x_train, y_train, c="black", s=20, zorder=3, label="Training Data"
    )

    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Composite Kernel: RBF * Periodic")
    plt.show()


if __name__ == "__main__":
    main()
