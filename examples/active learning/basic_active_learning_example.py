"""
Intro Active Learning example. This should be sufficient for most regular usage
and exposes the core API and most commonly used settings.
"""

import matplotlib.pyplot as plt
import numpy as np
from gplite import ActiveLearner, RBFKernel
from numpy.typing import NDArray

RMSE_THRESHOLD = 0.1
MAX_POINTS = 25


def target_function(x: NDArray) -> NDArray:
    """
    Synthetic target function we will be fitting the model to.
    """
    return (
        np.sin(x) * np.exp(-0.03 * x)
        + 0.5 * np.cos(2.5 * x)
        + 0.3 * np.sin(5 * x) / (1 + 0.05 * x**2)
    )


def main() -> None:
    rng = np.random.default_rng(0)

    x_pool = np.linspace(-10, 10, 200)
    # add random noise to the target pool to simulate real data
    y_pool = target_function(x_pool) + 0.05 * rng.standard_normal(200)
    y_true = target_function(x_pool)

    # initialize kernel and learner
    kernel = RBFKernel(length_scale=1.0)
    learner = ActiveLearner(kernel=kernel, x_full=x_pool, y_full=y_pool)

    # begin learning process
    learner.learn(
        max_points=MAX_POINTS,  # the maximum number of data points to allow for learning,
        rmse_threshold=RMSE_THRESHOLD,  # rmse threshold which, if reached, ends learning early,
        final_optimization_method="rmse",  # fine tune hyperparameters with an rmse target after learning finishes
    )

    # extract final Gaussian Process object that has been trained by the learner
    final_gp = learner.gp

    # extract the predicted output and standard deviation on the full input dataset
    y_preds, std = final_gp.predict(x_pool, return_std=True)

    plt.rcParams["figure.figsize"] = (12, 8)
    plt.plot(x_pool, y_true, color="blue", linewidth=2, label="True")
    plt.plot(
        x_pool,
        y_preds,
        color="red",
        linewidth=2,
        linestyle="--",
        label="Prediction",
    )

    # plot the points that the model learned with
    plt.scatter(
        learner.x_train,
        learner.y_train,
        c="black",
        s=30,
        zorder=3,
        label="Training Points",
    )

    # plot the 95% confidence interval of the model at each prediction point
    plt.fill_between(
        x_pool,
        y_preds - 2 * std,
        y_preds + 2 * std,
        alpha=0.2,
        color="steelblue",
        label="±2σ",
    )

    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Basic Active Learning Demo")
    plt.show()


if __name__ == "__main__":
    main()
