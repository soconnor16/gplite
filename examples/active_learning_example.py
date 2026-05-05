"""Compare active learning strategies on the same data and plot fits."""

import matplotlib.pyplot as plt
import numpy as np
from gplite import ActiveLearner, RBFKernel


def target_function(x):
    """Synthetic target function to fit to."""
    return (
        np.sin(x) * np.exp(-0.03 * x)
        + 0.5 * np.cos(2.5 * x)
        + 0.3 * np.sin(5 * x) / (1 + 0.05 * x**2)
    )


# generate input and output data pools for the learner
rng = np.random.default_rng()
x_pool = np.sort(rng.uniform(0, 20, 300)).reshape(-1, 1)
y_pool = target_function(x_pool).ravel() + 0.05 * rng.standard_normal(300)

# test points to predict on
x_test = np.linspace(0, 20, 500).reshape(-1, 1)
# true target function with no noise added for comparison
y_true = target_function(x_test).ravel()

# strategies to compare
# expected improvement should have the worst fit here, it is used for
# demonstration purposes, but in practice should only be used if you need
# to train on the minima (ei_min) or maxima (ei_max) of a function
strategies = [
    ("uncertainty", "Max Uncertainty"),
    ("mae", "Max Absolute Error"),
    ("ei_max", "Expected Improvement (Max)"),
    ("random", "Random"),
]

# max points from the pool; learning stops when reached or when RMSE is below
# threshold
MAX_POINTS = 40
RMSE_THRESHOLD = 0.1

# train with each strategy
results = []
for strategy_key, strategy_label in strategies:
    kernel = RBFKernel(length_scale=1.0)
    learner = ActiveLearner(
        kernel=kernel,
        x_full=x_pool.copy(),
        y_full=y_pool.copy(),
        max_points=MAX_POINTS,
        rmse_threshold=RMSE_THRESHOLD,
        optimize_interval=5,  # optimize hyperparameters once every 5 iterations
    )

    print(f"\n--- {strategy_label} ---")
    learner.learn(
        learning_strategy=strategy_key,
        batch_size=1,
        final_optimization_method="rmse",  # or "mae"
        update=True,
        log=True,
        update_interval=10,
        log_update_interval=5,
    )

    gp = learner.gp
    y_pred, y_std = gp.predict(x_test, return_std=True)
    # true rmse, will be different than what is in the log file because this is
    # without noise
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    results.append((strategy_label, learner, y_pred, y_std, rmse))

# plot the different models for comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)

for ax, (label, learner, y_pred, y_std, rmse) in zip(axes.ravel(), results):
    # plot 95% confidence interval
    ax.fill_between(
        x_test.ravel(),
        y_pred - 2 * y_std,
        y_pred + 2 * y_std,
        alpha=0.2,
        color="steelblue",
        label="±2σ",
    )

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
        learner.x_train,
        learner.y_train,
        c="black",
        s=20,
        zorder=3,
        label=f"Selected ({len(learner.y_train)})",
    )
    ax.set_title(f"{label}\nRMSE: {rmse:.4f}", fontsize=12)
    ax.set_xlabel("x", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

axes[0, 0].set_ylabel("y", fontsize=12)
axes[1, 0].set_ylabel("y", fontsize=12)
fig.suptitle("Active Learning Strategy Comparison — RBF Kernel", fontsize=16)
fig.tight_layout()
plt.show()
