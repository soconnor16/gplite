"""Save and load a GaussianProcess; verify predictions match after load."""

import matplotlib.pyplot as plt
import numpy as np
from gplite import GaussianProcess, RBFKernel


def target_function(x):
    """Synthetic function for demonstration."""
    return np.sin(x) * np.cos(0.5 * x) + 0.2 * np.sin(3 * x)


# generate training data
rng = np.random.default_rng()
x_train = np.linspace(0, 12, 30).reshape(-1, 1)
y_train = target_function(x_train).ravel() + 0.1 * rng.standard_normal(30)

# test points to predict on
x_test = np.linspace(0, 12, 500).reshape(-1, 1)
# true target data
y_true = target_function(x_test).ravel()

# train original model
kernel = RBFKernel(length_scale=0.2)
gp_original = GaussianProcess(kernel)
gp_original.fit(x_train, y_train, optimize=True)
print(gp_original.kernel.get_params())

# get predictions of original model
y_pred_original, y_std_original = gp_original.predict(x_test, return_std=True)

# save and load the model
model_path = "demo_model.pkl"
gp_original.save(model_path)
gp_loaded = GaussianProcess.load(model_path)

# get predictions of loaded model
y_pred_loaded, y_std_loaded = gp_loaded.predict(x_test, return_std=True)


# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for ax, title, y_pred, y_std in [
    (ax1, "Original Model", y_pred_original, y_std_original),
    (ax2, "Loaded Model", y_pred_loaded, y_std_loaded),
]:
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
        x_train,
        y_train,
        c="black",
        s=30,
        zorder=3,
        label=f"Training ({len(x_train)})",
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x", fontsize=12)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)

ax1.set_ylabel("y", fontsize=12)
fig.suptitle("Model Save / Load Demonstration", fontsize=16)
fig.tight_layout()
plt.show()
