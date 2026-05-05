"""
Combine kernels with + and * for trend+periodic and amplitude-modulated
patterns.
"""

import matplotlib.pyplot as plt
import numpy as np
from gplite import (
    ConstantKernel,
    GaussianProcess,
    PeriodicKernel,
    RBFKernel,
)

"""Example 1: Additive kernel (RBF + Periodic)"""

print("=" * 70)
print("Example 1: Additive kernel (RBF + Periodic)")
print("Use case: Functions with independent trend and periodic components")
print("=" * 70)


def target_function(x):
    """Linear trend + periodic component."""
    trend = 0.15 * x  # Linear trend
    periodic = 1.5 * np.sin(
        2 * np.pi * x / 4
    )  # Periodic component with period 4
    return trend + periodic


# generate training data
rng = np.random.default_rng(1)
x_train = np.sort(rng.uniform(0, 12, 25)).reshape(-1, 1)
y_train = target_function(x_train).ravel() + 0.1 * rng.standard_normal(25)

# test points
x_test = np.linspace(0, 12, 500).reshape(-1, 1)
y_true = target_function(x_test).ravel()

# fit with additive composite kernel
composite_kernel = RBFKernel(length_scale=5.0) + PeriodicKernel(
    length_scale=1.0, period=4.0
)
gp = GaussianProcess(composite_kernel)
gp.fit(x_train, y_train, optimize=True)

y_pred, y_std = gp.predict(x_test, return_std=True)
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

print(
    f"Kernel: RBF(length_scale={gp.kernel.kernels[0].get_params()[0]:.2f}) + "
    f"Periodic(length_scale={gp.kernel.kernels[1].get_params()[0]:.2f}, "
    f"period={gp.kernel.kernels[1].get_params()[1]:.2f})"
)
print(f"RMSE: {rmse:.4f}\n")

# plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

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
    x_test, y_true, color="tomato", linestyle="--", linewidth=2, label="True"
)
ax.scatter(
    x_train,
    y_train,
    c="black",
    s=40,
    zorder=3,
    label=f"Training ({len(x_train)})",
)
ax.set_title(
    "Additive Kernel: RBF + Periodic\n(Trend + Seasonal Pattern)", fontsize=14
)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()


"""Example 2: Product kernel (RBF x Periodic)"""

print("\n" + "=" * 70)
print("Example 2: Product kernel (RBF × Periodic)")
print("Use case: Periodic functions with smoothly varying amplitude")
print("=" * 70)


def modulated_function(x):
    """Periodic with smoothly varying amplitude (RBF × Periodic)."""
    # amplitude varies smoothly from 0.5 to 2.0
    amplitude = 0.5 + 1.5 * (1 + np.tanh((x - 6) / 2)) / 2
    periodic = np.sin(2 * np.pi * x / 3)  # Period 3
    return amplitude * periodic


x_train2 = np.sort(rng.uniform(0, 12, 25)).reshape(-1, 1)
y_train2 = modulated_function(x_train2).ravel() + 0.1 * rng.standard_normal(25)
x_test2 = np.linspace(0, 12, 500).reshape(-1, 1)
y_true2 = modulated_function(x_test2).ravel()

product_kernel = RBFKernel(length_scale=8.0) * PeriodicKernel(
    length_scale=1.0, period=3.0
)
gp2 = GaussianProcess(product_kernel)
gp2.fit(x_train2, y_train2, optimize=True)

y_pred2, y_std2 = gp2.predict(x_test2, return_std=True)
rmse2 = np.sqrt(np.mean((y_true2 - y_pred2) ** 2))

print(
    f"Kernel: RBF(length_scale={gp2.kernel.kernels[0].get_params()[0]:.2f}) × "
    f"Periodic(length_scale={gp2.kernel.kernels[1].get_params()[0]:.2f}, "
    f"period={gp2.kernel.kernels[1].get_params()[1]:.2f})"
)
print(f"RMSE: {rmse2:.4f}\n")

fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))

ax2.fill_between(
    x_test2.ravel(),
    y_pred2 - 2 * y_std2,
    y_pred2 + 2 * y_std2,
    alpha=0.2,
    color="steelblue",
    label="±2σ",
)
ax2.plot(x_test2, y_pred2, color="steelblue", linewidth=2, label="Prediction")
ax2.plot(
    x_test2, y_true2, color="tomato", linestyle="--", linewidth=2, label="True"
)
ax2.scatter(
    x_train2,
    y_train2,
    c="black",
    s=40,
    zorder=3,
    label=f"Training ({len(x_train2)})",
)
ax2.set_title(
    "Product Kernel: RBF × Periodic\n(Amplitude-Modulated Periodic Pattern)",
    fontsize=14,
)
ax2.set_xlabel("x", fontsize=12)
ax2.set_ylabel("y", fontsize=12)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(True, alpha=0.3)

fig2.tight_layout()
plt.show()


"""Example 3: ConstantKernel as scaling factor"""

print("\n" + "=" * 70)
print("Example 3: ConstantKernel as scaling factor")
print("=" * 70)

x_train3 = np.sort(rng.uniform(0, 10, 20)).reshape(-1, 1)
y_train3 = 2.5 * np.sin(x_train3).ravel() + 0.1 * rng.standard_normal(20)
x_test3 = np.linspace(0, 10, 500).reshape(-1, 1)
y_true3 = 2.5 * np.sin(x_test3).ravel()

scaled_kernel = ConstantKernel(constant=2.5) * RBFKernel(length_scale=1.0)
gp3 = GaussianProcess(scaled_kernel)
gp3.fit(x_train3, y_train3, optimize=True)

y_pred3, y_std3 = gp3.predict(x_test3, return_std=True)
rmse3 = np.sqrt(np.mean((y_true3 - y_pred3) ** 2))

print(
    f"Kernel: Constant({gp3.kernel.kernels[0].constant[0]:.2f}) × "
    f"RBF(length_scale={gp3.kernel.kernels[1].length_scale[0]:.2f})"
)
print(f"RMSE: {rmse3:.4f}\n")

fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))

ax3.fill_between(
    x_test3.ravel(),
    y_pred3 - 2 * y_std3,
    y_pred3 + 2 * y_std3,
    alpha=0.2,
    color="steelblue",
    label="±2σ",
)
ax3.plot(x_test3, y_pred3, color="steelblue", linewidth=2, label="Prediction")
ax3.plot(
    x_test3, y_true3, color="tomato", linestyle="--", linewidth=2, label="True"
)
ax3.scatter(
    x_train3,
    y_train3,
    c="black",
    s=40,
    zorder=3,
    label=f"Training ({len(x_train3)})",
)
ax3.set_title(
    "Product Kernel: Constant × RBF\n(Scaling the kernel output)", fontsize=14
)
ax3.set_xlabel("x", fontsize=12)
ax3.set_ylabel("y", fontsize=12)
ax3.legend(loc="upper left", fontsize=10)
ax3.grid(True, alpha=0.3)

fig3.tight_layout()
plt.show()
