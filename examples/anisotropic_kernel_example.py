"""
Anisotropic (ARD) kernels: per-dimension length scales for multi-dimensional
data.
"""

import matplotlib.pyplot as plt
import numpy as np
from gplite import GaussianProcess, RBFKernel


def target_function(x):
    """
    Synthetic 2D function for demonstration
    """
    x1, x2 = x[:, 0], x[:, 1]
    return np.sin(3 * x1) + 0.5 * np.cos(0.5 * x2)


# generate 2D training data
rng = np.random.default_rng()
x1_train = rng.uniform(0, 4, 40)
x2_train = rng.uniform(0, 10, 40)
x_train = np.column_stack([x1_train, x2_train])
y_train = target_function(x_train) + 0.05 * rng.standard_normal(40)

# test grid for prediction
x1_test = np.linspace(0, 4, 50)
x2_test = np.linspace(0, 10, 50)
X1, X2 = np.meshgrid(x1_test, x2_test)
x_test_grid = np.column_stack([X1.ravel(), X2.ravel()])
y_true_grid = target_function(x_test_grid).reshape(X1.shape)

# isotropic: single length scale (scalar)
isotropic_kernel = RBFKernel(1.0)
gp_iso = GaussianProcess(isotropic_kernel)
gp_iso.fit(x_train, y_train, optimize=True)
y_pred_iso = gp_iso.predict(x_test_grid).reshape(X1.shape)

# anisotropic (ARD): per-dimension length scales (array), isotropic=False
anisotropic_kernel = RBFKernel([0.5, 1.5], isotropic=False)
gp_aniso = GaussianProcess(anisotropic_kernel)
gp_aniso.fit(x_train, y_train, optimize=True)
y_pred_aniso = gp_aniso.predict(x_test_grid).reshape(X1.shape)

# compute RMSE for isotropic and non-isotropic models
rmse_iso = np.sqrt(np.mean((y_true_grid.ravel() - y_pred_iso.ravel()) ** 2))
rmse_aniso = np.sqrt(np.mean((y_true_grid.ravel() - y_pred_aniso.ravel()) ** 2))


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# plot true function
im0 = axes[0].contourf(X1, X2, y_true_grid, levels=20, cmap="viridis")
axes[0].scatter(
    x_train[:, 0], x_train[:, 1], c="white", s=20, edgecolors="black", zorder=3
)
axes[0].set_title("True Function", fontsize=12)
axes[0].set_xlabel("x₁", fontsize=11)
axes[0].set_ylabel("x₂", fontsize=11)
plt.colorbar(im0, ax=axes[0])

# plot isotropic prediction
im1 = axes[1].contourf(X1, X2, y_pred_iso, levels=20, cmap="viridis")
axes[1].scatter(
    x_train[:, 0], x_train[:, 1], c="white", s=20, edgecolors="black", zorder=3
)
axes[1].set_title(f"Isotropic Kernel\nRMSE: {rmse_iso:.4f}", fontsize=12)
axes[1].set_xlabel("x₁", fontsize=11)
axes[1].set_ylabel("x₂", fontsize=11)
plt.colorbar(im1, ax=axes[1])

# plot anisotropic prediction
im2 = axes[2].contourf(X1, X2, y_pred_aniso, levels=20, cmap="viridis")
axes[2].scatter(
    x_train[:, 0], x_train[:, 1], c="white", s=20, edgecolors="black", zorder=3
)
axes[2].set_title(
    f"Anisotropic Kernel (ARD)\nRMSE: {rmse_aniso:.4f}",
    fontsize=12,
)
axes[2].set_xlabel("x₁", fontsize=11)
axes[2].set_ylabel("x₂", fontsize=11)
plt.colorbar(im2, ax=axes[2])

fig.suptitle(
    "Anisotropic Kernels — Learning Per-Dimension Length Scales", fontsize=16
)
fig.tight_layout()
plt.show()
