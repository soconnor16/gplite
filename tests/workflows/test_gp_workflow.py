"""End-to-end workflow tests for GaussianProcess."""

import numpy as np
from gplite.GaussianProcess.gaussian_process import GaussianProcess
from gplite.Kernels.constant import ConstantKernel
from gplite.Kernels.matern import MaternKernel
from gplite.Kernels.rbf import RBFKernel

# ===========================================================================
# Full workflow: fit → predict on known functions
# ===========================================================================


class TestGPWorkflowSine:
    """GP should fit a noiseless sine wave and predict with low RMSE."""

    def _rmse(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    def test_rbf_sine_rmse(self):
        x = np.linspace(0, 2 * np.pi, 30).reshape(-1, 1)
        y = np.sin(x).ravel()
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        gp.fit(x, y)
        y_pred = gp.predict(x)
        assert self._rmse(y_pred, y) < 0.05

    def test_matern_15_sine_rmse(self):
        x = np.linspace(0, 2 * np.pi, 30).reshape(-1, 1)
        y = np.sin(x).ravel()
        gp = GaussianProcess(MaternKernel(length_scale=1.0, nu=1.5))
        gp.fit(x, y)
        y_pred = gp.predict(x)
        assert self._rmse(y_pred, y) < 0.1

    def test_matern_25_sine_rmse(self):
        x = np.linspace(0, 2 * np.pi, 30).reshape(-1, 1)
        y = np.sin(x).ravel()
        gp = GaussianProcess(MaternKernel(length_scale=1.0, nu=2.5))
        gp.fit(x, y)
        y_pred = gp.predict(x)
        assert self._rmse(y_pred, y) < 0.1

    def test_predict_on_held_out_points(self):
        """Train on every other point, predict on the rest - RBF should generalize."""
        x_all = np.linspace(0, 2 * np.pi, 40).reshape(-1, 1)
        y_all = np.sin(x_all).ravel()
        x_train, y_train = x_all[::2], y_all[::2]
        x_test, y_test = x_all[1::2], y_all[1::2]

        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        gp.fit(x_train, y_train)
        y_pred = gp.predict(x_test)
        assert self._rmse(y_pred, y_test) < 0.15


class TestGPWorkflowQuadratic:
    """GP should fit a quadratic in 1D and 2D."""

    def _rmse(self, y_pred, y_true):
        return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    def test_rbf_quadratic_1d(self):
        x = np.linspace(-3, 3, 30).reshape(-1, 1)
        y = (x**2).ravel()
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        gp.fit(x, y)
        y_pred = gp.predict(x)
        assert self._rmse(y_pred, y) < 0.2

    def test_rbf_quadratic_2d(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(-2, 2, size=(40, 2))
        y = x[:, 0] ** 2 + x[:, 1] ** 2
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        gp.fit(x, y)
        y_pred = gp.predict(x)
        assert self._rmse(y_pred, y) < 0.5


# ===========================================================================
# Full workflow: fit → optimize → predict
# ===========================================================================


class TestGPWorkflowWithOptimization:
    """After optimization, predictions should be at least as good as without."""

    def _rmse(self, y_pred, y_true):
        return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    def test_optimized_gp_rmse_sine(self):
        x = np.linspace(0, 2 * np.pi, 25).reshape(-1, 1)
        y = np.sin(x).ravel()
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        gp.fit(x, y, optimize=True)
        y_pred = gp.predict(x)
        assert self._rmse(y_pred, y) < 0.05

    def test_optimize_hyperparameters_explicitly(self):
        x = np.linspace(0, 2 * np.pi, 20).reshape(-1, 1)
        y = np.sin(x).ravel()
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        gp.fit(x, y)
        gp.optimize_hyperparameters(objective="lml", num_restarts=5)
        # just verify it didn't crash and the model is still usable
        y_pred = gp.predict(x)
        assert y_pred.shape == (20,)
        assert np.all(np.isfinite(y_pred))


# ===========================================================================
# Full workflow: fit → predict → save → load → predict
# ===========================================================================


class TestGPWorkflowSaveLoad:
    def test_save_load_predictions_consistent(self, tmp_path):
        x = np.linspace(0, 2 * np.pi, 20).reshape(-1, 1)
        y = np.sin(x).ravel()
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        gp.fit(x, y)

        x_test = np.linspace(0, 2 * np.pi, 10).reshape(-1, 1)
        y_before = gp.predict(x_test)

        filepath = tmp_path / "gp.pkl"
        gp.save(filepath)
        loaded = GaussianProcess.load(filepath)
        y_after = loaded.predict(x_test)

        np.testing.assert_allclose(y_before, y_after, atol=1e-10)


# ===========================================================================
# Composite kernel workflow
# ===========================================================================


class TestGPWorkflowCompositeKernel:
    def _rmse(self, y_pred, y_true):
        return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    def test_additive_kernel_sine_rmse(self):
        """RBF + Constant should still fit a sine wave."""
        x = np.linspace(0, 2 * np.pi, 25).reshape(-1, 1)
        y = np.sin(x).ravel()
        kernel = RBFKernel(length_scale=1.0) + ConstantKernel(constant=1.0)
        gp = GaussianProcess(kernel)
        gp.fit(x, y)
        y_pred = gp.predict(x)
        assert self._rmse(y_pred, y) < 0.1

    def test_product_kernel_runs(self):
        """Constant * RBF (scaling kernel) should fit and predict."""
        x = np.linspace(-2, 2, 20).reshape(-1, 1)
        y = (x**2).ravel()
        kernel = ConstantKernel(constant=2.0) * RBFKernel(length_scale=1.0)
        gp = GaussianProcess(kernel)
        gp.fit(x, y)
        y_pred = gp.predict(x)
        assert y_pred.shape == (20,)
        assert np.all(np.isfinite(y_pred))

    def test_anisotropic_2d_workflow(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(-2, 2, size=(30, 2))
        y = np.sin(x[:, 0]) + x[:, 1] ** 2
        kernel = RBFKernel(length_scale=[1.0, 2.0], isotropic=False)
        gp = GaussianProcess(kernel)
        gp.fit(x, y)
        y_pred = gp.predict(x)
        assert y_pred.shape == (30,)
        assert np.all(np.isfinite(y_pred))


# ===========================================================================
# Uncertainty is calibrated: known trends
# ===========================================================================


class TestGPUncertaintyCalibration:
    def test_uncertainty_higher_far_from_training(self):
        """GP should be more uncertain far outside the training range."""
        x_train = np.linspace(0, 1, 20).reshape(-1, 1)
        y_train = np.sin(x_train).ravel()
        gp = GaussianProcess(RBFKernel(length_scale=0.5))
        gp.fit(x_train, y_train)

        x_near = np.array([[0.5]])
        x_far = np.array([[10.0]])

        _, std_near = gp.predict(x_near, return_std=True)
        _, std_far = gp.predict(x_far, return_std=True)

        assert std_far[0] > std_near[0]
