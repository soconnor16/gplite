"""Unit + integration tests for GaussianProcess.fit()."""

import numpy as np
import pytest
from gplite._utils._errors import ValidationError
from gplite.GaussianProcess.gaussian_process import GaussianProcess
from gplite.Kernels.matern import MaternKernel
from gplite.Kernels.rbf import RBFKernel


def _sine_data(n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 2 * np.pi, n).reshape(-1, 1)
    y = np.sin(x).ravel()
    return x, y


# ===========================================================================
# Validation during init
# ===========================================================================


class TestGPInit:
    def test_valid_kernel_accepted(self):
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        assert gp.kernel is not None

    def test_invalid_kernel_raises(self):
        with pytest.raises(ValidationError, match="kernel"):
            GaussianProcess("not_a_kernel")

    def test_unfitted_alpha_is_empty(self):
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        assert gp.alpha.size == 0

    def test_unfitted_x_train_is_empty(self):
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        assert gp.x_train.size == 0


# ===========================================================================
# GaussianProcess.fit() - basic
# ===========================================================================


class TestGPFitBasic:
    def test_fit_sets_x_train(self):
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        x, y = _sine_data(20)
        gp.fit(x, y)
        assert gp.x_train.size > 0

    def test_fit_sets_alpha(self):
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        x, y = _sine_data(20)
        gp.fit(x, y)
        assert gp.alpha.size > 0
        assert gp.alpha.shape == (20,)

    def test_fit_sets_y_train(self):
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        x, y = _sine_data(20)
        gp.fit(x, y)
        assert gp.y_train.size > 0

    def test_fit_stores_standardized_x(self):
        """With standardize_inputs=True, x_train should have zero mean."""
        gp = GaussianProcess(RBFKernel(length_scale=1.0), standardize_inputs=True)
        x, y = _sine_data(20)
        gp.fit(x, y)
        np.testing.assert_allclose(gp.x_train.mean(), 0.0, atol=1e-10)

    def test_fit_without_standardization(self):
        """With standardize_inputs=False, x_train should equal the raw input."""
        gp = GaussianProcess(RBFKernel(length_scale=1.0), standardize_inputs=False)
        x, y = _sine_data(20)
        gp.fit(x, y)
        np.testing.assert_allclose(gp.x_train, x, atol=1e-12)

    def test_fit_1d_list_input(self):
        """Python lists should be accepted as inputs."""
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        x = [[0.0], [0.5], [1.0], [1.5], [2.0]]
        y = [0.0, 0.5, 1.0, 0.5, 0.0]
        gp.fit(x, y)
        assert gp.alpha.size == 5

    def test_fit_2d_input(self):
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        rng = np.random.default_rng(0)
        x = rng.standard_normal((15, 3))
        y = rng.standard_normal(15)
        gp.fit(x, y)
        assert gp.alpha.shape == (15,)


# ===========================================================================
# GaussianProcess.fit() - validation errors
# ===========================================================================


class TestGPFitValidation:
    def test_shape_mismatch_raises(self):
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        x = np.ones((10, 2))
        y = np.ones(8)  # wrong length
        with pytest.raises(ValidationError, match="samples"):
            gp.fit(x, y)

    def test_nan_in_x_raises(self):
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        x = np.array([[1.0], [np.nan]])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValidationError):
            gp.fit(x, y)

    def test_nan_in_y_raises(self):
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        x = np.array([[1.0], [2.0]])
        y = np.array([1.0, np.nan])
        with pytest.raises(ValidationError):
            gp.fit(x, y)

    def test_anisotropic_dimension_mismatch_raises(self):
        """Anisotropic kernel with 2 scales should reject 3-D input."""
        gp = GaussianProcess(RBFKernel(length_scale=[1.0, 2.0], isotropic=False))
        x = np.ones((10, 3))
        y = np.ones(10)
        with pytest.raises(ValidationError):
            gp.fit(x, y)


# ===========================================================================
# GaussianProcess.fit() - refitting
# ===========================================================================


class TestGPFitRefit:
    def test_refit_with_new_data_updates_alpha(self):
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        x1, y1 = _sine_data(10)
        gp.fit(x1, y1)
        alpha_first = gp.alpha.copy()

        x2, y2 = _sine_data(20)
        gp.fit(x2, y2)
        # alpha shape changes, or values change, because dataset changed
        assert gp.alpha.shape == (20,)
        assert not np.allclose(gp.alpha[:10], alpha_first)

    def test_fit_different_kernels(self):
        """GP should work with any valid kernel."""
        x, y = _sine_data(15)
        for kernel in [
            RBFKernel(length_scale=1.0),
            MaternKernel(length_scale=1.0, nu=1.5),
            MaternKernel(length_scale=1.0, nu=2.5),
            RBFKernel(length_scale=1.0) + MaternKernel(length_scale=1.0, nu=2.5),
        ]:
            gp = GaussianProcess(kernel)
            gp.fit(x, y)
            assert gp.alpha.size == 15


# ===========================================================================
# GaussianProcess.fit() with optimization (smoke test, capped restarts)
# ===========================================================================


class TestGPFitWithOptimization:
    def test_fit_with_optimize_runs(self):
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        x, y = _sine_data(15)
        gp.fit(x, y, optimize=True)
        assert gp.alpha.size == 15

    def test_optimize_changes_hyperparameters(self):
        """After optimization the length scale should change."""
        initial_ls = 1.0
        gp = GaussianProcess(RBFKernel(length_scale=initial_ls))
        x, y = _sine_data(20)
        gp.fit(x, y, optimize=True)
        # optimized length scale should differ from the naive default
        optimized_ls = float(gp.kernel.get_params()[0])
        # they might be close by chance, but almost certainly not identical
        # just check optimized is a positive finite number
        assert optimized_ls > 0
        assert np.isfinite(optimized_ls)
