"""Unit + integration tests for GaussianProcess.predict()."""

import numpy as np
import pytest
from gplite._utils._errors import ValidationError
from gplite.GaussianProcess.gaussian_process import GaussianProcess
from gplite.Kernels.rbf import RBFKernel


def _fitted_gp(n: int = 20) -> tuple[GaussianProcess, np.ndarray, np.ndarray]:
    x = np.linspace(0, 2 * np.pi, n).reshape(-1, 1)
    y = np.sin(x).ravel()
    gp = GaussianProcess(RBFKernel(length_scale=1.0))
    gp.fit(x, y)
    return gp, x, y


# ===========================================================================
# Unfitted model guard
# ===========================================================================


class TestGPPredictUnfitted:
    def test_predict_before_fit_raises(self):
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        x_test = np.linspace(0, 1, 5).reshape(-1, 1)
        with pytest.raises(RuntimeError, match="fitted"):
            gp.predict(x_test)


# ===========================================================================
# predict - mean only
# ===========================================================================


class TestGPPredictMeanOnly:
    def test_output_shape(self):
        gp, _x_train, _ = _fitted_gp()
        x_test = np.linspace(0, 2 * np.pi, 15).reshape(-1, 1)
        y_pred = gp.predict(x_test)
        assert y_pred.shape == (15,)

    def test_output_dtype(self):
        gp, _, _ = _fitted_gp()
        x_test = np.array([[0.5], [1.0]])
        y_pred = gp.predict(x_test)
        assert y_pred.dtype == np.float64

    def test_1d_input_accepted(self):
        """Flat 1-D array should be accepted."""
        gp, _, _ = _fitted_gp()
        x_test = np.linspace(0, 2 * np.pi, 10)  # shape (10,)
        y_pred = gp.predict(x_test)
        assert y_pred.shape == (10,)

    def test_list_input_accepted(self):
        gp, _, _ = _fitted_gp()
        x_test = [[0.5], [1.0], [1.5]]
        y_pred = gp.predict(x_test)
        assert y_pred.shape == (3,)

    def test_returns_array_not_tuple(self):
        gp, _, _ = _fitted_gp()
        result = gp.predict(np.array([[0.5]]))
        assert isinstance(result, np.ndarray)

    def test_nan_input_raises(self):
        gp, _, _ = _fitted_gp()
        x_test = np.array([[np.nan]])
        with pytest.raises(ValidationError):
            gp.predict(x_test)

    def test_predictions_close_to_training_at_training_points(self):
        """Predictions at training points should be close to training targets."""
        gp, x_train, y_train = _fitted_gp(20)
        y_pred = gp.predict(x_train)
        np.testing.assert_allclose(y_pred, y_train, atol=0.1)


# ===========================================================================
# predict - with standard deviation
# ===========================================================================


class TestGPPredictWithStd:
    def test_returns_tuple_of_two(self):
        gp, _, _ = _fitted_gp()
        x_test = np.linspace(0, 2 * np.pi, 10).reshape(-1, 1)
        result = gp.predict(x_test, return_std=True)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_mean_and_std_shapes(self):
        gp, _, _ = _fitted_gp()
        x_test = np.linspace(0, 2 * np.pi, 10).reshape(-1, 1)
        y_mean, y_std = gp.predict(x_test, return_std=True)
        assert y_mean.shape == (10,)
        assert y_std.shape == (10,)

    def test_std_is_nonnegative(self):
        gp, _, _ = _fitted_gp()
        x_test = np.linspace(0, 2 * np.pi, 15).reshape(-1, 1)
        _, y_std = gp.predict(x_test, return_std=True)
        assert np.all(y_std >= 0.0)

    def test_std_smaller_at_training_points(self):
        """Uncertainty should be low at (or very near) the training points."""
        gp, x_train, _ = _fitted_gp(20)
        x_far = np.array([[100.0]])  # far outside training range
        _, std_near = gp.predict(x_train[:1], return_std=True)
        _, std_far = gp.predict(x_far, return_std=True)
        assert std_near[0] < std_far[0]


# ===========================================================================
# predict - with covariance
# ===========================================================================


class TestGPPredictWithCov:
    def test_returns_tuple_of_two(self):
        gp, _, _ = _fitted_gp()
        x_test = np.linspace(0, 2 * np.pi, 5).reshape(-1, 1)
        result = gp.predict(x_test, return_cov=True)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_covariance_shape(self):
        gp, _, _ = _fitted_gp()
        x_test = np.linspace(0, 2 * np.pi, 5).reshape(-1, 1)
        y_mean, y_cov = gp.predict(x_test, return_cov=True)
        assert y_mean.shape == (5,)
        assert y_cov.shape == (5, 5)

    def test_covariance_is_symmetric(self):
        gp, _, _ = _fitted_gp()
        x_test = np.linspace(0, 2 * np.pi, 5).reshape(-1, 1)
        _, y_cov = gp.predict(x_test, return_cov=True)
        np.testing.assert_allclose(y_cov, y_cov.T, atol=1e-10)

    def test_covariance_diagonal_nonnegative(self):
        gp, _, _ = _fitted_gp()
        x_test = np.linspace(0, 2 * np.pi, 5).reshape(-1, 1)
        _, y_cov = gp.predict(x_test, return_cov=True)
        assert np.all(np.diag(y_cov) >= -1e-10)


# ===========================================================================
# predict - std and cov together
# ===========================================================================


class TestGPPredictStdAndCov:
    def test_returns_tuple_of_three(self):
        gp, _, _ = _fitted_gp()
        x_test = np.linspace(0, 2 * np.pi, 5).reshape(-1, 1)
        result = gp.predict(x_test, return_std=True, return_cov=True)
        assert len(result) == 3

    def test_std_equals_sqrt_of_cov_diag(self):
        gp, _, _ = _fitted_gp()
        x_test = np.linspace(0, 2 * np.pi, 8).reshape(-1, 1)
        _, y_std, y_cov = gp.predict(x_test, return_std=True, return_cov=True)
        expected_std = np.sqrt(np.maximum(np.diag(y_cov), 0.0))
        np.testing.assert_allclose(y_std, expected_std, atol=1e-8)


# ===========================================================================
# predict - 2D input
# ===========================================================================


class TestGPPredict2D:
    def test_2d_predict_output_shape(self):
        rng = np.random.default_rng(0)
        x_train = rng.standard_normal((20, 2))
        y_train = x_train[:, 0] ** 2 + x_train[:, 1] ** 2
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        gp.fit(x_train, y_train)
        x_test = rng.standard_normal((8, 2))
        y_pred = gp.predict(x_test)
        assert y_pred.shape == (8,)
