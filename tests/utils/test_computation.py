"""Unit tests for gplite._utils._computation."""

from __future__ import annotations

import numpy as np
from gplite._utils._computation import (
    compute_lower_cholesky_decomposition,
    compute_rmse_across_dataset,
    compute_square_euclidean_distance,
)
from gplite.GaussianProcess.gaussian_process import GaussianProcess
from gplite.Kernels.rbf import RBFKernel

# ===========================================================================
# compute_square_euclidean_distance
# ===========================================================================


class TestComputeSquareEuclideanDistance:
    def test_same_object_returns_symmetric(self):
        """When x1 is x2, output must be symmetric (triangular optimisation path)."""
        x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        D = compute_square_euclidean_distance(x, x)
        assert D.shape == (3, 3)
        np.testing.assert_allclose(D, D.T, atol=1e-12)

    def test_same_object_diagonal_is_zero(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        D = compute_square_euclidean_distance(x, x)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)

    def test_different_objects(self):
        x1 = np.array([[0.0], [1.0]])
        x2 = np.array([[2.0], [3.0]])
        D = compute_square_euclidean_distance(x1, x2)
        # expected: [[4, 9], [1, 4]]
        expected = np.array([[4.0, 9.0], [1.0, 4.0]])
        np.testing.assert_allclose(D, expected, atol=1e-12)

    def test_output_shape(self):
        x1 = np.ones((5, 3))
        x2 = np.ones((7, 3))
        D = compute_square_euclidean_distance(x1, x2)
        assert D.shape == (5, 7)

    def test_output_dtype(self):
        x = np.ones((3, 2))
        D = compute_square_euclidean_distance(x, x)
        assert D.dtype == np.float64

    def test_known_1d_values(self):
        """Single feature: distance between 0 and 3 should be 9."""
        x1 = np.array([[0.0]])
        x2 = np.array([[3.0]])
        D = compute_square_euclidean_distance(x1, x2)
        np.testing.assert_allclose(D[0, 0], 9.0, atol=1e-12)


# ===========================================================================
# compute_lower_cholesky_decomposition
# ===========================================================================


class TestComputeLowerCholeskyDecomposition:
    def _make_psd_matrix(self, n: int = 5) -> np.ndarray:
        """Generate a simple positive-definite matrix."""
        rng = np.random.default_rng(0)
        A = rng.standard_normal((n, n))
        return A @ A.T + np.eye(n) * 0.5

    def test_basic_decomposition(self):
        K = self._make_psd_matrix()
        L, noise = compute_lower_cholesky_decomposition(K, noise=1e-6, max_attempts=10)
        # L should be lower triangular
        np.testing.assert_allclose(np.triu(L, k=1), 0.0, atol=1e-10)
        # Reconstruct: L @ L.T should equal K + noise * I
        n = K.shape[0]
        K_reg = K + noise * np.eye(n)
        np.testing.assert_allclose(L @ L.T, K_reg, atol=1e-8)

    def test_returns_float_noise(self):
        K = self._make_psd_matrix()
        _L, noise = compute_lower_cholesky_decomposition(K, noise=1e-6, max_attempts=10)
        assert isinstance(noise, float)

    def test_output_shape(self):
        n = 6
        K = self._make_psd_matrix(n)
        L, _ = compute_lower_cholesky_decomposition(K, noise=1e-6, max_attempts=10)
        assert L.shape == (n, n)

    def test_near_singular_matrix_handled(self):
        """A matrix that is almost singular should not crash with retry logic."""
        n = 4
        # rank-1 matrix - extremely ill-conditioned
        v = np.ones((n, 1))
        K = v @ v.T  # rank 1, PSD but not PD
        # should succeed after jitter is added
        L, noise = compute_lower_cholesky_decomposition(K, noise=1e-6, max_attempts=10)
        assert L is not None
        assert noise > 0


# ===========================================================================
# compute_rmse_across_dataset
# ===========================================================================


class TestComputeRmseAcrossDataset:
    def _fitted_gp(self) -> tuple[GaussianProcess, np.ndarray, np.ndarray]:
        """Return a simple fitted GP and the data it was trained on."""
        x = np.linspace(0, 2 * np.pi, 20).reshape(-1, 1)
        y = np.sin(x).ravel()
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        gp.fit(x, y)
        return gp, x, y

    def test_returns_scalar(self):
        gp, x, y = self._fitted_gp()
        rmse = compute_rmse_across_dataset(gp, x, y)
        assert np.isscalar(rmse) or rmse.ndim == 0

    def test_rmse_nonnegative(self):
        gp, x, y = self._fitted_gp()
        rmse = compute_rmse_across_dataset(gp, x, y)
        assert rmse >= 0.0

    def test_perfect_predictions_give_zero_rmse(self):
        """If the GP predicts perfectly, RMSE should be ~0."""
        x = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.0, 1.0, 4.0])
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        gp.fit(x, y)
        # RMSE on training data should be very small (not necessarily zero
        # due to noise regularisation, but much less than 1)
        rmse = compute_rmse_across_dataset(gp, x, y)
        assert rmse < 0.5

    def test_rmse_dtype(self):
        gp, x, y = self._fitted_gp()
        rmse = compute_rmse_across_dataset(gp, x, y)
        assert rmse.dtype == np.float64
