"""Unit tests for ConstantKernel."""

import numpy as np
import pytest
from gplite._utils._errors import ValidationError
from gplite.Kernels.constant import ConstantKernel


def _make_x(n: int = 8, d: int = 2, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d))


def _is_psd(K: np.ndarray, tol: float = 1e-8) -> bool:
    """Tests whether a kernel is positive-semidefinite."""
    eigenvalues = np.linalg.eigvalsh(K)
    return bool(np.all(eigenvalues >= -tol))


# ===========================================================================
# Initialisation
# ===========================================================================


class TestConstantKernelInit:
    def test_valid_constant(self):
        k = ConstantKernel(constant=2.0)
        np.testing.assert_allclose(k.constant, [2.0])

    def test_zero_raises(self):
        with pytest.raises(ValidationError):
            ConstantKernel(constant=0.0)

    def test_negative_raises(self):
        with pytest.raises(ValidationError):
            ConstantKernel(constant=-1.0)

    def test_custom_bounds_accepted(self):
        k = ConstantKernel(constant=1.0, bounds={"constant": (0.01, 100.0)})
        assert k.bounds["constant"][0] == (np.float64(0.01), np.float64(100.0))


# ===========================================================================
# Hyperparameters
# ===========================================================================


class TestConstantHyperparameters:
    def test_hyperparameters_tuple(self):
        k = ConstantKernel(constant=1.0)
        assert k.hyperparameters == ("constant",)


# ===========================================================================
# get_params / set_params
# ===========================================================================


class TestConstantGetSetParams:
    def test_get_returns_array(self):
        k = ConstantKernel(constant=3.0)
        params = k.get_params()
        assert params.shape == (1,)
        np.testing.assert_allclose(params, [3.0])

    def test_set_updates_value(self):
        k = ConstantKernel(constant=1.0)
        k.set_params(np.array([7.0]))
        np.testing.assert_allclose(k.get_params(), [7.0])

    def test_nonpositive_set_raises(self):
        k = ConstantKernel(constant=1.0)
        with pytest.raises(ValidationError):
            k.set_params(np.array([-1.0]))

    def test_wrong_size_raises(self):
        k = ConstantKernel(constant=1.0)
        with pytest.raises(ValidationError):
            k.set_params(np.array([1.0, 2.0]))


# ===========================================================================
# compute
# ===========================================================================


class TestConstantKernelCompute:
    def test_output_shape_square(self):
        k = ConstantKernel(constant=2.0)
        x = _make_x(8, 2)
        K = k.compute(x, x)
        assert K.shape == (8, 8)

    def test_output_shape_rect(self):
        k = ConstantKernel(constant=2.0)
        x1 = _make_x(5, 2)
        x2 = _make_x(7, 2)
        K = k.compute(x1, x2)
        assert K.shape == (5, 7)

    def test_all_entries_equal_constant(self):
        c = 3.5
        k = ConstantKernel(constant=c)
        x = _make_x(6, 2)
        K = k._compute(x, x)
        np.testing.assert_allclose(K, c * np.ones((6, 6)), atol=1e-12)

    def test_symmetric(self):
        k = ConstantKernel(constant=2.0)
        x = _make_x(6, 2)
        K = k._compute(x, x)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_psd(self):
        """Constant kernel with c > 0 is PSD (all eigenvalues are 0 except one)."""
        k = ConstantKernel(constant=2.0)
        x = _make_x(6, 2)
        K = k._compute(x, x)
        assert _is_psd(K)

    def test_independent_of_input_values(self):
        """The kernel value should not depend on the actual input points."""
        k = ConstantKernel(constant=2.0)
        x1 = _make_x(5, 2, seed=0)
        x2 = _make_x(5, 2, seed=99)
        K1 = k._compute(x1, x1)
        K2 = k._compute(x2, x2)
        np.testing.assert_allclose(K1, K2, atol=1e-12)


# ===========================================================================
# gradient
# ===========================================================================


class TestConstantKernelGradient:
    def test_gradient_shape(self):
        """∂K/∂c = 1 everywhere, so gradient should be shape (n, m, 1)."""
        k = ConstantKernel(constant=2.0)
        x = _make_x(5, 2)
        (grad,) = k._gradient(x, x)
        assert grad.shape == (5, 5, 1)

    def test_gradient_all_ones(self):
        k = ConstantKernel(constant=2.0)
        x = _make_x(5, 2)
        (grad,) = k._gradient(x, x)
        np.testing.assert_allclose(grad[:, :, 0], np.ones((5, 5)), atol=1e-12)

    @pytest.mark.slow
    def test_gradient_matches_finite_difference(self):
        k = ConstantKernel(constant=2.0)
        x = _make_x(4, 2, seed=9)
        (grad,) = k._gradient(x, x)
        analytical = grad[:, :, 0]

        eps = 1e-5
        params = k.get_params().copy()
        k.set_params(params + eps, _validate=False)
        K_plus = k._compute(x, x)
        k.set_params(params - eps, _validate=False)
        K_minus = k._compute(x, x)
        k.set_params(params, _validate=False)
        fd = (K_plus - K_minus) / (2 * eps)

        np.testing.assert_allclose(analytical, fd, rtol=1e-4, atol=1e-6)


# ===========================================================================
# compute_with_gradient
# ===========================================================================


class TestConstantComputeWithGradient:
    def test_K_matches_compute(self):
        k = ConstantKernel(constant=2.0)
        x = _make_x(5, 2)
        K_direct = k._compute(x, x)
        K_cwg, _ = k._compute_with_gradient(x, x)
        np.testing.assert_allclose(K_cwg, K_direct, atol=1e-12)
