"""Unit + integration tests for MaternKernel (both nu, isotropic and anisotropic)."""

import numpy as np
import pytest
from gplite._utils._errors import ValidationError
from gplite.Kernels.matern import MaternKernel

# ---------------------------------------------------------------------------
# Shared helpers (same pattern as test_rbf.py)
# ---------------------------------------------------------------------------


def _make_x(n: int = 10, d: int = 1, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d))


def _is_psd(K: np.ndarray, tol: float = 1e-8) -> bool:
    """Tests whether a kernel is positive-semidefinite."""
    eigenvalues = np.linalg.eigvalsh(K)
    return bool(np.all(eigenvalues >= -tol))


def _finite_difference_gradient(kernel, x1, x2, param_index, eps=1e-5):
    params = kernel.get_params().copy()

    params_plus = params.copy()
    params_plus[param_index] += eps
    kernel.set_params(params_plus, _validate=False)
    K_plus = kernel._compute(x1, x2)

    params_minus = params.copy()
    params_minus[param_index] -= eps
    kernel.set_params(params_minus, _validate=False)
    K_minus = kernel._compute(x1, x2)

    kernel.set_params(params, _validate=False)
    return (K_plus - K_minus) / (2 * eps)


# ===========================================================================
# Initialisation
# ===========================================================================


class TestMaternKernelInit:
    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_valid_nu_isotropic(self, nu):
        k = MaternKernel(length_scale=1.0, nu=nu, isotropic=True)
        assert k.length_scale.shape == (1,)
        assert k.isotropic is True

    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_valid_nu_anisotropic(self, nu):
        k = MaternKernel(length_scale=[1.0, 2.0], nu=nu, isotropic=False)
        assert k.length_scale.shape == (2,)
        assert k.isotropic is False

    def test_invalid_nu_raises(self):
        with pytest.raises(ValidationError, match="nu"):
            MaternKernel(length_scale=1.0, nu=3.0)

    def test_zero_length_scale_raises(self):
        with pytest.raises(ValidationError):
            MaternKernel(length_scale=0.0, nu=1.5)

    def test_custom_bounds_accepted(self):
        k = MaternKernel(length_scale=1.0, nu=2.5, bounds={"length_scale": (0.1, 50.0)})
        assert k.bounds["length_scale"][0] == (
            np.float64(0.1),
            np.float64(50.0),
        )


# ===========================================================================
# Hyperparameters property
# ===========================================================================


class TestMaternHyperparameters:
    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_hyperparameters_tuple(self, nu):
        k = MaternKernel(length_scale=1.0, nu=nu)
        assert k.hyperparameters == ("length_scale",)


# ===========================================================================
# get_params / set_params round-trip
# ===========================================================================


class TestMaternGetSetParams:
    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_isotropic_roundtrip(self, nu):
        k = MaternKernel(length_scale=2.0, nu=nu)
        k.set_params(np.array([5.0]))
        np.testing.assert_allclose(k.get_params(), [5.0])

    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_anisotropic_roundtrip(self, nu):
        k = MaternKernel(length_scale=[1.0, 2.0], nu=nu, isotropic=False)
        k.set_params(np.array([3.0, 4.0]))
        np.testing.assert_allclose(k.get_params(), [3.0, 4.0])

    def test_nonpositive_raises(self):
        k = MaternKernel(length_scale=1.0, nu=2.5)
        with pytest.raises(ValidationError):
            k.set_params(np.array([-1.0]))


# ===========================================================================
# compute - isotropic, both nu values
# ===========================================================================


class TestMaternComputeIsotropic:
    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_output_shape_square(self, nu):
        k = MaternKernel(length_scale=1.0, nu=nu)
        x = _make_x(8, 2)
        K = k.compute(x, x)
        assert K.shape == (8, 8)

    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_output_shape_rect(self, nu):
        k = MaternKernel(length_scale=1.0, nu=nu)
        x1 = _make_x(5, 2)
        x2 = _make_x(7, 2)
        K = k.compute(x1, x2)
        assert K.shape == (5, 7)

    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_diagonal_is_one(self, nu):
        k = MaternKernel(length_scale=1.0, nu=nu)
        x = _make_x(8, 3)
        K = k._compute(x, x)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_symmetric(self, nu):
        k = MaternKernel(length_scale=1.0, nu=nu)
        x = _make_x(8, 2)
        K = k._compute(x, x)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_psd(self, nu):
        k = MaternKernel(length_scale=1.0, nu=nu)
        x = _make_x(8, 2)
        K = k._compute(x, x)
        assert _is_psd(K)

    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_values_in_zero_one(self, nu):
        k = MaternKernel(length_scale=1.0, nu=nu)
        x1 = _make_x(5, 2)
        x2 = _make_x(5, 2)
        K = k._compute(x1, x2)
        assert np.all(K >= 0.0 - 1e-12)
        assert np.all(K <= 1.0 + 1e-12)

    def test_nu_15_known_value(self):
        """K_{3/2}(0,1) = (1 + √3) * exp(-√3) with l=1."""
        k = MaternKernel(length_scale=1.0, nu=1.5)
        x1 = np.array([[0.0]])
        x2 = np.array([[1.0]])
        r = 1.0
        expected = (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
        K = k._compute(x1, x2)
        np.testing.assert_allclose(K[0, 0], expected, rtol=1e-6)

    def test_nu_25_known_value(self):
        """K_{5/2}(0,1) = (1 + √5 + 5/3) * exp(-√5) with l=1."""
        k = MaternKernel(length_scale=1.0, nu=2.5)
        x1 = np.array([[0.0]])
        x2 = np.array([[1.0]])
        r = 1.0
        expected = (1 + np.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-np.sqrt(5) * r)
        K = k._compute(x1, x2)
        np.testing.assert_allclose(K[0, 0], expected, rtol=1e-6)

    def test_nu_15_less_smooth_than_nu_25(self):
        """At very short distances, nu=1.5 should produce a lower covariance value
        than nu=2.5 because it decays more sharply.
        """
        k15 = MaternKernel(length_scale=1.0, nu=1.5)
        k25 = MaternKernel(length_scale=1.0, nu=2.5)
        x1 = np.array([[0.0]])
        x2 = np.array([[1.0]])
        K15 = k15._compute(x1, x2)[0, 0]
        K25 = k25._compute(x1, x2)[0, 0]
        assert K15 < K25  # ν=1.5 decays faster at this distance


# ===========================================================================
# compute - anisotropic, both nu values
# ===========================================================================


class TestMaternComputeAnisotropic:
    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_output_shape(self, nu):
        k = MaternKernel(length_scale=[1.0, 2.0], nu=nu, isotropic=False)
        x = _make_x(8, 2)
        K = k.compute(x, x)
        assert K.shape == (8, 8)

    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_symmetric(self, nu):
        k = MaternKernel(length_scale=[1.0, 2.0], nu=nu, isotropic=False)
        x = _make_x(8, 2)
        K = k._compute(x, x)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_psd(self, nu):
        k = MaternKernel(length_scale=[1.0, 2.0], nu=nu, isotropic=False)
        x = _make_x(8, 2)
        K = k._compute(x, x)
        assert _is_psd(K)

    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_diagonal_is_one(self, nu):
        k = MaternKernel(length_scale=[1.0, 2.0], nu=nu, isotropic=False)
        x = _make_x(6, 2)
        K = k._compute(x, x)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_dimension_mismatch_raises(self, nu):
        k = MaternKernel(length_scale=[1.0, 2.0], nu=nu, isotropic=False)
        x = _make_x(5, 3)
        with pytest.raises(ValidationError):
            k.compute(x, x)


# ===========================================================================
# gradient - isotropic
# ===========================================================================


class TestMaternGradientIsotropic:
    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_gradient_is_tuple(self, nu):
        k = MaternKernel(length_scale=1.0, nu=nu)
        x = _make_x(5, 2)
        grads = k._gradient(x, x)
        assert isinstance(grads, tuple)

    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_gradient_shape_isotropic(self, nu):
        k = MaternKernel(length_scale=1.0, nu=nu)
        x = _make_x(5, 2)
        grads = k._gradient(x, x)
        assert grads[0].shape == (5, 5, 1)

    @pytest.mark.slow
    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_gradient_matches_finite_difference_isotropic(self, nu):
        k = MaternKernel(length_scale=1.5, nu=nu)
        x = _make_x(4, 2, seed=3)
        grads = k._gradient(x, x)
        analytical = grads[0][:, :, 0]
        fd = _finite_difference_gradient(k, x, x, param_index=0)
        np.testing.assert_allclose(analytical, fd, rtol=1e-4, atol=1e-6)


# ===========================================================================
# gradient - anisotropic
# ===========================================================================


class TestMaternGradientAnisotropic:
    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_gradient_shape_anisotropic(self, nu):
        k = MaternKernel(length_scale=[1.0, 2.0], nu=nu, isotropic=False)
        x = _make_x(5, 2)
        grads = k._gradient(x, x)
        assert grads[0].shape == (5, 5, 2)

    @pytest.mark.slow
    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_gradient_matches_finite_difference_anisotropic(self, nu):
        k = MaternKernel(length_scale=[1.0, 2.0], nu=nu, isotropic=False)
        x = _make_x(4, 2, seed=4)
        grads = k._gradient(x, x)
        analytical = grads[0]
        for dim in range(2):
            fd = _finite_difference_gradient(k, x, x, param_index=dim)
            np.testing.assert_allclose(
                analytical[:, :, dim],
                fd,
                rtol=1e-4,
                atol=1e-6,
                err_msg=f"Gradient mismatch nu={nu} dim={dim}",
            )


# ===========================================================================
# compute_with_gradient
# ===========================================================================


class TestMaternComputeWithGradient:
    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_K_matches_compute_isotropic(self, nu):
        k = MaternKernel(length_scale=1.0, nu=nu)
        x = _make_x(5, 2)
        K_direct = k._compute(x, x)
        K_cwg, _ = k._compute_with_gradient(x, x)
        np.testing.assert_allclose(K_cwg, K_direct, atol=1e-12)

    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_K_matches_compute_anisotropic(self, nu):
        k = MaternKernel(length_scale=[1.0, 2.0], nu=nu, isotropic=False)
        x = _make_x(5, 2)
        K_direct = k._compute(x, x)
        K_cwg, _ = k._compute_with_gradient(x, x)
        np.testing.assert_allclose(K_cwg, K_direct, atol=1e-12)

    @pytest.mark.slow
    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_grad_matches_gradient_method_isotropic(self, nu):
        k = MaternKernel(length_scale=1.5, nu=nu)
        x = _make_x(4, 2)
        _, (grad_cwg,) = k._compute_with_gradient(x, x)
        (grad_sep,) = k._gradient(x, x)
        np.testing.assert_allclose(grad_cwg, grad_sep, atol=1e-10)
