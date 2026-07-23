"""Unit + integration tests for RBFKernel (isotropic and anisotropic)."""

import numpy as np
import pytest
from gplite._utils._errors import ValidationError
from gplite.Kernels.rbf import RBFKernel

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_x(n: int = 10, d: int = 1, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d))


def _is_psd(K: np.ndarray, tol: float = 1e-8) -> bool:
    """Tests whether a kernel is positive-semidefinite."""
    eigenvalues = np.linalg.eigvalsh(K)
    return bool(np.all(eigenvalues >= -tol))


def _finite_difference_gradient(kernel, x1, x2, param_index, eps=1e-5):
    """Numerical gradient of sum(K) w.r.t. the param at param_index."""
    params = kernel.get_params().copy()

    params_plus = params.copy()
    params_plus[param_index] += eps
    kernel.set_params(params_plus, _validate=False)
    K_plus = kernel._compute(x1, x2)

    params_minus = params.copy()
    params_minus[param_index] -= eps
    kernel.set_params(params_minus, _validate=False)
    K_minus = kernel._compute(x1, x2)

    # restore original params
    kernel.set_params(params, _validate=False)

    return (K_plus - K_minus) / (2 * eps)


# ===========================================================================
# Initialisation
# ===========================================================================


class TestRBFKernelInit:
    def test_isotropic_scalar(self):
        k = RBFKernel(length_scale=2.0)
        assert k.isotropic is True
        assert k.length_scale.shape == (1,)
        np.testing.assert_allclose(k.length_scale, [2.0])

    def test_anisotropic_list(self):
        k = RBFKernel(length_scale=[1.0, 2.0, 3.0], isotropic=False)
        assert k.isotropic is False
        assert k.length_scale.shape == (3,)

    def test_zero_length_scale_raises(self):
        with pytest.raises(ValidationError):
            RBFKernel(length_scale=0.0)

    def test_negative_length_scale_raises(self):
        with pytest.raises(ValidationError):
            RBFKernel(length_scale=-1.0)

    def test_custom_bounds_accepted(self):
        k = RBFKernel(length_scale=1.0, bounds={"length_scale": (0.1, 100.0)})
        assert k.bounds["length_scale"][0] == (
            np.float64(0.1),
            np.float64(100.0),
        )

    def test_invalid_bounds_raises(self):
        with pytest.raises(ValidationError):
            RBFKernel(length_scale=1.0, bounds={"bad_param": (0.1, 100.0)})


# ===========================================================================
# Hyperparameters property
# ===========================================================================


class TestRBFHyperparameters:
    def test_hyperparameters_tuple(self):
        k = RBFKernel(length_scale=1.0)
        assert k.hyperparameters == ("length_scale",)


# ===========================================================================
# get_params / set_params round-trip
# ===========================================================================


class TestRBFGetSetParams:
    def test_isotropic_get_returns_array(self):
        k = RBFKernel(length_scale=3.0)
        params = k.get_params()
        assert isinstance(params, np.ndarray)
        np.testing.assert_allclose(params, [3.0])

    def test_isotropic_set_updates_value(self):
        k = RBFKernel(length_scale=1.0)
        k.set_params(np.array([5.0]))
        np.testing.assert_allclose(k.length_scale, [5.0])

    def test_anisotropic_get_set_roundtrip(self):
        k = RBFKernel(length_scale=[1.0, 2.0], isotropic=False)
        original = k.get_params().copy()
        k.set_params(np.array([3.0, 4.0]))
        np.testing.assert_allclose(k.get_params(), [3.0, 4.0])
        k.set_params(original)
        np.testing.assert_allclose(k.get_params(), original)

    def test_nonpositive_set_raises(self):
        k = RBFKernel(length_scale=1.0)
        with pytest.raises(ValidationError):
            k.set_params(np.array([-1.0]))

    def test_isotropic_wrong_num_params_raises(self):
        k = RBFKernel(length_scale=1.0, isotropic=True)
        with pytest.raises(ValidationError, match="Wrong number"):
            k.set_params(np.array([1.0, 2.0]))


# ===========================================================================
# compute - isotropic
# ===========================================================================


class TestRBFComputeIsotropic:
    def setup_method(self):
        self.kernel = RBFKernel(length_scale=1.0)

    def test_output_shape_square(self):
        x = _make_x(8, 2)
        K = self.kernel.compute(x, x)
        assert K.shape == (8, 8)

    def test_output_shape_rect(self):
        x1 = _make_x(5, 2)
        x2 = _make_x(7, 2)
        K = self.kernel.compute(x1, x2)
        assert K.shape == (5, 7)

    def test_diagonal_is_one(self):
        """For an RBF kernel, K(x, x) = 1 for all x."""
        x = _make_x(10, 3)
        K = self.kernel._compute(x, x)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)

    def test_symmetric(self):
        x = _make_x(8, 2)
        K = self.kernel._compute(x, x)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_psd(self):
        x = _make_x(8, 2)
        K = self.kernel._compute(x, x)
        assert _is_psd(K)

    def test_values_between_zero_and_one(self):
        x1 = _make_x(5, 2)
        x2 = _make_x(5, 2)
        K = self.kernel._compute(x1, x2)
        assert np.all(K >= 0.0)
        assert np.all(K <= 1.0 + 1e-12)

    def test_known_value_1d(self):
        """K(0, 1) = exp(-0.5 * 1² / 1²) = exp(-0.5)."""
        x1 = np.array([[0.0]])
        x2 = np.array([[1.0]])
        K = self.kernel._compute(x1, x2)
        np.testing.assert_allclose(K[0, 0], np.exp(-0.5), rtol=1e-6)

    def test_feature_mismatch_raises(self):
        x1 = np.ones((5, 2))
        x2 = np.ones((5, 3))
        with pytest.raises(ValidationError):
            self.kernel.compute(x1, x2)

    def test_1d_input_accepted_via_compute(self):
        """Public compute() should accept 1-D arrays."""
        x = np.linspace(0, 1, 10)
        K = self.kernel.compute(x, x)
        assert K.shape == (10, 10)


# ===========================================================================
# compute - anisotropic
# ===========================================================================


class TestRBFComputeAnisotropic:
    def setup_method(self):
        self.kernel = RBFKernel(length_scale=[1.0, 2.0], isotropic=False)

    def test_output_shape(self):
        x = _make_x(8, 2)
        K = self.kernel.compute(x, x)
        assert K.shape == (8, 8)

    def test_symmetric(self):
        x = _make_x(8, 2)
        K = self.kernel._compute(x, x)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_psd(self):
        x = _make_x(8, 2)
        K = self.kernel._compute(x, x)
        assert _is_psd(K)

    def test_diagonal_is_one(self):
        x = _make_x(6, 2)
        K = self.kernel._compute(x, x)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)

    def test_dimension_mismatch_raises(self):
        """Anisotropic kernel with 2 length scales should reject 3-D data."""
        x = _make_x(5, 3)
        with pytest.raises(ValidationError):
            self.kernel.compute(x, x)

    def test_different_from_isotropic(self):
        """Anisotropic output should differ from isotropic on non-uniform data."""
        x = _make_x(6, 2, seed=7)
        k_iso = RBFKernel(length_scale=1.0, isotropic=True)
        k_aniso = RBFKernel(length_scale=[1.0, 3.0], isotropic=False)
        K_iso = k_iso._compute(x, x)
        K_aniso = k_aniso._compute(x, x)
        assert not np.allclose(K_iso, K_aniso)


# ===========================================================================
# compute_diag
# ===========================================================================


class TestRBFComputeDiag:
    def test_isotropic_diag_is_ones(self):
        k = RBFKernel(length_scale=1.0)
        x = _make_x(10, 3)
        diag = k._compute_diag(x)
        np.testing.assert_allclose(diag, np.ones(10), atol=1e-12)

    def test_output_shape(self):
        k = RBFKernel(length_scale=1.0)
        x = _make_x(7, 2)
        diag = k._compute_diag(x)
        assert diag.shape == (7,)


# ===========================================================================
# gradient - isotropic
# ===========================================================================


class TestRBFGradientIsotropic:
    def setup_method(self):
        self.kernel = RBFKernel(length_scale=1.5)

    def test_gradient_output_is_tuple(self):
        x = _make_x(5, 2)
        grads = self.kernel._gradient(x, x)
        assert isinstance(grads, tuple)

    def test_gradient_shape_isotropic(self):
        """Isotropic RBF gradient should have shape (n, m, 1)."""
        x = _make_x(5, 2)
        grads = self.kernel._gradient(x, x)
        assert grads[0].shape == (5, 5, 1)

    @pytest.mark.slow
    def test_gradient_matches_finite_difference_isotropic(self):
        """Analytical gradient vs finite-difference for isotropic length_scale."""
        x = _make_x(4, 2, seed=1)
        grads = self.kernel._gradient(x, x)
        analytical = grads[0][:, :, 0]

        fd = _finite_difference_gradient(self.kernel, x, x, param_index=0)
        np.testing.assert_allclose(analytical, fd, rtol=1e-4, atol=1e-6)


# ===========================================================================
# gradient - anisotropic
# ===========================================================================


class TestRBFGradientAnisotropic:
    def setup_method(self):
        self.kernel = RBFKernel(length_scale=[1.0, 2.0], isotropic=False)

    def test_gradient_shape_anisotropic(self):
        """Anisotropic RBF gradient should have shape (n, m, d)."""
        x = _make_x(5, 2)
        grads = self.kernel._gradient(x, x)
        assert grads[0].shape == (5, 5, 2)

    @pytest.mark.slow
    def test_gradient_matches_finite_difference_anisotropic(self):
        """Check each anisotropic dimension's analytical gradient vs FD."""
        x = _make_x(4, 2, seed=2)
        grads = self.kernel._gradient(x, x)
        analytical = grads[0]

        for dim in range(2):
            fd = _finite_difference_gradient(self.kernel, x, x, param_index=dim)
            np.testing.assert_allclose(
                analytical[:, :, dim],
                fd,
                rtol=1e-4,
                atol=1e-6,
                err_msg=f"Gradient mismatch at dimension {dim}",
            )


# ===========================================================================
# compute_with_gradient
# ===========================================================================


class TestRBFComputeWithGradient:
    def test_isotropic_K_matches_compute(self):
        k = RBFKernel(length_scale=1.0)
        x = _make_x(5, 2)
        K_direct = k._compute(x, x)
        K_with_grad, _ = k._compute_with_gradient(x, x)
        np.testing.assert_allclose(K_with_grad, K_direct, atol=1e-12)

    def test_anisotropic_K_matches_compute(self):
        k = RBFKernel(length_scale=[1.0, 2.0], isotropic=False)
        x = _make_x(5, 2)
        K_direct = k._compute(x, x)
        K_with_grad, _ = k._compute_with_gradient(x, x)
        np.testing.assert_allclose(K_with_grad, K_direct, atol=1e-12)

    @pytest.mark.slow
    def test_isotropic_grad_matches_gradient_method(self):
        k = RBFKernel(length_scale=1.5)
        x = _make_x(4, 2)
        _, (grad_cwg,) = k._compute_with_gradient(x, x)
        (grad_sep,) = k._gradient(x, x)
        np.testing.assert_allclose(grad_cwg, grad_sep, atol=1e-10)

    @pytest.mark.slow
    def test_anisotropic_grad_matches_gradient_method(self):
        k = RBFKernel(length_scale=[1.0, 2.0], isotropic=False)
        x = _make_x(4, 2)
        _, (grad_cwg,) = k._compute_with_gradient(x, x)
        (grad_sep,) = k._gradient(x, x)
        np.testing.assert_allclose(grad_cwg, grad_sep, atol=1e-10)


# ===========================================================================
# __call__ convenience
# ===========================================================================


class TestRBFCall:
    def test_single_arg_computes_self_kernel(self):
        k = RBFKernel(length_scale=1.0)
        x = _make_x(5, 2)
        K_call = k(x)
        K_compute = k.compute(x, x)
        np.testing.assert_allclose(K_call, K_compute, atol=1e-12)

    def test_two_args_computes_cross_kernel(self):
        k = RBFKernel(length_scale=1.0)
        x1 = _make_x(5, 2)
        x2 = _make_x(7, 2)
        K_call = k(x1, x2)
        K_compute = k.compute(x1, x2)
        np.testing.assert_allclose(K_call, K_compute, atol=1e-12)


# ===========================================================================
# _to_str
# ===========================================================================


class TestRBFToStr:
    def test_returns_string(self):
        k = RBFKernel(length_scale=1.0)
        x_train = np.array([[0.5, 1.0]])
        s = k._to_str(["x", "y"], np.float64(0.1), x_train[0])
        assert isinstance(s, str)
        assert "exp(" in s

    def test_string_contains_variable_names(self):
        k = RBFKernel(length_scale=1.0)
        x_train = np.array([[0.5]])
        s = k._to_str(["myvar"], np.float64(0.1), x_train[0])
        assert "myvar" in s
