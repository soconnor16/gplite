"""Unit + integration tests for PeriodicKernel (isotropic and anisotropic)."""

import numpy as np
import pytest
from gplite._utils._errors import ValidationError
from gplite.Kernels.periodic import PeriodicKernel

# ---------------------------------------------------------------------------
# Helpers
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


class TestPeriodicKernelInit:
    def test_isotropic_scalars(self):
        k = PeriodicKernel(length_scale=1.0, period=2.0)
        assert k.isotropic is True
        assert k.length_scale.shape == (1,)
        assert k.period.shape == (1,)
        np.testing.assert_allclose(k.period, [2.0])

    def test_anisotropic_arrays(self):
        k = PeriodicKernel(length_scale=[1.0, 1.5], period=[2.0, 3.0], isotropic=False)
        assert k.isotropic is False
        assert k.length_scale.shape == (2,)
        assert k.period.shape == (2,)

    def test_mismatched_aniso_sizes_raise(self):
        with pytest.raises(ValidationError):
            PeriodicKernel(length_scale=[1.0, 1.5], period=[2.0], isotropic=False)

    def test_zero_length_scale_raises(self):
        with pytest.raises(ValidationError):
            PeriodicKernel(length_scale=0.0, period=1.0)

    def test_zero_period_raises(self):
        with pytest.raises(ValidationError):
            PeriodicKernel(length_scale=1.0, period=0.0)

    def test_custom_bounds(self):
        k = PeriodicKernel(
            length_scale=1.0,
            period=2.0,
            bounds={"period": (0.5, 10.0)},
        )
        assert k.bounds["period"][0] == (np.float64(0.5), np.float64(10.0))


# ===========================================================================
# Hyperparameters property
# ===========================================================================


class TestPeriodicHyperparameters:
    def test_hyperparameters_tuple(self):
        k = PeriodicKernel(length_scale=1.0, period=2.0)
        assert k.hyperparameters == ("length_scale", "period")


# ===========================================================================
# get_params / set_params round-trip
# ===========================================================================


class TestPeriodicGetSetParams:
    def test_isotropic_get_concatenated(self):
        """get_params() concatenates [length_scale, period]."""
        k = PeriodicKernel(length_scale=1.0, period=2.0)
        params = k.get_params()
        assert params.shape == (2,)
        np.testing.assert_allclose(params, [1.0, 2.0])

    def test_isotropic_set_roundtrip(self):
        k = PeriodicKernel(length_scale=1.0, period=2.0)
        k.set_params(np.array([3.0, 4.0]))
        params = k.get_params()
        np.testing.assert_allclose(params, [3.0, 4.0])
        np.testing.assert_allclose(k.length_scale, [3.0])
        np.testing.assert_allclose(k.period, [4.0])

    def test_anisotropic_get_concatenated(self):
        k = PeriodicKernel(length_scale=[1.0, 1.5], period=[2.0, 3.0], isotropic=False)
        params = k.get_params()
        assert params.shape == (4,)
        np.testing.assert_allclose(params, [1.0, 1.5, 2.0, 3.0])

    def test_anisotropic_set_roundtrip(self):
        k = PeriodicKernel(length_scale=[1.0, 1.5], period=[2.0, 3.0], isotropic=False)
        k.set_params(np.array([2.0, 2.5, 5.0, 6.0]))
        params = k.get_params()
        np.testing.assert_allclose(params, [2.0, 2.5, 5.0, 6.0])

    def test_nonpositive_raises(self):
        k = PeriodicKernel(length_scale=1.0, period=2.0)
        with pytest.raises(ValidationError):
            k.set_params(np.array([-1.0, 2.0]))


# ===========================================================================
# compute - isotropic
# ===========================================================================


class TestPeriodicComputeIsotropic:
    def setup_method(self):
        self.kernel = PeriodicKernel(length_scale=1.0, period=2.0)

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
        """K(x, x) = exp(0) = 1 for all x."""
        x = _make_x(8, 3)
        K = self.kernel._compute(x, x)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_symmetric(self):
        x = _make_x(8, 2)
        K = self.kernel._compute(x, x)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_psd(self):
        x = _make_x(8, 2)
        K = self.kernel._compute(x, x)
        assert _is_psd(K)

    def test_values_in_zero_one(self):
        x1 = _make_x(5, 2)
        x2 = _make_x(5, 2)
        K = self.kernel._compute(x1, x2)
        assert np.all(K >= 0.0 - 1e-12)
        assert np.all(K <= 1.0 + 1e-12)

    def test_known_value_1d(self):
        """K(0, p/2) should be exp(-2/l²) for the isotropic periodic kernel,
        because sin²(π * (p/2) / p) = sin²(π/2) = 1.
        """
        p = 2.0
        l = 1.0
        k = PeriodicKernel(length_scale=l, period=p)
        x1 = np.array([[0.0]])
        x2 = np.array([[p / 2]])  # half period away
        K = k._compute(x1, x2)
        expected = np.exp(-2.0 / l**2)
        np.testing.assert_allclose(K[0, 0], expected, rtol=1e-6)

    def test_periodicity(self):
        """K(x, x') should equal K(x, x' + period) for a periodic kernel."""
        k = PeriodicKernel(length_scale=1.0, period=3.0)
        x1 = np.array([[1.0]])
        x2 = np.array([[2.0]])
        x2_shifted = x2 + 3.0  # add one full period
        K_orig = k._compute(x1, x2)
        K_shifted = k._compute(x1, x2_shifted)
        np.testing.assert_allclose(K_orig, K_shifted, atol=1e-10)


# ===========================================================================
# compute - anisotropic
# ===========================================================================


class TestPeriodicComputeAnisotropic:
    def setup_method(self):
        self.kernel = PeriodicKernel(
            length_scale=[1.0, 1.5],
            period=[2.0, 3.0],
            isotropic=False,
        )

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
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_dimension_mismatch_raises(self):
        x = _make_x(5, 3)
        with pytest.raises(ValidationError):
            self.kernel.compute(x, x)


# ===========================================================================
# gradient - isotropic
# ===========================================================================


class TestPeriodicGradientIsotropic:
    def setup_method(self):
        self.kernel = PeriodicKernel(length_scale=1.0, period=2.0)

    def test_gradient_is_tuple_of_two(self):
        """Periodic kernel has two gradient components: length_scale, period."""
        x = _make_x(5, 2)
        grads = self.kernel._gradient(x, x)
        assert isinstance(grads, tuple)
        assert len(grads) == 2

    def test_gradient_shapes_isotropic(self):
        x = _make_x(5, 2)
        grad_ls, grad_p = self.kernel._gradient(x, x)
        assert grad_ls.shape == (5, 5, 1)
        assert grad_p.shape == (5, 5, 1)

    @pytest.mark.slow
    def test_gradient_ls_matches_fd_isotropic(self):
        k = PeriodicKernel(length_scale=1.5, period=2.0)
        x = _make_x(4, 2, seed=5)
        grad_ls, _ = k._gradient(x, x)
        fd = _finite_difference_gradient(k, x, x, param_index=0)
        np.testing.assert_allclose(grad_ls[:, :, 0], fd, rtol=1e-4, atol=1e-6)

    @pytest.mark.slow
    def test_gradient_period_matches_fd_isotropic(self):
        k = PeriodicKernel(length_scale=1.0, period=2.0)
        x = _make_x(4, 2, seed=6)
        _, grad_p = k._gradient(x, x)
        fd = _finite_difference_gradient(k, x, x, param_index=1)
        np.testing.assert_allclose(grad_p[:, :, 0], fd, rtol=1e-4, atol=1e-6)


# ===========================================================================
# gradient - anisotropic
# ===========================================================================


class TestPeriodicGradientAnisotropic:
    def setup_method(self):
        self.kernel = PeriodicKernel(
            length_scale=[1.0, 1.5],
            period=[2.0, 3.0],
            isotropic=False,
        )

    def test_gradient_shapes_anisotropic(self):
        x = _make_x(5, 2)
        grad_ls, grad_p = self.kernel._gradient(x, x)
        assert grad_ls.shape == (5, 5, 2)
        assert grad_p.shape == (5, 5, 2)

    @pytest.mark.slow
    def test_gradient_ls_matches_fd_anisotropic(self):
        k = PeriodicKernel(
            length_scale=[1.0, 1.5],
            period=[2.0, 3.0],
            isotropic=False,
        )
        x = _make_x(4, 2, seed=7)
        grad_ls, _ = k._gradient(x, x)
        for dim in range(2):
            fd = _finite_difference_gradient(k, x, x, param_index=dim)
            np.testing.assert_allclose(
                grad_ls[:, :, dim],
                fd,
                rtol=1e-4,
                atol=1e-6,
                err_msg=f"ls gradient mismatch at dim={dim}",
            )

    @pytest.mark.slow
    def test_gradient_period_matches_fd_anisotropic(self):
        k = PeriodicKernel(
            length_scale=[1.0, 1.5],
            period=[2.0, 3.0],
            isotropic=False,
        )
        x = _make_x(4, 2, seed=8)
        _, grad_p = k._gradient(x, x)
        for dim in range(2):
            fd = _finite_difference_gradient(k, x, x, param_index=2 + dim)
            np.testing.assert_allclose(
                grad_p[:, :, dim],
                fd,
                rtol=1e-4,
                atol=1e-6,
                err_msg=f"period gradient mismatch at dim={dim}",
            )


# ===========================================================================
# compute_with_gradient
# ===========================================================================


class TestPeriodicComputeWithGradient:
    def test_K_matches_compute_isotropic(self):
        k = PeriodicKernel(length_scale=1.0, period=2.0)
        x = _make_x(5, 2)
        K_direct = k._compute(x, x)
        K_cwg, _ = k._compute_with_gradient(x, x)
        np.testing.assert_allclose(K_cwg, K_direct, atol=1e-12)

    def test_K_matches_compute_anisotropic(self):
        k = PeriodicKernel(
            length_scale=[1.0, 1.5],
            period=[2.0, 3.0],
            isotropic=False,
        )
        x = _make_x(5, 2)
        K_direct = k._compute(x, x)
        K_cwg, _ = k._compute_with_gradient(x, x)
        np.testing.assert_allclose(K_cwg, K_direct, atol=1e-12)

    @pytest.mark.slow
    def test_grad_matches_gradient_method_isotropic(self):
        k = PeriodicKernel(length_scale=1.0, period=2.0)
        x = _make_x(4, 2)
        _, (grad_ls_cwg, grad_p_cwg) = k._compute_with_gradient(x, x)
        grad_ls_sep, grad_p_sep = k._gradient(x, x)
        np.testing.assert_allclose(grad_ls_cwg, grad_ls_sep, atol=1e-10)
        np.testing.assert_allclose(grad_p_cwg, grad_p_sep, atol=1e-10)
