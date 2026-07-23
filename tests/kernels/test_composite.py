"""Unit + integration tests for composite kernels (Additive and Product)."""

import numpy as np
import pytest
from gplite._utils._errors import ValidationError
from gplite.Kernels._composite import AdditiveKernel, ProductKernel
from gplite.Kernels.constant import ConstantKernel
from gplite.Kernels.matern import MaternKernel
from gplite.Kernels.periodic import PeriodicKernel
from gplite.Kernels.rbf import RBFKernel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_x(n: int = 8, d: int = 2, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d))


def _is_psd(K: np.ndarray, tol: float = 1e-8) -> bool:
    """Tests whether a kernel is positive-semidefinite."""
    eigenvalues = np.linalg.eigvalsh(K)
    return bool(np.all(eigenvalues >= -tol))


# ===========================================================================
# Initialisation via + and * operators (Kernel.__add__, __mul__)
# ===========================================================================


class TestCompositeKernelConstruction:
    def test_add_returns_additive_kernel(self):
        k = RBFKernel(length_scale=1.0) + MaternKernel(length_scale=1.0, nu=2.5)
        assert isinstance(k, AdditiveKernel)

    def test_mul_returns_product_kernel(self):
        k = RBFKernel(length_scale=1.0) * ConstantKernel(constant=2.0)
        assert isinstance(k, ProductKernel)

    def test_invalid_operand_raises(self):
        with pytest.raises(ValidationError, match="Kernel"):
            AdditiveKernel(RBFKernel(length_scale=1.0), "not_a_kernel")

    def test_chaining_three_additive_kernels(self):
        """k1 + k2 + k3 should remain flat (not doubly nested)."""
        k1 = RBFKernel(length_scale=1.0)
        k2 = MaternKernel(length_scale=1.0, nu=2.5)
        k3 = ConstantKernel(constant=1.0)
        composite = k1 + k2 + k3
        assert isinstance(composite, AdditiveKernel)
        assert len(composite.kernels) == 3

    def test_chaining_three_product_kernels(self):
        k1 = RBFKernel(length_scale=1.0)
        k2 = ConstantKernel(constant=2.0)
        k3 = MaternKernel(length_scale=1.0, nu=1.5)
        composite = k1 * k2 * k3
        assert isinstance(composite, ProductKernel)
        assert len(composite.kernels) == 3


# ===========================================================================
# AdditiveKernel - compute
# ===========================================================================


class TestAdditiveKernelCompute:
    def _make(self):
        k1 = RBFKernel(length_scale=1.0)
        k2 = MaternKernel(length_scale=1.0, nu=2.5)
        return k1 + k2, k1, k2

    def test_output_shape_square(self):
        k, _, _ = self._make()
        x = _make_x(8, 2)
        K = k.compute(x, x)
        assert K.shape == (8, 8)

    def test_output_shape_rect(self):
        k, _, _ = self._make()
        x1 = _make_x(5, 2)
        x2 = _make_x(7, 2)
        K = k.compute(x1, x2)
        assert K.shape == (5, 7)

    def test_equals_sum_of_children(self):
        """K_sum(x, x') should equal K1(x, x') + K2(x, x')."""
        k_sum, k1, k2 = self._make()
        x = _make_x(6, 2)
        K_add = k_sum._compute(x, x)
        K_expected = k1._compute(x, x) + k2._compute(x, x)
        np.testing.assert_allclose(K_add, K_expected, atol=1e-12)

    def test_symmetric(self):
        k, _, _ = self._make()
        x = _make_x(8, 2)
        K = k._compute(x, x)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_psd(self):
        k, _, _ = self._make()
        x = _make_x(8, 2)
        K = k._compute(x, x)
        assert _is_psd(K)

    def test_anisotropic_component(self):
        k = RBFKernel(length_scale=[1.0, 2.0], isotropic=False) + MaternKernel(
            length_scale=[0.5, 1.5], nu=1.5, isotropic=False
        )
        x = _make_x(6, 2)
        K = k.compute(x, x)
        assert K.shape == (6, 6)
        assert _is_psd(K)


# ===========================================================================
# ProductKernel - compute
# ===========================================================================


class TestProductKernelCompute:
    def _make(self):
        k1 = RBFKernel(length_scale=1.0)
        k2 = ConstantKernel(constant=2.0)
        return k1 * k2, k1, k2

    def test_output_shape_square(self):
        k, _, _ = self._make()
        x = _make_x(8, 2)
        K = k.compute(x, x)
        assert K.shape == (8, 8)

    def test_output_shape_rect(self):
        k, _, _ = self._make()
        x1 = _make_x(5, 2)
        x2 = _make_x(7, 2)
        K = k.compute(x1, x2)
        assert K.shape == (5, 7)

    def test_equals_product_of_children(self):
        """K_prod(x, x') should equal K1(x, x') * K2(x, x')."""
        k_prod, k1, k2 = self._make()
        x = _make_x(6, 2)
        K_prod = k_prod._compute(x, x)
        K_expected = k1._compute(x, x) * k2._compute(x, x)
        np.testing.assert_allclose(K_prod, K_expected, atol=1e-12)

    def test_symmetric(self):
        k, _, _ = self._make()
        x = _make_x(8, 2)
        K = k._compute(x, x)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_psd(self):
        k, _, _ = self._make()
        x = _make_x(8, 2)
        K = k._compute(x, x)
        assert _is_psd(K)

    def test_constant_product_scales_kernel(self):
        """RBF * Constant(c) should equal c * RBF."""
        rbf = RBFKernel(length_scale=1.0)
        c = 3.0
        k_prod = rbf * ConstantKernel(constant=c)
        x = _make_x(5, 2)
        K_prod = k_prod._compute(x, x)
        K_rbf = rbf._compute(x, x)
        np.testing.assert_allclose(K_prod, c * K_rbf, atol=1e-12)

    def test_anisotropic_component(self):
        k = RBFKernel(length_scale=[1.0, 2.0], isotropic=False) * ConstantKernel(
            constant=2.0
        )
        x = _make_x(6, 2)
        K = k.compute(x, x)
        assert K.shape == (6, 6)
        assert _is_psd(K)


# ===========================================================================
# Hyperparameters and bounds
# ===========================================================================


class TestCompositeHyperparameters:
    def test_additive_hyperparameters_concatenated(self):
        k = RBFKernel(length_scale=1.0) + ConstantKernel(constant=2.0)
        assert k.hyperparameters == ("length_scale", "constant")

    def test_product_hyperparameters_concatenated(self):
        k = RBFKernel(length_scale=1.0) * ConstantKernel(constant=2.0)
        assert k.hyperparameters == ("length_scale", "constant")

    def test_bounds_has_unique_keys(self):
        """When two same-type kernels are composed, bounds keys must be unique."""
        k = RBFKernel(length_scale=1.0) + RBFKernel(length_scale=2.0)
        bounds = k.bounds
        assert "kernel_0_length_scale" in bounds
        assert "kernel_1_length_scale" in bounds

    def test_bounds_length_matches_params(self):
        k = PeriodicKernel(length_scale=1.0, period=2.0) + RBFKernel(length_scale=1.0)
        flat_bounds = k._bounds
        params = k.get_params()
        assert len(flat_bounds) == len(params)


# ===========================================================================
# get_params / set_params round-trip
# ===========================================================================


class TestCompositeGetSetParams:
    def test_additive_get_concatenated(self):
        k = RBFKernel(length_scale=1.5) + ConstantKernel(constant=2.5)
        params = k.get_params()
        np.testing.assert_allclose(params, [1.5, 2.5])

    def test_additive_set_dispatches_to_children(self):
        k1 = RBFKernel(length_scale=1.0)
        k2 = ConstantKernel(constant=1.0)
        k = k1 + k2
        k.set_params(np.array([3.0, 4.0]))
        np.testing.assert_allclose(k1.get_params(), [3.0])
        np.testing.assert_allclose(k2.get_params(), [4.0])

    def test_product_get_set_roundtrip(self):
        k = RBFKernel(length_scale=1.0) * ConstantKernel(constant=2.0)
        original = k.get_params().copy()
        k.set_params(np.array([5.0, 6.0]))
        np.testing.assert_allclose(k.get_params(), [5.0, 6.0])
        k.set_params(original)
        np.testing.assert_allclose(k.get_params(), original)

    def test_anisotropic_composite_get_set(self):
        """Composite with anisotropic children should correctly concatenate params."""
        k1 = RBFKernel(length_scale=[1.0, 2.0], isotropic=False)
        k2 = MaternKernel(length_scale=[0.5, 1.5], nu=2.5, isotropic=False)
        k = k1 + k2
        params = k.get_params()
        np.testing.assert_allclose(params, [1.0, 2.0, 0.5, 1.5])
        k.set_params(np.array([2.0, 3.0, 1.0, 2.0]))
        np.testing.assert_allclose(k.get_params(), [2.0, 3.0, 1.0, 2.0])


# ===========================================================================
# Gradient
# ===========================================================================


class TestCompositeGradient:
    def test_additive_gradient_is_tuple(self):
        k = RBFKernel(length_scale=1.0) + ConstantKernel(constant=2.0)
        x = _make_x(5, 2)
        grads = k._gradient(x, x)
        assert isinstance(grads, tuple)

    def test_additive_gradient_num_components(self):
        """Additive kernel has one gradient per child hyperparameter."""
        k = RBFKernel(length_scale=1.0) + ConstantKernel(constant=2.0)
        x = _make_x(5, 2)
        grads = k._gradient(x, x)
        # 1 gradient for RBF ls + 1 gradient for Constant c = 2 total
        assert len(grads) == 2

    def test_product_gradient_is_tuple(self):
        k = RBFKernel(length_scale=1.0) * ConstantKernel(constant=2.0)
        x = _make_x(5, 2)
        grads = k._gradient(x, x)
        assert isinstance(grads, tuple)

    @pytest.mark.slow
    def test_additive_gradient_matches_fd(self):
        """Finite-difference check for additive kernel gradient."""
        k = RBFKernel(length_scale=1.5) + ConstantKernel(constant=2.0)
        x = _make_x(4, 2, seed=10)
        grads = k._gradient(x, x)

        params = k.get_params().copy()
        eps = 1e-5
        for i, grad_tensor in enumerate(grads):
            analytical = grad_tensor[:, :, 0]
            p_plus = params.copy()
            p_plus[i] += eps
            k.set_params(p_plus, _validate=False)
            K_plus = k._compute(x, x)

            p_minus = params.copy()
            p_minus[i] -= eps
            k.set_params(p_minus, _validate=False)
            K_minus = k._compute(x, x)

            k.set_params(params, _validate=False)
            fd = (K_plus - K_minus) / (2 * eps)
            np.testing.assert_allclose(
                analytical,
                fd,
                rtol=1e-4,
                atol=1e-6,
                err_msg=f"Gradient mismatch at param index {i}",
            )

    @pytest.mark.slow
    def test_product_gradient_matches_fd(self):
        """Finite-difference check for product kernel gradient."""
        k = RBFKernel(length_scale=1.5) * ConstantKernel(constant=2.0)
        x = _make_x(4, 2, seed=11)
        grads = k._gradient(x, x)

        params = k.get_params().copy()
        eps = 1e-5
        for i, grad_tensor in enumerate(grads):
            analytical = grad_tensor[:, :, 0]
            p_plus = params.copy()
            p_plus[i] += eps
            k.set_params(p_plus, _validate=False)
            K_plus = k._compute(x, x)

            p_minus = params.copy()
            p_minus[i] -= eps
            k.set_params(p_minus, _validate=False)
            K_minus = k._compute(x, x)

            k.set_params(params, _validate=False)
            fd = (K_plus - K_minus) / (2 * eps)
            np.testing.assert_allclose(
                analytical,
                fd,
                rtol=1e-4,
                atol=1e-6,
                err_msg=f"Product gradient mismatch at param index {i}",
            )


# ===========================================================================
# compute_with_gradient
# ===========================================================================


class TestCompositeComputeWithGradient:
    def test_additive_K_matches_compute(self):
        k = RBFKernel(length_scale=1.0) + MaternKernel(length_scale=1.0, nu=2.5)
        x = _make_x(5, 2)
        K_direct = k._compute(x, x)
        K_cwg, _ = k._compute_with_gradient(x, x)
        np.testing.assert_allclose(K_cwg, K_direct, atol=1e-12)

    def test_product_K_matches_compute(self):
        k = RBFKernel(length_scale=1.0) * ConstantKernel(constant=2.0)
        x = _make_x(5, 2)
        K_direct = k._compute(x, x)
        K_cwg, _ = k._compute_with_gradient(x, x)
        np.testing.assert_allclose(K_cwg, K_direct, atol=1e-12)
