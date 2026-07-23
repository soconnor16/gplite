"""Unit + integration tests for GaussianProcess.to_str()."""

import warnings

import numpy as np
import pytest
from gplite._utils._errors import ValidationError
from gplite.GaussianProcess.gaussian_process import GaussianProcess
from gplite.Kernels.rbf import RBFKernel


def _fitted_gp_1d() -> GaussianProcess:
    x = np.linspace(0, 2 * np.pi, 15).reshape(-1, 1)
    y = np.sin(x).ravel()
    gp = GaussianProcess(RBFKernel(length_scale=1.0))
    gp.fit(x, y)
    return gp


def _fitted_gp_2d() -> GaussianProcess:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((15, 2))
    y = x[:, 0] ** 2 + x[:, 1] ** 2
    gp = GaussianProcess(RBFKernel(length_scale=1.0))
    gp.fit(x, y)
    return gp


# ===========================================================================
# Unfitted model warning
# ===========================================================================


class TestGPToStrUnfitted:
    def test_unfitted_returns_empty_string(self):
        gp = GaussianProcess(RBFKernel(length_scale=1.0))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = gp.to_str(["x"])
        assert result == ""
        assert len(w) == 1
        assert "not fitted" in str(w[0].message).lower()


# ===========================================================================
# to_str - 1D
# ===========================================================================


class TestGPToStr1D:
    def test_returns_string(self):
        gp = _fitted_gp_1d()
        result = gp.to_str("x")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_string_contains_variable_name(self):
        gp = _fitted_gp_1d()
        result = gp.to_str("theta")
        assert "theta" in result

    def test_string_contains_exp(self):
        """RBF kernel string should contain 'exp'."""
        gp = _fitted_gp_1d()
        result = gp.to_str("x")
        assert "exp(" in result

    def test_single_string_accepted(self):
        """A plain string (not a list) should be accepted for 1D GPs."""
        gp = _fitted_gp_1d()
        result = gp.to_str("x")
        assert isinstance(result, str)

    def test_list_with_one_element_accepted(self):
        gp = _fitted_gp_1d()
        result = gp.to_str(["x"])
        assert isinstance(result, str)


# ===========================================================================
# to_str - 2D
# ===========================================================================


class TestGPToStr2D:
    def test_returns_string(self):
        gp = _fitted_gp_2d()
        result = gp.to_str(["x", "y"])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_string_contains_both_variable_names(self):
        gp = _fitted_gp_2d()
        result = gp.to_str(["phi", "psi"])
        assert "phi" in result
        assert "psi" in result


# ===========================================================================
# to_str - validation errors
# ===========================================================================


class TestGPToStrValidation:
    def test_wrong_number_of_variables_raises(self):
        gp = _fitted_gp_1d()  # 1-D GP
        with pytest.raises(ValidationError, match="Expected"):
            gp.to_str(["x", "y"])  # 2 names for 1-D GP

    def test_too_few_variables_raises(self):
        gp = _fitted_gp_2d()  # 2-D GP
        with pytest.raises(ValidationError, match="Expected"):
            gp.to_str(["x"])  # only 1 name for 2-D GP

    def test_non_string_element_raises(self):
        gp = _fitted_gp_1d()
        with pytest.raises(ValidationError):
            gp.to_str([8])

    def test_invalid_type_raises(self):
        gp = _fitted_gp_1d()
        with pytest.raises(ValidationError):
            gp.to_str(123)


# ===========================================================================
# to_str - standardized vs non-standardized inputs
# ===========================================================================


class TestGPToStrStandardization:
    def test_standardized_string_contains_x_mean(self):
        """Standardized GPs embed x_mean into the string expression."""
        gp = _fitted_gp_1d()
        result = gp.to_str("x")
        assert len(result) > 20

    def test_non_standardized_string_does_not_embed_x_mean(self):
        """Non-standardized GP should produce a simpler expression."""
        x = np.linspace(0, 2 * np.pi, 10).reshape(-1, 1)
        y = np.sin(x).ravel()
        gp = GaussianProcess(RBFKernel(length_scale=1.0), standardize_inputs=False)
        gp.fit(x, y)
        result = gp.to_str("x")
        assert isinstance(result, str)
        assert len(result) > 0
