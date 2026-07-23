"""Unit tests for gplite._utils._validation."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from gplite._utils._errors import ValidationError
from gplite._utils._validation import (
    validate_anisotropic_hyperparameter,
    validate_anisotropic_hyperparameter_shape,
    validate_bounds_dict,
    validate_input_and_target_data,
    validate_input_arrays,
    validate_isotropic_hyperparameter,
    validate_multiple_anisotropic_hyperparameter_size,
    validate_numeric_array,
    validate_numeric_value,
    validate_set_params,
    validate_variable_names,
)

# ===========================================================================
# validate_numeric_value
# ===========================================================================


class TestValidateNumericValue:
    def test_valid_positive_float(self):
        result = validate_numeric_value(3.14, "x", allow_nonpositive=True)
        assert isinstance(result, np.float64)
        np.testing.assert_almost_equal(result, 3.14)

    def test_valid_int_coerced(self):
        result = validate_numeric_value(5, "x", allow_nonpositive=True)
        assert result == np.float64(5)

    def test_valid_string_float(self):
        result = validate_numeric_value("2.5", "x", allow_nonpositive=True)
        assert result == np.float64(2.5)

    def test_nan_raises(self):
        with pytest.raises(ValidationError, match="nan"):
            validate_numeric_value(float("nan"), "x", allow_nonpositive=True)

    def test_inf_raises(self):
        with pytest.raises(ValidationError, match="inf"):
            validate_numeric_value(float("inf"), "x", allow_nonpositive=True)

    def test_non_numeric_raises(self):
        with pytest.raises(ValidationError):
            validate_numeric_value("hello", "x", allow_nonpositive=True)

    def test_nonpositive_zero_raises_when_disallowed(self):
        with pytest.raises(ValidationError, match="positive"):
            validate_numeric_value(0.0, "x", allow_nonpositive=False)

    def test_negative_raises_when_disallowed(self):
        with pytest.raises(ValidationError, match="positive"):
            validate_numeric_value(-1.0, "x", allow_nonpositive=False)

    def test_negative_allowed(self):
        result = validate_numeric_value(-5.0, "x", allow_nonpositive=True)
        assert result == np.float64(-5.0)

    def test_zero_allowed(self):
        result = validate_numeric_value(0.0, "x", allow_nonpositive=True)
        assert result == np.float64(0.0)


# ===========================================================================
# validate_numeric_array
# ===========================================================================


class TestValidateNumericArray:
    def test_valid_list(self):
        result = validate_numeric_array([1.0, 2.0, 3.0], "arr", allow_nonpositive=True)
        assert result.dtype == np.float64
        assert result.shape == (3,)

    def test_valid_2d_array(self):
        arr = np.ones((4, 3))
        result = validate_numeric_array(arr, "arr", allow_nonpositive=True)
        assert result.shape == (4, 3)

    def test_empty_raises(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_numeric_array([], "arr", allow_nonpositive=True)

    def test_nan_raises(self):
        with pytest.raises(ValidationError, match="nan"):
            validate_numeric_array([1.0, np.nan], "arr", allow_nonpositive=True)

    def test_inf_raises(self):
        with pytest.raises(ValidationError, match="inf"):
            validate_numeric_array([1.0, np.inf], "arr", allow_nonpositive=True)

    def test_nonpositive_raises_when_disallowed(self):
        with pytest.raises(ValidationError, match="positive"):
            validate_numeric_array([1.0, -1.0], "arr", allow_nonpositive=False)

    def test_zero_raises_when_disallowed(self):
        with pytest.raises(ValidationError, match="positive"):
            validate_numeric_array([1.0, 0.0], "arr", allow_nonpositive=False)

    def test_negatives_allowed(self):
        result = validate_numeric_array([-1.0, -2.0], "arr", allow_nonpositive=True)
        np.testing.assert_array_equal(result, np.array([-1.0, -2.0]))

    def test_non_numeric_raises(self):
        with pytest.raises(ValidationError):
            validate_numeric_array(["a", "b"], "arr", allow_nonpositive=True)


# ===========================================================================
# validate_isotropic_hyperparameter
# ===========================================================================


class TestValidateIsotropicHyperparameter:
    def test_scalar_returns_1d_array(self):
        result = validate_isotropic_hyperparameter(2.0, "ls")
        assert result.shape == (1,)
        assert result[0] == np.float64(2.0)

    def test_zero_raises(self):
        with pytest.raises(ValidationError):
            validate_isotropic_hyperparameter(0.0, "ls")

    def test_negative_raises(self):
        with pytest.raises(ValidationError):
            validate_isotropic_hyperparameter(-1.0, "ls")


# ===========================================================================
# validate_anisotropic_hyperparameter
# ===========================================================================


class TestValidateAnisotropicHyperparameter:
    def test_list_returns_1d_array(self):
        result = validate_anisotropic_hyperparameter([1.0, 2.0, 3.0], "ls")
        assert result.ndim == 1
        assert result.shape == (3,)

    def test_nonpositive_raises(self):
        with pytest.raises(ValidationError):
            validate_anisotropic_hyperparameter([1.0, 0.0], "ls")

    def test_2d_input_flattened(self):
        result = validate_anisotropic_hyperparameter([[1.0], [2.0]], "ls")
        assert result.ndim == 1
        assert result.shape == (2,)


# ===========================================================================
# validate_input_arrays
# ===========================================================================


class TestValidateInputArrays:
    def test_compatible_shapes(self):
        x1 = np.ones((5, 3))
        x2 = np.ones((7, 3))
        out1, out2 = validate_input_arrays(x1, "x1", x2, "x2")
        assert out1.shape == (5, 3)
        assert out2.shape == (7, 3)

    def test_1d_inputs_reshaped_to_2d(self):
        x1 = np.ones(5)
        x2 = np.ones(7)
        out1, out2 = validate_input_arrays(x1, "x1", x2, "x2")
        assert out1.ndim == 2
        assert out2.ndim == 2

    def test_feature_mismatch_raises(self):
        x1 = np.ones((5, 3))
        x2 = np.ones((7, 2))
        with pytest.raises(ValidationError, match="features"):
            validate_input_arrays(x1, "x1", x2, "x2")


# ===========================================================================
# validate_anisotropic_hyperparameter_shape
# ===========================================================================


class TestValidateAnisotropicHyperparameterShape:
    def test_matching_shapes_ok(self):
        x = np.ones((10, 3))
        param = np.array([1.0, 2.0, 3.0])
        # should not raise
        validate_anisotropic_hyperparameter_shape(x, param)

    def test_mismatched_raises(self):
        x = np.ones((10, 3))
        param = np.array([1.0, 2.0])  # wrong number of dims
        with pytest.raises(ValidationError):
            validate_anisotropic_hyperparameter_shape(x, param)


# ===========================================================================
# validate_multiple_anisotropic_hyperparameter_size
# ===========================================================================


class TestValidateMultipleAnisotropicHyperparameterSize:
    def test_matching_sizes_ok(self):
        params = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        names = ["ls", "period"]
        # should not raise
        validate_multiple_anisotropic_hyperparameter_size(params, names)

    def test_mismatched_sizes_raise(self):
        params = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])]
        names = ["ls", "period"]
        with pytest.raises(ValidationError, match="Mismatch"):
            validate_multiple_anisotropic_hyperparameter_size(params, names)


# ===========================================================================
# validate_set_params
# ===========================================================================


class TestValidateSetParams:
    def test_isotropic_correct_length_ok(self):
        result = validate_set_params(
            np.array([2.0]), "ls", isotropic=True, expected_length=1
        )
        assert result[0] == np.float64(2.0)

    def test_isotropic_wrong_length_raises(self):
        with pytest.raises(ValidationError, match="Wrong number"):
            validate_set_params(
                np.array([1.0, 2.0]), "ls", isotropic=True, expected_length=1
            )

    def test_anisotropic_wrong_length_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_set_params(
                np.array([1.0, 2.0, 3.0]),
                "ls",
                isotropic=False,
                expected_length=2,
            )
        assert len(w) == 1
        assert "different length" in str(w[0].message).lower()

    def test_nonpositive_raises(self):
        with pytest.raises(ValidationError):
            validate_set_params(
                np.array([-1.0]), "ls", isotropic=True, expected_length=1
            )


# ===========================================================================
# validate_bounds_dict
# ===========================================================================


class TestValidateBoundsDict:
    def test_valid_simple_bounds(self):
        result = validate_bounds_dict(
            {"length_scale": [1e-3, 1e2]},
            expected_params=["length_scale"],
            kernel_name="RBFKernel",
        )
        assert "length_scale" in result
        assert result["length_scale"][0] == (np.float64(1e-3), np.float64(1e2))

    def test_not_dict_raises(self):
        with pytest.raises(ValidationError, match="dictionary"):
            validate_bounds_dict("bad", ["length_scale"], "RBFKernel")

    def test_unknown_param_raises(self):
        with pytest.raises(ValidationError):
            validate_bounds_dict({"foo": [1, 10]}, ["length_scale"], "RBFKernel")

    def test_lower_greater_than_upper_raises(self):
        with pytest.raises(ValidationError, match="[Ll]ower"):
            validate_bounds_dict(
                {"length_scale": [10, 1]}, ["length_scale"], "RBFKernel"
            )

    def test_non_string_key_raises(self):
        with pytest.raises(ValidationError):
            validate_bounds_dict({1: [1, 10]}, ["length_scale"], "RBFKernel")

    def test_case_and_space_normalised(self):
        """'Length Scale' should be accepted and normalised to 'length_scale'."""
        result = validate_bounds_dict(
            {"Length Scale": [1e-3, 1e2]},
            expected_params=["length_scale"],
            kernel_name="RBFKernel",
        )
        assert "length_scale" in result


# ===========================================================================
# validate_input_and_target_data
# ===========================================================================


class TestValidateInputAndTargetData:
    def test_matching_shapes_ok(self):
        x = np.ones((10, 2))
        y = np.ones(10)
        out_x, out_y = validate_input_and_target_data(x, y)
        assert out_x.shape == (10, 2)
        assert out_y.shape == (10,)

    def test_1d_x_reshaped(self):
        x = np.linspace(0, 1, 10)
        y = np.ones(10)
        out_x, _out_y = validate_input_and_target_data(x, y)
        assert out_x.ndim == 2
        assert out_x.shape == (10, 1)

    def test_sample_mismatch_raises(self):
        x = np.ones((10, 2))
        y = np.ones(8)
        with pytest.raises(ValidationError, match="samples"):
            validate_input_and_target_data(x, y)

    def test_nan_in_x_raises(self):
        x = np.array([[1.0], [np.nan]])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValidationError):
            validate_input_and_target_data(x, y)


# ===========================================================================
# validate_variable_names
# ===========================================================================


class TestValidateVariableNames:
    def test_single_string_ok(self):
        result = validate_variable_names("x", 1)
        assert result == ["x"]

    def test_list_ok(self):
        result = validate_variable_names(["x", "y"], 2)
        assert result == ["x", "y"]

    def test_wrong_count_raises(self):
        with pytest.raises(ValidationError, match="Expected"):
            validate_variable_names(["x"], 2)

    def test_non_string_element_raises(self):
        with pytest.raises(ValidationError, match="strings"):
            validate_variable_names(["x", 9], 2)

    def test_not_str_or_list_raises(self):
        with pytest.raises(ValidationError):
            validate_variable_names(123, 1)
