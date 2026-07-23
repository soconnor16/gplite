"""Unit tests for gplite._utils._data."""

from __future__ import annotations

import numpy as np
import pytest
from gplite._utils._data import (
    distribute_anisotropic_hyperparameters,
    resolve_bounds_shape,
    standardize_input_data,
    standardize_target_data,
)
from gplite._utils._errors import ValidationError

# ===========================================================================
# distribute_anisotropic_hyperparameters
# ===========================================================================


class TestDistributeAnisotropicHyperparameters:
    def test_even_split(self):
        params = np.array([1.0, 2.0, 3.0, 4.0])
        result = distribute_anisotropic_hyperparameters(params, 2)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], [1.0, 2.0])
        np.testing.assert_array_equal(result[1], [3.0, 4.0])

    def test_uneven_split_raises(self):
        params = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValidationError, match="uneven"):
            distribute_anisotropic_hyperparameters(params, 2)

    def test_single_group(self):
        params = np.array([1.0, 2.0, 3.0])
        result = distribute_anisotropic_hyperparameters(params, 1)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], params)

    def test_scalar_per_group(self):
        params = np.array([1.0, 2.0])
        result = distribute_anisotropic_hyperparameters(params, 2)
        assert result[0].shape == (1,)
        assert result[1].shape == (1,)


# ===========================================================================
# resolve_bounds_shape
# ===========================================================================


class TestResolveBoundsShape:
    def test_exact_match_returns_unchanged(self):
        bounds = [(1.0, 10.0), (2.0, 20.0)]
        result = resolve_bounds_shape(bounds, n_dims=2, param_name="ls")
        assert result == bounds

    def test_single_bound_expanded_to_n_dims(self):
        bounds = [(1.0, 10.0)]
        result = resolve_bounds_shape(bounds, n_dims=3, param_name="ls")
        assert result == [(1.0, 10.0), (1.0, 10.0), (1.0, 10.0)]

    def test_mismatch_raises(self):
        bounds = [(1.0, 10.0), (2.0, 20.0)]
        with pytest.raises(ValidationError, match="Shape mismatch"):
            resolve_bounds_shape(bounds, n_dims=3, param_name="ls")

    def test_single_bound_single_dim_unchanged(self):
        bounds = [(1.0, 10.0)]
        result = resolve_bounds_shape(bounds, n_dims=1, param_name="ls")
        assert result == [(1.0, 10.0)]


# ===========================================================================
# standardize_input_data
# ===========================================================================


class TestStandardizeInputData:
    def test_output_shapes(self):
        arr = np.random.default_rng(0).standard_normal((20, 3))
        standardized, mean, std = standardize_input_data(arr)
        assert standardized.shape == arr.shape
        assert mean.shape == (3,)
        assert std.shape == (3,)

    def test_zero_mean_unit_variance(self):
        arr = np.random.default_rng(1).standard_normal((50, 2))
        standardized, _, _ = standardize_input_data(arr)
        np.testing.assert_allclose(standardized.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(standardized.std(axis=0), 1.0, atol=1e-10)

    def test_constant_feature_std_set_to_one(self):
        """A constant feature should not produce division by zero."""
        arr = np.ones((10, 2))
        standardized, _, std = standardize_input_data(arr)
        # std should have been set to 1 to prevent division by zero
        assert np.all(std >= 1.0)
        # standardized values should be zero (mean-centred constant)
        np.testing.assert_allclose(standardized, 0.0, atol=1e-10)

    def test_reversible(self):
        """Applying the returned mean/std should recover original data."""
        arr = np.random.default_rng(2).standard_normal((20, 3)) * 5 + 3
        standardized, mean, std = standardize_input_data(arr)
        recovered = standardized * std + mean
        np.testing.assert_allclose(recovered, arr, atol=1e-10)


# ===========================================================================
# standardize_target_data
# ===========================================================================


class TestStandardizeTargetData:
    def test_output_types(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        standardized, mean, std = standardize_target_data(arr)
        assert standardized.dtype == np.float64
        assert isinstance(mean, np.float64)
        assert isinstance(std, np.float64)

    def test_zero_mean_unit_variance(self):
        arr = np.random.default_rng(3).standard_normal(50) * 10 + 5
        standardized, _, _ = standardize_target_data(arr)
        np.testing.assert_allclose(standardized.mean(), 0.0, atol=1e-10)
        np.testing.assert_allclose(standardized.std(), 1.0, atol=1e-10)

    def test_constant_target_std_set_to_one(self):
        arr = np.full(10, 3.14)
        standardized, _mean, std = standardize_target_data(arr)
        assert std >= 1.0
        np.testing.assert_allclose(standardized, 0.0, atol=1e-10)

    def test_reversible(self):
        arr = np.array([10.0, 20.0, 30.0, 40.0])
        standardized, mean, std = standardize_target_data(arr)
        recovered = standardized * std + mean
        np.testing.assert_allclose(recovered, arr, atol=1e-10)
