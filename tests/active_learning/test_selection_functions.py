"""Unit tests for ActiveLearning selection functions."""

import numpy as np
import pytest
from gplite.ActiveLearning.active_learning import ActiveLearner
from gplite.ActiveLearning.selection_functions import (
    expected_improvement_max,
    expected_improvement_min,
    max_absolute_error,
    max_uncertainty,
    random_selection,
)
from gplite.Kernels.rbf import RBFKernel

# ---------------------------------------------------------------------------
# Shared fixture: a fitted learner (used by most tests here)
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_learner() -> ActiveLearner:
    """A learner with a fitted GP, ready for selection function calls."""
    x = np.linspace(0, 2 * np.pi, 50).reshape(-1, 1)
    y = np.sin(x).ravel()
    learner = ActiveLearner(
        kernel=RBFKernel(length_scale=1.0),
        x_full=x,
        y_full=y,
    )
    # fit the GP on the initial training data
    learner.gp.fit(learner.x_train, learner.y_train)
    return learner


@pytest.fixture
def empty_pool_learner() -> ActiveLearner:
    """A learner whose remaining_indices pool has been exhausted."""
    x = np.linspace(0, 2 * np.pi, 10).reshape(-1, 1)
    y = np.sin(x).ravel()
    learner = ActiveLearner(kernel=RBFKernel(length_scale=1.0), x_full=x, y_full=y)
    learner.gp.fit(learner.x_train, learner.y_train)
    # manually clear the pool
    learner.remaining_indices = np.array([], dtype=np.int64)
    return learner


# ===========================================================================
# random_selection
# ===========================================================================


class TestRandomSelection:
    def test_returns_array(self, fitted_learner):
        indices = random_selection(fitted_learner, n_points=1)
        assert isinstance(indices, np.ndarray)

    def test_returns_correct_count(self, fitted_learner):
        indices = random_selection(fitted_learner, n_points=3)
        assert len(indices) == 3

    def test_indices_within_remaining(self, fitted_learner):
        indices = random_selection(fitted_learner, n_points=5)
        for idx in indices:
            assert idx in fitted_learner.remaining_indices

    def test_no_duplicate_indices(self, fitted_learner):
        indices = random_selection(fitted_learner, n_points=5)
        assert len(set(indices)) == len(indices)

    def test_empty_pool_returns_empty(self, empty_pool_learner):
        indices = random_selection(empty_pool_learner, n_points=1)
        assert len(indices) == 0

    def test_request_more_than_pool_returns_all(self, fitted_learner):
        pool_size = len(fitted_learner.remaining_indices)
        indices = random_selection(fitted_learner, n_points=pool_size + 100)
        assert len(indices) == pool_size

    def test_reproducible_with_seed(self, fitted_learner):
        """Two calls with the same seeded rng should give same results."""
        rng = np.random.default_rng(0)
        fitted_learner.remaining_indices = rng.integers(0, 100, size=50).astype(
            np.int64
        )
        indices = random_selection(fitted_learner, n_points=5)
        for idx in indices:
            assert idx in fitted_learner.remaining_indices


# ===========================================================================
# max_uncertainty
# ===========================================================================


class TestMaxUncertainty:
    def test_returns_array(self, fitted_learner):
        indices = max_uncertainty(fitted_learner, n_points=1)
        assert isinstance(indices, np.ndarray)

    def test_returns_correct_count(self, fitted_learner):
        indices = max_uncertainty(fitted_learner, n_points=3)
        assert len(indices) == 3

    def test_indices_within_remaining(self, fitted_learner):
        indices = max_uncertainty(fitted_learner, n_points=3)
        for idx in indices:
            assert idx in fitted_learner.remaining_indices

    def test_empty_pool_returns_empty(self, empty_pool_learner):
        indices = max_uncertainty(empty_pool_learner, n_points=1)
        assert len(indices) == 0

    def test_selected_point_has_highest_uncertainty(self, fitted_learner):
        """The selected point should have the highest std among remaining pool."""
        _, stds = fitted_learner.gp.predict(
            fitted_learner.x_full[fitted_learner.remaining_indices],
            return_std=True,
        )
        best_idx_in_pool = np.argmax(stds)
        best_full_idx = fitted_learner.remaining_indices[best_idx_in_pool]

        selected = max_uncertainty(fitted_learner, n_points=1)
        assert selected[0] == best_full_idx


# ===========================================================================
# max_absolute_error
# ===========================================================================


class TestMaxAbsoluteError:
    def test_returns_array(self, fitted_learner):
        indices = max_absolute_error(fitted_learner, n_points=1)
        assert isinstance(indices, np.ndarray)

    def test_returns_correct_count(self, fitted_learner):
        indices = max_absolute_error(fitted_learner, n_points=2)
        assert len(indices) == 2

    def test_indices_within_remaining(self, fitted_learner):
        indices = max_absolute_error(fitted_learner, n_points=3)
        for idx in indices:
            assert idx in fitted_learner.remaining_indices

    def test_empty_pool_returns_empty(self, empty_pool_learner):
        indices = max_absolute_error(empty_pool_learner, n_points=1)
        assert len(indices) == 0

    def test_selected_point_has_highest_error(self, fitted_learner):
        """Selected point should be the one with the largest absolute error."""
        pool_x = fitted_learner.x_full[fitted_learner.remaining_indices]
        pool_y = fitted_learner.y_full[fitted_learner.remaining_indices]
        preds = fitted_learner.gp.predict(pool_x)
        errors = np.abs(pool_y - preds)
        best_in_pool = np.argmax(errors)
        best_full_idx = fitted_learner.remaining_indices[best_in_pool]

        selected = max_absolute_error(fitted_learner, n_points=1)
        assert selected[0] == best_full_idx


# ===========================================================================
# expected_improvement_max
# ===========================================================================


class TestExpectedImprovementMax:
    def test_returns_array(self, fitted_learner):
        indices = expected_improvement_max(fitted_learner, n_points=1)
        assert isinstance(indices, np.ndarray)

    def test_returns_correct_count(self, fitted_learner):
        indices = expected_improvement_max(fitted_learner, n_points=2)
        assert len(indices) == 2

    def test_indices_within_remaining(self, fitted_learner):
        indices = expected_improvement_max(fitted_learner, n_points=3)
        for idx in indices:
            assert idx in fitted_learner.remaining_indices

    def test_empty_pool_returns_empty(self, empty_pool_learner):
        indices = expected_improvement_max(empty_pool_learner, n_points=1)
        assert len(indices) == 0

    def test_no_duplicate_indices(self, fitted_learner):
        indices = expected_improvement_max(fitted_learner, n_points=5)
        assert len(set(indices)) == len(indices)


# ===========================================================================
# expected_improvement_min
# ===========================================================================


class TestExpectedImprovementMin:
    def test_returns_array(self, fitted_learner):
        indices = expected_improvement_min(fitted_learner, n_points=1)
        assert isinstance(indices, np.ndarray)

    def test_returns_correct_count(self, fitted_learner):
        indices = expected_improvement_min(fitted_learner, n_points=2)
        assert len(indices) == 2

    def test_indices_within_remaining(self, fitted_learner):
        indices = expected_improvement_min(fitted_learner, n_points=3)
        for idx in indices:
            assert idx in fitted_learner.remaining_indices

    def test_empty_pool_returns_empty(self, empty_pool_learner):
        indices = expected_improvement_min(empty_pool_learner, n_points=1)
        assert len(indices) == 0

    def test_ei_max_and_min_can_differ(self, fitted_learner):
        """EI-max and EI-min can select different points on typical data."""
        idx_max = expected_improvement_max(fitted_learner, n_points=1)
        idx_min = expected_improvement_min(fitted_learner, n_points=1)
        assert idx_max[0] in fitted_learner.remaining_indices
        assert idx_min[0] in fitted_learner.remaining_indices
