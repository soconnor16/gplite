"""Unit + integration tests for the ActiveLearner class."""

import warnings
from pathlib import Path

import numpy as np
import pytest
from gplite._utils._errors import ValidationError
from gplite.ActiveLearning.active_learning import ActiveLearner
from gplite.ActiveLearning.selection_functions import random_selection
from gplite.Kernels.rbf import RBFKernel

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _sine_dataset(n: int = 50) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 2 * np.pi, n).reshape(-1, 1)
    y = np.sin(x).ravel()
    return x, y


# ===========================================================================
# Initialisation
# ===========================================================================


class TestActiveLearnerInit:
    def test_valid_init(self):
        x, y = _sine_dataset(30)
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        assert learner.x_full.shape == (30, 1)
        assert learner.y_full.shape == (30,)

    def test_invalid_kernel_raises(self):
        x, y = _sine_dataset()
        with pytest.raises(ValidationError, match="kernel"):
            ActiveLearner("not_a_kernel", x, y)

    def test_mismatched_shapes_raise(self):
        x = np.ones((20, 1))
        y = np.ones(15)  # wrong length
        with pytest.raises(ValidationError):
            ActiveLearner(RBFKernel(length_scale=1.0), x, y)

    def test_nan_in_data_raises(self):
        x = np.array([[1.0], [np.nan]])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValidationError):
            ActiveLearner(RBFKernel(length_scale=1.0), x, y)

    def test_gp_is_initialized(self):
        x, y = _sine_dataset()
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        assert learner.gp is not None

    def test_training_data_initialized(self):
        x, y = _sine_dataset(50)
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        assert learner.x_train.shape[0] == 3
        assert learner.y_train.shape[0] == 3

    def test_remaining_indices_initialized(self):
        x, y = _sine_dataset(50)
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        assert len(learner.remaining_indices) == 47


# ===========================================================================
# _initialize_training_data edge cases
# ===========================================================================


class TestInitTrainingData:
    def test_less_than_3_samples_uses_full_dataset(self):
        x = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        assert len(w) == 1
        assert "< 3 samples" in str(w[0].message)
        np.testing.assert_array_equal(learner.x_train, x)

    def test_exactly_3_samples_initializes_all(self):
        x = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.0, 1.0, 4.0])
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        assert len(learner.remaining_indices) == 0
        assert learner.x_train.shape[0] == 3


# ===========================================================================
# select_next_point
# ===========================================================================


class TestSelectNextPoint:
    def test_select_returns_indices(self):
        x, y = _sine_dataset(50)
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        learner.gp.fit(learner.x_train, learner.y_train)

        indices = learner.select_next_point(random_selection, n_points=1)
        assert len(indices) == 1

    def test_custom_callable_accepted(self):
        x, y = _sine_dataset(50)
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        learner.gp.fit(learner.x_train, learner.y_train)

        # custom strategy: always return the first remaining index
        def first_remaining(learner, n_points):
            return learner.remaining_indices[:n_points]

        indices = learner.select_next_point(first_remaining, n_points=2)
        assert len(indices) == 2


# ===========================================================================
# learn() - validation
# ===========================================================================


class TestActiveLearnerLearnValidation:
    def test_invalid_strategy_raises(self):
        x, y = _sine_dataset(30)
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        with pytest.raises(ValidationError, match="not a valid"):
            learner.learn(learning_strategy="nonexistent_strategy", max_points=5)

    def test_non_callable_non_string_strategy_raises(self):
        x, y = _sine_dataset(30)
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        with pytest.raises(ValidationError):
            learner.learn(learning_strategy=10, max_points=5)

    def test_invalid_rmse_threshold_raises(self):
        x, y = _sine_dataset(30)
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        with pytest.raises(ValidationError):
            learner.learn(rmse_threshold=-0.1, max_points=5)


# ===========================================================================
# learn() - smoke tests for all built-in strategies
# ===========================================================================


class TestActiveLearnerLearnStrategies:
    """Smoke tests: just verify learn() runs to completion for each strategy."""

    @pytest.mark.parametrize(
        "strategy",
        [
            "random",
            "uncertainty",
            "mae",
            "ei_max",
            "ei_min",
        ],
    )
    def test_learn_completes_with_strategy(self, strategy):
        x, y = _sine_dataset(30)
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        learner.learn(
            learning_strategy=strategy,
            max_points=10,
            optimize_interval=None,
            rmse_threshold=1e-6,  # set very low so loop runs to max_points
        )
        assert learner.x_train.shape[0] > 3

    def test_learn_with_custom_callable_strategy(self):
        x, y = _sine_dataset(30)
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)

        def my_strategy(learner, n_points):

            return random_selection(learner, n_points)

        learner.learn(
            learning_strategy=my_strategy,
            max_points=8,
            optimize_interval=None,
            rmse_threshold=1e-6,
        )
        assert learner.x_train.shape[0] > 3


# ===========================================================================
# learn() - stopping criteria
# ===========================================================================


class TestActiveLearnerStoppingCriteria:
    def test_stops_when_rmse_threshold_met(self):
        x, y = _sine_dataset(30)
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        learner.learn(
            learning_strategy="random",
            max_points=50,
            optimize_interval=None,
            rmse_threshold=1e10,  # nearly impossible to exceed
        )
        # should have stopped very early with almost no points added
        assert learner.x_train.shape[0] <= 6  # 3 initial + maybe 1-2

    def test_stops_at_max_points(self):
        x, y = _sine_dataset(50)
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        learner.learn(
            learning_strategy="random",
            max_points=10,
            optimize_interval=None,
            rmse_threshold=1e-10,  # very low so won't trigger early
        )
        assert learner.x_train.shape[0] <= 10

    def test_stops_when_pool_exhausted(self):
        x, y = _sine_dataset(6)  # tiny dataset
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        learner.learn(
            learning_strategy="random",
            max_points=50,
            optimize_interval=None,
            rmse_threshold=1e-10,
        )
        # should stop when remaining_indices is empty
        assert len(learner.remaining_indices) == 0


# ===========================================================================
# learn() - batch_size
# ===========================================================================


class TestActiveLearnerBatchSize:
    def test_batch_size_greater_than_one(self):
        x, y = _sine_dataset(30)
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        learner.learn(
            learning_strategy="random",
            max_points=12,
            batch_size=3,
            optimize_interval=None,
            rmse_threshold=1e-10,
        )
        assert learner.x_train.shape[0] <= 12


# ===========================================================================
# learn() - log file
# ===========================================================================


class TestActiveLearnerLogFile:
    def test_log_file_created(self, tmp_path: Path):
        x, y = _sine_dataset(20)
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        log_path = tmp_path / "log.csv"
        learner.learn(
            learning_strategy="random",
            max_points=8,
            log_file=log_path,
            log_interval=1,
            optimize_interval=None,
            rmse_threshold=1e-10,
        )
        assert log_path.exists()

    def test_log_file_has_header(self, tmp_path: Path):
        x, y = _sine_dataset(20)
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        log_path = tmp_path / "log.csv"
        learner.learn(
            learning_strategy="random",
            max_points=5,
            log_file=log_path,
            log_interval=1,
            optimize_interval=None,
            rmse_threshold=1e-10,
        )
        content = log_path.read_text()
        assert "iteration" in content
        assert "rmse" in content
