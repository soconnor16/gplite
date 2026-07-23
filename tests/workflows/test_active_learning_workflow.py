"""End-to-end workflow tests for ActiveLearner."""

import numpy as np
import pytest
from gplite.ActiveLearning.active_learning import ActiveLearner
from gplite.Kernels.matern import MaternKernel
from gplite.Kernels.rbf import RBFKernel

# ---------------------------------------------------------------------------
# Dataset factories
# ---------------------------------------------------------------------------


def _sine_dataset(n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 2 * np.pi, n).reshape(-1, 1)
    y = np.sin(x).ravel()
    return x, y


def _quadratic_dataset(n: int = 50) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-3, 3, n).reshape(-1, 1)
    y = (x**2).ravel()
    return x, y


def _rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


# ===========================================================================
# Full active learning loop - correctness
# ===========================================================================


class TestActiveLearningWorkflowCorrectness:
    """Verify that the learner actually improves RMSE as points are added."""

    def test_rmse_decreases_over_iterations(self):
        """RMSE should generally decrease as more points are added."""
        x, y = _sine_dataset(60)
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)

        rmse_snapshots = []
        for _ in range(8):
            learner.gp.fit(learner.x_train, learner.y_train)
            current_rmse = _rmse(learner.gp.predict(x), y)
            rmse_snapshots.append(current_rmse)

            if len(learner.remaining_indices) == 0:
                break

            # add one point via uncertainty
            from gplite.ActiveLearning.selection_functions import (
                max_uncertainty,
            )

            selected = learner.select_next_point(max_uncertainty, n_points=1)
            idx = np.asarray(selected)
            learner.x_train = np.vstack([learner.x_train, x[idx]])
            learner.y_train = np.append(learner.y_train, y[idx])
            learner.remaining_indices = np.setdiff1d(learner.remaining_indices, idx)

        # RMSE at the end should be less than at the beginning
        assert rmse_snapshots[-1] < rmse_snapshots[0]

    def test_final_rmse_below_threshold_sine(self):
        """After learning with uncertainty strategy, RMSE should be < 0.15."""
        x, y = _sine_dataset(60)
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        learner.learn(
            learning_strategy="uncertainty",
            max_points=25,
            optimize_interval=None,
            rmse_threshold=0.15,  # generous threshold
        )
        final_rmse = _rmse(learner.gp.predict(x), y)
        assert final_rmse < 0.15

    def test_final_rmse_below_threshold_quadratic(self):
        x, y = _quadratic_dataset(50)
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        learner.learn(
            learning_strategy="uncertainty",
            max_points=20,
            optimize_interval=None,
            rmse_threshold=0.1,
        )
        final_rmse = _rmse(learner.gp.predict(x), y)
        assert final_rmse < 0.1


# ===========================================================================
# Full active learning loop - all strategies
# ===========================================================================


class TestActiveLearningWorkflowAllStrategies:
    """Smoke tests: every strategy should reach a reasonable RMSE."""

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
    def test_strategy_reduces_rmse(self, strategy):
        x, y = _sine_dataset(50)
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)

        # record initial RMSE
        learner.gp.fit(learner.x_train, learner.y_train)
        initial_rmse = _rmse(learner.gp.predict(x), y)

        learner.learn(
            learning_strategy=strategy,
            max_points=15,
            optimize_interval=None,
            rmse_threshold=1e-10,  # force loop to run to max_points
        )
        final_rmse = _rmse(learner.gp.predict(x), y)

        # final RMSE should be better than initial for all strategies
        assert final_rmse < initial_rmse, (
            f"Strategy '{strategy}': final RMSE {final_rmse:.4f} >= "
            f"initial RMSE {initial_rmse:.4f}"
        )


# ===========================================================================
# Full active learning loop - with hyperparameter optimization
# ===========================================================================


class TestActiveLearningWorkflowWithOptimization:
    def test_learn_with_optimize_interval(self):
        """optimize_interval=5 should run without errors."""
        x, y = _sine_dataset(40)
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        learner.learn(
            learning_strategy="uncertainty",
            max_points=12,
            optimize_interval=5,
            rmse_threshold=1e-10,
        )
        assert learner.x_train.shape[0] <= 12
        assert np.all(np.isfinite(learner.gp.predict(x)))

    def test_learn_with_final_optimization(self):
        """final_optimization_method='rmse' should run without errors."""
        x, y = _sine_dataset(40)
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        learner.learn(
            learning_strategy="random",
            max_points=10,
            optimize_interval=None,
            rmse_threshold=1e-10,
            final_optimization_method="rmse",
        )
        assert np.all(np.isfinite(learner.gp.predict(x)))


# ===========================================================================
# Full active learning loop - different kernels
# ===========================================================================


class TestActiveLearningWorkflowKernels:
    @pytest.mark.parametrize(
        "kernel",
        [
            RBFKernel(length_scale=1.0),
            MaternKernel(length_scale=1.0, nu=1.5),
            MaternKernel(length_scale=1.0, nu=2.5),
            RBFKernel(length_scale=1.0) + MaternKernel(length_scale=1.0, nu=2.5),
        ],
    )
    def test_active_learning_with_kernel(self, kernel):
        x, y = _sine_dataset(40)
        learner = ActiveLearner(kernel, x, y)
        learner.learn(
            learning_strategy="uncertainty",
            max_points=10,
            optimize_interval=None,
            rmse_threshold=1e-10,
        )
        y_pred = learner.gp.predict(x)
        assert y_pred.shape == (40,)
        assert np.all(np.isfinite(y_pred))


# ===========================================================================
# Batch active learning
# ===========================================================================


class TestActiveLearningBatchWorkflow:
    def test_batch_learning_runs(self):
        x, y = _sine_dataset(60)
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        learner.learn(
            learning_strategy="uncertainty",
            max_points=20,
            batch_size=3,
            optimize_interval=None,
            rmse_threshold=1e-10,
        )
        # with batch_size=3, we add 3 at a time, shouldn't exceed max_points
        assert learner.x_train.shape[0] <= 20

    def test_batch_rmse_similar_to_sequential(self):
        """Batch learning should achieve comparable quality to sequential."""
        x, y = _sine_dataset(60)

        learner_seq = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        learner_seq.learn(
            learning_strategy="uncertainty",
            max_points=18,
            batch_size=1,
            optimize_interval=None,
            rmse_threshold=1e-10,
        )

        learner_batch = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        learner_batch.learn(
            learning_strategy="uncertainty",
            max_points=18,
            batch_size=3,
            optimize_interval=None,
            rmse_threshold=1e-10,
        )

        rmse_seq = _rmse(learner_seq.gp.predict(x), y)
        rmse_batch = _rmse(learner_batch.gp.predict(x), y)

        # batch should be within 2x of sequential (very generous bound)
        assert rmse_batch < rmse_seq * 2.0


# ===========================================================================
# Active learner better than naive baseline
# ===========================================================================


class TestActiveLearningVsBaseline:
    def test_uncertainty_beats_trivial_baseline(self):
        """A GP trained on just 3 fixed points (no AL) should be worse
        than one that used active learning to pick the best 12 points.
        """
        x, y = _sine_dataset(60)

        # baseline: naive 3-point GP (first/mid/last)
        x_baseline = x[[0, 30, 59]]
        y_baseline = y[[0, 30, 59]]
        from gplite.GaussianProcess.gaussian_process import GaussianProcess

        gp_baseline = GaussianProcess(RBFKernel(length_scale=1.0))
        gp_baseline.fit(x_baseline, y_baseline)
        rmse_baseline = _rmse(gp_baseline.predict(x), y)

        # active learner: starts at same 3 points, adds 9 more
        learner = ActiveLearner(RBFKernel(length_scale=1.0), x, y)
        learner.learn(
            learning_strategy="uncertainty",
            max_points=12,
            optimize_interval=None,
            rmse_threshold=1e-10,
        )
        rmse_al = _rmse(learner.gp.predict(x), y)

        assert rmse_al < rmse_baseline
