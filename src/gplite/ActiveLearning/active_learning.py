"""
Active learning implementation for intelligent data sampling with Gaussian
Process models.

Active learning aims to achieve high model accuracy with minimal labeled data
by strategically selecting the most informative points from an unlabeled pool.

Common acquisition strategies:
    - Uncertainty Sampling: Select points where σ²(x) is highest, exploring
      regions where the model is least confident.
    - Expected Improvement: Select points that maximize the expected improvement
      over the current best observation (separate variants for maximization and
      minimization objectives).
    - Error-based: Select points with highest |y - μ(x)|, focusing on regions
      where the model performs worst.
    - Random: Baseline strategy with uniform random selection.

The learning loop:
    1. Train GP on current labeled set
    2. Compute acquisition scores for unlabeled points
    3. Select and label highest-scoring point(s)
    4. Repeat until stopping criterion (RMSE threshold, budget, etc.)
"""

import csv
import logging
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np

from gplite._utils._computation import compute_rmse_across_dataset
from gplite._utils._errors import ValidationError
from gplite._utils._types import (
    ActiveLearningLossFunction,
    Arrf64,
    Arri64,
    NumericArray,
    NumericValue,
    SelectionFunction,
    f64,
)
from gplite._utils._validation import (
    validate_input_and_target_data,
    validate_numeric_value,
)
from gplite.ActiveLearning.selection_functions import (
    expected_improvement_max,
    expected_improvement_min,
    max_absolute_error,
    max_uncertainty,
    random_selection,
)
from gplite.GaussianProcess.gaussian_process import GaussianProcess
from gplite.Kernels._base import Kernel
from gplite.Optimization.active_learning.optimization import (
    optimize_hyperparameters,
)

logger = logging.getLogger(__name__)

LEARNING_STRATEGIES: dict[str, Callable] = {
    "random": random_selection,
    "random_choice": random_selection,
    "uncertainty": max_uncertainty,
    "max_uncertainty": max_uncertainty,
    "mae": max_absolute_error,
    "max_absolute_error": max_absolute_error,
    "ei_max": expected_improvement_max,
    "expected_improvement_max": expected_improvement_max,
    "ei_min": expected_improvement_min,
    "expected_improvement_min": expected_improvement_min,
}


class ActiveLearner:
    """
    Active learning system that intelligently selects training points to
    minimize required data while maintaining model accuracy.

    Uses a Gaussian Process model to identify the most informative points
    from a pool of unlabeled data based on various selection strategies.

    Attributes:
        - gp: GaussianProcess
            - The underlying Gaussian Process model.
        - x_full: Arrf64
            - Complete pool of input features.
        - y_full: Arrf64
            - Complete pool of target values.
        - x_train: Arrf64
            - Current training input features.
        - y_train: Arrf64
            - Current training target values.
        - remaining_indices: Arri64
            - Indices of points not yet in training set.
    """

    def __init__(
        self,
        kernel: Kernel,
        x_full: NumericArray,
        y_full: NumericArray,
    ) -> None:
        """
        Initializes an active learner with the given kernel and data pool.

        Args:
            - kernel: Kernel
                - Kernel instance for the underlying GP model.
            - x_full: NumericArray
                - Full dataset input features of shape (n, d).
            - y_full: NumericArray
                - Full dataset target values of shape (n,).

        Raises:
            ValidationError: If kernel is invalid or data arrays are
                              incompatible.
        """
        if not isinstance(kernel, Kernel):
            err_msg = "Error: 'kernel' must be a valid Kernel instance"
            raise ValidationError(err_msg)

        self.x_full, self.y_full = validate_input_and_target_data(
            x_full, y_full
        )

        self.kernel = kernel

        self.gp = GaussianProcess(self.kernel)

        # initialize training sets and pool of points that remain
        # available to be picked
        self.x_train: Arrf64 = np.array([])
        self.y_train: Arrf64 = np.array([])
        self.remaining_indices: Arri64 = np.array([])

        self._initialize_training_data()

    def _initialize_training_data(self) -> None:
        """
        Initializes the training set with three points: first, middle, and last
        from the dataset. Sets up the remaining indices pool for active
        selection.

        Warns:
            UserWarning: If dataset has fewer than 3 samples.
        """
        num_samples = self.x_full.shape[0]

        if num_samples < 3:
            warning_msg = (
                "Warning: Active Learning data has < 3 samples. Using full "
                "dataset for training."
            )
            warnings.warn(warning_msg)

            self.x_train = self.x_full
            self.y_train = self.y_full

            return

        # first, middle, and last points of the dataset
        initial_indices = [0, num_samples // 2, num_samples - 1]

        self.x_train = self.x_full[initial_indices]
        self.y_train = self.y_full[initial_indices]

        # remove indices from training pool
        self.remaining_indices = np.setdiff1d(
            np.arange(num_samples), initial_indices
        )

        return

    def select_next_point(
        self, selection_function: Callable, n_points: int = 1
    ) -> Arri64:
        """
        Selects the next point(s) to add to the training set using the given
        selection strategy.

        Args:
            - selection_function: Callable
                - Function that takes learner and n_points and returns indices.
            - n_points: int
                - Number of points to select. Defaults to 1.

        Returns:
            Arri64: Indices of selected points from the full dataset.
        """
        return selection_function(self, n_points)

    def _update_log(self, iteration: int, rmse: f64, log_file: Path) -> None:
        """
        Private method to update the log file of the learning loop.

        Args:
            - iteration: int
                - The current iteration number being logged.
            - rmse: f64
                - The rmse of the model during that iteration.
            - log_file: Path
                - The file path to write the log to.
        """
        with log_file.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([iteration, self.x_train.shape[0], rmse])

        return

    def _acquire_data(self, strategy: str | Callable, num_points: int) -> None:
        """
        Private helper method to append new points to the training set.

        Args:
            - strategy: str
                - The learning strategy we are using for point selection.
            - num_points: int
                - The number of points to select at once.
        """
        if callable(strategy):
            selection_function = strategy
        else:
            selection_function = LEARNING_STRATEGIES[strategy]
        selected_indices = self.select_next_point(
            selection_function=selection_function, n_points=num_points
        )

        self.x_train = np.vstack([self.x_train, self.x_full[selected_indices]])
        self.y_train = np.append(self.y_train, self.y_full[selected_indices])
        self.remaining_indices = np.setdiff1d(
            self.remaining_indices, selected_indices
        )

        return

    def learn(
        self,
        learning_strategy: str | SelectionFunction = "uncertainty",
        max_points: int | None = None,
        rmse_threshold: NumericValue = np.float64(0.5),
        optimize_interval: int | None = 1,
        batch_size: int = 1,
        final_optimization_method: str
        | ActiveLearningLossFunction
        | None = None,
        log_file: str | Path | None = None,
        log_interval: int = 5,
    ) -> None:
        """
        Executes the active learning loop, iteratively selecting and adding
        points until a stopping criterion is met.

        Stopping criteria include: reaching RMSE threshold, exhausting all
        points, or reaching max_points limit.

        Args:
            - learning_strategy: str | SelectionFunction
                - Point selection strategy. Options: 'random', 'uncertainty',
                  'mae', 'ei_max', 'ei_min', or a custom function. Defaults to
                  'uncertainty'.
            - max_points: int | None
                - Maximum training points to use. Defaults to full dataset size.
            - rmse_threshold: NumericValue
                - RMSE target for stopping criterion. Defaults to 0.5.
            - optimize_interval: int | None
                - Iterations between hyperparameter optimization. A None value
                  disables optimization. Defaults to 1.
            - batch_size: int
                - Points to add per iteration. Defaults to 1.
            - final_optimization_method: str | ActiveLearningLossFunction | None
                - Objective for final optimization. Options: 'mae', 'rmse',
                  'None', or a custom loss function. A value of None disables
                  final optimization. Defaults to None.
            - log_file: str | Path | None
                - The file path to write csv logging data to. A value of None
                  disables logging to a file. Defaults to None.
            - log_interval: int
                - Iterations between log progress updates. Defaults to 5.

        Raises:
            ValidationError: If learning_strategy is not recognized or invalid.

        Warns:
            UserWarning: If learning stops early due to errors.
        """
        rmse_threshold = validate_numeric_value(
            rmse_threshold, "Active Learner RMSE Threshold", False
        )

        if max_points:
            max_points = int(
                validate_numeric_value(
                    max_points, "Active Learner Max Points", False
                )
            )
        else:
            max_points = len(self.y_full)

        if optimize_interval:
            optimize_interval = int(
                validate_numeric_value(
                    optimize_interval, "Active Learner Optimize Interval", False
                )
            )
        else:
            optimize_interval = None

        batch_size = int(
            validate_numeric_value(
                batch_size, "Batch Size", allow_nonpositive=False
            )
        )
        log_interval = int(
            validate_numeric_value(
                log_interval, "Log Interval", allow_nonpositive=False
            )
        )

        if isinstance(learning_strategy, str):
            if learning_strategy not in LEARNING_STRATEGIES:
                err_msg = (
                    f"Error: {learning_strategy} is not a valid learning "
                    "strategy. Valid strategies include: "
                    f"{list(LEARNING_STRATEGIES.keys())}"
                )
                raise ValidationError(err_msg)
        elif not callable(learning_strategy):
            err_msg = (
                "Error: 'learning_strategy' must be a string or a callable "
                "function."
            )
            raise ValidationError(err_msg)

        if log_file is not None:
            log_file = Path(log_file).resolve()
            with log_file.open("w", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["iteration", "num_points_used", "rmse"])

        # default stop reason
        stop_reason: str = "Max points reached"

        for iteration in range(max_points):
            should_optimize = (
                optimize_interval is not None
                and iteration % optimize_interval == 0
            )
            should_log = (iteration + 1) % log_interval == 0

            # step 1: fit and evaluate model
            self.gp.fit(self.x_train, self.y_train, optimize=should_optimize)
            current_rmse = compute_rmse_across_dataset(
                self.gp, self.x_full, self.y_full
            )

            if should_log:
                logger.info(f"Iteration {iteration + 1}: RMSE: {current_rmse}")

                if log_file is not None:
                    self._update_log(
                        iteration=iteration + 1,
                        rmse=current_rmse,
                        log_file=log_file,
                    )

            # step 2: Check stopping criteria
            if current_rmse <= rmse_threshold:
                stop_reason = "RMSE threshold Reached"
                break

            if len(self.remaining_indices) == 0:
                stop_reason = "All points used"
                break

            remaining_budget = max_points - len(self.y_train)
            if remaining_budget <= 0:
                stop_reason = "Max points reached"
                break

            # step 3: Label and choose next training points
            try:
                points_to_add = min(batch_size, remaining_budget)
                self._acquire_data(
                    strategy=learning_strategy, num_points=points_to_add
                )
            except ValueError as exc:
                # usually due to running out of points
                warning_msg = f"Warning: Learning stopped early: {exc!s}"
                warnings.warn(warning_msg)
                break

        # exit block
        if final_optimization_method:
            optimize_hyperparameters(
                learner=self, objective_func=final_optimization_method
            )

        final_rmse = compute_rmse_across_dataset(
            gp=self.gp, x_full=self.x_full, y_full=self.y_full
        )

        if log_file is not None:
            self._update_log(
                iteration=iteration + 1, rmse=final_rmse, log_file=log_file
            )

        logger.info(
            f"{stop_reason}:\n"
            f"Final RMSE: {final_rmse:.4f}\n"
            f"Points used: {len(self.y_train)}"
        )

        return
