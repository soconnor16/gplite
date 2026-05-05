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
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
from gplite._utils._computation import compute_rmse_across_dataset
from gplite._utils._errors import ValidationError
from gplite._utils._types import Arrf64, Arri64, f64
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
        - gp (GaussianProcess): The underlying Gaussian Process model.
        - x_full (Arrf64): Complete pool of input features.
        - y_full (Arrf64): Complete pool of target values.
        - x_train (Arrf64): Current training input features.
        - y_train (Arrf64): Current training target values.
        - remaining_indices (Arri64): Indices of points not yet in training set.
    """

    def __init__(
        self,
        kernel: Kernel,
        x_full: Arrf64,
        y_full: Arrf64,
        max_points: int | None = None,
        rmse_threshold: f64 = np.float64(0.5),
        optimize_interval: int | None = 1,
    ) -> None:
        """
        Initializes an active learner with the given kernel and data pool.

        Args:
            - kernel (Kernel): Kernel instance for the underlying GP model.
            - x_full (Arrf64): Full dataset input features of shape (n, d).
            - y_full (Arrf64): Full dataset target values of shape (n,).
            - max_points (int | None): Maximum training points to use. Defaults
                                       to full dataset size.
            - rmse_threshold (f64): RMSE target for stopping criterion. Defaults
                                    to 0.5.
            - optimize_interval (int | None): Iterations between hyperparameter
                                              optimization. None disables.

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

        self.rmse_threshold = validate_numeric_value(
            rmse_threshold, "Active Learner RMSE Threshold", False
        )

        if max_points:
            self.max_points = int(
                validate_numeric_value(
                    max_points, "Active Learner Max Points", False
                )
            )

        else:
            self.max_points = len(self.y_full)

        if optimize_interval:
            self.optimize_interval = int(
                validate_numeric_value(
                    optimize_interval, "Active Learner Optimize Interval", False
                )
            )
        else:
            self.optimize_interval = None

        # initialize training sets and pool of points that remain
        # available to be picked
        self.x_train = np.array([])
        self.y_train = np.array([])
        self.remaining_indices = np.array([])

        self._initialize_training_data()

    def _initialize_training_data(self) -> None:
        """
        Initializes the training set with three strategically selected points:
        first, middle, and last from the dataset. Sets up the remaining
        indices pool for active selection.

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

    def select_next_point(
        self, selection_function: Callable, n_points: int = 1
    ) -> Arri64:
        """
        Selects the next point(s) to add to the training set using the given
        selection strategy.

        Args:
            - selection_function (Callable): Function that takes learner and
                                             n_points and returns indices.
            - n_points (int): Number of points to select. Defaults to 1.

        Returns:
            Arri64: Indices of selected points from the full dataset.
        """
        return selection_function(self, n_points)

    def _update_log(self, rmse, log_file: Path) -> None:
        """
        Private method to update the log file of the learning loop.
        """
        with log_file.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.x_train.shape[0], rmse])

    def learn(
        self,
        learning_strategy: str,
        batch_size: int = 1,
        final_optimization_method: str = "rmse",
        update: bool = False,
        log: bool = False,
        update_interval: int = 10,
        log_update_interval: int = 5,
    ) -> None:
        """
        Executes the active learning loop, iteratively selecting and adding
        points until a stopping criterion is met.

        Stopping criteria include: reaching RMSE threshold, exhausting all
        points, or reaching max_points limit.

        Args:
            - learning_strategy (str): Point selection strategy. Options:
                                       'random', 'uncertainty', 'mae'.
            - batch_size (int): Points to add per iteration. Defaults to 1.
            - final_optimization_method (str): Objective for final optimization.
                                               Defaults to 'rmse'.
            - update (bool): Whether to print progress updates. Defaults to
                             False.
            - log (bool): Whether to log status updates. Works like the
                          "update" parameter, but updates are put into a log
                          file rather than stdout. Defaults to False.
            - update_interval (int): Iterations between progress updates.
                                     Defaults to 10.
            - log_update_interval (int): Iterations between log progress updates.
                                         Defaults to 5.

        Raises:
            ValidationError: If learning_strategy is not recognized.

        Warns:
            UserWarning: If learning stops early due to errors.
        """
        batch_size = int(
            validate_numeric_value(
                batch_size, "Number of Learning Points", allow_nonpositive=False
            )
        )
        update_interval = int(
            validate_numeric_value(
                update_interval, "Update Interval", allow_nonpositive=False
            )
        )

        log_update_interval = int(
            validate_numeric_value(
                log_update_interval,
                "Log Update Interval",
                allow_nonpositive=False,
            )
        )

        if learning_strategy not in LEARNING_STRATEGIES:
            err_msg = (
                f"Error: {learning_strategy} is not a valid learning strategy. "
                f"Valid strategies include: {list(LEARNING_STRATEGIES.keys())}"
            )
            raise ValidationError(err_msg)

        log_file = None
        if log:
            log_file = Path(
                f"./active_learning_{learning_strategy}.csv"
            ).resolve()

            with log_file.open("w", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["num_points_used", "rmse"])

        for iteration in range(self.max_points):
            should_optimize = (
                self.optimize_interval is not None
                and iteration % self.optimize_interval == 0
            )
            should_update = update and (iteration + 1) % update_interval == 0
            should_log = log and (iteration + 1) % log_update_interval == 0

            # step 1: fit model to current training data, optimize with lml
            self.gp.fit(self.x_train, self.y_train, optimize=should_optimize)

            # step 2: compute rmse and check if the threshold has been reached
            current_rmse = compute_rmse_across_dataset(
                self.gp, self.x_full, self.y_full
            )

            if should_update:
                print(f"Iteration {iteration + 1}: RMSE: {current_rmse}")

            if should_log and log_file is not None:
                self._update_log(current_rmse, log_file)

            if current_rmse <= self.rmse_threshold:
                optimize_hyperparameters(self, final_optimization_method)
                final_rmse = compute_rmse_across_dataset(
                    self.gp, self.x_full, self.y_full
                )

                if update:
                    # make sure the update printing and final printing have a
                    # new line between them
                    print()

                if log and log_file is not None:
                    self._update_log(final_rmse, log_file)

                print(
                    "\033[4mRMSE threshold reached\033[0m:",
                    f"\nFinal RMSE: {final_rmse:.4f}",
                    f"\nPoints used: {len(self.y_train)}",
                )

                break

            if len(self.remaining_indices) == 0:
                optimize_hyperparameters(self, final_optimization_method)
                final_rmse = compute_rmse_across_dataset(
                    self.gp, self.x_full, self.y_full
                )

                if update:
                    # make sure the update printing and final printing have a
                    # new line between them
                    print()

                if log and log_file is not None:
                    self._update_log(final_rmse, log_file)

                print(
                    "\033[4mAll points used\033[0m:",
                    f"\nFinal RMSE: {final_rmse:.4f}",
                    f"\nPoints used: {len(self.y_train)}",
                )

                break

            remaining_budget = self.max_points - len(self.y_train)
            if remaining_budget <= 0:
                optimize_hyperparameters(self, final_optimization_method)
                final_rmse = compute_rmse_across_dataset(
                    self.gp, self.x_full, self.y_full
                )

                if update:
                    # make sure the update printing and final printing have a
                    # new line between them
                    print()

                if log and log_file is not None:
                    self._update_log(final_rmse, log_file)

                print(
                    "\033[4mMax points reached\033[0m:",
                    f"\nFinal RMSE: {final_rmse:.4f}",
                    f"\nPoints used: {len(self.y_train)}",
                )

                break

            try:
                points_to_add = min(batch_size, remaining_budget)
                selection_function = LEARNING_STRATEGIES[learning_strategy]
                selected_indices = self.select_next_point(
                    selection_function, points_to_add
                )
                self.x_train = np.vstack(
                    [self.x_train, self.x_full[selected_indices]]
                )
                self.y_train = np.append(
                    self.y_train, self.y_full[selected_indices]
                )

                self.remaining_indices = np.setdiff1d(
                    self.remaining_indices, selected_indices
                )

            except ValueError as exc:
                # usually due to running out of points
                warning_msg = f"Warning: Learning stopped early: {exc!s}"
                warnings.warn(warning_msg)

                break

        return
