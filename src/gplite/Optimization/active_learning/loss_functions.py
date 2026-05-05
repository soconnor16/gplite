"""
Loss functions for active learning hyperparameter optimization.

These functions evaluate model performance across the full dataset to guide
final hyperparameter tuning after active learning completes.

RMSE (Root Mean Squared Error):
    RMSE = sqrt(1/n * Σᵢ (yᵢ - ŷᵢ)²)

    Penalizes large errors more heavily due to squaring.

MAE (Mean Absolute Error):
    MAE = 1/n * Σᵢ |yᵢ - ŷᵢ|

    More robust to outliers than RMSE.
"""

from typing import TYPE_CHECKING

import numpy as np
from gplite._utils._types import f64

if TYPE_CHECKING:
    from gplite.ActiveLearning.active_learning import ActiveLearner


def root_mean_squared_error(learner: "ActiveLearner") -> f64:
    """
    Computes RMSE between GP predictions and true values across the full
    dataset.

    Args:
        - learner (ActiveLearner): Active learner with fitted GP model.

    Returns:
        f64: Root mean squared error value.
    """
    pred_target_values = learner.gp.predict(learner.x_full)
    real_target_values = learner.y_full

    return np.sqrt(np.mean((pred_target_values - real_target_values) ** 2))


def mean_absolute_error(learner: "ActiveLearner") -> f64:
    """
    Computes MAE between GP predictions and true values across the full
    dataset.

    Args:
        - learner (ActiveLearner): Active learner with fitted GP model.

    Returns:
        f64: Mean absolute error value.
    """
    pred_target_values = learner.gp.predict(learner.x_full)
    real_target_values = learner.y_full

    return np.float64(np.mean(np.abs(pred_target_values - real_target_values)))
