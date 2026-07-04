"""Loss functions for active learning hyperparameter optimization.

These functions evaluate model performance across the full dataset to guide
final hyperparameter tuning after active learning completes.

Built-in loss functions include:

    RMSE (Root Mean Squared Error):
        RMSE = sqrt(1/n * Σᵢ (yᵢ - ŷᵢ)²)

        Penalizes large errors more heavily due to squaring.

    MAE (Mean Absolute Error):
        MAE = 1/n *  |yᵢ - ŷᵢ|

        More robust to outliers than RMSE.

This package also supports custom loss functions whose expected function
signatures can be found in the ActiveLearning and GaussianProcess module-level
README files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gplite._utils._types import f64
    from gplite.ActiveLearning.active_learning import ActiveLearner


def root_mean_squared_error(learner: ActiveLearner) -> f64:
    """Computes RMSE between GP predictions and true values.

    To avoid overfitting to training data, the RMSE value here is computed
    across the full dataset rather than the training pool.

    Args:
        learner: Active learner with fitted GP model.

    Returns:
        Root mean squared error value.
    """
    pred_target_values = learner.gp.predict(learner.x_full)
    real_target_values = learner.y_full

    return np.sqrt(np.mean((pred_target_values - real_target_values) ** 2))


def mean_absolute_error(learner: ActiveLearner) -> f64:
    """Computes MAE between GP predictions and true values.

    To avoid overfitting to training data, the MAE value here is computed
    across the full dataset rather than the training pool.

    Args:
        learner: Active learner with fitted GP model.

    Returns:
        Mean absolute error value.
    """
    pred_target_values = learner.gp.predict(learner.x_full)
    real_target_values = learner.y_full

    return np.float64(np.mean(np.abs(pred_target_values - real_target_values)))
