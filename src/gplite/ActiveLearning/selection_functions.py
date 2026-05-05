"""
Selection functions for active learning point acquisition strategies.
"""

from typing import TYPE_CHECKING

import numpy as np
from gplite._utils._types import Arri64
from scipy.stats import norm

if TYPE_CHECKING:
    from gplite.ActiveLearning.active_learning import ActiveLearner


def random_selection(learner: "ActiveLearner", n_points: int = 1) -> Arri64:
    """
    Randomly selects points from the remaining pool for training.

    Args:
        - learner (ActiveLearner): The active learner instance containing
                                   the data pool.
        - n_points (int): Number of points to select. Defaults to 1.

    Returns:
        Arri64: Indices of randomly selected points from the full dataset.
                Returns empty array if no points remain.
    """
    num_points_in_pool = len(learner.remaining_indices)

    if num_points_in_pool == 0:
        return np.array([], dtype=np.int64)

    num_selection_points = min(n_points, num_points_in_pool)
    rng = np.random.default_rng()
    selected_indices = rng.choice(
        num_points_in_pool, num_selection_points, replace=False
    )

    return learner.remaining_indices[selected_indices]


def max_uncertainty(learner: "ActiveLearner", n_points: int = 1) -> Arri64:
    """
    Selects points with highest predictive uncertainty (variance).

    This strategy prioritizes points where the model is most uncertain,
    helping to explore regions with limited training data coverage.

    Args:
        - learner (ActiveLearner): The active learner instance with fitted GP.
        - n_points (int): Number of points to select. Defaults to 1.

    Returns:
        Arri64: Indices of points with highest uncertainty, sorted by
                decreasing uncertainty. Returns empty array if no points
                remain.
    """
    if len(learner.remaining_indices) == 0:
        return np.array([], dtype=np.int64)

    _, var = learner.gp.predict(
        learner.x_full[learner.remaining_indices], return_std=True
    )

    flat_var = var.flatten()
    sorted_indices = np.argsort(flat_var)
    top_indices = sorted_indices[-n_points:][::-1]

    return learner.remaining_indices[top_indices]


def expected_improvement_max(
    learner: "ActiveLearner", n_points: int = 1
) -> Arri64:
    """
    Selects points with highest Expected Improvement for maximization.

    Expected Improvement measures the expected amount by which a candidate
    point will exceed the current best observed value. Balances exploitation
    (high mean) with exploration (high uncertainty).

    expected_improvement(x) = (μ(x) - f_best) * Φ(Z) + σ(x) * φ(Z)
    where Z = (μ(x) - f_best) / σ(x), f_best = max(y_train)

    Args:
        - learner (ActiveLearner): The active learner instance with fitted GP.
        - n_points (int): Number of points to select. Defaults to 1.

    Returns:
        Arri64: Indices of points with highest expected improvement, sorted by
                decreasing expected_improvement. Returns empty array if no
                points remain.
    """
    if len(learner.remaining_indices) == 0:
        return np.array([], dtype=np.int64)

    candidate_x = learner.x_full[learner.remaining_indices]
    mu, sigma = learner.gp.predict(candidate_x, return_std=True)

    target_max = np.max(learner.y_train)

    expected_improvement = np.zeros_like(mu)
    mask = sigma > 0
    z = (mu[mask] - target_max) / sigma[mask]
    expected_improvement[mask] = (mu[mask] - target_max) * norm.cdf(z) + sigma[
        mask
    ] * norm.pdf(z)

    sorted_indices = np.argsort(expected_improvement)
    top_indices = sorted_indices[-n_points:][::-1]

    return learner.remaining_indices[top_indices]


def expected_improvement_min(
    learner: "ActiveLearner", n_points: int = 1
) -> Arri64:
    """
    Selects points with highest Expected Improvement for minimization.

    Expected Improvement measures the expected amount by which a candidate
    point will fall below the current best (lowest) observed value. Balances
    exploitation (low mean) with exploration (high uncertainty).

    expected_improvement(x) = (f_best - μ(x)) * Φ(Z) + σ(x) * φ(Z)
    where Z = (f_best - μ(x)) / σ(x), f_best = min(y_train)

    Args:
        - learner (ActiveLearner): The active learner instance with fitted GP.
        - n_points (int): Number of points to select. Defaults to 1.

    Returns:
        Arri64: Indices of points with highest expected improvement, sorted by
                decreasing expected_improvement. Returns empty array if no
                points remain.
    """
    if len(learner.remaining_indices) == 0:
        return np.array([], dtype=np.int64)

    candidate_x = learner.x_full[learner.remaining_indices]
    mu, sigma = learner.gp.predict(candidate_x, return_std=True)

    target_min = np.min(learner.y_train)

    expected_improvement = np.zeros_like(mu)
    mask = sigma > 0
    z = (target_min - mu[mask]) / sigma[mask]
    expected_improvement[mask] = (target_min - mu[mask]) * norm.cdf(z) + sigma[
        mask
    ] * norm.pdf(z)

    sorted_indices = np.argsort(expected_improvement)
    top_indices = sorted_indices[-n_points:][::-1]

    return learner.remaining_indices[top_indices]


def max_absolute_error(learner: "ActiveLearner", n_points: int = 1) -> Arri64:
    """
    Selects points with highest absolute prediction error.

    This strategy prioritizes points where the current model makes the
    largest errors, focusing learning on the most challenging regions.

    Args:
        - learner (ActiveLearner): The active learner instance with fitted GP.
        - n_points (int): Number of points to select. Defaults to 1.

    Returns:
        Arri64: Indices of points with highest absolute error, sorted by
                decreasing error. Returns empty array if no points remain.
    """
    if len(learner.remaining_indices) == 0:
        return np.array([], dtype=np.int64)

    preds = learner.gp.predict(learner.x_full[learner.remaining_indices])
    target_data = learner.y_full[learner.remaining_indices]

    # absolute error
    absolute_error = np.abs(target_data - preds)
    sorted_indices = np.argsort(absolute_error)
    top_indices = sorted_indices[-n_points:][::-1]

    return learner.remaining_indices[top_indices]
