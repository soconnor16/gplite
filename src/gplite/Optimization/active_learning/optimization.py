"""
Hyperparameter optimization for active learning models.

Unlike GP optimization which uses log marginal likelihood, active learning
optimization directly minimizes prediction error (RMSE or MAE) across the
full dataset. This ensures the final model is tuned for predictive accuracy
on the specific data distribution.

Uses a two-phase hybrid optimization strategy:
    1. Global Screening: Quick L-BFGS-B runs from multiple starting points
       to identify promising basins in the hyperparameter space.
    2. Local Refinement: Thorough optimization of the top candidates.

Initial points are sampled using Latin Hypercube Sampling in log-space.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import scipy
from gplite._utils._constants import GLOBAL_MAXITER, LOCAL_MAXITER, N_REFINE
from gplite.Optimization.active_learning.loss_functions import (
    mean_absolute_error,
    root_mean_squared_error,
)
from scipy.optimize import minimize
from scipy.stats import qmc

if TYPE_CHECKING:
    from gplite.ActiveLearning.active_learning import ActiveLearner

# local constant to define valid loss functions
LOSS_FUNCTIONS: dict[str, Callable] = {
    "rmse": root_mean_squared_error,
    "root_mean_squared_error": root_mean_squared_error,
    "mae": mean_absolute_error,
    "mean_absolute_error": mean_absolute_error,
}


def _generate_starting_points(
    initial_theta: np.ndarray,
    bounds: list[tuple[float, float]],
    n_restarts: int,
) -> list[np.ndarray]:
    """
    Generates starting points for optimization using Latin Hypercube Sampling
    in log-space.

    Args:
        - initial_theta (np.ndarray): Current hyperparameter values.
        - bounds (list[tuple[float, float]]): Bounds for each hyperparameter.
        - n_restarts (int): Number of random starting points to generate.

    Returns:
        list[np.ndarray]: List of starting points including initial_theta.
    """
    starting_points = [initial_theta]

    if n_restarts > 0:
        sampler = qmc.LatinHypercube(d=len(bounds))
        samples = sampler.random(n_restarts)

        for sample in samples:
            theta = []
            for j, (low, high) in enumerate(bounds):
                # sample in log space: 10^uniform(log10(low), log10(high))
                log_low, log_high = np.log10(low), np.log10(high)
                log_sample = log_low + sample[j] * (log_high - log_low)
                theta.append(10**log_sample)
            starting_points.append(np.array(theta))

    return starting_points


def optimize_hyperparameters(
    learner: "ActiveLearner", objective_func: str, n_restarts: int = 0
) -> None:
    """
    Optimizes active learner hyperparameters using a hybrid two-phase
    optimization strategy with prediction error metrics.

    Phase 1 (Global Screening): Runs quick optimizations from multiple
    starting points to identify promising basins.

    Phase 2 (Local Refinement): Takes the top candidates and performs
    thorough optimization to find the best solution.

    Args:
        - learner (ActiveLearner): The active learner to optimize.
        - objective_func (str): Loss function to minimize. Options: 'rmse',
                                'mae', 'none'/'no' (skip optimization).
        - n_restarts (int): Number of random restarts for global screening.
                            Defaults to 0 (only optimize from current params).

    Raises:
        ValueError: If objective_func is not a recognized loss function.
    """
    if objective_func.lower() in ["none", "no"]:
        return

    if objective_func.lower() not in LOSS_FUNCTIONS:
        err_msg = (
            f"Error: '{objective_func}' is not an available objective function."
            f" Available functions are: {list(LOSS_FUNCTIONS.keys())}"
        )
        raise ValueError(err_msg)

    loss_fn = LOSS_FUNCTIONS[objective_func.lower()]

    initial_kernel_params = learner.gp.kernel.get_params()
    initial_noise = np.array([learner.gp._noise])
    initial_theta = np.concatenate([initial_kernel_params, initial_noise])

    noise_bounds = [(1e-6, 1e1)]
    bounds = learner.gp.kernel._get_expanded_bounds() + noise_bounds

    # function wrapper for scipy optimizer
    def func_wrapper(theta):
        learner.gp.kernel.set_params(theta[:-1], validate=False)
        learner.gp._noise = theta[-1]
        learner.gp._fit_without_optimization()
        return loss_fn(learner)

    # generate all starting points
    starting_points = _generate_starting_points(
        initial_theta, bounds, n_restarts
    )

    # phase 1: global screening
    # quick optimization from each starting point to identify promising basins
    screening_results = []

    for start_theta in starting_points:
        try:
            result = minimize(
                func_wrapper,
                start_theta,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": GLOBAL_MAXITER},
            )
            screening_results.append((result.fun, result.x))

        except (scipy.linalg.LinAlgError, ValueError):
            continue

    # if all screening runs failed, fall back to initial parameters
    if not screening_results:
        learner.gp.kernel.set_params(initial_theta[:-1], validate=False)
        learner.gp._noise = initial_theta[-1]
        learner.gp._fit_without_optimization()
        return

    # sort by loss and select top candidates for refinement
    screening_results.sort(key=lambda x: x[0])
    n_to_refine = min(N_REFINE, len(screening_results))
    top_candidates = [theta for _, theta in screening_results[:n_to_refine]]

    # phase 2: local refinement
    # more thoroughly optimize the most promising candidates
    best_theta = top_candidates[0]
    best_loss = np.inf

    for candidate_theta in top_candidates:
        try:
            result = minimize(
                func_wrapper,
                candidate_theta,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": LOCAL_MAXITER},
            )

            if result.fun < best_loss:
                best_loss = result.fun
                best_theta = result.x

        except (scipy.linalg.LinAlgError, ValueError):
            continue

    # set final best hyperparameters
    learner.gp.kernel.set_params(best_theta[:-1], validate=False)
    learner.gp._noise = best_theta[-1]

    learner.gp._fit_without_optimization()

    return
