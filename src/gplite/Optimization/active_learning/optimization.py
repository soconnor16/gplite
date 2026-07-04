"""Hyperparameter optimization for active learning models.

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

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import LinAlgError
from scipy.optimize import minimize
from scipy.stats import qmc

from gplite._utils._constants import GLOBAL_MAXITER, LOCAL_MAXITER, N_REFINE
from gplite._utils._errors import ValidationError
from gplite.Optimization.active_learning.loss_functions import (
    mean_absolute_error,
    root_mean_squared_error,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from gplite._utils._types import ActiveLearningLossFunction, Arrf64
    from gplite.ActiveLearning.active_learning import ActiveLearner


# local constant to define valid loss functions
LOSS_FUNCTIONS: dict[str, Callable] = {
    "rmse": root_mean_squared_error,
    "root_mean_squared_error": root_mean_squared_error,
    "mae": mean_absolute_error,
    "mean_absolute_error": mean_absolute_error,
}


def _get_objective_wrapper(
    learner: ActiveLearner,
    objective_func: str | ActiveLearningLossFunction | None,
) -> Callable | None:
    """Validates objective functions and returns wrappers for optimization.

    Args:
        learner: The ActiveLearner instance whose hyperparameters are being
            optimized.
        objective_func: The objective function used to calculate the loss
            value to be minimized during optimization.

    Returns:
        The loss function, or None if final optimization is to be skipped.

    Raises:
        ValidationError: If the objective function is not a string, Callable, or
            "None".
        ValueError: If the objective function is a string, but not a valid
            built-in objective function choice.
    """
    # handle "None" (skip optimization)
    if objective_func is None or (
        isinstance(objective_func, str)
        and objective_func.lower() in ["none", "no"]
    ):
        return None

    # handle Custom Callables
    if callable(objective_func):

        def custom_wrapper(theta: Arrf64) -> float:
            try:
                learner.gp.kernel.set_params(theta[:-1], _validate=False)
                learner.gp._noise = theta[-1]
                learner.gp._fit_without_optimization()
                return float(objective_func(learner))
            except (LinAlgError, ValueError):
                return float(np.inf)

        return custom_wrapper

    # if the objective func is not a callable type, it must be a string by this
    # point
    if not isinstance(objective_func, str):
        err_msg = "Error: 'objective_func' must be a string, Callable, or None."
        raise ValidationError(err_msg)

    obj_lower = objective_func.lower()
    if obj_lower not in LOSS_FUNCTIONS:
        err_msg = (
            f"Error: '{objective_func}' is not an available objective function."
            f" Available functions are: {list(LOSS_FUNCTIONS.keys())}."
        )
        raise ValueError(err_msg)

    loss_fn = LOSS_FUNCTIONS[obj_lower]

    def builtin_wrapper(theta: Arrf64) -> float:
        try:
            learner.gp.kernel.set_params(theta[:-1], _validate=False)
            learner.gp._noise = theta[-1]
            learner.gp._fit_without_optimization()
            return float(loss_fn(learner))
        except (LinAlgError, ValueError):
            return float(np.inf)

    return builtin_wrapper


def _generate_starting_points(
    initial_theta: Arrf64,
    bounds: list[tuple[float, float]],
    n_restarts: int,
) -> list[Arrf64]:
    """Generates starting points for optimization with Latin Hypercube Sampling.

    Sampling is performed in log-space. Because Gaussian Process hyperparameters
    (such as length scales and variances) are strictly positive and can span
    multiple orders of magnitude, uniform sampling in linear space severely
    under-samples small values. Log-space sampling ensures the optimizer
    explores all orders of magnitude uniformly.

    Args:
        initial_theta: Current hyperparameter values.
        bounds: Bounds for each hyperparameter.
        n_restarts: Number of random starting points to generate.

    Returns:
        List of starting points including initial_theta.
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
            starting_points.append(np.asarray(theta, dtype=np.float64))

    return starting_points


def optimize_hyperparameters(
    learner: ActiveLearner,
    objective_func: str | ActiveLearningLossFunction | None,
    n_restarts: int = 0,
) -> None:
    """Optimizes active learner hyperparameters.

    Hyperparameters are optimized using a hybrid two-phase optimization strategy
    described below:

    Phase 1 (Global Screening): Runs quick optimizations from multiple
    starting points to identify promising basins.

    Phase 2 (Local Refinement): Takes the top candidates and performs
    thorough optimization to find the best solution.

    Args:
        learner: The active learner to optimize.
        objective_func: Loss function to minimize. Options: 'rmse', 'mae',
            'none'/'no' (skip optimization).
        n_restarts: Number of random restarts for global screening. Defaults to
            0 (only optimize from current params).
    """
    func_wrapper = _get_objective_wrapper(learner, objective_func)

    if func_wrapper is None:
        return

    initial_kernel_params = learner.gp.kernel.get_params()
    initial_noise = np.array([learner.gp._noise], dtype=np.float64)
    initial_theta = np.concatenate([initial_kernel_params, initial_noise])

    noise_bounds = [(1e-6, 1e1)]
    bounds = learner.gp.kernel._bounds + noise_bounds

    # generate all starting points
    starting_points = _generate_starting_points(
        initial_theta,
        bounds,
        n_restarts,
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

        except (LinAlgError, ValueError):
            continue

    # if all screening runs failed, fall back to initial parameters
    if not screening_results:
        learner.gp.kernel.set_params(initial_theta[:-1], _validate=False)
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
    best_loss = screening_results[0][0]

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

        except (LinAlgError, ValueError):
            continue

    # set final best hyperparameters
    learner.gp.kernel.set_params(best_theta[:-1], _validate=False)
    learner.gp._noise = best_theta[-1]

    learner.gp._fit_without_optimization()

    return
