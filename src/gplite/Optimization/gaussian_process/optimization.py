"""
Hyperparameter optimization for Gaussian Process models.

Uses a two-phase hybrid optimization strategy:
    1. Global Screening: Quick L-BFGS-B runs from multiple starting points
       to identify promising basins in the hyperparameter space.
    2. Local Refinement: Thorough optimization of the top candidates.

This approach is more efficient than full optimization from all starting
points, as most random restarts land in poor basins and would waste
computation.

Initial points are sampled using Latin Hypercube Sampling in log-space
for better coverage of the typically log-scaled hyperparameter space.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from gplite._utils._constants import LOCAL_MAXITER, N_REFINE
from gplite.Optimization.gaussian_process.loss_functions import (
    negative_log_marginal_likelihood,
)
from scipy.optimize import minimize
from scipy.stats import qmc

if TYPE_CHECKING:
    from gplite.GaussianProcess.gaussian_process import GaussianProcess

# loss functions implemented for gaussian process in this package
LOSS_FUNCTIONS: dict[str, Callable] = {
    "lml": negative_log_marginal_likelihood,
    "log_marginal_likelihood": negative_log_marginal_likelihood,
}

# whether the loss function defined has gradient implementation for optimization
LOSS_FUNCTION_HAS_GRAD: dict[str, bool] = {
    "lml": True,
    "log_marginal_likelihood": True,
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
    gp: "GaussianProcess", objective_func: str = "lml", n_restarts: int = 0
) -> None:
    """
    Optimizes kernel hyperparameters and noise using a hybrid two-phase
    optimization strategy.

    Phase 1 (Global Screening): Runs quick optimizations from multiple
    starting points to identify promising basins.

    Phase 2 (Local Refinement): Takes the top candidates and performs
    thorough optimization to find the best solution.

    Args:
        - gp (GaussianProcess): The Gaussian Process model to optimize.
        - objective_func (str): Loss function to minimize. Options: 'lml'
                                (log marginal likelihood). Defaults to 'lml'.
        - n_restarts (int): Number of random restarts for global screening.
                            Defaults to 0 (only optimize from current params).

    Raises:
        ValueError: If objective_func is not a recognized loss function.
    """
    if objective_func not in LOSS_FUNCTIONS:
        err_msg = (
            f"Error: '{objective_func}' is not an available objective function."
            f" Available functions are: {list(LOSS_FUNCTIONS.keys())}"
        )
        raise ValueError(err_msg)

    loss_fn = LOSS_FUNCTIONS[objective_func]
    use_grad = LOSS_FUNCTION_HAS_GRAD[objective_func]

    initial_kernel_params = gp.kernel.get_params()
    initial_noise = np.array([gp._noise])
    initial_theta = np.concatenate([initial_kernel_params, initial_noise])

    noise_bounds = [(1e-6, 1e1)]
    bounds = gp.kernel._get_expanded_bounds() + noise_bounds

    # function wrappers for scipy optimizer
    def func_wrapper_grad(theta):
        gp.kernel.set_params(theta[:-1], validate=False)
        gp._noise = theta[-1]
        return loss_fn(gp, return_gradient=True)

    def func_wrapper_no_grad(theta):
        gp.kernel.set_params(theta[:-1], validate=False)
        gp._noise = theta[-1]
        return loss_fn(gp)

    func_wrapper = func_wrapper_grad if use_grad else func_wrapper_no_grad

    # generate all starting points
    starting_points = _generate_starting_points(
        initial_theta, bounds, n_restarts
    )

    # phase 1: global screening
    # quick optimization from each starting point to identify promising basins
    screening_results = []

    for start_theta in starting_points:
        try:
            # evaluate the loss function for candidates
            loss = func_wrapper_no_grad(start_theta)
            screening_results.append((loss, start_theta))
        except np.linalg.LinAlgError:
            continue

    # if all screening runs failed, fall back to initial parameters
    if not screening_results:
        gp.kernel.set_params(initial_theta[:-1], validate=False)
        gp._noise = initial_theta[-1]
        gp._fit_without_optimization()
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
                jac=use_grad,
                bounds=bounds,
                options={
                    "maxiter": LOCAL_MAXITER,
                    "ftol": 1e-5,
                    "gtol": 1e-4,
                },
            )

            if result.fun < best_loss:
                best_loss = result.fun
                best_theta = result.x

        except np.linalg.LinAlgError:
            continue

    # set final best hyperparameters
    gp.kernel.set_params(best_theta[:-1])
    gp._noise = best_theta[-1]

    gp._fit_without_optimization()
    return
