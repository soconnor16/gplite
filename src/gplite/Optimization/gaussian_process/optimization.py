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
from numpy.typing import NDArray
from scipy.linalg import LinAlgError
from scipy.optimize import minimize
from scipy.stats import qmc

from gplite._utils._constants import LOCAL_MAXITER, N_REFINE
from gplite._utils._types import Arrf64, GaussianProcessLossFunction
from gplite.Optimization.gaussian_process.loss_functions import (
    negative_log_marginal_likelihood,
)

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


def _get_objective_wrappers(
    gp: "GaussianProcess", objective_func: str | GaussianProcessLossFunction
) -> tuple[Callable, Callable, bool]:
    """
    Validates objective functions and returns SciPy-compatible wrappers for the
    main optimization loop.

    Args:
        - gp: GaussianProcess
            - The GaussianProcess instance whose hyperparameters are being
              optimized.
        - objective_func: str | GaussianProcessLossFunction
            - The objective function used to calculate the loss value to be
              minimized during optimization.

    Returns:
        tuple[Callable, Callable, bool]: A tuple of two loss functions and a
                                         bool which is used to determine whether
                                         to use gradients or not. The reason for
                                         two loss functions is because when a
                                         custom loss function is not passed, it
                                         is much cheaper to do the initial
                                         screening without gradient calculation.
    """
    if callable(objective_func):
        use_grad = False

        def custom_wrapper(theta: Arrf64) -> float:
            try:
                gp.kernel.set_params(theta[:-1], _validate=False)
                gp._noise = theta[-1]
                gp._fit_without_optimization()
                return float(objective_func(gp))
            except (LinAlgError, ValueError):
                return float(np.inf)

        return custom_wrapper, custom_wrapper, use_grad

    if not isinstance(objective_func, str):
        raise ValueError(
            "Error: 'objective_func' must be a string or Callable."
        )

    obj_lower = objective_func.lower()
    if obj_lower not in LOSS_FUNCTIONS:
        err_msg = (
            f"Error: '{objective_func}' is not an available objective function."
            f" Available functions are: {list(LOSS_FUNCTIONS.keys())}. "
            "Alternatively, you can provide a custom Callable function."
        )
        raise ValueError(err_msg)

    loss_fn = LOSS_FUNCTIONS[obj_lower]
    use_grad = LOSS_FUNCTION_HAS_GRAD[obj_lower]

    def func_wrapper_grad(theta: Arrf64) -> tuple[float, NDArray]:
        try:
            gp.kernel.set_params(theta[:-1], _validate=False)
            gp._noise = theta[-1]
            return loss_fn(gp, return_gradient=True)
        except (LinAlgError, ValueError):
            return float(np.inf), np.zeros_like(theta)

    def func_wrapper_no_grad(theta: Arrf64) -> float:
        try:
            gp.kernel.set_params(theta[:-1], _validate=False)
            gp._noise = theta[-1]
            return float(loss_fn(gp, return_gradient=False))
        except (LinAlgError, ValueError):
            return float(np.inf)

    func_wrapper = func_wrapper_grad if use_grad else func_wrapper_no_grad
    return func_wrapper, func_wrapper_no_grad, use_grad


def _generate_starting_points(
    initial_theta: Arrf64,
    bounds: list[tuple[float, float]],
    n_restarts: int,
) -> list[np.ndarray]:
    """
    Generates starting points for optimization using Latin Hypercube Sampling
    in log-space.

    Args:
        - initial_theta: Arrf64
            - Current hyperparameter values.
        - bounds: list[tuple[float, float]]
            - Bounds for each hyperparameter.
        - n_restarts: int
            - Number of random starting points to generate.

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
    gp: "GaussianProcess",
    objective_func: str | GaussianProcessLossFunction = "lml",
    n_restarts: int = 0,
) -> None:
    """
    Optimizes kernel hyperparameters and noise using a hybrid two-phase
    optimization strategy.

    Phase 1 (Global Screening): Runs quick optimizations from multiple
    starting points to identify promising basins.

    Phase 2 (Local Refinement): Takes the top candidates and performs
    thorough optimization to find the best solution.

    Args:
        - gp: GaussianProcess
            - The Gaussian Process model to optimize.
        - objective_func: str | Callable
            - Loss function to minimize. Options: 'lml'
              (log marginal likelihood), or a custom loss function.
              Defaults to 'lml'.
        - n_restarts: int
            - Number of random restarts for global screening. Defaults to 0
              (only optimize from current params).

    Raises:
        ValueError: If objective_func is not a recognized loss function.
    """
    func_wrapper, func_wrapper_screening, use_grad = _get_objective_wrappers(
        gp, objective_func
    )

    # extract initial parameters and bounds
    initial_kernel_params = gp.kernel.get_params()
    initial_noise = np.array([gp._noise], dtype=np.float64)
    initial_theta = np.concatenate([initial_kernel_params, initial_noise])

    noise_bounds = [(1e-6, 1e1)]
    bounds = gp.kernel._get_expanded_bounds() + noise_bounds

    # generate starting points
    starting_points = _generate_starting_points(
        initial_theta, bounds, n_restarts
    )

    # Phase 1: Global Screening
    screening_results = []
    for start_theta in starting_points:
        try:
            loss = func_wrapper_screening(start_theta)
            if loss != np.inf:
                screening_results.append((loss, start_theta))
        except (LinAlgError, ValueError):
            continue

    if not screening_results:
        gp.kernel.set_params(initial_theta[:-1], _validate=False)
        gp._noise = initial_theta[-1]
        gp._fit_without_optimization()
        return

    screening_results.sort(key=lambda x: x[0])
    n_to_refine = min(N_REFINE, len(screening_results))
    top_candidates = [theta for _, theta in screening_results[:n_to_refine]]

    # Phase 2: Local Refinement
    best_theta = top_candidates[0]
    best_loss = screening_results[0][0]

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

        except (LinAlgError, ValueError):
            continue

    # finalize
    gp.kernel.set_params(best_theta[:-1], _validate=False)
    gp._noise = best_theta[-1]
    gp._fit_without_optimization()
    return
