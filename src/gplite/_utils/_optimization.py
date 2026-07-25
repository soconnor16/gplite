"""Shared utility functions for hyperparameter optimization routines.

These helpers are used by both the Gaussian Process and Active Learning
optimization sub-packages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import qmc

if TYPE_CHECKING:
    from gplite._utils._types import Arrf64


def generate_starting_points(
    initial_log_theta: Arrf64,
    log_bounds: list[tuple[float, float]],
    n_restarts: int,
) -> list[Arrf64]:
    """Generates starting points for optimization with Latin Hypercube Sampling.

    Sampling is performed in log-space. Because Gaussian Process hyperparameters
    (such as length scales and variances) are strictly positive and can span
    multiple orders of magnitude, uniform sampling in linear space severely
    under-samples small values. Log-space sampling ensures the optimizer
    explores all orders of magnitude uniformly.

    Args:
        initial_log_theta: Current hyperparameter values in natural log-space.
        log_bounds: Natural log-space bounds for each hyperparameter.
        n_restarts: Number of random starting points to generate.

    Returns:
        List of starting points including initial_theta.
    """
    starting_points = [initial_log_theta]

    if n_restarts > 0:
        sampler = qmc.LatinHypercube(d=len(log_bounds))
        samples = sampler.random(n_restarts)

        for sample in samples:
            log_theta = []
            for j, (log_low, log_high) in enumerate(log_bounds):
                log_sample = log_low + sample[j] * (log_high - log_low)
                log_theta.append(log_sample)
            starting_points.append(np.asarray(log_theta, dtype=np.float64))

    return starting_points
