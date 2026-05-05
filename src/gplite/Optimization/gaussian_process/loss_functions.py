"""
Loss functions for Gaussian Process hyperparameter optimization.

The primary objective for GP optimization is the log marginal likelihood (LML):

    log p(y|X,θ) = -0.5 * y.T @ K⁻¹ @ y    (data fit term)
                  - 0.5 * log|K|           (complexity penalty)
                  - n/2 * log(2π)          (normalization constant)

The data fit term measures how well the model explains the training data,
while the complexity penalty discourages overfitting by penalizing complex
(high-variance) models. This automatic Occam's razor is a key advantage of
Bayesian methods.

Gradients with respect to hyperparameter θ:

    ∂log p(y|X,θ)/∂θ = 0.5 * tr((α @ α.T - K⁻¹) @ ∂K/∂θ)

where α = K⁻¹ @ y.
"""

from typing import TYPE_CHECKING

import numpy as np
from gplite._utils._computation import compute_lower_cholesky_decomposition
from gplite._utils._types import Arrf64
from scipy import linalg

if TYPE_CHECKING:
    from gplite.GaussianProcess.gaussian_process import GaussianProcess


def negative_log_marginal_likelihood(
    gp: "GaussianProcess", return_gradient: bool = True
) -> float | tuple[float, Arrf64]:
    """
    Computes the negative log marginal likelihood and optionally its gradient.

    Args:
        - gp (GaussianProcess): Fitted Gaussian Process model.
        - return_gradient (bool): Whether to compute and return gradients.
                                  Defaults to True.

    Returns:
        float | tuple[float, Arrf64]: Negative LML value, or tuple of
            (negative LML, gradient array) if return_gradient is True.
            Returns (inf, zeros) if Cholesky decomposition fails.
    """
    X, y = gp.x_train, gp.y_train

    if return_gradient:
        K, grads_raw = gp.kernel._compute_with_gradient(X, X)
    else:
        K = gp.kernel._compute(X, X)
        grads_raw = []

    # attempt to compute lower cholesky decomposition
    try:
        L, _ = compute_lower_cholesky_decomposition(
            K, gp._noise, max_attempts=10
        )

    # return infinity and zeros for hyperparameters if decomposition fails
    except ValueError:
        val = np.inf
        n_params = len(gp.kernel.get_params()) + 1
        return (val, np.zeros(n_params)) if return_gradient else val

    # compute alpha
    alpha_vec = linalg.cho_solve((L, True), y, check_finite=False)

    # log marginal likelihood terms
    data_fit = -0.5 * (y.ravel() @ alpha_vec.ravel())
    complexity = -np.sum(np.log(np.diag(L)))
    constant = -0.5 * X.shape[0] * np.log(2 * np.pi)

    # combine lml terms to get the scalar value
    n_lml = -(data_fit + complexity + constant)

    if not return_gradient:
        return n_lml

    # gradient computation: K_inv needed for trace term (α @ α.T - K⁻¹)
    n = K.shape[0]
    K_inv = linalg.cho_solve(
        (L, True), np.eye(n, dtype=K.dtype), check_finite=False
    )
    inner_term = np.outer(alpha_vec, alpha_vec) - K_inv

    kernel_grads = []

    for g in grads_raw:
        if g.ndim == 2:
            # scalar parameter: 0.5 * Tr(inner @ g)
            grad_val = 0.5 * np.einsum("ji,ij->", inner_term, g)
            kernel_grads.append(np.atleast_1d(grad_val))

        else:
            # vector parameter (N, N, D)
            grad_vals = 0.5 * np.einsum("ji,ijk->k", inner_term, g)
            kernel_grads.append(grad_vals)

    # noise gradient
    grad_noise = 0.5 * np.trace(inner_term)

    all_grads = np.concatenate(kernel_grads + [np.atleast_1d(grad_noise)])

    return n_lml, -all_grads
