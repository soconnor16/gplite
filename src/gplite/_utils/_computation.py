from typing import TYPE_CHECKING

import numpy as np
from gplite._utils._constants import EPSILON
from gplite._utils._types import Arrf64, f64
from scipy import linalg
from scipy.spatial.distance import cdist, pdist, squareform

if TYPE_CHECKING:
    from gplite.GaussianProcess.gaussian_process import GaussianProcess


def compute_square_euclidean_distance(x1: Arrf64, x2: Arrf64) -> Arrf64:
    """
    Helper function which computes the square euclidean distance (x - x_i)^2
    between two input arrays.

    Args:
        - x1 (Arrf64): Input array 1 of shape
        - x2 (Arrf64): Input array 2 of shape

    Returns:
        Arrf64: The square euclidean distance matrix between input arrays.
    """
    # check if x1 and x2 are the exact same array object in memory
    if x1 is x2:
        # if they are, calculate only the upper triangle
        sq_dist = pdist(x1, metric="sqeuclidean")
        return squareform(sq_dist).astype(np.float64)

    square_dist = cdist(x1, x2, metric="sqeuclidean")
    return square_dist.astype(np.float64)


def compute_lower_cholesky_decomposition(
    K: Arrf64, noise: float, max_attempts: int
) -> tuple[Arrf64, float]:
    """
    Computes the lower triangular Cholesky decomposition of a kernel matrix
    with adaptive noise regularization for numerical stability.

    Uses a two-phase strategy:
        1. Retry with exponentially growing noise (handles most cases)
        2. Eigenvalue-based correction as a final fallback

    Args:
        - K (Arrf64): Kernel matrix of shape (n, n)
        - noise (float): Initial noise/jitter level to add to diagonal
        - max_attempts (int): Maximum number of decomposition attempts
                              before falling back to eigenvalue correction

    Returns:
        tuple[Arrf64, float]: Lower triangular Cholesky factor L and the
            final noise level used, where K + noise * I = L @ L.T

    Raises:
        ValueError: If decomposition fails after all strategies are exhausted.
    """
    n = K.shape[0]

    # copy the matrix exactly once before the loop
    K_reg = K.copy()
    current_noise = 0.0

    # phase 1: retry with exponentially growing noise
    for attempt in range(max_attempts):
        # calculate how much additional noise is needed for this attempt
        noise_to_add = noise - current_noise

        # add the noise in-place to the diagonal
        K_reg.flat[:: n + 1] += noise_to_add
        current_noise = noise

        try:
            return linalg.cholesky(K_reg, lower=True, check_finite=False), noise
        except linalg.LinAlgError:
            pass

        if attempt == 0:
            k_scale = float(np.mean(np.diag(K)))
            noise = max(noise, k_scale * EPSILON)
        noise *= 10

    # phase 2: eigenvalue-based correction
    # note: K_reg already has `current_noise` added to its diagonal.
    # we just compute the eigenvalues of the current regularized matrix.
    eigenvalues = linalg.eigvalsh(K_reg, check_finite=False)
    min_eig = float(np.min(eigenvalues))

    if min_eig >= 0.0:
        err_msg = (
            "Error: Cholesky decomposition of the kernel matrix "
            "failed despite positive eigenvalues."
        )
        raise ValueError(err_msg)

    # add the final jitter to the diagonal in-place
    jitter = abs(min_eig) + EPSILON
    K_reg.flat[:: n + 1] += jitter
    noise += jitter

    try:
        return linalg.cholesky(K_reg, lower=True, check_finite=False), noise
    except linalg.LinAlgError as exc:
        err_msg = (
            "Error: Numerical instability during kernel matrix "
            f"decomposition: {exc!s}"
        )
        raise ValueError(err_msg) from exc


def compute_rmse_across_dataset(
    gp: "GaussianProcess", x_full: Arrf64, y_full: Arrf64
) -> f64:
    """
    Computes the root mean squared error (RMSE) of Gaussian process predictions
    across a dataset.

    Args:
        - gp (GaussianProcess): Fitted Gaussian process model
        - x_full (Arrf64): Input features of shape (n, d)
        - y_full (Arrf64): True target values of shape (n,)

    Returns:
        f64: RMSE = sqrt(mean((y_pred - y_true)^2))
    """
    y_pred = gp.predict(x_full)

    return np.float64(np.sqrt(np.mean((y_pred - y_full) ** 2)))
