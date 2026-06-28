from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist, pdist, squareform

from gplite._utils._constants import EPSILON
from gplite._utils._types import Arrf64, f64

if TYPE_CHECKING:
    from gplite.GaussianProcess.gaussian_process import GaussianProcess


def compute_square_euclidean_distance(x1: Arrf64, x2: Arrf64) -> Arrf64:
    """Computes the square euclidean distance between x1 and x2.

    This function computes the square euclidean distance (x - xᵢ)² between two
    input arrays for use in kernel covariance functions (RBF and Matern). If the
    x1 and x2 arguments are identical, the symmetry of the resulting distance
    matrix is leveraged by calculating only the upper triangular distance and
    converting it to a square matrix before returning.

    Args:
        x1: Input array 1 of shape (n_samples, n_features).
        x2: Input array 2 of shape (m_samples, n_features).

    Returns:
        The square euclidean distance matrix between input arrays.
    """
    # check if x1 and x2 are the exact same array object in memory
    if x1 is x2:
        # if they are, calculate only the upper triangle
        sq_dist = pdist(x1, metric="sqeuclidean")
        return squareform(sq_dist).astype(np.float64)

    square_dist = cdist(x1, x2, metric="sqeuclidean")
    return square_dist.astype(np.float64)


def compute_lower_cholesky_decomposition(
    K: Arrf64,
    noise: float,
    max_attempts: int,
) -> tuple[Arrf64, float]:
    """Computes the lower triangular Cholesky decomposition of a kernel matrix.

    In typical Gaussian Process Regression, an expensive, direct inversion of
    the kernel's covariance matrix is necessary when fitting the model for
    prediction. Cholesky decomposition provides a more computationally efficient
    and numerically stable alternative to this direct inversion, which is
    especially important when training datasets must remain unstandardized or
    many matrix inversions are necessary (e.g., during hyperparameter
    optimization where the model is fitted many times). While more numerically
    stable than direct matrix inversion, Cholesky decomposition may still fail
    due to floating point imprecision. This function attempts to automatically
    correct and handle these failures by employing a two-phase strategy if
    failure occurs:
        1. Retry with exponentially growing noise (handles most cases)
        2. Eigenvalue-based correction as a final fallback (more expensive)

    Args:
        K: kernel matrix of shape (n, n)
        noise: Initial noise/jitter level to add to diagonal.
        max_attempts: Maximum number of decomposition attempts before falling
            back to eigenvalue correction.

    Returns:
        A tuple with the Lower Cholesky decomposition L and the final noise
        value used to compute it. The original kernel matrix (K), noise value,
        and L are related via:
            K + noise * I = L @ L^T

    Raises:
        ValueError: If decomposition fails after all strategies are exhausted.
    """
    n = K.shape[0]

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
    gp: "GaussianProcess",
    x_full: Arrf64,
    y_full: Arrf64,
) -> f64:
    """Computes the root mean squared error (RMSE) of a model's predictions.

    While direct error-based optimization is mostly avoided in this package,
    rmse-based optimization is provided as a final optimization method for the
    ActiveLearning class. Here, we compute the rmse across the entire data pool
    given to the learner (as opposed to just the data it picked for training) as
    to avoid overfitting risks.

    Args:
        gp: Fitted Gaussian process model.
        x_full: Input features of shape (n, d).
        y_full: True target values of shape (n,).

    Returns:
        The computed RMSE value where:
            RMSE = sqrt(mean((y_pred - y_true)²))
    """
    y_pred = gp.predict(x_full)

    return np.float64(np.sqrt(np.mean((y_pred - y_full) ** 2)))
