import numpy as np
from gplite._utils._constants import EPSILON
from gplite._utils._errors import ValidationError
from gplite._utils._types import Arrf64, f64

### Kernel Data Handling ###


def distribute_anisotropic_hyperparameters(
    params: Arrf64, num_anisotropic_kernel_params: int
) -> list[Arrf64]:
    """
    Distributes a flat array of hyperparameters into separate arrays for
    anisotropic kernel parameters (e.g., separate length scales per dimension).

    Args:
        - params (Arrf64): Flat array of hyperparameters to distribute
        - num_anisotropic_kernel_params (int): Number of anisotropic parameter
                                               groups to split into

    Returns:
        list[Arrf64]: List of parameter arrays, one per anisotropic group

    Raises:
        ValidationError: If params cannot be evenly split into the specified
            number of groups
    """

    try:
        split_params = np.split(params, num_anisotropic_kernel_params)

    except ValueError as exc:
        err_msg = (
            "Error: The number of hyperparameters given leads to an uneven "
            "distribution to the kernel's anisotropic hyperparameters "
            "(i.e, they have different lengths)."
        )
        raise ValidationError(err_msg) from exc

    return split_params


def expand_kernel_bounds(
    params: Arrf64,
    bounds: list[tuple[f64, f64]],
    num_anisotropic_kernel_params: int,
) -> list[tuple[f64, f64]]:
    """
    Expands kernel bounds to match the dimensionality of anisotropic
    hyperparameters. Replicates each bound entry to cover all dimensions
    for that parameter type.

    Args:
        - params (Arrf64): Hyperparameter array defining total size
        - bounds (list[tuple[f64, f64]]): Bounds for each anisotropic parameter
                                          type
        - num_anisotropic_kernel_params (int): Number of distinct anisotropic
                                               parameter types

    Returns:
        list[tuple[f64, f64]]: Expanded bounds list matching params.size

    Raises:
        ValueError: If params.size is not evenly divisible by
                    num_anisotropic_kernel_params
    """
    if params.size % num_anisotropic_kernel_params != 0:
        err_msg = (
            "Error: The number of hyperparameters given leads to an uneven "
            "distribution to the kernel's anisotropic hyperparameters "
            "(i.e, they have different lengths)."
        )
        raise ValueError(err_msg)

    num_bounds_per_hyperparameter = int(
        params.size / num_anisotropic_kernel_params
    )

    bounds_list = []

    for i in range(num_anisotropic_kernel_params):
        for _ in range(num_bounds_per_hyperparameter):
            bounds_list.append(bounds[i])

    return bounds_list


### Gaussian Process Data Handling ###


def normalize_input_data(arr: Arrf64) -> tuple[Arrf64, Arrf64, Arrf64]:
    """
    Normalizes input features to zero mean and unit variance (standardization).

    Args:
        - arr (Arrf64): Input array of shape (n, d) to normalize

    Returns:
        tuple[Arrf64, Arrf64, Arrf64]: Normalized array, mean values, and
                                       standard deviations. For constant
                                       features (std=0), uses std=1 to avoid
                                       division by zero.
    """
    arr_mean = np.mean(arr, axis=0)
    arr_std = np.std(arr, axis=0)
    # prevents division by std of 0 if a feature is constant
    arr_std[arr_std == 0] = 1

    arr_normalized = (arr - arr_mean) / arr_std

    return arr_normalized, arr_mean, arr_std


def normalize_target_data(arr: Arrf64) -> tuple[Arrf64, f64, f64]:
    """
    Normalizes target values to zero mean and unit variance (standardization).

    Args:
        - arr (Arrf64): Target array of shape (n,) to normalize

    Returns:
        tuple[Arrf64, f64, f64]: Normalized array, mean value, and standard
                                 deviation. For constant targets (std≈0), uses
                                 std=1 to avoid division by zero.
    """
    arr_mean = np.mean(arr)
    arr_std = np.std(arr)
    # prevent division by zero if target data is constant
    if np.abs(arr_std - 0) <= EPSILON:
        arr_std = 1.0

    arr_normalized = (arr - arr_mean) / arr_std

    return arr_normalized, np.float64(arr_mean), np.float64(arr_std)
