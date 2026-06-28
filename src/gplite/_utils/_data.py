import numpy as np

from gplite._utils._constants import EPSILON
from gplite._utils._errors import ValidationError
from gplite._utils._types import Arrf64, f64


### Kernel Data Handling ###
def distribute_anisotropic_hyperparameters(
    params: Arrf64,
    num_anisotropic_kernel_params: int,
) -> list[Arrf64]:
    """Distributes flat hyperparameters into per-dimension anisotropic arrays.

    This function takes a 1D array of hyperparameters and distributes them into
    separate arrays for anisotropic kernel parameters (e.g., partitioning the
    length_scale and period hyperparameter values in an anisotropic Periodic
    kernel). This is especially useful for compatibility with scikit-learn's
    optimizers which interact with the direct '.set_params()' method far more
    often than the user does.

    Args:
        params: Flat array of hyperparameters to distribute.
        num_anisotropic_kernel_params: Number of anisotropic parameter groups to
            split into.

    Returns:
        A list of parameter arrays, one per anisotropic hyperparameter.

    Raises:
        ValidationError: If params cannot be evenly split into the specified
            number of groups.
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


def resolve_bounds_shape(
    bounds_list: list[tuple[f64, f64]],
    n_dims: int,
    param_name: str,
) -> list[tuple[f64, f64]]:
    """Resolves and validates the proper shape of custom kernel bound inputs.

    This function resolves the shape of kernel bounds after initialization, when
    the shape of the data being fit to is known (e.g., during hyperparameter
    optimization) to ensure that improper bound inputs are caught before causing
    optimization to crash. Three branches exist for this function:

        1. The number of tuples in the bounds list exactly matches the number
            of dimensions we expected bounds for. The function then returns the
            list of bounds untouched.
        2. There is only one tuple in the bounds list, but multiple dimensions
            that need bounds. The function returns a list with a list of that
            tuple and as many entries as there are hyperparameter dimensions.
        3. The number of tuples in the bounds list is greater than one, but does
            not match the number we were expecting. The function raises a
            ValidationError with an explanatory error message.

    Args:
        bounds_list: List of one more more tuples of hyperparameter bounds.
        n_dims: The number of dimensions the kernel is currently initialized for
            (e.g., the number of dimensions its hyperparameters have).
        param_name: The name of the hyperparameter being resolved.

    Returns:
        A list of shape-validated bounds, which are composed of tuples
        containing two floats: an upper and lower bound, respectively.

    Raises:
        ValidationError: If more than one tuple of bounds is passed in the list
            and that number does not match the dimensionality of the kernel's
            hyperparameters.

    """
    # the exact correct amount of bounds is provided, do nothing
    if len(bounds_list) == n_dims:
        return bounds_list

    # only one bound is provided and the kernel is anisotropic, simply
    # expand that bound for each feature dimension
    if len(bounds_list) == 1 and n_dims > 1:
        return bounds_list * n_dims

    err_msg = (
        f"Error: Shape mismatch for '{param_name}'. Expected 1 or {n_dims} "
        f"bounds (to match feature dimensions), but got {len(bounds_list)}."
    )
    raise ValidationError(err_msg)


### Gaussian Process Data Handling ###
def standardize_input_data(arr: Arrf64) -> tuple[Arrf64, Arrf64, Arrf64]:
    """Standardizes input features to zero mean and unit variance.

    Standardization of training data is crucial for machine learning and
    particularly for GPR, where the numerical stability of the kernel matrix is
    crucial for hyperparameter optimization and fitting without crashing. This
    function standardizes input data only if "standardize_inputs" is true when
    a GaussianProcess instance is initialized.

    Args:
        arr: Input array of shape (n, d) to standardize.

    Returns:
        A tuple containing the standardized array, mean values, and standard
        deviations of the standardized data. For constant features (std=0),
        uses std=1 to avoid division by zero.
    """
    arr_mean = np.mean(arr, axis=0)
    arr_std = np.std(arr, axis=0)
    # prevents division by std of 0 if a feature is constant
    arr_std[np.abs(arr_std) < EPSILON] = 1

    arr_standardized = (arr - arr_mean) / arr_std

    return arr_standardized, arr_mean, arr_std


def standardize_target_data(arr: Arrf64) -> tuple[Arrf64, f64, f64]:
    """Standardizes target values to zero mean and unit variance.

    Standardization of training data is crucial for machine learning and
    particularly for GPR, where the numerical stability of the kernel matrix is
    crucial for hyperparameter optimization and fitting without crashing. This
    function standardizes target data automatically for any GaussianProcess
    instance, regardless of whether the input data was standardized or not.

    Args:
        arr: Target array of shape (n,) to standardize.

    Returns:
        A tuple containing the standardized array, mean values, and standard
        deviations of the standardized data. For constant features (std=0),
        uses std=1 to avoid division by zero.
    """
    arr_mean = np.mean(arr)
    arr_std = np.std(arr)
    # prevent division by zero if target data is constant
    if np.abs(arr_std) < EPSILON:
        arr_std = 1.0

    arr_standardized = (arr - arr_mean) / arr_std

    return arr_standardized, np.float64(arr_mean), np.float64(arr_std)
