import warnings
from typing import Any

import numpy as np

from gplite._utils._errors import ValidationError
from gplite._utils._types import Arrf64, NumericArray, NumericValue, f64


### General Validation ###
def validate_numeric_value(
    value: Any,
    name: str,
    allow_nonpositive: bool,
) -> f64:
    """Validates individual values that are expected to be numeric before use.

    Args:
        value: Value to be validated before use expected as a NumPy float64 type
            or an object that can be converted to a NumPy float64 type.
        name: "name" of the value for more descriptive error messages.
        allow_nonpositive: Whether the value should be allowed to be negative.


    Returns:
        The validated value as a 64 bit NumPy float.

    Raises:
        ValidationError: If the value cannot be converted to a NumPy float64
            type, if the value is "nan" or "inf", or if the value is nonpositive
            while allow_nonpositive is false.
    """
    try:
        value = np.float64(float(value))
    except (OverflowError, TypeError, ValueError) as exc:
        err_msg = f"Error processing '{name}': {exc!s}"
        raise ValidationError(err_msg) from exc

    if np.isnan(value):
        err_msg = f"Error: '{name}' cannot be 'nan'."
        raise ValidationError(err_msg)
    if np.isinf(value):
        err_msg = f"Error: '{name}' cannot be 'inf'."
        raise ValidationError(err_msg)
    if not allow_nonpositive and value <= 0:
        err_msg = f"Error: '{name}' must be a positive, non-zero value."
        raise ValidationError(err_msg)

    return value


def validate_numeric_array(
    array: Any,
    name: str,
    allow_nonpositive: bool,
) -> Arrf64:
    """Function to validate numeric arrays and array-like types.

    Args:
        array: The array to validate. Expected as an array of 64 bit floats or
            an object that can be converted to an array of 64 bit floats.
        name: The "name" of the object being validated.
        allow_nonpositive: Whether nonpositive values should be allowed in the
            array.

    Returns:
        The validated array as a NumPy array of 64 bit float types.

    Raises:
        ValidationError: If the array is empty, contains any 'nan' or 'inf'
            values, or any values less than or equal to 0 when allow_nonpositive
            is false.
    """
    try:
        array = np.asarray(array, dtype=np.float64)

    except (OverflowError, TypeError, ValueError) as exc:
        err_msg = f"Error processing '{name}': {exc!s}"
        raise ValidationError(err_msg) from exc

    if not array.size:
        err_msg = f"Error: '{name}' cannot be an empty array."
        raise ValidationError(err_msg)
    if np.any(np.isinf(array)):
        err_msg = f"Error: '{name}' cannot contain any 'inf' values."
        raise ValidationError(err_msg)
    if np.any(np.isnan(array)):
        err_msg = f"Error: '{name}' cannot contain any 'nan' values."
        raise ValidationError(err_msg)
    if not allow_nonpositive and np.any(array <= 0):
        err_msg = (
            f"Error: '{name}' must contain only positive, non-zero values."
        )
        raise ValidationError(err_msg)

    return array


### Kernel Validation ###


def validate_isotropic_hyperparameter(param: Any, name: str) -> Arrf64:
    """Validates isotropic kernel hyperparameters during initialization.

    This function validates that the hyperparameter is a valid numeric value,
    that is greater than zero and returns it as a flat, single-valued array for
    consistency with anisotropic hyperparameters (which are also validated to be
    arrays).

    Args:
        param: The hyperparameter value to validate.
        name: The "name" of the hyperparameter for better error messages.

    Returns:
        The validated hyperparameter as a 1D array.
    """
    param = validate_numeric_value(param, name, allow_nonpositive=False)

    return np.asarray([param], dtype=np.float64)


def validate_anisotropic_hyperparameter(param: Any, name: str) -> Arrf64:
    """Validates anisotropic kernel hyperparameters during initialization.

    This function validates that the hyperparameters are valid numeric arrays
    with all positive values and returns them as flat arrays.

    Args:
        param: The hyperparameter values to validate.
        name: The "name" of the hyperparameter for better error messages.

    Returns:
        The validated hyperparameter as a 1D array.
    """
    return validate_numeric_array(
        param,
        name,
        allow_nonpositive=False,
    ).flatten()


def validate_input_arrays(
    arr1: NumericArray,
    name1: str,
    arr2: NumericArray,
    name2: str,
) -> tuple[Arrf64, Arrf64]:
    """Validates input arrays passed as input to a kernel's 'compute' method.

    This function validates that the arrays are valid numeric arrays and that
    they have compatible shape (same number of features).

    Args:
        arr1: First array to validate.
        name1: "Name" of first array for better error messages.
        arr2: Second array to validate.
        name2: "Name" of second array for better error messages.

    Returns:
        A tuple of both validated input arrays as NumPy Arrays of 64 bit
        float types.

    Raises:
        ValidationError: If input arrays have different numbers of features.
    """
    arr1 = validate_numeric_array(arr1, name1, allow_nonpositive=True)
    arr2 = validate_numeric_array(arr2, name2, allow_nonpositive=True)

    # ensures that all arrays are at least column vectors
    # otherwise arrays like (100,) and (50,) would fail the next check
    arr1 = arr1.reshape(-1, 1) if arr1.ndim == 1 else arr1
    arr2 = arr2.reshape(-1, 1) if arr2.ndim == 1 else arr2

    if arr1.shape[1] != arr2.shape[1]:
        err_msg = "Error: Input arrays do not have the same number of features."
        raise ValidationError(err_msg)

    return arr1, arr2


def validate_anisotropic_hyperparameter_shape(
    x1: Arrf64,
    param: Arrf64,
) -> None:
    """Validates the length of anisotropic hyperparameters.

    This function validates that the size of each anisotropic hyperparameter
    array is equivalent to the number of features a kernel is training on.

    Args:
        x1: Input data array whose shape is used for reference.
        param: Anisotropic hyperparameter whose shape is being validated.

    Raises:
        ValidationError: If the number of features in x1 has a different value
            than the length of the hyperparameter being validated.
    """
    if x1.shape[1] != param.size:
        err_msg = (
            "Error: 1 or more anisotropic hyperparameters have incorrect"
            " shape. Hyperparameter length should be the same as the number"
            " of data features."
            f"Length: {param.size} Number of features: {x1.shape[1]}."
        )

        raise ValidationError(err_msg)


def validate_multiple_anisotropic_hyperparameter_size(
    params: list[Arrf64],
    names: list[str],
) -> None:
    """Enforces a consistent size across aniotropic hyperparameters in a kernel.

    Args:
        params: List of anisotropic hyperparameters to validate.
        names: List of the names of the hyperparameters to be valdidated.

    Raises:
        ValidationError: If the anisotropic hyperparameters being validated do
            not all have the same length.
    """
    # use the first parameter as the reference
    ref_size = params[0].size
    ref_name = names[0]

    for i in range(1, len(params)):
        curr_param = params[i]
        curr_name = names[i]

        if curr_param.size != ref_size:
            err_msg = (
                "Anisotropic Hyperparameter Mismatch: "
                f"'{curr_name}' has {curr_param.size} dimensions, "
                f"but '{ref_name}' has {ref_size} dimensions. "
                "All anisotropic parameters must match in size."
            )
            raise ValidationError(err_msg)


def validate_set_params(
    params: NumericArray | NumericValue,
    name: str,
    isotropic: bool,
    expected_length: int,
) -> Arrf64:
    """Validates and prepares hyperparameters for kernel's set_params method.

    Args:
        params: Hyperparameter array to validate.
        name: Parameter name for error messages.
        isotropic: If True, strictly enforce expected_length; if False, only
            warn about length mismatches (e.g., if the new hyperparameters have
            a different size than the current hyperparameters).
        expected_length: Expected number of unique hyperparameters.

    Returns:
        Validated and flattened hyperparameters as a NumPy array of 64 bit float
        types.

    Raises:
        ValidationError: If params contains non-positive values, or if
            isotropic=True and params.size != expected_length.

    Warns:
        UserWarning: If isotropic=False and params.size != expected_length.
    """
    params = validate_numeric_array(params, name, allow_nonpositive=False)
    params = params.flatten()

    if isotropic and (params.size != expected_length):
        err_msg = (
            "Error: Wrong number of parameters passed to 'set_params'. "
            f"Expected {expected_length}, got {params.size}."
        )

        raise ValidationError(err_msg)

    # if the first check passes, just warn here as the user could just be
    # changing the shape of the data they are using, and an error will be raised
    # in compute if they pass an array of the wrong size anyways
    if params.size != expected_length:
        warning = (
            "Warning: New hyperparameters have a different length than the "
            "previous kernel hyperparameters. Ensure this is purposeful."
        )
        warnings.warn(warning, stacklevel=1)

    return params


def validate_bounds_dict(
    bounds: Any,
    expected_params: list[str],
    kernel_name: str,
) -> dict[str, list[tuple[f64, f64]]]:
    """Validates the structure of custom bounds dictionaries for kernels.

    This function validates the structure of custom bounds dictionaries passed
    during kernel initialization but not validate the shape, which is unknown at
    when the bounds are passed during kernel initialization.

    Args:
        bounds: The bounds object to be validated.
        expected_params: A list of the names of the expected hyperparameters to
            be checked during validation.
        kernel_name: The name of the kernel whose bounds are being validated,
            used for error messages.

    Returns:
        The validated bounds dictionary.

    Raises:
        ValidationError: If the bounds object is not a dictionary, if it
            contains any non-string key values, if the key values are not valid
            names for the expected hyperparameters, if any of its bounds
            containers have more than two dimensions, if any of its bounds
            containers don't have exactly two values, or if any lower bound is a
            larger value than the upper bound.
    """
    if not isinstance(bounds, dict):
        err_msg = f"Error: Bounds for {kernel_name} must be a dictionary."
        raise ValidationError(err_msg)

    validated = {}

    for param_name, bound in bounds.items():
        if not isinstance(param_name, str):
            err_msg = (
                "Error: hyperparameter names are expected as strings. Got "
                f"{param_name}"
            )
            raise ValidationError(err_msg)

        # format the param name to be all lower case and replace spaces with
        # underscores for flexibility
        # e.g., lets param names like "Length Scale" pass
        param_name = param_name.lower().replace(" ", "_")

        if param_name not in expected_params:
            err_msg = (
                f"Error: '{param_name} is not a valid hyperparameter for "
                f"{kernel_name}"
            )
            raise ValidationError(err_msg)

        bound_arr = validate_numeric_array(
            bound,
            f"{param_name} bounds",
            allow_nonpositive=False,
        )

        # standardize shape so the array is never flat, then we can always check
        # whether each bound has two values by using arr.shape[1]
        if bound_arr.ndim == 1:
            bound_arr = bound_arr.reshape(1, -1)
        elif bound_arr.ndim > 2:
            err_msg = (
                f"Error: Bounds for {param_name} have too many dimensions."
            )
            raise ValidationError(err_msg)

        # verify the container has exactly two values per bound
        if bound_arr.shape[1] != 2:
            err_msg = (
                f"Error: Each bound for '{param_name}' must contain exactly two"
                " numbers."
            )
            raise ValidationError(err_msg)

        if np.any(bound_arr[:, 0] > bound_arr[:, 1]):
            err_msg = (
                "Error: Lower bound must be less than or equal to upper bound "
                f"for {param_name}."
            )
            raise ValidationError(err_msg)

        validated[param_name] = [(row[0], row[1]) for row in bound_arr]

    return validated


### Gaussian Process Validation ###


def validate_input_and_target_data(
    input_data: NumericArray,
    target_data: NumericArray,
) -> tuple[Arrf64, Arrf64]:
    """Validates and reshapes data used for Gaussian Process fitting.

    This function validates that the input and target values passed to a
    GaussianProcess object for fitting are valid data types and formatted
    correctly for the fitting process. Both the input and target data are
    expected as numeric Array-like objects which can be converted to NumPy
    arrays of 64 bit float types for use in internal computation. Target value
    arrays are reshaped to be column vectors if they are found to be flat (1D)
    arrays. The input and target data is validated to enforce dimensional
    consistency (i.e., they must have identical numbers of samples).

    Args:
        input_data: Input features array.
        target_data: Target values array.

    Returns:
        Validated input array of shape (n, d) and target array of shape
        (n,1).

    Raises:
        ValidationError: If arrays contain non-numeric values, or if the number
            of samples in input_data and target_data don't match.
    """
    input_data = validate_numeric_array(
        input_data,
        "Gaussian Process input data",
        allow_nonpositive=True,
    )
    target_data = validate_numeric_array(
        target_data,
        "Gaussian Process target data",
        allow_nonpositive=True,
    ).ravel()

    input_data = (
        input_data.reshape(-1, 1) if input_data.ndim == 1 else input_data
    )

    if target_data.shape[0] != input_data.shape[0]:
        err_msg = (
            "Error: input data should have the same number of samples as target"
            " data."
        )
        raise ValidationError(err_msg)

    return input_data, target_data


def validate_variable_names(
    variable_names: str | list[str],
    expected_num_variables: int,
) -> list[str]:
    """Validates that variable names are strings and match the expected count.

    This function is used to validate the variable names passed to the
    'GaussianProcess.to_str()' method. It validates that the variable names
    are strings and that the expected amount are passed (the number of input
    features).

    Args:
        variable_names: Variable name string or list of variable name strings.
        expected_num_variables: Expected number of variables.

    Returns:
        Validated variable names. This is always returned as a list of strings,
        though passing one single string value (for data with only one feature)
        is a valid input.

    Raises:
        ValidationError: If not all elements are strings, or if the length
            doesn't match the expected number of variables.
    """
    if not isinstance(variable_names, (str, list)):
        err_msg = (
            "Error: 'variable_names' argument must be a string or "
            "list of strings."
        )
        raise ValidationError(err_msg)

    if isinstance(variable_names, str):
        variable_names = [variable_names]

    if not all(isinstance(v, str) for v in variable_names):
        err_msg = "Error: Not all elements of 'variable_names' are strings."
        raise ValidationError(err_msg)

    if len(variable_names) != expected_num_variables:
        err_msg = (
            f"Error: Expected {expected_num_variables} variable names, "
            f"got {len(variable_names)}."
        )
        raise ValidationError(err_msg)

    return variable_names
