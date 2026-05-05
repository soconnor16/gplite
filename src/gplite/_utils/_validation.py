import warnings
from typing import Any

import numpy as np
from gplite._utils._errors import ValidationError
from gplite._utils._types import Arrf64, f64


### General Validation ###
def validate_numeric_value(
    value: Any, name: str, allow_nonpositive: bool
) -> f64:
    """
    Function to validate individual numeric values and reject others.

    Args:
        - value: Any
          - Value to be validated before use.

        - name: str
          - "name" of the value for more descriptive error messages.

        - allow_negative: bool
          - Whether the value should be allowed to be negative.


    Returns:
        - f64: Validated value as a 64 bit NumPy float.
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
    array: Any, name: str, allow_nonpositive: bool
) -> Arrf64:
    """
    Function to validate numeric arrays and array-like types and reject others.

    Args:
        - array (Any): The value to validate. Expected as an array of 64 bit
                       floats or an object that can be converted to an array of
                       64 bit floats.

        - name (str): The "name" of the object being validated.

        - allow_nonpositive (bool): Whether nonpositive values should be allowed
                                    in the array.

    Returns:
        - Arrf64: The validated array if it passes all validation checks.

    Raises:
        ValidationError: If the array is empty, contains any 'nan' or 'inf'
                         values, or any values less than or equal to 0 when
                         allow_nonpositive is false.
    """

    try:
        array = np.array(array, dtype=np.float64)

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


### Kernel validation ###


def validate_isotropic_hyperparameter(param: Any, name: str) -> Arrf64:
    """
    Validates isotropic kernel hyperparameters which are expected as floats.
    Returns them as 1D arrays for consistency with anisotropic hyperparameters
    which are expected as arrays.

    Args:
        - param (Any): The param to validate.
        - name (str): The "name" of the hyperparameter for better error
                      messages.

    Returns:
        - Arrf64: The validated hyperparameter as a 1D array.
    """
    param = validate_numeric_value(param, name, allow_nonpositive=False)

    return np.array([param], dtype=np.float64)


def validate_anisotropic_hyperparameter(param: Any, name: str) -> Arrf64:
    """
    Validates anisotropic kernel hyperparameters which are expected as arrays
    of floats. Returns them as flattened 1D arrays.

    Args:
        - param (Any): The hyperparameters to validate.
        - name (str): The "name" of the hyperparameter for better error
                      messages.

    Returns:
        - Arrf64: The validated hyperparameter as a 1D array.
    """
    return validate_numeric_array(
        param, name, allow_nonpositive=False
    ).flatten()


def validate_input_arrays(
    arr1: Arrf64, name1: str, arr2: Arrf64, name2: str
) -> tuple[Arrf64, Arrf64]:
    """
    Validates input arrays passed to a kernel's 'compute' method. Validates
    that the arrays are valid numeric arrays and that they have compatible
    shape (same number of features).

    Args:
        - arr1 (Arrf64): First array to validate.
        - name1 (str): "Name" of first array for better error messages.
        - arr2 (Arrf64): Second array to validate.
        - name2 (str): "Name" of second array for better error messages.

    Returns:
        - tuple[Arrf64, Arrf64]: A tuple of validated arrays.

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
    x1: Arrf64, param: Arrf64
) -> None:
    """
    Validates that anisotropic hyperparameters have the same length as their
    input data has features.

    Args:
        - x1 (Arrf64): Input data array whose shape is used for reference
        - param (Arrf64): Anisotropic hyperparameter whose shape is being
                          validated.

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

    return


def validate_multiple_anisotropic_hyperparameter_size(
    params: list[Arrf64], names: list[str]
) -> None:
    """
    Validates that in kernels that have multiple possible anisotropic
    hyperparameters, all anisotropic hyperparameters have the same length.

    Args:
        - params (Arrf64): List of anisotropic hyperparameters to validate.
        - names (list[str]): List of the names of the hyperparameters to be
                             valdidated.

    Raises:
        ValidationError: If the anisotropic hyperparameters being validated do
                         not have the same length.
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

    return


def validate_set_params(
    params: Arrf64, name: str, isotropic: bool, expected_length: int
) -> Arrf64:
    """
    Validates and prepares hyperparameters for kernel's set_params method.

    Args:
        - params (Arrf64): Hyperparameter array to validate.
        - name (str): Parameter name for error messages.
        - isotropic (bool): If True, strictly enforce expected_length;
                            if False, only warn about length mismatches.
        - expected_length (int): Expected number of parameters.

    Returns:
        Arrf64: Validated and flattened parameter array.

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


### Gaussian Process Validation ###


def validate_input_and_target_data(
    input_data: Arrf64, target_data: Arrf64
) -> tuple[Arrf64, Arrf64]:
    """
    Validates and reshapes input features and target values for Gaussian
    process fitting.

    Args:
        - input_data (Arrf64): Input features array.
        - target_data (Arrf64): Target values array.

    Returns:
        tuple[Arrf64, Arrf64]: Validated input array of shape (n, d) and
                               target array of shape (n,).

    Raises:
        ValidationError: If arrays contain non-numeric values, or if the
                         number of samples in input_data and target_data don't
                         match.
    """

    input_data = validate_numeric_array(
        input_data, "Gaussian Process input data", allow_nonpositive=True
    )
    target_data = validate_numeric_array(
        target_data, "Gaussian Process target data", allow_nonpositive=True
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
    variable_names: list[str], expected_num_variables: int
) -> list[str]:
    """
    Validates that variable names are strings and match the expected count.

    Args:
        - variable_names (list[str]): List of variable name strings.
        - expected_num_variables (int): Expected number of variables.

    Returns:
        list[str]: Validated variable names list.

    Raises:
        ValidationError: If not all elements are strings, or if the length
                         doesn't match expected_num_variables.
    """
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
