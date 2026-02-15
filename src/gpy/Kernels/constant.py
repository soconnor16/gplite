"""
Constant kernel implementation that returns a constant covariance value
regardless of input distance.

The constant kernel is defined as:

    K(x, x') = c

where c is a positive constant hyperparameter. The gradient is simply:

    ∂K/∂c = 1

This kernel is primarily used as a component in composite kernels:
    - Added to other kernels: provides a baseline/bias term
    - Multiplied with other kernels: acts as a scaling factor (amplitude)

For example, c * RBF(l) scales the RBF kernel's output magnitude.
"""

import numpy as np

from gpy._utils._types import Arrf64, f64
from gpy._utils._validation import (
    validate_isotropic_hyperparameter,
    validate_set_params,
)
from gpy.Kernels._base import Kernel


class ConstantKernel(Kernel):
    """
    Constant kernel that returns the same covariance value for all input pairs.
    K(x, x') = c, where c is the constant hyperparameter.

    Useful as a bias term when combined with other kernels.
    """

    def __init__(self, constant: f64) -> None:
        """
        Initializes a constant kernel with the specified constant value.

        Args:
            - constant (f64): The constant covariance value. Must be positive.

        Raises:
            ValidationError: If constant is not a positive numeric value.
        """
        self.constant = validate_isotropic_hyperparameter(
            constant, "Constant Kernel Constant"
        )

    @property
    def hyperparameters(self) -> tuple[str, ...]:
        """
        Returns the names of the kernel's hyperparameters.

        Returns:
            tuple[str, ...]: Tuple containing 'constant'.
        """
        return ("constant",)

    @property
    def bounds(self) -> list[tuple[f64, f64]]:
        """
        Returns the optimization bounds for the constant hyperparameter.

        Returns:
            list[tuple[f64, f64]]: Bounds for constant in range [1e-6, 1e5].
        """
        return [(f64(1e-6), f64(1e5))]

    def _compute(self, x1: Arrf64, x2: Arrf64) -> Arrf64:
        """
        Computes the constant kernel matrix filled with the constant value.

        Args:
            - x1 (Arrf64): First input array of shape (n, d).
            - x2 (Arrf64): Second input array of shape (m, d).

        Returns:
            Arrf64: Kernel matrix of shape (n, m) filled with constant value.
        """
        return np.full((x1.shape[0], x2.shape[0]), self.constant[0])

    def _gradient(self, x1: Arrf64, x2: Arrf64) -> tuple[Arrf64, ...]:
        """
        Computes the gradient of the kernel with respect to the constant.

        Args:
            - x1 (Arrf64): First input array of shape (n, d).
            - x2 (Arrf64): Second input array of shape (m, d).

        Returns:
            tuple[Arrf64, ...]: Gradient tensor of ones with shape (n, m, 1).
        """
        return (np.ones((x1.shape[0], x2.shape[0], 1)),)

    def _validate_anisotropic_hyperparameter_shape(self, x: Arrf64) -> None:
        """No-op since constant kernel has no anisotropic hyperparameters."""
        pass

    def get_params(self) -> Arrf64:
        """
        Returns the current constant hyperparameter value.

        Returns:
            Arrf64: Array containing the constant value.
        """
        return self.constant

    def set_params(self, params: Arrf64) -> None:
        """
        Sets new hyperparameter values for the kernel.

        Args:
            - params (Arrf64): New constant value as an array.

        Raises:
            ValidationError: If params contains invalid values.
        """
        self.constant = validate_set_params(
            params, "New Constant Kernel Hyperparameter", True, 1
        )

    def _compute_with_gradient(
        self, x1: Arrf64, x2: Arrf64
    ) -> tuple[Arrf64, tuple[Arrf64, ...]]:
        """
        Computes kernel matrix and gradient together.

        Args:
            - x1 (Arrf64): First input array.
            - x2 (Arrf64): Second input array.

        Returns:
            tuple[Arrf64, tuple[Arrf64, ...]]: Kernel matrix and gradient tuple.
        """
        K = np.full((x1.shape[0], x2.shape[0]), self.constant[0])
        grad = np.ones((x1.shape[0], x2.shape[0], 1))

        return K, (grad,)

    def _to_str(
        self, variable_names: list[str], alpha: f64, training_point: Arrf64
    ) -> str:
        """
        Creates a string representation of the constant kernel term.

        Args:
            - variable_names (list[str]): Input variable names (unused).
            - alpha (f64): Weight coefficient.
            - training_point (Arrf64): Training point (unused).

        Returns:
            str: String representation 'alpha * constant'.
        """
        return f"{alpha:.6e} * {self.constant[0]:.6e}"

    def _get_expanded_bounds(self) -> list[tuple[f64, f64]]:
        """
        Returns bounds for optimization.

        Returns:
            list[tuple[f64, f64]]: Same as bounds property for isotropic kernel.
        """
        return self.bounds

    def _validate_input_data(
        self, x1: Arrf64, x2: Arrf64, name1: str, name2: str
    ) -> tuple[Arrf64, Arrf64]:
        """
        Validates input arrays using parent class validation.

        Args:
            - x1 (Arrf64): First input array.
            - x2 (Arrf64): Second input array.
            - name1 (str): Name of first array for error messages.
            - name2 (str): Name of second array for error messages.

        Returns:
            tuple[Arrf64, Arrf64]: Validated input arrays.
        """
        return super()._validate_input_data(x1, x2, name1, name2)
