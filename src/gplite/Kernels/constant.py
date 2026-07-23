"""Constant kernel class for modeling data with a constant variance.

The constant kernel is defined as:

    K(x, x') = c

where c is a positive constant hyperparameter. The gradient is simply:

    ∂K/∂c = 1

This kernel is primarily used as a component in composite kernels:
    - Added to other kernels: provides a baseline/bias term
    - Multiplied with other kernels: acts as a scaling factor (amplitude)

More detailed documentation about many methods below can be found in the Kernel
base class in 'Kernels/_base.py'.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gplite._utils._data import resolve_bounds_shape
from gplite._utils._types import (
    Arrf64,
    KernelBounds,
    NumericArray,
    NumericValue,
    f64,
)
from gplite._utils._validation import (
    validate_bounds_dict,
    validate_isotropic_hyperparameter,
    validate_set_params,
)
from gplite.Kernels._base import Kernel

if TYPE_CHECKING:
    from gplite._utils._types import (
        Arrf64,
        KernelBounds,
        NumericArray,
        NumericValue,
        f64,
    )


class ConstantKernel(Kernel):
    """Constant kernel that returns the same covariance value for all inputs.

    The constant kernel is defined as:
        K(x, x') = c
    where c is the constant hyperparameter.

    Attributes:
        bounds: A dictionary of kernel hyperparameter names and their current
            bounds.
        hyperparameters: A tuple of the kernel's hyperparameter names as
            strings.
        isotropic: A boolean indicating whether the kernel is isotropic.
        constant: The current value for the constant hyperparameter.
    """

    constant: Arrf64

    def __init__(
        self,
        constant: NumericValue,
        bounds: KernelBounds | None = None,
    ) -> None:
        """Initializes a constant kernel.

        Args:
            constant: The constant covariance value. Must be positive.

            bounds: Custom hyperparameter bounds. Must be a dictionary where
                keysare the hyperparameter names (strings) and values are either
                a single tuple (min, max) or a list of tuples [(min, max), ...].
                For example: {"constant": (5, 10)}. Defaults to None.

        Raises:
            ValidationError: If constant is not a positive numeric value.
        """
        self.constant = validate_isotropic_hyperparameter(
            constant,
            "Constant Kernel Constant",
        )

        self._bound_config = {"constant": [(np.float64(1e-6), np.float64(1e5))]}

        if bounds is not None:
            validated_bounds = validate_bounds_dict(
                bounds=bounds,
                expected_params=["constant"],
                kernel_name="ConstantKernel",
            )
            self._bound_config.update(validated_bounds)

    @property
    def hyperparameters(self) -> tuple[str, ...]:
        """Defines kernel hyperparameters.

        Returns:
            The names of the hyperparameters in a given kernel as a tuple of
            strings.
        """
        return ("constant",)

    @property
    def bounds(self) -> dict[str, list[tuple[f64, f64]]]:
        """Defines kernel bounds.

        Returns:
            A dictionary with the kernel's hyperparameter names and their
            respective bounds.
        """
        return {
            "constant": resolve_bounds_shape(
                self._bound_config["constant"],
                1,  # the constant kernel is always isotropic
                "constant",
            ),
        }

    @property
    def _bounds(self) -> list[tuple[f64, f64]]:
        """Exposes the bounds defined for the kernel hyperparameters internally.

        Returns:
            A flat list of tuples representing the bounds for a kernel's
            hyperparameters.
        """
        return self.bounds["constant"]

    def _compute(self, x1: Arrf64, x2: Arrf64) -> Arrf64:
        """Computes the similarity matrix of a kernel.

        Args:
            x1: First array of points used to compute the kernel matrix.
            x2: Second array of points used to compute the kernel matrix.

        Returns:
            Kernel covariance matrix calculated between x1 and x2.
        """
        return np.full((x1.shape[0], x2.shape[0]), self.constant[0])

    def _gradient(self, x1: Arrf64, x2: Arrf64) -> tuple[Arrf64, ...]:
        """Compute the kernel's gradient with respect to its hyperparameters.

        Args:
            x1: First array of points used to compute the kernel gradient.
            x2: Second array of points used to compute the kernel gradient.

        Returns:
            Tuple of a kernel's gradients with respect to each of its
            hyperparameters.
        """
        return (np.ones((x1.shape[0], x2.shape[0], 1)),)

    def _compute_with_gradient(
        self,
        x1: Arrf64,
        x2: Arrf64,
    ) -> tuple[Arrf64, tuple[Arrf64, ...]]:
        """Computes the kernel's matrix and gradients together.

        Args:
            x1: First input array of shape (n, d).
            x2: Second input array of shape (m, d).

        Returns:
            Tuple containing the kernel matrix K of shape (n, m) and a
            tuple of gradient tensors.
        """
        K = np.full((x1.shape[0], x2.shape[0]), self.constant[0])
        grad = np.ones((x1.shape[0], x2.shape[0], 1))

        return K, (grad,)

    def get_params(self) -> Arrf64:
        """Returns the current constant hyperparameter value.

        Returns:
            Single-valued array containing the constant value.
        """
        return self.constant

    def set_params(
        self,
        params: NumericArray | NumericValue,
        _validate: bool = True,
    ) -> None:
        """Method to set new hyperparameter values for the kernel.

        Args:
            params: New hyperparameter values to be set for the kernel.
            _validate: Whether to validate the hyperparameters before setting
                them. This is intended to be used for internal usage such as
                optimization loops where skipping the small overhead from
                validation saves a lot of time. If _validate is false, it is
                assumed you know what you are doing. Defaults to True.
        """
        if _validate:
            self.constant = validate_set_params(
                params,
                "New Constant Kernel Hyperparameter",
                True,
                1,
            )
        else:
            self.constant = np.asarray(params, dtype=np.float64).ravel()

    def _to_str(
        self,
        variable_names: list[str],
        alpha: f64,
        training_point: Arrf64,
    ) -> str:
        """Creates a string representation of a kernel at a single data point.

        Args:
            variable_names: Names of variables to be used in the string
                (e.g., ['x', 'y']).
            alpha: The computed weight for this data point.
            training_point: The specific training point to center the expression
                along.

        Returns:
            A string representation of the mathematical definition of the
            kernel function at the given training point.
        """
        return f"{(alpha * self.constant[0]):.15e}"

    def _compute_diag(self, x: Arrf64) -> Arrf64:
        """Computes the diagonal of the kernel matrix K(x, x).

        Args:
            x: Input array of shape (n, d).

        Returns:
            The diagonal of K(x, x) as a flat array.
        """
        return np.full(x.shape[0], self.constant[0], dtype=np.float64)

    def _validate_anisotropic_hyperparameter_shape(self, x: Arrf64) -> None:
        """Pass since the constant kernel has no anisotropic hyperparameters."""
