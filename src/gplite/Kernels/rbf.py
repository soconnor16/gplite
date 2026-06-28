"""Radial Basis Function (RBF) kernel for modeling smooth functions.

The RBF kernel is defined as:
    K(x, x') = exp(-0.5 * Σᵢ (xᵢ - x'ᵢ)² / lᵢ²)
where lᵢ is the length scale hyperparameter. Larger length scales
produce smoother functions; smaller values allow more rapid variation.

The gradient with respect to length scale l is:

    ∂K/∂l = K(x, x') * ||x - x'||² / l³

This kernel produces infinitely differentiable (very smooth) functions and
is the most commonly used kernel in Gaussian Process regression.

More detailed documentation about many methods below can be found in the Kernel
base class in 'Kernels/_base.py'.
"""

from typing import cast

import numpy as np

from gplite._utils._computation import compute_square_euclidean_distance
from gplite._utils._constants import EPSILON
from gplite._utils._data import resolve_bounds_shape
from gplite._utils._types import (
    Arrf64,
    KernelBounds,
    NumericArray,
    NumericValue,
    f64,
)
from gplite._utils._validation import (
    validate_anisotropic_hyperparameter,
    validate_anisotropic_hyperparameter_shape,
    validate_bounds_dict,
    validate_isotropic_hyperparameter,
    validate_set_params,
)
from gplite.Kernels._base import Kernel


class RBFKernel(Kernel):
    """Radial Basis Function (RBF) kernel for smooth function interpolation.

    The rbf kernel matrix is defined as:
        K(x, x') = exp(-0.5 * sum((x - x')² / l²))
    where l is the length scale hyperparameter controlling smoothness.

    Attributes:
        bounds: A dictionary of kernel hyperparameter names and their current
            bounds.
        hyperparameters: A tuple of the kernel's hyperparameter names as
            strings.
        isotropic: A boolean indicating whether the kernel is isotropic.
        length_scale: The current value for the length_scale hyperparameter.
    """

    length_scale: Arrf64
    isotropic: bool

    def __init__(
        self,
        length_scale: NumericArray | NumericValue,
        isotropic: bool = True,
        bounds: KernelBounds | None = None,
    ) -> None:
        """Initializes an rbf kernel.

        Args:
            length_scale: Length scale hyperparameter controlling smoothness.
                Scalar for isotropic, array for anisotropic.
            isotropic: If True, uses single length scale and period for all
                dimensions. Defaults to True.
            bounds: Custom hyperparameter bounds. Must be a dictionary where
                keysare the hyperparameter names (strings) and values are either
                a single tuple (min, max) or a list of tuples [(min, max), ...].
                For example: {"length_scale": (10, 50)}. Defaults to None.

        Raises:
            ValidationError: If hyperparameter values are invalid.
        """
        self.length_scale = (
            validate_isotropic_hyperparameter(length_scale, "RBF Length Scale")
            if isotropic
            else validate_anisotropic_hyperparameter(
                length_scale,
                "RBF Length Scale",
            )
        )
        self.isotropic = isotropic

        # default kernel bounds
        self._bound_config = {
            "length_scale": [(np.float64(1e-6), np.float64(5e2))],
        }

        if bounds is not None:
            validated_bounds = validate_bounds_dict(
                bounds=bounds,
                expected_params=["length_scale"],
                kernel_name="RBFKernel",
            )
            self._bound_config.update(validated_bounds)

    @property
    def hyperparameters(self) -> tuple[str, ...]:
        """Defines kernel hyperparameters.

        Returns:
            The names of the hyperparameters in a given kernel as a tuple of
            strings.
        """
        return ("length_scale",)

    @property
    def bounds(self) -> dict[str, list[tuple[f64, f64]]]:
        """Defines kernel bounds.

        Returns:
            A dictionary with the kernel's hyperparameter names and their
            respective bounds.
        """
        length_scale_dimensions = np.atleast_1d(self.length_scale).size

        return {
            "length_scale": resolve_bounds_shape(
                self._bound_config["length_scale"],
                length_scale_dimensions,
                "length_scale",
            ),
        }

    @property
    def _bounds(self) -> list[tuple[f64, f64]]:
        """Exposes the bounds defined for the kernel hyperparameters internally.

        Returns:
            A flat list of tuples representing the bounds for a kernel's
            hyperparameters.
        """
        return self.bounds["length_scale"]

    def _compute(self, x1: Arrf64, x2: Arrf64) -> Arrf64:
        """Computes the similarity matrix of the kernel.

        Args:
            x1: First array of points used to compute the kernel matrix.
            x2: Second array of points used to compute the kernel matrix.

        Returns:
            Kernel covariance matrix calculated between x1 and x2.
        """
        # hot path for K(x, x): sq_dist will be calculated with a
        # triangular matrix instead of the full matrix
        if x1 is x2:
            x_scaled = x1 / self.length_scale

            square_dist_scaled = compute_square_euclidean_distance(
                x_scaled,
                x_scaled,
            )

        else:
            x1_scaled = x1 / self.length_scale
            x2_scaled = x2 / self.length_scale

            square_dist_scaled = compute_square_euclidean_distance(
                x1_scaled,
                x2_scaled,
            )

        return np.exp(-0.5 * square_dist_scaled)

    def _gradient(self, x1: Arrf64, x2: Arrf64) -> tuple[Arrf64, ...]:
        """Compute the kernel's gradient with respect to its hyperparameters.

        Args:
            x1: First array of points used to compute the kernel gradient.
            x2: Second array of points used to compute the kernel gradient.

        Returns:
            Tuple of a kernel's gradients with respect to each of its
            hyperparameters.
        """
        K = self._compute(x1, x2)

        num_rows, num_columns = K.shape
        num_features = x1.shape[1]

        num_params = self.length_scale.size

        # initialize the length scale array with zeros
        grad_length_scale = np.zeros((num_rows, num_columns, num_params))

        for dim in range(num_features):
            l_d = (
                self.length_scale[0]
                if self.isotropic
                else self.length_scale[dim]
            )

            # difference array between x1 and x2 for this dimension
            diff_d = x1[:, dim : dim + 1] - x2[:, dim : dim + 1].T
            # square distance array between x1 and x2 for this dimension
            sq_dist_d = diff_d**2

            # partial derivative term for this dimension
            term = K * (sq_dist_d / (l_d**3))

            if self.isotropic:
                # terms should accumulate with only one length scale
                grad_length_scale[:, :, 0] += term
            else:
                # otherwise each length scale dimension should be assigned
                # to a separate gradient in the tensor
                grad_length_scale[:, :, dim] = term

        return (grad_length_scale,)

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
        x1_scaled = x1 / self.length_scale
        x2_scaled = x2 / self.length_scale
        scaled_dist = compute_square_euclidean_distance(x1_scaled, x2_scaled)
        K = np.exp(-0.5 * scaled_dist)

        n, m = K.shape

        if self.isotropic:
            # we need the unscaled distance for the gradient
            sq_dist = compute_square_euclidean_distance(x1, x2)
            grad = K * (sq_dist / (self.length_scale[0] ** 3))
            grad = grad[
                :,
                :,
                np.newaxis,
            ]  # reshape to (N, M, 1) to match expected output
        else:
            d = x1.shape[1]
            grad = np.empty((n, m, d), dtype=K.dtype)
            for i in range(d):
                # calculate difference for this dimension only
                diff_i = x1[:, i : i + 1] - x2[:, i : i + 1].T
                grad[:, :, i] = K * (diff_i**2 / (self.length_scale[i] ** 3))

        return K, (grad,)

    def get_params(self) -> Arrf64:
        """Returns the current hyperparameter values.

        Returns:
            Array of the kernel's hyperparameter values.
        """
        return self.length_scale

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

        Raises:
            ValidationError: If the input parameter values are invalid.

        Warns:
            UserWarning: If anisotropic params have different length than
                current hyperparameters.
        """
        if _validate:
            expected_num_hyperparameters = len(self.length_scale)
            params = validate_set_params(
                params,
                "New RBF Kernel Hyperparameters",
                self.isotropic,
                expected_num_hyperparameters,
            )

        self.length_scale = cast("Arrf64", params)

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
        difference_parts = []

        # pre-compute the exponent coefficient for more compact strings
        coeffs = -0.5 / (self.length_scale**2)

        for i, var in enumerate(variable_names):
            coeff = coeffs[0] if self.isotropic else coeffs[i]
            tp = float(training_point[i])

            # handle different training point cases to save tokens
            # when possible
            if abs(tp) < EPSILON:
                diff_str = f"{coeff:.6e}*{var}²"
            elif tp < 0.0:
                diff_str = f"{coeff:.6e}*({var}+{abs(tp):.6e})²"
            else:
                diff_str = f"{coeff:.6e}*({var}-{tp:.6e})²"

            difference_parts.append(diff_str)

        full_dist_str = " + ".join(difference_parts)

        return f"{alpha:.6e}*exp({full_dist_str})"

    def _compute_diag(self, x: Arrf64) -> Arrf64:
        """Computes the diagonal of the kernel matrix K(x, x).

        Args:
            x: Input array of shape (n, d).

        Returns:
            The diagonal of K(x, x) as a flat array.
        """
        return np.ones(x.shape[0])

    def _validate_anisotropic_hyperparameter_shape(self, x: Arrf64) -> None:
        """Validates the shapes of anisotropic hyperparameters.

        Args:
            x: Input data array to be validated.

        Raises:
            ValidationError: If the hyperparameter length doesn't match the
                number of features.
        """
        if not self.isotropic:
            validate_anisotropic_hyperparameter_shape(x, self.length_scale)
