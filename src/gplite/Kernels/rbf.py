"""
Radial Basis Function (RBF) kernel, also known as the Squared Exponential
or Gaussian kernel.

The RBF kernel is defined as:

    K(x, x') = exp(-0.5 * ||x - x'||² / l²)

For anisotropic (ARD - Automatic Relevance Determination) variant:

    K(x, x') = exp(-0.5 * Σᵢ (xᵢ - x'ᵢ)² / lᵢ²)

where l (or lᵢ) is the length scale hyperparameter. Larger length scales
produce smoother functions; smaller values allow more rapid variation.

The gradient with respect to length scale l is:

    ∂K/∂l = K(x, x') * ||x - x'||² / l³

This kernel produces infinitely differentiable (very smooth) functions and
is the most commonly used kernel in Gaussian Process regression.
"""

import numpy as np
from gplite._utils._computation import compute_square_euclidean_distance
from gplite._utils._data import expand_kernel_bounds
from gplite._utils._types import Arrf64, f64
from gplite._utils._validation import (
    validate_anisotropic_hyperparameter,
    validate_anisotropic_hyperparameter_shape,
    validate_isotropic_hyperparameter,
    validate_set_params,
)
from gplite.Kernels._base import Kernel


class RBFKernel(Kernel):
    """
    Radial Basis Function (RBF) kernel for smooth function interpolation.
    K(x, x') = exp(-0.5 * sum((x - x')^2 / l^2))

    where l is the length scale hyperparameter controlling smoothness.
    Also known as the Squared Exponential or Gaussian kernel.
    """

    def __init__(self, length_scale: Arrf64, isotropic: bool = True) -> None:
        """
        Initializes an RBF kernel with the specified length scale.

        Args:
            - length_scale (Arrf64): Length scale hyperparameter controlling
                                     smoothness. Scalar for isotropic, array
                                     for anisotropic (one per dimension).
            - isotropic (bool): If True, uses single length scale for all
                                dimensions. Defaults to True.

        Raises:
            ValidationError: If length_scale contains invalid values.
        """
        self.length_scale = (
            validate_isotropic_hyperparameter(length_scale, "RBF Length Scale")
            if isotropic
            else validate_anisotropic_hyperparameter(
                length_scale, "RBF Length Scale"
            )
        )
        self.isotropic = isotropic

        return

    @property
    def hyperparameters(self) -> tuple[str, ...]:
        """
        Returns the names of the kernel's hyperparameters.

        Returns:
            tuple[str, ...]: Tuple containing 'length scale'.
        """
        return ("length scale",)

    @property
    def bounds(self) -> list[tuple[f64, f64]]:
        """
        Returns the optimization bounds for the length scale.

        Returns:
            list[tuple[f64, f64]]: Bounds for length scale in range [1e-6, 5e2].
        """
        return [
            (np.float64(1e-6), np.float64(5e2)),
        ]

    def _compute(self, x1: Arrf64, x2: Arrf64) -> Arrf64:
        """
        Computes the RBF kernel matrix between two input arrays.

        Args:
            - x1 (Arrf64): First input array of shape (n, d).
            - x2 (Arrf64): Second input array of shape (m, d).

        Returns:
            Arrf64: Kernel matrix of shape (n, m).
        """
        # hot path for K(x, x): sq_dist will be calculated with a
        # triangular matrix instead of the full matrix
        if x1 is x2:
            x_scaled = x1 / self.length_scale

            square_dist_scaled = compute_square_euclidean_distance(
                x_scaled, x_scaled
            )

        else:
            x1_scaled = x1 / self.length_scale
            x2_scaled = x2 / self.length_scale

            square_dist_scaled = compute_square_euclidean_distance(
                x1_scaled, x2_scaled
            )

        return np.exp(-0.5 * square_dist_scaled)

    def _gradient(self, x1: Arrf64, x2: Arrf64) -> tuple[Arrf64, ...]:
        """
        Computes the gradient of the RBF kernel with respect to length scale.

        Args:
            - x1 (Arrf64): First input array of shape (n, d).
            - x2 (Arrf64): Second input array of shape (m, d).

        Returns:
            tuple[Arrf64, ...]: Tuple containing the length scale gradient
                                tensor.
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
        self, x1: Arrf64, x2: Arrf64
    ) -> tuple[Arrf64, tuple[Arrf64, ...]]:
        """
        Computes kernel matrix and gradient together for efficiency by
        reusing intermediate calculations.

        Args:
            - x1 (Arrf64): First input array of shape (n, d).
            - x2 (Arrf64): Second input array of shape (m, d).

        Returns:
            tuple[Arrf64, tuple[Arrf64, ...]]: Kernel matrix and tuple
                containing the length scale gradient tensor.
        """
        # 1. Compute K using your fast, already-implemented cdist wrapper
        x1_scaled = x1 / self.length_scale
        x2_scaled = x2 / self.length_scale
        scaled_dist = compute_square_euclidean_distance(x1_scaled, x2_scaled)
        K = np.exp(-0.5 * scaled_dist)

        n, m = K.shape

        # 2. Compute gradients without broadcasting a full (N, M, D) difference tensor
        if self.isotropic:
            # We need the unscaled distance for the gradient
            sq_dist = compute_square_euclidean_distance(x1, x2)
            grad = K * (sq_dist / (self.length_scale[0] ** 3))
            grad = grad[
                :, :, np.newaxis
            ]  # Reshape to (N, M, 1) to match expected output
        else:
            d = x1.shape[1]
            grad = np.empty((n, m, d), dtype=K.dtype)
            for i in range(d):
                # Calculate difference for this dimension only
                diff_i = x1[:, i : i + 1] - x2[:, i : i + 1].T
                grad[:, :, i] = K * (diff_i**2 / (self.length_scale[i] ** 3))

        return K, (grad,)

    def get_params(self) -> Arrf64:
        """
        Returns the current length scale hyperparameters.

        Returns:
            Arrf64: Array of length scale values.
        """
        return self.length_scale

    def set_params(self, params: Arrf64, validate: bool = True) -> None:
        """
        Sets new length scale values for the kernel.

        Args:
            - params (Arrf64): New length scale values as an array.

        Raises:
            ValidationError: If params contains invalid values or wrong size
                             for isotropic kernel.

        Warns:
            UserWarning: If anisotropic params have different length than
                         current hyperparameters.
        """
        if validate:
            expected_num_hyperparameters = len(self.length_scale)
            params = validate_set_params(
                params,
                "New RBF Kernel Hyperparameters",
                self.isotropic,
                expected_num_hyperparameters,
            )

        self.length_scale = params

    def _to_str(
        self, variable_names: list[str], alpha: f64, training_point: Arrf64
    ) -> str:
        """
        Creates a string representation of the RBF kernel expression.

        Args:
            - variable_names (list[str]): Names of input variables.
            - alpha (f64): Weight coefficient.
            - training_point (Arrf64): Training point to center expression on.

        Returns:
            str: Mathematical expression string for the kernel.
        """
        difference_parts = []
        length_scale_squared = self.length_scale**2

        for i, var in enumerate(variable_names):
            if self.isotropic:
                difference_str = (
                    f"( {var} - {training_point[i]:.6e} )^2 / "
                    f"{length_scale_squared[0]:.6e}"
                )

            else:
                difference_str = (
                    f"( {var} - {training_point[i]:.6e} )^2 / "
                    f"{length_scale_squared[i]:.6e}"
                )

            difference_parts.append(difference_str)

        full_dist_str = " + ".join(difference_parts)

        return f"( {alpha:.6e} * exp( -0.5 * ( {full_dist_str} ) ) )"

    def _compute_diag(self, x: Arrf64) -> Arrf64:
        """
        Returns the diagonal of K(x, x). For the RBF kernel, k(x, x) = 1
        for all x since the squared distance is zero.

        Args:
            - x (Arrf64): Input array of shape (n, d).

        Returns:
            Arrf64: Array of ones with shape (n,).
        """
        return np.ones(x.shape[0])

    def _get_expanded_bounds(self) -> list[tuple[f64, f64]]:
        """
        Returns expanded bounds for anisotropic hyperparameters.

        Returns:
            list[tuple[f64, f64]]: Bounds list expanded to match total number
                                   of hyperparameters.
        """
        if self.isotropic:
            return self.bounds

        return expand_kernel_bounds(self.length_scale, self.bounds, 1)

    def _validate_anisotropic_hyperparameter_shape(self, x: Arrf64) -> None:
        """
        Validates that anisotropic length scales match input dimensionality.

        Args:
            - x (Arrf64): Input data for shape reference.

        Raises:
            ValidationError: If length scale size doesn't match number of
                             features.
        """
        if not self.isotropic:
            validate_anisotropic_hyperparameter_shape(x, self.length_scale)
