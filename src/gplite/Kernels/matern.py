"""
Matérn kernel implementation for Gaussian Process regression with
controllable smoothness.

The Matérn kernel family is parameterized by a smoothness parameter ν (nu)
that controls the differentiability of the resulting functions:

    - ν = 3/2: Once differentiable (C¹) functions
    - ν = 5/2: Twice differentiable (C²) functions
    - ν → ∞: Infinitely differentiable, equivalent to the RBF kernel

For ν = 3/2:

    K(x, x') = (1 + √3 r) exp(-√3 r)

For ν = 5/2:

    K(x, x') = (1 + √5 r + 5r²/3) exp(-√5 r)

where r = ||x - x'|| / l for isotropic, or
r = √(Σᵢ (xᵢ - x'ᵢ)² / lᵢ²) for anisotropic (ARD).

Gradients with respect to length scale l:

    ν = 3/2:  ∂K/∂l = 3r² / l³ · exp(-√3 r/l)
    ν = 5/2:  ∂K/∂l = 5r²(1 + √5 r/l) / (3l³) · exp(-√5 r/l)

The Matérn kernel is widely used in practice as a less restrictive
alternative to the RBF kernel, particularly for physical processes
with finite differentiability.
"""

import numpy as np
from gplite._utils._computation import compute_square_euclidean_distance
from gplite._utils._constants import VALID_NU
from gplite._utils._data import expand_kernel_bounds
from gplite._utils._errors import ValidationError
from gplite._utils._types import Arrf64, f64
from gplite._utils._validation import (
    validate_anisotropic_hyperparameter,
    validate_anisotropic_hyperparameter_shape,
    validate_isotropic_hyperparameter,
    validate_set_params,
)
from gplite.Kernels._base import Kernel


class MaternKernel(Kernel):
    """
    Matérn kernel for controlling function smoothness in GP regression.
    Supports ν = 3/2 (once differentiable) and ν = 5/2 (twice differentiable).

    K_{3/2}(x, x') = (1 + √3 r) exp(-√3 r)
    K_{5/2}(x, x') = (1 + √5 r + 5r²/3) exp(-√5 r)

    where r is the (optionally ARD-scaled) Euclidean distance.
    """

    def __init__(
        self,
        length_scale: Arrf64,
        nu: float = 2.5,
        isotropic: bool = True,
    ) -> None:
        """
        Initializes a Matérn kernel with the specified length scale and
        smoothness parameter.

        Args:
            - length_scale (Arrf64): Length scale hyperparameter controlling
                                     correlation distance. Scalar for isotropic,
                                     array for anisotropic (one per dimension).
            - nu (float): Smoothness parameter. Must be 1.5 or 2.5.
                          Defaults to 2.5.
            - isotropic (bool): If True, uses single length scale for all
                                dimensions. Defaults to True.

        Raises:
            ValidationError: If length_scale contains invalid values or nu
                             is not 1.5 or 2.5.
        """
        if nu not in VALID_NU:
            err_msg = f"Error: 'nu' must be 1.5 or 2.5, got {nu}."
            raise ValidationError(err_msg)

        self.length_scale = (
            validate_isotropic_hyperparameter(
                length_scale, "Matérn Length Scale"
            )
            if isotropic
            else validate_anisotropic_hyperparameter(
                length_scale, "Matérn Length Scale"
            )
        )
        self._nu = nu
        self.isotropic = isotropic

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
        Computes the Matérn kernel matrix between two input arrays.

        Args:
            - x1 (Arrf64): First input array of shape (n, d).
            - x2 (Arrf64): Second input array of shape (m, d).

        Returns:
            Arrf64: Kernel matrix of shape (n, m).
        """
        x1_scaled = x1 / self.length_scale
        x2_scaled = x2 / self.length_scale

        sq_dist = compute_square_euclidean_distance(x1_scaled, x2_scaled)
        r = np.sqrt(np.maximum(sq_dist, 0.0))

        if self._nu == 1.5:
            z = np.sqrt(3.0) * r
            return (1.0 + z) * np.exp(-z)

        # nu == 2.5
        z = np.sqrt(5.0) * r
        return (1.0 + z + z**2 / 3.0) * np.exp(-z)

    def _gradient(self, x1: Arrf64, x2: Arrf64) -> tuple[Arrf64, ...]:
        """
        Computes the gradient of the Matérn kernel with respect to the
        length scale.

        Args:
            - x1 (Arrf64): First input array of shape (n, d).
            - x2 (Arrf64): Second input array of shape (m, d).

        Returns:
            tuple[Arrf64, ...]: Tuple containing the length scale gradient
                                tensor.
        """
        diff = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]
        sq_diff = diff**2

        if self.isotropic:
            sum_sq = np.sum(sq_diff, axis=2)
            r = np.sqrt(np.maximum(sum_sq, 0.0)) / self.length_scale[0]
        else:
            r = np.sqrt(
                np.maximum(np.sum(sq_diff / self.length_scale**2, axis=2), 0.0)
            )

        l_cubed = self.length_scale**3

        if self._nu == 1.5:
            exp_term = np.exp(-np.sqrt(3.0) * r)[:, :, np.newaxis]

            if self.isotropic:
                grad = (
                    3.0
                    * np.sum(sq_diff, axis=2, keepdims=True)
                    / l_cubed[0]
                    * exp_term
                )
            else:
                grad = 3.0 * sq_diff / l_cubed * exp_term

        else:  # nu == 2.5
            sqrt5_r = np.sqrt(5.0) * r
            exp_term = np.exp(-sqrt5_r)[:, :, np.newaxis]
            factor = (1.0 + sqrt5_r)[:, :, np.newaxis]

            if self.isotropic:
                grad = (
                    5.0
                    * factor
                    * np.sum(sq_diff, axis=2, keepdims=True)
                    / (3.0 * l_cubed[0])
                    * exp_term
                )
            else:
                grad = 5.0 * factor * sq_diff / (3.0 * l_cubed) * exp_term

        return (grad,)

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
        diff = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]
        sq_diff = diff**2

        if self.isotropic:
            sum_sq = np.sum(sq_diff, axis=2)
            r = np.sqrt(np.maximum(sum_sq, 0.0)) / self.length_scale[0]
        else:
            r = np.sqrt(
                np.maximum(np.sum(sq_diff / self.length_scale**2, axis=2), 0.0)
            )

        l_cubed = self.length_scale**3

        if self._nu == 1.5:
            z = np.sqrt(3.0) * r
            exp_neg_z = np.exp(-z)
            K = (1.0 + z) * exp_neg_z

            exp_3d = exp_neg_z[:, :, np.newaxis]
            if self.isotropic:
                grad = (
                    3.0
                    * np.sum(sq_diff, axis=2, keepdims=True)
                    / l_cubed[0]
                    * exp_3d
                )
            else:
                grad = 3.0 * sq_diff / l_cubed * exp_3d

        else:  # nu == 2.5
            z = np.sqrt(5.0) * r
            exp_neg_z = np.exp(-z)
            K = (1.0 + z + z**2 / 3.0) * exp_neg_z

            exp_3d = exp_neg_z[:, :, np.newaxis]
            factor = (1.0 + z)[:, :, np.newaxis]
            if self.isotropic:
                grad = (
                    5.0
                    * factor
                    * np.sum(sq_diff, axis=2, keepdims=True)
                    / (3.0 * l_cubed[0])
                    * exp_3d
                )
            else:
                grad = 5.0 * factor * sq_diff / (3.0 * l_cubed) * exp_3d

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
                "New Matérn Kernel Hyperparameters",
                self.isotropic,
                expected_num_hyperparameters,
            )

        self.length_scale = params

    def _to_str(
        self, variable_names: list[str], alpha: f64, training_point: Arrf64
    ) -> str:
        """
        Creates a string representation of the Matérn kernel expression.

        Args:
            - variable_names (list[str]): Names of input variables.
            - alpha (f64): Weight coefficient.
            - training_point (Arrf64): Training point to center expression on.

        Returns:
            str: Mathematical expression string for the kernel.
        """
        ls_squared = self.length_scale**2

        dist_parts = []
        for i, var in enumerate(variable_names):
            ls_val = ls_squared[0] if self.isotropic else ls_squared[i]
            dist_parts.append(
                f"( {var} - {training_point[i]:.6e} )^2 / {ls_val:.6e}"
            )

        dist_sum = " + ".join(dist_parts)
        r_str = f"sqrt( {dist_sum} )"

        if self._nu == 1.5:
            c = np.sqrt(3.0)
            return (
                f"( {alpha:.6e} * ( 1 + {c:.6e} * {r_str} ) "
                f"* exp( -{c:.6e} * {r_str} ) )"
            )

        # nu == 2.5
        c = np.sqrt(5.0)
        c_sq = 5.0 / 3.0
        return (
            f"( {alpha:.6e} * ( 1 + {c:.6e} * {r_str} "
            f"+ {c_sq:.6e} * ( {dist_sum} ) ) "
            f"* exp( -{c:.6e} * {r_str} ) )"
        )

    def _compute_diag(self, x: Arrf64) -> Arrf64:
        """
        Returns the diagonal of K(x, x). For the Matérn kernel, k(x, x) = 1
        for all x since the distance r is zero.

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
