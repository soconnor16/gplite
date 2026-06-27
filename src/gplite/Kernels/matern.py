"""Matérn kernel class for data that needs a controllable smoothness.

The Matérn kernel family is parameterized by a smoothness parameter ν (nu)
that controls the differentiability of the resulting functions:

    - ν = 3/2: Once differentiable
    - ν = 5/2: Twice differentiable
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

More detailed documentation about many methods below can be found in the Kernel
base class in 'Kernels/_base.py'.
"""

from typing import cast

import numpy as np

from gplite._utils._computation import compute_square_euclidean_distance
from gplite._utils._constants import VALID_NU
from gplite._utils._data import resolve_bounds_shape
from gplite._utils._errors import ValidationError
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


class MaternKernel(Kernel):
    """Matérn kernel for controlling function smoothness in GP regression.

    Supports ν = 3/2 (once differentiable) and ν = 5/2 (twice differentiable).

    The kernel matrices for v = 3/2 and v = 5/2 are defined as:
        K_{3/2}(x, x') = (1 + √3 r) exp(-√3 r)
    and
        K_{5/2}(x, x') = (1 + √5 r + 5r²/3) exp(-√5 r)
    where r is the Euclidean distance between input points.

    Attributes:
        bounds: A dictionary of kernel hyperparameter names and their current
            bounds.
        hyperparameters: A tuple of the kernel's hyperparameter names as
            strings.
        isotropic: A boolean indicating whether the kernel is isotropic.
        length_scale: The current value for the length_scale hyperparameter.
        nu: The current value for the kernel's nu value (3/2 or 5/2).
    """

    length_scale: Arrf64
    nu: float
    isotropic: bool

    def __init__(
        self,
        length_scale: NumericArray | NumericValue,
        nu: float = 2.5,
        isotropic: bool = True,
        bounds: KernelBounds | None = None,
    ) -> None:
        """Initializes a Matérn kernel.

        Args:
            length_scale: Length scale hyperparameter controlling correlation
                distance. Scalar for isotropic, array for anisotropic.
            nu: Smoothness parameter. Must be 1.5 or 2.5. Defaults to 2.5.
            isotropic: If True, uses single length scale for all dimensions.
                Defaults to True.
            bounds: Custom hyperparameter bounds. Must be a dictionary where
                keysare the hyperparameter names (strings) and values are either
                a single tuple (min, max) or a list of tuples [(min, max), ...].
                For example: {"length_scale": (5, 10)}. Defaults to None.

        Raises:
            ValidationError: If length_scale contains invalid values or nu is
                not 1.5 or 2.5.
        """
        if nu not in VALID_NU:
            err_msg = f"Error: 'nu' must be 1.5 or 2.5, got {nu}."
            raise ValidationError(err_msg)

        self.length_scale = (
            validate_isotropic_hyperparameter(
                length_scale,
                "Matérn Length Scale",
            )
            if isotropic
            else validate_anisotropic_hyperparameter(
                length_scale,
                "Matérn Length Scale",
            )
        )
        self._nu = nu
        self.isotropic = isotropic

        # default kernel bounds
        self._bound_config = {
            "length_scale": [(np.float64(1e-6), np.float64(5e2))],
        }

        if bounds is not None:
            validated_bounds = validate_bounds_dict(
                bounds=bounds,
                expected_params=["length_scale"],
                kernel_name="Matern",
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
        x1_scaled = x1 / self.length_scale
        x2_scaled = x2 / self.length_scale

        sq_dist = compute_square_euclidean_distance(x1_scaled, x2_scaled)
        r = np.sqrt(np.maximum(sq_dist, 0.0))

        if self._nu == 1.5:
            z = np.sqrt(3.0) * r
            return (1.0 + z) * np.exp(-z)

        # nu = 2.5
        z = np.sqrt(5.0) * r
        return (1.0 + z + z**2 / 3.0) * np.exp(-z)

    def _gradient(self, x1: Arrf64, x2: Arrf64) -> tuple[Arrf64, ...]:
        """Compute the kernel's gradient with respect to its hyperparameters.

        Args:
            x1: First array of points used to compute the kernel gradient.
            x2: Second array of points used to compute the kernel gradient.

        Returns:
            Tuple of a kernel's gradients with respect to each of its
            hyperparameters.
        """
        diff = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]
        sq_diff = diff**2

        if self.isotropic:
            sum_sq = np.sum(sq_diff, axis=2)
            r = np.sqrt(np.maximum(sum_sq, 0.0)) / self.length_scale[0]
        else:
            r = np.sqrt(
                np.maximum(np.sum(sq_diff / self.length_scale**2, axis=2), 0.0),
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

        else:  # nu = 2.5
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
        diff = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]
        sq_diff = diff**2

        if self.isotropic:
            sum_sq = np.sum(sq_diff, axis=2)
            r = np.sqrt(np.maximum(sum_sq, 0.0)) / self.length_scale[0]
        else:
            r = np.sqrt(
                np.maximum(np.sum(sq_diff / self.length_scale**2, axis=2), 0.0),
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

        else:  # nu = 2.5
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
        """Returns the current hyperparameter values.

        Returns:
            Array of the kernel's 'length_scale' hyperparameter.
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
                "New Matérn Kernel Hyperparameters",
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
        ls_squared = self.length_scale**2

        dist_parts = []
        for i, var in enumerate(variable_names):
            ls_val = ls_squared[0] if self.isotropic else ls_squared[i]
            dist_parts.append(
                f"( {var} - {training_point[i]:.6e} )² / {ls_val:.6e}",
            )

        dist_sum = " + ".join(dist_parts)
        r_str = f"sqrt( {dist_sum} )"

        if self._nu == 1.5:
            c = np.sqrt(3.0)
            return (
                f"( {alpha:.6e} * ( 1 + {c:.6e} * {r_str} ) "
                f"* exp( -{c:.6e} * {r_str} ) )"
            )

        # nu = 2.5
        c = np.sqrt(5.0)
        c_sq = 5.0 / 3.0
        return (
            f"( {alpha:.6e} * ( 1 + {c:.6e} * {r_str} "
            f"+ {c_sq:.6e} * ( {dist_sum} ) ) "
            f"* exp( -{c:.6e} * {r_str} ) )"
        )

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
