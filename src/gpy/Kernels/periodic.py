"""
Periodic kernel implementation for modeling repeating patterns in data.

The periodic kernel is defined as:

    K(x, x') = exp(-2 * Σᵢ sin²(π|xᵢ - x'ᵢ| / pᵢ) / lᵢ²)

where:
    - p is the period hyperparameter (repetition interval)
    - l is the length scale hyperparameter (smoothness within each period)

The gradients with respect to hyperparameters are:

    ∂K/∂l = K * 4 * sin²(π|x - x'| / p) / l³

    ∂K/∂p = K * 4π|x - x'| * sin(π|x - x'| / p) * cos(π|x - x'| / p) / (l² * p²)

This kernel is ideal for data with known or learnable periodicity, such as
seasonal patterns, cyclical phenomena, or any repeating structures.
"""

import numpy as np

from gpy._utils._computation import compute_absolute_distance
from gpy._utils._data import (
    distribute_anisotropic_hyperparameters,
    expand_kernel_bounds,
)
from gpy._utils._types import Arrf64, f64
from gpy._utils._validation import (
    validate_anisotropic_hyperparameter,
    validate_anisotropic_hyperparameter_shape,
    validate_isotropic_hyperparameter,
    validate_multiple_anisotropic_hyperparameter_size,
    validate_set_params,
)
from gpy.Kernels._base import Kernel


class PeriodicKernel(Kernel):
    """
    Periodic kernel for modeling data with repeating patterns.
    K(x, x') = exp(-2 * sum(sin(pi * |x - x'| / p)^2 / l^2))

    where p is the period and l is the length scale.
    """

    def __init__(
        self,
        length_scale: Arrf64,
        period: Arrf64,
        isotropic: bool = True,
    ) -> None:
        """
        Initializes a periodic kernel with length scale and period parameters.

        Args:
            - length_scale (Arrf64): Length scale hyperparameter controlling
                                     smoothness. Scalar for isotropic, array
                                     for anisotropic.
            - period (Arrf64): Period hyperparameter defining the repetition
                               interval. Scalar for isotropic, array for
                               anisotropic.
            - isotropic (bool): If True, uses single length scale and period
                                for all dimensions. Defaults to True.

        Raises:
            ValidationError: If hyperparameters are invalid or anisotropic
                             parameters have mismatched sizes.
        """
        self.length_scale = (
            validate_isotropic_hyperparameter(
                length_scale, "Periodic Length Scale"
            )
            if isotropic
            else validate_anisotropic_hyperparameter(
                length_scale, "Periodic Length Scale"
            )
        )
        self.period = (
            validate_isotropic_hyperparameter(period, "Periodic period")
            if isotropic
            else validate_anisotropic_hyperparameter(period, "Periodic period")
        )

        if not isotropic:
            validate_multiple_anisotropic_hyperparameter_size(
                [self.length_scale, self.period],
                ["Periodic Length Scale", "Periodic period"],
            )
        self.isotropic = isotropic

    @property
    def hyperparameters(self) -> tuple[str, ...]:
        """
        Returns the names of the kernel's hyperparameters.

        Returns:
            tuple[str, ...]: Tuple containing 'length scale' and 'period'.
        """
        return ("length scale", "period")

    @property
    def bounds(self) -> list[tuple[f64, f64]]:
        """
        Returns the optimization bounds for hyperparameters.

        Returns:
            list[tuple[f64, f64]]: Bounds for length scale and period, both
                                   in range [1e-6, 5e2].
        """
        return [
            (np.float64(1e-6), np.float64(5e2)),
            (np.float64(1e-6), np.float64(5e2)),
        ]

    def _compute(self, x1: Arrf64, x2: Arrf64) -> Arrf64:
        """
        Computes the periodic kernel matrix between two input arrays.

        Args:
            - x1 (Arrf64): First input array of shape (n, d).
            - x2 (Arrf64): Second input array of shape (m, d).

        Returns:
            Arrf64: Kernel matrix of shape (n, m).
        """
        absolute_dist = compute_absolute_distance(x1, x2)

        sine_term = np.sin((np.pi / self.period) * absolute_dist)
        ls = self.length_scale
        scaled_sine_squared = (sine_term * sine_term) / (ls * ls)

        exponent = np.sum(scaled_sine_squared, axis=2)

        return np.exp(-2.0 * exponent)

    def _gradient(self, x1: Arrf64, x2: Arrf64) -> tuple[Arrf64, ...]:
        """
        Computes the gradient of the periodic kernel with respect to its
        hyperparameters.

        Args:
            - x1 (Arrf64): First input array of shape (n, d).
            - x2 (Arrf64): Second input array of shape (m, d).

        Returns:
            tuple[Arrf64, ...]: Tuple of gradient tensors for length scale
                                and period.
        """
        K = self._compute(x1, x2)
        absolute_dist = compute_absolute_distance(x1, x2)

        n_rows, n_cols = K.shape
        n_features = x1.shape[1]

        # determine output sizes based on parameters
        n_ls_params = self.length_scale.size
        n_p_params = self.period.size

        grad_length_scale = np.zeros((n_rows, n_cols, n_ls_params))
        grad_period = np.zeros((n_rows, n_cols, n_p_params))

        for dim in range(n_features):
            # use correct hyperparameters:
            # first (only) element in the array if isotropic is true
            # otherwise use the array corresponding to the iteration's dimension
            scale_d = (
                self.length_scale[0]
                if self.isotropic
                else self.length_scale[dim]
            )
            period_d = self.period[0] if self.isotropic else self.period[dim]

            # determine the indices for the gradient array
            scale_idx = 0 if self.isotropic else dim
            period_idx = 0 if self.isotropic else dim

            # calculate distance for this dimension
            dimension_distance = absolute_dist[:, :, dim]

            # common argument for trig functions for kernel gradient
            arg_val = np.pi * dimension_distance / period_d

            sin_val = np.sin(arg_val)
            cos_val = np.cos(arg_val)

            # dK/dl = K * 4 * sin²(...) / l³
            dK_dl = 4.0 * (sin_val**2) / (scale_d**3)
            # accumulates for isotropic, assigns per-dimension for anisotropic
            grad_length_scale[:, :, scale_idx] += K * dK_dl

            # dK/dp = K * 4π * dist * sin(...) * cos(...) / (l² * p²)
            numerator = 4.0 * np.pi * dimension_distance * sin_val * cos_val
            denominator = (scale_d**2) * (period_d**2)
            dK_dp = numerator / denominator
            # accumulates for isotropic, assigns per-dimension for anisotropic
            grad_period[:, :, period_idx] += K * dK_dp

        return grad_length_scale, grad_period

    def _compute_with_gradient(
        self, x1: Arrf64, x2: Arrf64
    ) -> tuple[Arrf64, tuple[Arrf64, ...]]:
        """
        Computes kernel matrix and gradients together for efficiency by
        reusing intermediate calculations.

        Args:
            - x1 (Arrf64): First input array of shape (n, d).
            - x2 (Arrf64): Second input array of shape (m, d).

        Returns:
            tuple[Arrf64, tuple[Arrf64, ...]]: Kernel matrix and tuple of
                gradient tensors.
        """
        # shared distance computation
        absolute_dist = compute_absolute_distance(x1, x2)

        ls = self.length_scale
        per = self.period

        # shared trig terms - computed once for K and gradients
        # broadcasting handles both isotropic (scalar) and anisotropic (d,)
        arg_val = (np.pi / per) * absolute_dist
        sin_val = np.sin(arg_val)
        cos_val = np.cos(arg_val)

        # compute K
        sin_squared = sin_val * sin_val
        scaled_sine_squared = sin_squared / (ls * ls)
        exponent = np.sum(scaled_sine_squared, axis=2)
        K = np.exp(-2.0 * exponent)

        # compute gradients (reusing sin_squared, sin_val, cos_val)
        K_expanded = K[:, :, np.newaxis]
        ls_cubed = ls * ls * ls

        # dK/dl = K * 4 * sin² / l³
        dK_dl_terms = (4.0 / ls_cubed) * sin_squared

        # dK/dp = K * 4π * dist * sin * cos / (l² * p²)
        ls_sq_per_sq = (ls * ls) * (per * per)
        dK_dp_terms = (
            (4.0 * np.pi / ls_sq_per_sq) * absolute_dist * sin_val * cos_val
        )

        if self.isotropic:
            # sum gradients across dimensions for single parameter
            grad_length_scale = np.sum(
                K_expanded * dK_dl_terms, axis=2, keepdims=True
            )
            grad_period = np.sum(
                K_expanded * dK_dp_terms, axis=2, keepdims=True
            )
        else:
            # keep dimensions separate for anisotropic
            grad_length_scale = K_expanded * dK_dl_terms
            grad_period = K_expanded * dK_dp_terms

        return K, (grad_length_scale, grad_period)

    def get_params(self) -> Arrf64:
        """
        Returns the current hyperparameters as a concatenated array.

        Returns:
            Arrf64: Array of [length_scale, period] values.
        """
        return np.concatenate([self.length_scale, self.period])

    def set_params(self, params: Arrf64) -> None:
        """
        Sets new hyperparameter values for the kernel.

        Args:
            - params (Arrf64): Flat array containing length scale values
                               followed by period values.

        Raises:
            ValidationError: If params contains invalid values or wrong size
                             for isotropic kernel.

        Warns:
            UserWarning: If anisotropic params have different length than
                         current hyperparameters.
        """
        expected_num_hyperparameters = len(self.length_scale) + len(self.period)
        params = validate_set_params(
            params,
            "New Periodic Kernel Hyperparameters",
            self.isotropic,
            expected_num_hyperparameters,
        )

        if self.isotropic:
            self.length_scale = params[:1]
            self.period = params[1:]
        else:
            self.length_scale, self.period = (
                distribute_anisotropic_hyperparameters(params, 2)
            )

        return

    def _to_str(
        self, variable_names: list[str], alpha: f64, training_point: Arrf64
    ) -> str:
        """
        Creates a string representation of the periodic kernel expression.

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
            # handle length scale and period, they should remain constant
            # if isotropic is true, and change with the dimension otherwise
            scale_squared = (
                length_scale_squared[0]
                if self.isotropic
                else length_scale_squared[i]
            )
            period = self.period[0] if self.isotropic else self.period[i]

            # prepare the sine squared term string:
            # sin( pi * (x - x') / p )^2 / l^2
            diff_term = f"( {var} - {training_point[i]:.6e} )"
            sine_term = f"sin( {np.pi:.6e} * {diff_term} / {period:.6e} )^2"
            term_str = f"{sine_term} / {scale_squared:.6e}"

            difference_parts.append(term_str)

        exponent_sum = " + ".join(difference_parts)

        return f"( {alpha:.6e} * exp( -2.0 * ( {exponent_sum} ) ) )"

    def _get_expanded_bounds(self) -> list[tuple[f64, f64]]:
        """
        Returns expanded bounds for anisotropic hyperparameters.

        Returns:
            list[tuple[f64, f64]]: Bounds list expanded to match total number
                                   of hyperparameters.
        """
        if self.isotropic:
            return self.bounds

        return expand_kernel_bounds(
            np.concatenate([self.length_scale, self.period]), self.bounds, 2
        )

    def _validate_anisotropic_hyperparameter_shape(self, x: Arrf64) -> None:
        """
        Validates that anisotropic hyperparameters match input dimensionality.

        Args:
            - x (Arrf64): Input data for shape reference.

        Raises:
            ValidationError: If hyperparameter length doesn't match number of
                             features.
        """
        if not self.isotropic:
            validate_anisotropic_hyperparameter_shape(x, self.length_scale)
            validate_anisotropic_hyperparameter_shape(x, self.period)

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
