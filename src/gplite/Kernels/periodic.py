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
from gplite._utils._data import (
    distribute_anisotropic_hyperparameters,
    expand_kernel_bounds,
)
from gplite._utils._types import Arrf64, f64
from gplite._utils._validation import (
    validate_anisotropic_hyperparameter,
    validate_anisotropic_hyperparameter_shape,
    validate_isotropic_hyperparameter,
    validate_multiple_anisotropic_hyperparameter_size,
    validate_set_params,
)
from gplite.Kernels._base import Kernel


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
        n_rows = x1.shape[0]
        n_cols = x2.shape[0]
        n_features = x1.shape[1]

        exponent = np.zeros((n_rows, n_cols))

        # Calculate the exponent dimension-by-dimension to avoid 3D allocations
        for dim in range(n_features):
            p_d = self.period[0] if self.isotropic else self.period[dim]
            l_d = (
                self.length_scale[0]
                if self.isotropic
                else self.length_scale[dim]
            )

            # Compute 1D absolute distance: shape (N, M)
            dist_d = np.abs(x1[:, dim : dim + 1] - x2[:, dim : dim + 1].T)

            sine_term = np.sin((np.pi / p_d) * dist_d)
            exponent += (sine_term * sine_term) / (l_d * l_d)

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
        _, gradients = self._compute_with_gradient(x1, x2)

        return gradients

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
        n_rows = x1.shape[0]
        n_cols = x2.shape[0]
        n_features = x1.shape[1]

        exponent = np.zeros((n_rows, n_cols))

        # pre-allocate gradient arrays based on isotropy
        if self.isotropic:
            grad_ls = np.zeros((n_rows, n_cols, 1))
            grad_p = np.zeros((n_rows, n_cols, 1))
        else:
            grad_ls = np.empty((n_rows, n_cols, n_features))
            grad_p = np.empty((n_rows, n_cols, n_features))

        for dim in range(n_features):
            p_d = self.period[0] if self.isotropic else self.period[dim]
            l_d = (
                self.length_scale[0]
                if self.isotropic
                else self.length_scale[dim]
            )

            # compute 1D absolute distance (N, M)
            dist_d = np.abs(x1[:, dim : dim + 1] - x2[:, dim : dim + 1].T)

            arg_val = (np.pi / p_d) * dist_d
            sin_val = np.sin(arg_val)
            cos_val = np.cos(arg_val)

            sin_squared = sin_val * sin_val
            exponent += sin_squared / (l_d * l_d)

            # compute partial derivatives (before multiplying by K)
            d_ls = 4.0 * sin_squared / (l_d**3)
            d_p = (4.0 * np.pi * dist_d * sin_val * cos_val) / (
                (l_d**2) * (p_d**2)
            )

            if self.isotropic:
                grad_ls[:, :, 0] += d_ls
                grad_p[:, :, 0] += d_p
            else:
                grad_ls[:, :, dim] = d_ls
                grad_p[:, :, dim] = d_p

        # compute the final K matrix
        K = np.exp(-2.0 * exponent)

        # multiply the partial derivatives by K in-place to complete the chain
        # rule.
        K_expanded = K[:, :, np.newaxis]
        grad_ls *= K_expanded
        grad_p *= K_expanded

        return K, (grad_ls, grad_p)

    def get_params(self) -> Arrf64:
        """
        Returns the current hyperparameters as a concatenated array.

        Returns:
            Arrf64: Array of [length_scale, period] values.
        """
        return np.concatenate([self.length_scale, self.period])

    def set_params(self, params: Arrf64, validate: bool = True) -> None:
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
        if validate:
            expected_num_hyperparameters = len(self.length_scale) + len(
                self.period
            )
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

    def _compute_diag(self, x: Arrf64) -> Arrf64:
        """
        Returns the diagonal of K(x, x). For the periodic kernel, k(x, x) = 1
        for all x since sin(0) = 0 makes the exponent zero.

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
