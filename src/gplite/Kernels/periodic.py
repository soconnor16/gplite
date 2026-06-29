"""Periodic kernel class for modeling repeating patterns in data.

The periodic kernel is defined as:
    K(x, x') = exp(-2 * Σᵢ sin²(π|xᵢ - x'ᵢ| / pᵢ) / lᵢ²)

where:
    - p is the period hyperparameter (repetition interval)
    - l is the length scale hyperparameter (smoothness within each period)

The gradients with respect to hyperparameters (length scale, period) are:
    ∂K/∂l = K * 4 * sin²(π|x - x'| / p) / l³

    ∂K/∂p = K * 4π|x - x'| * sin(π|x - x'| / p) * cos(π|x - x'| / p) / (l² * p²)

This kernel is ideal for data with known or learnable periodicity, such as
seasonal patterns, cyclical phenomena, or any repeating structures.

More detailed documentation about many methods below can be found in the Kernel
base class in 'Kernels/_base.py'.
"""

from typing import cast

import numpy as np

from gplite._utils._constants import EPSILON
from gplite._utils._data import (
    distribute_anisotropic_hyperparameters,
    resolve_bounds_shape,
)
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
    validate_multiple_anisotropic_hyperparameter_size,
    validate_set_params,
)
from gplite.Kernels._base import Kernel


class PeriodicKernel(Kernel):
    """Periodic kernel for modeling data with repeating patterns.

    The periodic kernel matrix is defined as:
        K(x, x') = exp(-2 * sum(sin(pi * |x - x'| / p)² / l²))
    where p is the period and l is the length scale.

    Attributes:
        bounds: A dictionary of kernel hyperparameter names and their current
            bounds.
        hyperparameters: A tuple of the kernel's hyperparameter names as
            strings.
        isotropic: A boolean indicating whether the kernel is isotropic.
        length_scale: The current value for the length_scale hyperparameter.
        period: The current value for the kernel's period hyperparameter.
    """

    length_scale: Arrf64
    period: Arrf64
    isotropic: bool

    def __init__(
        self,
        length_scale: NumericArray | NumericValue,
        period: NumericArray | NumericValue,
        isotropic: bool = True,
        bounds: KernelBounds | None = None,
    ) -> None:
        """Initializes a periodic kernel.

        Args:
            length_scale: Length scale hyperparameter controlling smoothness.
                Scalar for isotropic, array for anisotropic.
            period: Period hyperparameter defining the repetition interval.
                Scalar for isotropic, array for anisotropic.
            isotropic: If True, uses single length scale and period for all
                dimensions. Defaults to True.
            bounds: Custom hyperparameter bounds. Must be a dictionary where
                keysare the hyperparameter names (strings) and values are either
                a single tuple (min, max) or a list of tuples [(min, max), ...].
                For example: {"period": (3.1415, 6.2830)}. Defaults to None.

        Raises:
            ValidationError: If hyperparameter values are invalid or anisotropic
                hyperparameters have mismatched sizes.
        """
        self.length_scale = (
            validate_isotropic_hyperparameter(
                length_scale,
                "Periodic Length Scale",
            )
            if isotropic
            else validate_anisotropic_hyperparameter(
                length_scale,
                "Periodic Length Scale",
            )
        )
        self.period = (
            validate_isotropic_hyperparameter(period, "Periodic Period")
            if isotropic
            else validate_anisotropic_hyperparameter(period, "Periodic Period")
        )

        if not isotropic:
            validate_multiple_anisotropic_hyperparameter_size(
                [self.length_scale, self.period],
                ["Periodic Length Scale", "Periodic Period"],
            )
        self.isotropic = isotropic

        self._bound_config = {
            "length_scale": [(np.float64(1e-6), np.float64(5e2))],
            "period": [(np.float64(1e-6), np.float64(5e2))],
        }

        if bounds is not None:
            validated_bounds = validate_bounds_dict(
                bounds=bounds,
                expected_params=["length_scale", "period"],
                kernel_name="PeriodicKernel",
            )
            self._bound_config.update(validated_bounds)

    @property
    def hyperparameters(self) -> tuple[str, ...]:
        """Defines kernel hyperparameters.

        Returns:
            The names of the hyperparameters in a given kernel as a tuple of
            strings.
        """
        return ("length_scale", "period")

    @property
    def bounds(self) -> dict[str, list[tuple[f64, f64]]]:
        """Defines kernel bounds.

        Returns:
            A dictionary with the kernel's hyperparameter names and their
            respective bounds.
        """
        length_scale_dimensions = np.atleast_1d(self.length_scale).size
        period_dimensions = np.atleast_1d(self.period).size

        return {
            "length_scale": resolve_bounds_shape(
                self._bound_config["length_scale"],
                length_scale_dimensions,
                "length_scale",
            ),
            "period": resolve_bounds_shape(
                self._bound_config["period"],
                period_dimensions,
                "period",
            ),
        }

    @property
    def _bounds(self) -> list[tuple[f64, f64]]:
        """Exposes the bounds defined for the kernel hyperparameters internally.

        Returns:
            A flat list of tuples representing the bounds for a kernel's
            hyperparameters.
        """
        return self.bounds["length_scale"] + self.bounds["period"]

    def _compute(self, x1: Arrf64, x2: Arrf64) -> Arrf64:
        """Computes the similarity matrix of the kernel.

        Args:
            x1: First array of points used to compute the kernel matrix.
            x2: Second array of points used to compute the kernel matrix.

        Returns:
            Kernel covariance matrix calculated between x1 and x2.
        """
        n_rows = x1.shape[0]
        n_cols = x2.shape[0]
        n_features = x1.shape[1]

        exponent = np.zeros((n_rows, n_cols))

        # calculate the exponent dimension-by-dimension to avoid 3D allocations
        for dim in range(n_features):
            p_d = self.period[0] if self.isotropic else self.period[dim]
            l_d = (
                self.length_scale[0]
                if self.isotropic
                else self.length_scale[dim]
            )

            # compute 1D absolute distance: shape (N, M)
            dist_d = np.abs(x1[:, dim : dim + 1] - x2[:, dim : dim + 1].T)

            sine_term = np.sin((np.pi / p_d) * dist_d)
            exponent += (sine_term * sine_term) / (l_d * l_d)

        return np.exp(-2.0 * exponent)

    def _gradient(self, x1: Arrf64, x2: Arrf64) -> tuple[Arrf64, ...]:
        """Compute the kernel's gradient with respect to its hyperparameters.

        Args:
            x1: First array of points used to compute the kernel gradient.
            x2: Second array of points used to compute the kernel gradient.

        Returns:
            Tuple of a kernel's gradients with respect to each of its
            hyperparameters.
        """
        _, gradients = self._compute_with_gradient(x1, x2)

        return gradients

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
        """Returns the current hyperparameter values.

        Returns:
            Array of the kernel's hyperparameter values.
        """
        return np.concatenate([self.length_scale, self.period])

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
            expected_num_hyperparameters = len(self.length_scale) + len(
                self.period,
            )
            params = validate_set_params(
                params,
                "New Periodic Kernel Hyperparameters",
                self.isotropic,
                expected_num_hyperparameters,
            )

        params = cast("Arrf64", params)

        if self.isotropic:
            self.length_scale = params[:1]
            self.period = params[1:]
        else:
            self.length_scale, self.period = (
                distribute_anisotropic_hyperparameters(params, 2)
            )

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

        # precompute frequencies and amplitudes
        freqs = np.pi / self.period
        amplitudes = -2.0 / (self.length_scale**2)

        for i, var in enumerate(variable_names):
            freq = freqs[0] if self.isotropic else freqs[i]
            amp = amplitudes[0] if self.isotropic else amplitudes[i]
            tp = float(training_point[i])

            # handle different training point cases to save tokens
            # when possible
            if abs(tp) < EPSILON:
                inner = f"{freq:.6e}*{var}"
            elif tp < 0.0:
                inner = f"{freq:.6e}*({var}+{abs(tp):.6e})"
            else:
                inner = f"{freq:.6e}*({var}-{tp:.6e})"

            term = f"{amp:.6e}*sin({inner})^2"
            difference_parts.append(term)

        exponent_sum = "+".join(difference_parts)

        return f"{alpha:.6e}*exp({exponent_sum})"

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
            validate_anisotropic_hyperparameter_shape(x, self.period)
