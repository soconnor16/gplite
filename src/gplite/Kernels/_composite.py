"""Kernel classes for combining kernels through addition or multiplication.

Composite kernels are particularly useful for data that contains multiple
distinct patterns, such as periodicity with random noise.

Composite kernels can be constructed in two ways:

Additive Kernels (K_1 + K_2):
    K_sum(x, x') = K_1(x, x') + K_2(x, x')

    The resulting function can be seen as the sum of independent functions,
    each drawn from one of the component GPs. Useful for modeling functions
    with multiple additive components (e.g., trend + seasonality).

    Gradients are computed independently: ∂K_sum/∂θᵢ = ∂Kᵢ/∂θᵢ

Product Kernels (K_1 * K_2):
    K_prod(x, x') = K_1(x, x') * K_2(x, x')

    The product kernel models interactions between components. Useful when
    one pattern modulates another (e.g., varying amplitude over time).

    Gradients are combined via the product rule:
    ∂K_prod/∂θ₁ = (∂K₁/∂θ₁) * K₂
    ∂K_prod/∂θ₂ = K₁ * (∂K₂/∂θ₂)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from gplite._utils._errors import ValidationError
from gplite.Kernels._base import Kernel

if TYPE_CHECKING:
    from gplite._utils._types import Arrf64, NumericArray, NumericValue, f64


class CompositeKernel(Kernel):
    """Abstract base class for composite kernels.

    This provides shared functionality for managing child kernels and their
    hyperparameters to both additive and summation kernels.
    """

    def __init__(self, *kernels: Kernel) -> None:
        """Initializes a composite kernel with one or more child kernels.

        Args:
            kernels: Variable number of kernel instances to combine.

        Raises:
            ValidationError: If any operand is not a valid Kernel instance.
        """
        self.kernels = list(kernels)
        self._validate_kernels()

    def _validate_kernels(self) -> None:
        """Validates that all child kernels are valid Kernel instances.

        Raises:
            ValidationError: If any kernel is not a Kernel instance.
        """
        for k in self.kernels:
            if not isinstance(k, Kernel):
                err_msg = "Error: All operands must be Kernel instances"
                raise ValidationError(err_msg)

    @property
    def bounds(self) -> dict[str, list[tuple[f64, f64]]]:
        """Returns a combined dictionary of bounds from all child kernels.

        Composite kernels may be comprised of two or more separate kernels
        of the same kernel class. To maintain readability, each kernel is
        given a unique name determined by their index in the internal kernel
        list.

        Returns:
            Dictionary of combined hyperparameter bounds.
        """
        composite_bounds = {}

        for i, k in enumerate(self.kernels):
            for param_name, bound_list in k.bounds.items():
                # creates unique keys like: "kernel_0_length_scale" in case
                # there are multiple of the same kind of kernel in a composite
                # kernel
                composite_bounds[f"kernel_{i}_{param_name}"] = bound_list

        return composite_bounds

    @property
    def _bounds(self) -> list[tuple[f64, f64]]:
        """Exposes a flat bound list defined for the kernel hyperparameters."""
        all_bounds = []

        for k in self.kernels:
            all_bounds.extend(k._bounds)

        return all_bounds

    @property
    def hyperparameters(self) -> tuple[str, ...]:
        """Returns the concatenated hyperparameter names from all child kernels.

        Returns:
            A combined tuple of hyperparameter names.
        """
        all_hyperparameters = []

        for k in self.kernels:
            all_hyperparameters.extend(k.hyperparameters)

        return tuple(all_hyperparameters)

    def get_params(self) -> Arrf64:
        """Returns the concatenated hyperparameters from all child kernels.

        Returns:
            Flat array of all hyperparameter values.
        """
        params_list = [k.get_params() for k in self.kernels]

        return np.concatenate(params_list)

    def set_params(
        self,
        params: NumericArray | NumericValue,
        _validate: bool = True,
    ) -> None:
        """Distributes and sets hyperparameters to each child kernel.

        Args:
            params: Flat array of hyperparameters to distribute across child
                kernels.
            _validate: Whether to validate the hyperparameters before setting
                them. This is intended to be used for internal usage such as
                optimization loops where skipping the small overhead from
                validation saves a lot of time. If _validate is false, it is
                assumed you know what you are doing. Defaults to True.
        """
        if _validate:
            if not hasattr(params, "__getitem__") and not isinstance(
                params,
                (int, float, np.integer, np.floating),
            ):
                err_msg = (
                    "Error: Composite kernels require an array-like object "
                    "of hyperparameters, but received type "
                    f"'{type(params).__name__}'. Ensure you are passing a list,"
                    " tuple, or numpy array."
                )
                raise ValidationError(err_msg)

            params = np.atleast_1d(np.asarray(params, dtype=np.float64))

        params = cast("NumericArray", params)
        idx = 0

        for k in self.kernels:
            num_params = len(k.get_params())
            kernel_params = params[idx : idx + num_params]
            k.set_params(kernel_params, _validate)
            idx += num_params

    def _to_str(
        self,
        variable_names: list[str],
        alpha: f64,
        training_point: Arrf64,
    ) -> str:
        """Creates a string representation of the composite kernel expression.

        Args:
            variable_names: Names of input variables.
            alpha: Weight coefficient for the expression.
            training_point: Training point to center expression on.

        Returns:
            Mathematical expression string for the composite kernel.

        Raises:
            NotImplementedError: If called on an unknown composite kernel type.
        """
        # pass alpha=1.0 to child kernels, real alpha is added to the whole
        # expression after
        parts = [
            k._to_str(variable_names, np.float64(1.0), training_point)
            for k in self.kernels
        ]

        if isinstance(self, AdditiveKernel):
            combined_parts = " + ".join(parts)
        elif isinstance(self, ProductKernel):
            combined_parts = " * ".join(parts)
        else:
            err_msg = "Error: Unknown composite kernel type"
            raise NotImplementedError(err_msg)

        return f"( {alpha:.6e} * ( {combined_parts} ) )"

    def _validate_anisotropic_hyperparameter_shape(self, x: Arrf64) -> None:
        """Validates anisotropic hyperparameter shapes for all child kernels.

        Args:
            x: Input data array used for shape validation.
        """
        for k in self.kernels:
            k._validate_anisotropic_hyperparameter_shape(x)

    def _validate_input_data(
        self,
        x1: NumericArray,
        x2: NumericArray,
        name1: str,
        name2: str,
    ) -> tuple[Arrf64, Arrf64]:
        """Validates input data before it is used for computation.

        Args:
            x1: First input array.
            x2: Second input array.
            name1: Name of first array for error messages.
            name2: Name of second array for error messages.

        Returns:
            A tuple of the validated inputs.
        """
        return self.kernels[0]._validate_input_data(x1, x2, name1, name2)


class AdditiveKernel(CompositeKernel):
    """Composite kernel representing the sum of multiple kernels.

    Additive kernels are defined as:
        K_additive(x, x') = K1(x, x') + K2(x, x') + ...
    """

    def _compute_diag(self, x: Arrf64) -> Arrf64:
        """Computes the diagonal of the additive kernel matrix.

        Additive kernel diagonals are computed via a of summation of each of its
            child kernels' diagonals.

        Args:
            x: Input array of shape (n, d).

        Returns:
            Sum of child kernel diagonals.
        """
        diag = self.kernels[0]._compute_diag(x)

        for k in self.kernels[1:]:
            diag = diag + k._compute_diag(x)

        return diag

    def _compute(self, x1: Arrf64, x2: Arrf64) -> Arrf64:
        """Computes the sum of kernel matrices from all child kernels.

        Args:
            x1: First input array.
            x2: Second input array.

        Returns:
            Sum of child kernel matrices.
        """
        kernel_matrix = self.kernels[0]._compute(x1, x2)

        for k in self.kernels[1:]:
            kernel_matrix += k._compute(x1, x2)

        return kernel_matrix

    def _gradient(self, x1: Arrf64, x2: Arrf64) -> tuple[Arrf64, ...]:
        """Computes gradients from all child kernels.

        Additive kernel gradients are independent under addition and are
            computed via a summation of the child kernel gradients.

        Args:
            x1: First input array.
            x2: Second input array.

        Returns:
            Concatenated gradients from all child kernels.
        """
        all_grads = []

        for k in self.kernels:
            all_grads.extend(k._gradient(x1, x2))

        return tuple(all_grads)

    def _compute_with_gradient(
        self,
        x1: Arrf64,
        x2: Arrf64,
    ) -> tuple[Arrf64, tuple[Arrf64, ...]]:
        """Computes kernel matrix and gradients together for efficiency.

        Args:
            x1: First input array.
            x2: Second input array.

        Returns:
            Composite kernel matrix and gradients.
        """
        results = [k._compute_with_gradient(x1, x2) for k in self.kernels]

        k_matrices, grad_list = zip(*results, strict=True)

        k_total = k_matrices[0].copy()
        for k in k_matrices[1:]:
            k_total += k

        all_grads = []
        for grad_tuple in grad_list:
            all_grads.extend(grad_tuple)

        return k_total, tuple(all_grads)

    def __add__(self, other: Kernel) -> AdditiveKernel:
        """Adds another kernel to this additive kernel.

        Args:
            other: Kernel to add.

        Returns:
            New additive kernel containing all component kernels.
        """
        # flatten: (A + B) + C -> AdditiveKernel(A, B, C)
        if isinstance(other, AdditiveKernel):
            return AdditiveKernel(*(self.kernels + other.kernels))
        return AdditiveKernel(*([*self.kernels, other]))


class ProductKernel(CompositeKernel):
    """Composite kernel representing the product of multiple kernels.

    Product kernels are defined as:
        K_prod(x, x') = K1(x, x') * K2(x, x') * ...
    """

    def _compute_diag(self, x: Arrf64) -> Arrf64:
        """Computes the diagonal of the product kernel matrix.

        Product kernels  diagonals are computed via the element-wise product of
            each of the child kernels' diagonals.

        Args:
            x: Input array of shape (n, d).

        Returns:
            Product of child kernel diagonals as a flat array.
        """
        diag = self.kernels[0]._compute_diag(x)

        for k in self.kernels[1:]:
            diag = diag * k._compute_diag(x)

        return diag

    def _compute(self, x1: Arrf64, x2: Arrf64) -> Arrf64:
        """Computes the product of kernel matrices from all child kernels.

        Args:
            x1: First input array.
            x2: Second input array.

        Returns:
            Product of child kernel matrices.
        """
        result = self.kernels[0]._compute(x1, x2)

        for k in self.kernels[1:]:
            result *= k._compute(x1, x2)

        return result

    def _gradient(self, x1: Arrf64, x2: Arrf64) -> tuple[Arrf64, ...]:
        """Computes gradients using the product rule across all child kernels.

        Args:
            x1: First input array.
            x2: Second input array.

        Returns:
            Tuple of gradients scaled by product of other kernels.
        """
        # product rule: d(ABC)/dθ = (dA * BC) + (dB * AC) + ...
        results = [k._compute_with_gradient(x1, x2) for k in self.kernels]
        k_matrices, grad_list = zip(*results, strict=True)
        all_grads = []
        for i, k_grads in enumerate(grad_list):
            k_other = np.ones_like(k_matrices[0])
            for j, k_matrix in enumerate(k_matrices):
                if i != j:
                    k_other *= k_matrix
            scaled_grads = [g * k_other[..., np.newaxis] for g in k_grads]
            all_grads.extend(scaled_grads)

        return tuple(all_grads)

    def _compute_with_gradient(
        self,
        x1: Arrf64,
        x2: Arrf64,
    ) -> tuple[Arrf64, tuple[Arrf64, ...]]:
        """Computes kernel matrix and gradients together for efficiency.

        Args:
            x1: First input array.
            x2: Second input array.

        Returns:
            Composite kernel matrix and gradients.
        """
        results = [k._compute_with_gradient(x1, x2) for k in self.kernels]

        k_matrices, grad_list = zip(*results, strict=True)

        k_total = k_matrices[0].copy()
        for k in k_matrices[1:]:
            k_total *= k

        all_grads = []

        for i, k_grads in enumerate(grad_list):
            k_other = np.ones_like(k_matrices[0])

            for j, k_matrix in enumerate(k_matrices):
                if i != j:
                    k_other *= k_matrix

            scaled_grads = [grad * k_other[..., np.newaxis] for grad in k_grads]
            all_grads.extend(scaled_grads)

        return k_total, tuple(all_grads)

    def __mul__(self, other: Kernel) -> ProductKernel:
        """Multiplies another kernel with this product kernel.

        Args:
            other: Kernel to multiply.

        Returns:
            New product kernel containing all component kernels.
        """
        # flatten: (A * B) * C -> ProductKernel(A, B, C)
        if isinstance(other, ProductKernel):
            return ProductKernel(*(self.kernels + other.kernels))
        return ProductKernel(*([*self.kernels, other]))
