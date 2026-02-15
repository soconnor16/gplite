"""
Composite kernel classes for combining multiple kernels through addition
or multiplication.

Kernels can be combined to create more expressive covariance functions:

Additive Kernels (K₁ + K₂):
    K_sum(x, x') = K₁(x, x') + K₂(x, x')

    The resulting function can be seen as the sum of independent functions,
    each drawn from one of the component GPs. Useful for modeling functions
    with multiple additive components (e.g., trend + seasonality).

    Gradients are computed independently: ∂K_sum/∂θᵢ = ∂Kᵢ/∂θᵢ

Product Kernels (K₁ * K₂):
    K_prod(x, x') = K₁(x, x') * K₂(x, x')

    The product kernel models interactions between components. Useful when
    one pattern modulates another (e.g., varying amplitude over time).

    Gradients use the product rule:
    ∂K_prod/∂θ₁ = (∂K₁/∂θ₁) * K₂
    ∂K_prod/∂θ₂ = K₁ * (∂K₂/∂θ₂)
"""

import numpy as np

from gpy._utils._errors import ValidationError
from gpy._utils._types import Arrf64, f64
from gpy.Kernels._base import Kernel


class CompositeKernel(Kernel):
    """
    Abstract base class for composite kernels that combine multiple kernels.
    Provides common functionality for managing child kernels and their
    hyperparameters.
    """

    def __init__(self, *kernels: Kernel) -> None:
        """
        Initializes a composite kernel with one or more child kernels.

        Args:
            - *kernels (Kernel): Variable number of kernel instances to combine.

        Raises:
            ValidationError: If any operand is not a valid Kernel instance.
        """
        self.kernels = list(kernels)
        self._validate_kernels()

    def _validate_kernels(self) -> None:
        """
        Validates that all child kernels are valid Kernel instances.

        Raises:
            ValidationError: If any kernel is not a Kernel instance.
        """
        for k in self.kernels:
            if not isinstance(k, Kernel):
                err_msg = "Error: All operands must be Kernel instances"
                raise ValidationError(err_msg)

    @property
    def bounds(self) -> list[tuple[f64, f64]]:
        """
        Returns the concatenated bounds from all child kernels.

        Returns:
            list[tuple[f64, f64]]: Combined list of hyperparameter bounds.
        """
        all_bounds = []

        for k in self.kernels:
            all_bounds.extend(k.bounds)

        return all_bounds

    @property
    def hyperparameters(self) -> tuple[str, ...]:
        """
        Returns the concatenated hyperparameter names from all child kernels.

        Returns:
            tuple[str, ...]: Combined tuple of hyperparameter names.
        """
        all_hyperparameters = []

        for k in self.kernels:
            all_hyperparameters.extend(k.hyperparameters)

        return tuple(all_hyperparameters)

    def get_params(self) -> Arrf64:
        """
        Returns the concatenated hyperparameters from all child kernels.

        Returns:
            Arrf64: Flat array of all hyperparameter values.
        """
        params_list = [k.get_params() for k in self.kernels]

        return np.concatenate(params_list)

    def set_params(self, params: Arrf64) -> None:
        """
        Distributes and sets hyperparameters to each child kernel.

        Args:
            - params (Arrf64): Flat array of hyperparameters to distribute
                               across child kernels.
        """
        idx = 0

        for k in self.kernels:
            num_params = len(k.get_params())
            kernel_params = params[idx : idx + num_params]
            k.set_params(kernel_params)
            idx += num_params

        return

    def _get_expanded_bounds(self) -> list[tuple[f64, f64]]:
        """
        Returns expanded bounds from all child kernels for optimization.

        Returns:
            list[tuple[f64, f64]]: Combined expanded bounds list.
        """
        all_expanded_bounds = []

        for k in self.kernels:
            all_expanded_bounds.extend(k._get_expanded_bounds())

        return all_expanded_bounds

    def _to_str(
        self, variable_names: list[str], alpha: f64, training_point: Arrf64
    ) -> str:
        """
        Creates a string representation of the composite kernel expression.

        Args:
            - variable_names (list[str]): Names of input variables.
            - alpha (f64): Weight coefficient for the expression.
            - training_point (Arrf64): Training point to center expression on.

        Returns:
            str: Mathematical expression string for the composite kernel.

        Raises:
            NotImplementedError: If called on an unknown composite kernel type.
        """
        # pass alpha=1.0 to child kernels, real alpha is added to the whole
        # expression after
        parts = [
            k._to_str(variable_names, f64(1.0), training_point)
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
        """
        Validates anisotropic hyperparameter shapes for all child kernels.

        Args:
            - x (Arrf64): Input data array used for shape validation.
        """
        for k in self.kernels:
            k._validate_anisotropic_hyperparameter_shape(x)

    def _validate_input_data(
        self, x1: Arrf64, x2: Arrf64, name1: str, name2: str
    ) -> tuple[Arrf64, Arrf64]:
        """
        Validates input data using the first child kernel's validation.

        Args:
            - x1 (Arrf64): First input array.
            - x2 (Arrf64): Second input array.
            - name1 (str): Name of first array for error messages.
            - name2 (str): Name of second array for error messages.

        Returns:
            tuple[Arrf64, Arrf64]: Validated input arrays.
        """
        return self.kernels[0]._validate_input_data(x1, x2, name1, name2)


class AdditiveKernel(CompositeKernel):
    """
    Composite kernel representing the sum of multiple kernels.
    K_sum(x, x') = K1(x, x') + K2(x, x') + ...
    """

    def _compute(self, x1: Arrf64, x2: Arrf64) -> Arrf64:
        """
        Computes the sum of kernel matrices from all child kernels.

        Args:
            - x1 (Arrf64): First input array.
            - x2 (Arrf64): Second input array.

        Returns:
            Arrf64: Sum of kernel matrices.
        """
        kernel_matrix = self.kernels[0]._compute(x1, x2)

        for k in self.kernels[1:]:
            kernel_matrix += k._compute(x1, x2)

        return kernel_matrix

    def _gradient(self, x1: Arrf64, x2: Arrf64) -> tuple[Arrf64, ...]:
        """
        Computes gradients from all child kernels. Gradients are independent
        under addition.

        Args:
            - x1 (Arrf64): First input array.
            - x2 (Arrf64): Second input array.

        Returns:
            tuple[Arrf64, ...]: Concatenated gradients from all child kernels.
        """
        # gradients are independent under addition
        all_grads = []

        for k in self.kernels:
            all_grads.extend(k._gradient(x1, x2))

        return tuple(all_grads)

    def _compute_with_gradient(
        self, x1: Arrf64, x2: Arrf64
    ) -> tuple[Arrf64, tuple[Arrf64, ...]]:
        """
        Computes kernel matrix and gradients together for efficiency.

        Args:
            - x1 (Arrf64): First input array.
            - x2 (Arrf64): Second input array.

        Returns:
            tuple[Arrf64, tuple[Arrf64, ...]]: Kernel matrix sum and
                concatenated gradients.
        """
        results = [k._compute_with_gradient(x1, x2) for k in self.kernels]

        k_matrices, grad_list = zip(*results)

        k_total = k_matrices[0].copy()
        for k in k_matrices[1:]:
            k_total += k

        all_grads = []
        for grad_tuple in grad_list:
            all_grads.extend(grad_tuple)

        return k_total, tuple(all_grads)

    def __add__(self, other: Kernel) -> "AdditiveKernel":
        """
        Adds another kernel to this additive kernel, flattening the structure.

        Args:
            - other (Kernel): Kernel to add.

        Returns:
            AdditiveKernel: New additive kernel containing all component
                            kernels.
        """
        # flatten: (A + B) + C -> AdditiveKernel(A, B, C)
        if isinstance(other, AdditiveKernel):
            return AdditiveKernel(*(self.kernels + other.kernels))
        return AdditiveKernel(*(self.kernels + [other]))


class ProductKernel(CompositeKernel):
    """
    Composite kernel representing the product of multiple kernels.
    K_prod(x, x') = K1(x, x') * K2(x, x') * ...
    """

    def _compute(self, x1: Arrf64, x2: Arrf64) -> Arrf64:
        """
        Computes the element-wise product of kernel matrices from all child
        kernels.

        Args:
            - x1 (Arrf64): First input array.
            - x2 (Arrf64): Second input array.

        Returns:
            Arrf64: Product of kernel matrices.
        """
        result = self.kernels[0]._compute(x1, x2)

        for k in self.kernels[1:]:
            result *= k._compute(x1, x2)

        return result

    def _gradient(self, x1: Arrf64, x2: Arrf64) -> tuple[Arrf64, ...]:
        """
        Computes gradients using the product rule across all child kernels.
        d(ABC)/d_param = (dA * BC) + (dB * AC) + ...

        Args:
            - x1 (Arrf64): First input array.
            - x2 (Arrf64): Second input array.

        Returns:
            tuple[Arrf64, ...]: Gradients scaled by product of other kernels.
        """
        # product rule: d(ABC)/dθ = (dA * BC) + (dB * AC) + ...
        k_matrices = [k._compute(x1, x2) for k in self.kernels]

        all_grads = []

        for i, k in enumerate(self.kernels):
            # compute product of all other kernels
            k_other = np.ones_like(k_matrices[0])

            for j, k_matrix in enumerate(k_matrices):
                if i != j:
                    k_other *= k_matrix

            k_grads = k._gradient(x1, x2)

            # apply product rule: grad * (everything else)
            scaled_grads = [g * k_other[..., np.newaxis] for g in k_grads]

            all_grads.extend(scaled_grads)

        return tuple(all_grads)

    def _compute_with_gradient(
        self, x1: Arrf64, x2: Arrf64
    ) -> tuple[Arrf64, tuple[Arrf64, ...]]:
        """
        Computes kernel product and gradients together for efficiency.

        Args:
            - x1 (Arrf64): First input array.
            - x2 (Arrf64): Second input array.

        Returns:
            tuple[Arrf64, tuple[Arrf64, ...]]: Kernel matrix product and
                product-rule scaled gradients.
        """
        results = [k._compute_with_gradient(x1, x2) for k in self.kernels]

        k_matrices, grad_list = zip(*results)

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

    def __mul__(self, other: Kernel) -> "ProductKernel":
        """
        Multiplies another kernel with this product kernel, flattening the
        structure.

        Args:
            - other (Kernel): Kernel to multiply.

        Returns:
            ProductKernel: New product kernel containing all component kernels.
        """
        # flatten: (A * B) * C -> ProductKernel(A, B, C)
        if isinstance(other, ProductKernel):
            return ProductKernel(*(self.kernels + other.kernels))
        return ProductKernel(*(self.kernels + [other]))
