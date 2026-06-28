"""Abstract base class defining the kernel interface.

Kernels (covariance functions) are the core of Gaussian Process regression.
A kernel k(x, x') measures the similarity between two input points and
determines the shape and smoothness of functions the GP can represent.

Key properties of valid kernels:
    - Symmetric: k(x, x') = k(x', x)
    - Positive semi-definite: any kernel matrix K where Kᵢⱼ = k(xᵢ, xⱼ)
      must have non-negative eigenvalues

The kernel matrix K(X, X) captures the covariance structure of the training
data and is central to GP predictions:
    - K(X, X): covariance between training points (n × n)
    - K(X*, X): covariance between test and training points (m × n)
    - K(X*, X*): covariance between test points (m × m)
"""

from abc import ABC, abstractmethod

import numpy as np

from gplite._utils._types import Arrf64, NumericArray, NumericValue, f64
from gplite._utils._validation import validate_input_arrays


class Kernel(ABC):
    """Kernel base class to define the general kernel interface.

    Each kernel that follows this structure represents a fundamentally prior
    assumption that could be made about input data. The covariance matrix
    (accessed via the 'kernel.compute()' method) is most efficient at
    describing the relationships found in certain data patterns (e.g, periodic
    data, data with an underlying function mixed with random noise, etc.),
    though many kernels can fit to many types of data. Kernel choice is
    important for efficiency and as such, it is usually helpful to visualize
    data prior to picking your kernel.

    Attributes:
        bounds: A dictionary of kernel hyperparameter names and their current
            bounds.
        hyperparameters: A tuple of the kernel's hyperparameter names as
            strings.
        isotropic: A boolean indicating whether the kernel is isotropic.
    """

    @property
    @abstractmethod
    def hyperparameters(self) -> tuple[str, ...]:
        """Defines kernel hyperparameters.

        Method in which kernels define their hyperparameters and the order in
        which they are expected during initialization, or when new
        hyperparameters are set for an existing kernel.

        Returns:
            The names of the hyperparameters in a given kernel as a tuple of
            strings.
        """

    @property
    @abstractmethod
    def bounds(self) -> dict[str, list[tuple[f64, f64]]]:
        """Defines kernel bounds.

        Method in which kernels expose the bounds defined for their
        hyperparameters. Custom hyperparameter bounds can be passed during
        initialization. If they are not, the default bounds are returned here.

        Returns:
            A dictionary with the kernel's hyperparameter names and their
            respective bounds.
        """

    # ----------------------------- PUBLIC METHODS --------------------------- #
    def compute(self, x1: NumericArray, x2: NumericArray) -> Arrf64:
        """Computes the kernel's similarity matrix between two input arrays.

        Kernel functions are the core of Gaussian Process Regression (GPR).
        They are used to define the correlation (or "similarity") between input
        points which is necessary for predicting the output of new points and
        the shape of the predictive output function.

        Input arrays must have the same number of features (columns), but can
        have different numbers of samples (rows).
        Given an x1 with shape (n_samples1, n_features) and x2 with shape
        (n_samples2, n_features), the resulting covariance matrix K(x1, x2)
        will have shape (n_samples1, n_samples2)


        Args:
            x1: First array of points used to compute the kernel matrix.
            x2: Second array of points used to compute the kernel matrix.

        Returns:
            Kernel covariance matrix calculated between x1 and x2.

        Raises:
            ValidationError: If the input values are invalid (mismatched shapes,
                incorrect types, non-numeric values, etc.).
        """
        name = self.__class__.__name__
        x1, x2 = self._validate_input_data(
            x1,
            x2,
            f"{name} Compute Input 1",
            f"{name} Compute Input 2",
        )

        self._validate_anisotropic_hyperparameter_shape(x1)

        return self._compute(x1, x2)

    def gradient(
        self,
        x1: NumericArray,
        x2: NumericArray,
    ) -> tuple[Arrf64, ...]:
        """Compute a kernel's gradient with respect to its hyperparameters.

        Kernel gradients are primarily leveraged during hyperparameter
        optimization a gradient based optimization methods are sped up
        significantly when passed analytical gradient values.

        Args:
            x1: First array of points used to compute the kernel gradient.
            x2: Second array of points used to compute the kernel gradient.

        Returns:
            Tuple of a kernel's gradients with respect to each of its
            hyperparameters.

        Raises:
            ValidationError: If the input values are invalid (mismatched shapes,
                incorrect types, non-numeric values, etc.).
        """
        name = self.__class__.__name__
        x1, x2 = self._validate_input_data(
            x1,
            x2,
            f"{name} Gradient Input 1",
            f"{name} Gradient Input 2",
        )

        self._validate_anisotropic_hyperparameter_shape(x1)

        return self._gradient(x1, x2)

    def compute_with_gradient(
        self,
        x1: NumericArray,
        x2: NumericArray,
    ) -> tuple[Arrf64, tuple[Arrf64, ...]]:
        """Computes the kernel matrix and its gradients.

        During hyperparameter optimization, both the kernel matrix and its
        gradients are required repeatedly. This function computes both at the
        same time to save computational cost by reusing expensive calculations
        needed by both the kernel matrix and gradient computation.

        Args:
            x1: First input array of shape (n, d).
            x2: Second input array of shape (m, d).

        Returns:
            Tuple containing the kernel matrix K of shape (n, m) and a
            tuple of gradient tensors.

        Raises:
            ValidationError: If the input values are invalid (mismatched shapes,
                incorrect types, non-numeric values, etc.).
        """
        name = self.__class__.__name__
        x1, x2 = self._validate_input_data(
            x1,
            x2,
            f"{name} Compute with Gradient Input 1",
            f"{name} Compute with Gradient Input 2",
        )
        self._validate_anisotropic_hyperparameter_shape(x1)

        return self._compute_with_gradient(x1, x2)

    @abstractmethod
    def get_params(self) -> Arrf64:
        """Gets the current kernel hyperparameters.

        Returns:
            Current kernel hyperparameters as a flat array, hyperparameter
            values are returned in the order given in the kernel's
                'hyperparameters' property.
        """

    @abstractmethod
    def set_params(
        self,
        params: NumericArray | NumericValue,
        _validate: bool,
    ) -> None:
        """Method to set new hyperparameter values for the kernel.

        Hyperparameter values should be passed as a flat array in the order
        consistent with the kernel's 'hyperparameters' property.

        Args:
            params: New hyperparameter values to be set for the kernel.
            _validate: Whether to validate the hyperparameters before setting
                them. This is intended to be used for internal usage such as
                optimization loops where skipping the small overhead from
                validation saves a lot of time. If _validate is false, it is
                assumed you know what you are doing. Defaults to True.
        """

    # ---------------------------- PRIVATE METHODS --------------------------- #
    @property
    @abstractmethod
    def _bounds(self) -> list[tuple[f64, f64]]:
        """Exposes the bounds defined for the kernel hyperparameters internally.

        The public 'bounds()' method is designed for exposing bounds for user
        access. this private method is for exposing bounds during optimization,
        where they will be passed to scipy.

        Returns:
            A flat list of tuples representing the bounds for a kernel's
            hyperparameters.
        """

    @abstractmethod
    def _compute(self, x1: Arrf64, x2: Arrf64) -> Arrf64:
        """Computes the similarity matrix of a kernel.

        The public "compute" method calls this function with added data
        validation, while this method is called internally when data is known
        to be validated and time can be saved by avoiding the validation
        overhead.

        Args:
            x1: First array of points used to compute the kernel matrix.
            x2: Second array of points used to compute the kernel matrix.

        Returns:
            Kernel covariance matrix calculated between x1 and x2.
        """

    @abstractmethod
    def _gradient(self, x1: Arrf64, x2: Arrf64) -> tuple[Arrf64, ...]:
        """Computes the kernel's gradient.

        The public 'gradient' method calls this function with added data
        validation, while this method is called internally when data is known
        to be validated and time can be saved by avoiding the validation
        overhead.

        Args:
            x1: First array of points used to compute the kernel gradient.
            x2: Second array of points used to compute the kernel gradient.

        Returns:
            Tuple of a kernel's gradients with respect to each of its
            hyperparameters.
        """

    @abstractmethod
    def _compute_with_gradient(
        self,
        x1: Arrf64,
        x2: Arrf64,
    ) -> tuple[Arrf64, tuple[Arrf64, ...]]:
        """Computes the kernel's matrix and gradients together.

        The public 'compute_with_gradient' method calls this function with added
        data validation, while this method is called internally when data is
        known to be validated and time can be saved by avoiding the validation
        overhead.

        Args:
            x1: First input array of shape (n, d).
            x2: Second input array of shape (m, d).

        Returns:
            Tuple containing the kernel matrix K of shape (n, m) and a
            tuple of gradient tensors.
        """

    @abstractmethod
    def _to_str(
        self,
        variable_names: list[str],
        alpha: f64,
        training_point: Arrf64,
    ) -> str:
        """Creates a string representation of a kernel at a single data point.

        This is a utility function for the creation of larger string
        representations of a the full kernel function. The public api for this
        full representation can be found at the 'GaussianProcess.learn()'
        method.

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

    @abstractmethod
    def _validate_anisotropic_hyperparameter_shape(self, x: Arrf64) -> None:
        """Validates the shapes of anisotropic hyperparameters.

        This is a utility function to validate whether the input data used for
        kernel matrix calculation has consistent dimensionality with the current
        kernel hyperparameters (i.e., checks that the number of data features
        and the number of hyperparameters are the same).

        Args:
            x: Input data array to be validated.
        """

    def _compute_diag(self, x: Arrf64) -> Arrf64:
        """Computes the diagonal of the kernel matrix K(x, x).

        This default implementation computes the full matrix and extracts
        the diagonal. Subclasses should override this for efficiency when
        the diagonal has a known closed form (e.g., stationary kernels
        always return ones).

        Args:
            x: Input array of shape (n, d).

        Returns:
            The diagonal of K(x, x) as a flat array.
        """
        return np.diag(self._compute(x, x))

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
        return validate_input_arrays(x1, name1, x2, name2)

    # ----------------------------- MAGIC METHODS ---------------------------- #
    def __call__(
        self,
        x1: NumericArray,
        x2: NumericArray | None = None,
    ) -> Arrf64:
        """Compute the kernel matrix K(x1, x2).

        This method allows for kernels to be called directly (e.g.,
        kernel(x1,x2) as opposed to kernel.compute(x1,x2)). If only passed one
        array, it is assumed that the kernel matrix to be computed is that of
        the given array and itself.

        Args:
            x1: First array of points to compute the kernel matrix with.
            x2: Second array of points to compute the kernel matrix with.
                Defaults to None.

        Returns:
            The kernel matrix K(x1, x2).
        """
        if x2 is None:
            return self.compute(x1, x1)

        return self.compute(x1, x2)

    def __add__(self, other: "Kernel") -> "Kernel":
        """Returns a composite kernel as the sum of this kernel and another.

        This is equivalent to K_sum(x, x') = K(x, x') + K_other(x, x').

        Args:
            other: The kernel to add.

        Returns:
            The composite sum kernel.
        """
        # import locally to avoid circular import crashes
        from gplite.Kernels._composite import AdditiveKernel

        if isinstance(self, AdditiveKernel):
            return self.__add__(other)

        return AdditiveKernel(self, other)

    def __mul__(self, other: "Kernel") -> "Kernel":
        """Returns a composite kernel as the product of this kernel and another.

        This is equivalent to K_prod(x, x') = K(x, x') * K_other(x, x').

        Args:
            other: The kernel to multiply

        Returns:
            The composite product kernel.
        """
        # import locally to avoid circular import crashes
        from gplite.Kernels._composite import ProductKernel

        if isinstance(self, ProductKernel):
            return self.__mul__(other)

        return ProductKernel(self, other)
