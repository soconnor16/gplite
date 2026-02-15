"""
Abstract base class defining the kernel interface for Gaussian Process models.

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

from gpy._utils._types import Arrf64, f64
from gpy._utils._validation import validate_input_arrays


class Kernel(ABC):
    @property
    @abstractmethod
    def hyperparameters(self) -> tuple[str, ...]:
        """
        Method in which kernels define their hyperparameters and the order in
        which they are expected during initialization, or when new
        hyperparameters are set for an existing kernel.

        Args:
            - None

        Returns:
            - list[str]: List of the names of the hyperparameters in a given
                         kernel.
        """

    @property
    @abstractmethod
    def bounds(self) -> list[tuple[f64, f64]]:
        """
        Method in which kernels define the bounds for their hyperparameters
        during optimization. The order of the tuples matches the order
        of the hyperparameters in the "hyperparameters" method.

        Args:
            - None

        Returns:
            - list[tuple[float, float]]: List of the bounds of each of a
                                         kernel's hyperparameters.
        """

    # ----------------------------- PUBLIC METHODS --------------------------- #

    def compute(self, x1: Arrf64, x2: Arrf64) -> Arrf64:
        """
        Method to compute a kernel's similarity matrix between two given input
        arrays.

        Kernel functions are the core of Gaussian Process Regression (GPR).
        They are used to define the correlation (or "similarity") between input
        points which is necessary for predicting the output of new points and
        the shape of the predictive output function.

        Input arrays must have the same number of features (columns), but can
        have different numbers of features (rows).
        Given an x1 with shape (n_samples1, n_features) and x2 with shape
        (n_samples2, n_features), the resulting covariance matrix K(x1, x2)
        will have shape (n_samples1, n_samples2)


        Args:
            - x1: Arrf64
                - First array of points used to compute the kernel matrix.

            - x2: Arrf64
                - Second array of points used to compute the kernel matrix.


        Returns:
            - Arrf64: Kernel covariance matrix calculated between x1 and x2.
        """

        name = self.__class__.__name__
        x1, x2 = self._validate_input_data(
            x1, x2, f"{name} Compute Input 1", f"{name} Compute Input 2"
        )

        self._validate_anisotropic_hyperparameter_shape(x1)

        return self._compute(x1, x2)

    def gradient(self, x1: Arrf64, x2: Arrf64) -> tuple[Arrf64, ...]:
        """
        Method to compute a kernel's gradient with respect to each of its
        hyperparameters with input arrays x1 and x2.

        Kernel gradients are used for more efficient hyperparameter
        optimization.

        Args:
            - x1: Arrf64
                - First array of points used to compute the kernel gradient.

            - x2: Arrf64
                - Second array of points used to compute the kernel gradient.


        Returns:
            - tuple[Arrf64, ...]: Tuple of a kernel's gradients with respect
                                 to each of its hyperparameters.

        """
        name = self.__class__.__name__
        x1, x2 = self._validate_input_data(
            x1, x2, f"{name} Gradient Input 1", f"{name} Gradient Input 2"
        )

        self._validate_anisotropic_hyperparameter_shape(x1)

        return self._gradient(x1, x2)

    def compute_with_gradient(
        self, x1: Arrf64, x2: Arrf64
    ) -> tuple[Arrf64, tuple[Arrf64, ...]]:
        """
        Computes the kernel matrix and its gradients with respect to
        hyperparameters in a single call for efficiency.

        Args:
            - x1 (Arrf64): First input array of shape (n, d).
            - x2 (Arrf64): Second input array of shape (m, d).

        Returns:
            tuple[Arrf64, tuple[Arrf64, ...]]: Tuple containing the kernel
                matrix K of shape (n, m) and a tuple of gradient tensors.
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
        """
        Method to obtain the current hyperparameters of the kernel in the order
        defined by the "hyperparameters" method.

        Args:
            - None

        Returns:
            - list[float]: Current kernel hyperparameters as a list of Python
                           floats
        """

    @abstractmethod
    def set_params(self, params: Arrf64) -> None:
        """
        Method to set new hyperparameter values for the kernel. hyperparameter
        values should be passed as a flat array in the order defined by the
        "hyperparameters" method.

        Args:
            - params: Arrf64
                - New hyperparameter values to be set for the kernel.
        Returns:
            - None
        """

    # ---------------------------- PRIVATE METHODS --------------------------- #
    @abstractmethod
    def _compute(self, x1: Arrf64, x2: Arrf64) -> Arrf64:
        """
        Private method to compute the similarity matrix of a kernel. The public
        "compute" method calls this function with added data validation.
        """

    @abstractmethod
    def _gradient(self, x1: Arrf64, x2: Arrf64) -> tuple[Arrf64, ...]:
        """
        Private method to compute the kernel's gradient. The public "gradient"
        method calls this function with added data validation.
        """

    @abstractmethod
    def _compute_with_gradient(
        self, x1: Arrf64, x2: Arrf64
    ) -> tuple[Arrf64, tuple[Arrf64, ...]]:
        """
        Private method to compute kernel matrix and gradients together.
        The public compute_with_gradient method calls this with validation.
        """

    @abstractmethod
    def _to_str(
        self, variable_names: list[str], alpha: f64, training_point: Arrf64
    ) -> str:
        """
        Method to create string representations of the kernel function at a
        specific training point. This is a private function which should not
        be used directly by users. It is a utility function for the creation
        of larger string representations of a the full kernel function. The
        public api for this full representation is in the GaussianProcess
        class.

        Args:
            - variable_names: list[str]
                - Names of variables to be used in the string (e.g ['x', 'y']
                  for a kernel trained on data with two features).

            - training_point: Arrf64
                - The specific training point to center the expression along
                  (e.g '(x - 4.0)' for a 1D kernel with a training point input
                  of [4.0].


        Returns:
            - str: A string representation of the mathematical definition of the
                   kernel function at this specific training point.
        """

    @abstractmethod
    def _get_expanded_bounds(self) -> list[tuple[f64, f64]]:
        """ """

    @abstractmethod
    def _validate_anisotropic_hyperparameter_shape(self, x: Arrf64) -> None:
        """
        Private method to validate the shapes of anisotropic hyperparameters
        to ensure that they are valid for the current input data.
        """

    @abstractmethod
    def _validate_input_data(
        self, x1: Arrf64, x2: Arrf64, name1: str, name2: str
    ) -> tuple[Arrf64, Arrf64]:
        return validate_input_arrays(x1, name1, x2, name2)

    # ----------------------------- MAGIC METHODS ---------------------------- #
    def __call__(self, x1: Arrf64, x2: Arrf64 | None) -> Arrf64:
        """
        Compute the kernel matrix K(x1, x2) and allows the kernel instance to
        be called directly.

        Args:
            - x1: Arrf64
              - First array of points to compute the kernel matrix with.
            - x2: Arrf64
              - Second array of points to compute the kernel matrix with.

        Returns:
            - Arrf64: The kernel matrix K(x1, x2)
        """
        if x2 is None:
            return self.compute(x1, x1)

        return self.compute(x1, x2)

    def __add__(self, other: "Kernel") -> "Kernel":
        """
        Returns a composite kernel representing the sum of this kernel and
        another.

        This is equivelant to K_sum(x, x') = K(x, x') + K_other(x, x').

        Args:
            - other: Kernel
              - The kernel to add

        Returns:
            - Kernel: The composite sum kernel
        """

        # import locally to avoid circular import crashes
        from gpy.Kernels._composite import AdditiveKernel

        if isinstance(self, AdditiveKernel):
            return self.__add__(other)

        return AdditiveKernel(self, other)

    def __mul__(self, other: "Kernel") -> "Kernel":
        """
        Returns a composite kernel representing the product of this kernel and
        another.

        This is equivelant to K_prod(x, x') = K(x, x') * K_other(x, x').

        Args:
            - other: Kernel
              - The kernel to multiply

        Returns:
            - Kernel: The composite product kernel
        """
        # import locally to avoid circular import crashes
        from gpy.Kernels._composite import ProductKernel

        if isinstance(self, ProductKernel):
            return self.__mul__(other)

        return ProductKernel(self, other)
