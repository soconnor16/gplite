"""Gaussian Process Regression class for probabilistic machine learning.

Gaussian Processes define a distribution over functions f(x) ~ GP(m(x), k(x,x'))
where m(x) is the mean function (assumed zero here) and k(x,x') is the
covariance/kernel function.

Given training data (X, y), predictions at new points X* are computed as:

    Mean:     μ* = K(X*, X) @ α,  where α = K(X, X)^(-1) @ y
    Variance: σ²* = K(X*, X*) - K(X*, X) @ K(X, X)^(-1) @ K(X, X*)

The Cholesky decomposition L (where K = L @ L.T) is used for numerical
stability, allowing efficient computation of α via solving L @ L.T @ α = y.

Hyperparameters (kernel parameters and noise) can be optimized by maximizing
the log marginal likelihood:

    log p(y|X,θ) = -0.5 * y.T @ K^(-1) @ y - 0.5 * log|K| - n/2 * log(2π)
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg

from gplite._utils._computation import compute_lower_cholesky_decomposition
from gplite._utils._constants import EPSILON
from gplite._utils._data import (
    standardize_input_data,
    standardize_target_data,
)
from gplite._utils._errors import ValidationError
from gplite._utils._validation import (
    validate_input_and_target_data,
    validate_numeric_array,
    validate_variable_names,
)
from gplite.Kernels._base import Kernel
from gplite.Optimization.gaussian_process.optimization import (
    optimize_hyperparameters,
)

if TYPE_CHECKING:
    from gplite._utils._types import (
        Arrf64,
        GaussianProcessLossFunction,
        NumericArray,
    )


class GaussianProcess:
    """Gaussian Process class for regression.

    The GaussianProcess class is designed to aid the easy training of GPR
    models. Gaussian Process Regression (GPR) models can be extremely useful for
    applications in which little data is available and some prior knowledge of
    the data is had. The flexibility of the kernels used by GPR allows for the
    injection of prior knowledge into the model, which can make training much
    more efficient, though also makes kernel selection important. GPR models can
    leverage log-marginal likelihood targets to automate the tuning of kernel
    hyperparameters, which can drastically improve the fit of the model while
    decreasing the manual tuning required from the user. Gaussian Processes are
    also known to require less training data than other models such as neural
    networks, making them ideal for situations in which the shape of the data
    is known but few data points are available. When used for prediction,
    GPR models return not only predicted values but also their covariance, which
    can be thought of as the model's "certainty" about the predicted point.

    GPR models may not be the best fit for high-dimensional data or large
    dataset as the time and memory complexity of fitting a model scales roughly
    with the number of data points cubed.

    Attributes:
        kernel: The kernel function used to compute covariance.
        x_train: Training input values. Standardized to zero mean and unit
            variance if standardize_inputs was left as True, untouched
            otherwise.
        y_train: Training target values after standardization.
        alpha: Weights computed during fitting for predictions.
    """

    def __init__(self, kernel: Kernel, standardize_inputs: bool = True) -> None:
        """Initializes a Gaussian Process model with the specified kernel.

        Args:
            kernel: A kernel instance defining the covariance function used by
                the GPR model.
            standardize_inputs: Whether to standardize input features to zero
                mean and unit variance. Defaults to True.

        Raises:
            ValidationError: If kernel is not a valid Kernel subclass.
        """
        if not isinstance(kernel, Kernel):
            err_msg = (
                "Error: 'kernel' argument must be a valid kernel subclass."
            )
            raise ValidationError(err_msg)

        self.kernel = kernel
        self._standardize_inputs = standardize_inputs

        # simulates gaussian noise in data, optimized with kernel
        # hyperparameters helps stabilize fitting with numerically unstable data
        self._noise = 1e-3

        # training and testing data
        self.x_train = np.array([])
        self.y_train = np.array([])

        # GP weight parameter (alpha = K^(-1) @ y)
        self.alpha = np.array([])

        # lower cholesky decomposition of the kernel matrix K
        self._lower_chol = np.array([])

        # target standardization stats (always used)
        self._y_mean = 0
        self._y_std = 1

        # input standardization stats (arrays to handle multiple features)
        # only used if standardize_inputs is True; otherwise set to identity
        # transformation
        if self._standardize_inputs:
            self._x_mean = np.array([])
            self._x_std = np.array([])

    def optimize_hyperparameters(
        self,
        objective: str | GaussianProcessLossFunction = "lml",
        num_restarts: int = 10,
    ) -> None:
        """Optimizes the kernel hyperparameters.

        Args:
            objective: The objective function to minimize. Options include 'lml'
                (log-marginal-likelihood) or a custom loss function. Defaults to
                'lml'.
            num_restarts: Number of random restarts to avoid local minima.
                Defaults to 10.
        """
        optimize_hyperparameters(self, objective, num_restarts)

    def _fit_without_optimization(self) -> None:
        """Fits the Gaussian Process to training data without optimization.

        This method fits the kernel matrix, its Cholesky decomposition, and the
        alpha weights used for prediction.
        """
        # kernel matrix K(x_train, x_train)
        K = self.kernel._compute(self.x_train, self.x_train)

        self._lower_chol, self._noise = compute_lower_cholesky_decomposition(
            K,
            self._noise,
            max_attempts=10,
        )

        self.alpha = linalg.cho_solve(
            (self._lower_chol, True),
            self.y_train,
            check_finite=False,
        )

    def fit(
        self,
        x: NumericArray,
        y: NumericArray,
        optimize: bool = False,
        objective: str | GaussianProcessLossFunction = "lml",
    ) -> None:
        """Fits the Gaussian Process model to training data.

        Args:
            x: Input features of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).
            optimize: Whether to optimize hyperparameters before fitting.
                Defaults to False.
            objective: Objective function for optimization if optimize is True.
                Defaults to 'lml'.

        Raises:
            ValidationError: If input and target arrays have incompatible shapes
                or contain invalid values.
        """
        x, y = validate_input_and_target_data(x, y)

        if self._standardize_inputs:
            self.x_train, self._x_mean, self._x_std = standardize_input_data(x)

        else:
            self.x_train = x
            self._x_mean = np.zeros(x.shape[1])
            self._x_std = np.ones(x.shape[1])

        self.kernel._validate_anisotropic_hyperparameter_shape(x)

        self.y_train, self._y_mean, self._y_std = standardize_target_data(y)

        if optimize:
            self.optimize_hyperparameters(objective)

        else:
            self._fit_without_optimization()

    def predict(
        self,
        x: NumericArray,
        return_std: bool = False,
        return_cov: bool = False,
    ) -> Arrf64 | tuple[Arrf64, ...]:
        """Predicts target values for new input data.

        This method takes a fitted gaussian process model and calculates the
        predicted target values on new input points, optionally returning
        the model's covariance or standard deviation (the square root of the
        diagonal of the covariance matrix) along with the predicted values.

        Args:
            x: Input features of shape (n_samples, n_features).
            return_std: Whether to return standard deviation of predictions.
                Defaults to False.
            return_cov: Whether to return full covariance matrix. Defaults to
                False.

        Returns:
            Predicted mean values, and optionally standard deviation and/or
            covariance matrix. If both return_std and return_cov are True,
                returns (mean, std, cov). All values are returned as NumPy
                arrays of 64 bit floating point types.

        Raises:
            RuntimeError: If the model has not been fitted before prediction.
            ValidationError: If input contains invalid values.
        """
        # model must be fitted before prediction
        if self.alpha.size == 0 or self._lower_chol.size == 0:
            err_msg = (
                "Error: Model needs to be fitted before it can be used for "
                "prediction."
            )
            raise RuntimeError(err_msg)

        x = validate_numeric_array(
            x,
            "Gaussian Process Prediction input",
            allow_nonpositive=True,
        )
        x = x.reshape(-1, 1) if x.ndim == 1 else x

        # standardize new input data
        # this is safe even if _standardize_inputs is False because self._x_mean
        # and self._x_std are initialized to default values of 0 and 1 anyways
        x_norm = (x - self._x_mean) / self._x_std

        # compute mean prediction: μ* = K(x*, X) @ α
        k_test_train = self.kernel.compute(x_norm, self.x_train)

        y_mean_norm = k_test_train @ self.alpha

        # unstandardize y_mean for returning
        y_mean = (y_mean_norm * self._y_std) + self._y_mean

        if not (return_std or return_cov):
            return y_mean

        # compute variance / covariance
        # v = L^(-1) * k_test_train.T
        variance = linalg.solve_triangular(
            self._lower_chol,
            k_test_train.T,
            lower=True,
            check_finite=False,
        )

        if return_cov:
            k_test_test = self.kernel.compute(x_norm, x_norm)

            y_cov_norm = k_test_test - variance.T @ variance
            y_cov = y_cov_norm * (self._y_std**2)
            if return_std:
                y_std = np.sqrt(np.maximum(np.diag(y_cov), 0.0))
                return y_mean, y_std, y_cov
            return y_mean, y_cov

        k_diag = self.kernel._compute_diag(x_norm)

        y_var_norm = k_diag - np.einsum("ij,ij->j", variance, variance)

        y_var_norm = np.maximum(y_var_norm, 0.0)
        y_std = np.sqrt(y_var_norm) * self._y_std

        return y_mean, y_std

    # TODO: Make json based saving
    def save(self, filepath: str | Path) -> None:
        """Saves the Gaussian Process model to a file.

        Args:
            filepath: Path to save the model to.
        """
        if not isinstance(filepath, (Path, str)):
            err_msg = "Error: 'filepath' must be a str type or Path object"
            raise ValidationError(err_msg)

        filepath = Path(filepath).resolve()

        with filepath.open("wb") as f:
            pickle.dump(self, f)

    # TODO: Make json based loading
    @classmethod
    def load(cls, filepath: str | Path) -> GaussianProcess:
        """Loads a Gaussian Process model from a file.

        Args:
            filepath: Path to the saved model file.

        Returns:
            The loaded GaussianProcess instance.

        Raises:
            TypeError: If the loaded object is not a GaussianProcess instance.
            FileNotFoundError: If the file does not exist.
        """
        if not isinstance(filepath, (Path, str)):
            err_msg = "Error: 'filepath' must be a str type or Path object"
            raise ValidationError(err_msg)

        filepath = Path(filepath).resolve()

        if not filepath.exists():
            err_msg = f"Error: '{filepath}' does not exist."
            raise FileNotFoundError(err_msg)

        with filepath.open("rb") as f:
            model = pickle.load(f)

        if not isinstance(model, cls):
            err_msg = (
                f"Error: Expected a GaussianProcess instance, "
                f"got '{type(model).__name__}'."
            )
            raise TypeError(err_msg)

        if model.alpha.size == 0 or model._lower_chol.size == 0:
            warning_msg = "Warning: Loaded model is not fitted."
            warnings.warn(warning_msg, stacklevel=2)

        return model

    def to_str(self, variable_names: list[str]) -> str:
        """Generates a string representation of the fitted GPR model.

        By the Representer Theorem, the predictive mean function of a fitted
        Gaussian Process can be expressed as a linear combination of kernel
        evaluations centered on the training points:
            μ(x) = Σ ⍺ᵢ * k(x, xᵢ)
        where ⍺ᵢ is the learned weight at training index i, and k(x, xᵢ)
        is the scalar covariance between the new input x and the training
        point xᵢ. We can expand this representation to create a string
        representation of that summation.

        This project was originally created for Δ-Machine Learning applications,
        and as such, all model strings are built to be compatible with the
        format expected by OpenMM for custom string-specified forces.

        Args:
            variable_names: Names of input variables to use in the expression
                (e.g., ['x', 'y']).

        Returns:
            Mathematical string expression representing the GP prediction
            function.

        Warns:
            UserWarning: If the model has not been fitted.

        Raises:
            ValidationError: If the variable_names length doesn't match number
                of features in the data the model was trained on.
        """
        if self.alpha.size == 0:
            warning_msg = (
                "Warning: Gaussian Process is not fitted, returning empty "
                "string."
            )
            warnings.warn(warning_msg, stacklevel=2)
            return ""

        variable_names = validate_variable_names(
            variable_names,
            self.x_train.shape[1],
        )

        # standardize variables to match standardized data
        # if the x standardization option was set to false, the mean and std
        # are set to 0 and 1 respectively and the unstandardization is skipped
        if self._standardize_inputs:
            standardized_vars = []
            for i, var in enumerate(variable_names):
                standardized_vars.append(
                    f"(({var} - {self._x_mean[i]:.6e}) / {self._x_std[i]:.6e})",
                )
        else:
            standardized_vars = variable_names

        terms = []
        for x_i, alpha_i in zip(self.x_train, self.alpha, strict=True):
            # absorb y_std directly into alpha to save OpenMM from
            # multiplying the entire sum at every integration step
            alpha_scaled = alpha_i * self._y_std
            k_str = self.kernel._to_str(standardized_vars, alpha_scaled, x_i)
            terms.append(k_str)

        full_expression = " + ".join(terms)

        # target data is always standardized, this unstandardizes it if the
        # unstandardized mean was not already 0
        if self._y_mean > EPSILON:
            return f"{full_expression} + {self._y_mean:.6e}"

        return full_expression
