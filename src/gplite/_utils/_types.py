from typing import TYPE_CHECKING, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

### EXTERNAL TYPES ###
# These types are designed to be more flexible and used in the type hints of
# public function inputs.

# overarching type for arrays of numeric values
NumericArray: TypeAlias = (
    list[float]
    | list[int]
    | tuple[float, ...]
    | tuple[int, ...]
    | NDArray[np.floating]
    | NDArray[np.integer]
)

# overarching type for singular numeric values
NumericValue: TypeAlias = int | float | np.integer | np.floating

### INTERNAL TYPES ###
# These types are designed to be more rigid and used in the type hints of
# private function inputs as well as all function outputs.

# more concise 64 bit floating point type
f64: TypeAlias = np.float64

# more concise 64 bit int type
i64: TypeAlias = np.int64

# arbitrarily sized array of 64 bit floats
Arrf64: TypeAlias = NDArray[f64]

# arbitrarily sized array of 64 bit ints
Arri64: TypeAlias = NDArray[i64]

if TYPE_CHECKING:
    from gplite.ActiveLearning.active_learning import ActiveLearner
    from gplite.GaussianProcess.gaussian_process import GaussianProcess


class SelectionFunction(Protocol):
    """
    Protocol for custom active learning point selection strategies.
    """

    def __call__(self, learner: "ActiveLearner", n_points: int) -> Arri64: ...


class ActiveLearningLossFunction(Protocol):
    """
    Protocol for custom active learning hyperparameter optimization.
    """

    def __call__(self, learner: "ActiveLearner") -> float: ...


class GaussianProcessLossFunction(Protocol):
    """
    Protocol for custom GP hyperparameter optimization.
    """

    def __call__(self, gp: "GaussianProcess") -> float: ...
