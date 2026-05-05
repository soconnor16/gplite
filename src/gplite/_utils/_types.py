from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# more concise floating point type
f64: TypeAlias = np.float64

# more concise int type
i64: TypeAlias = np.int64

# arbitrarily sized array of 64 bit floats
Arrf64: TypeAlias = NDArray[f64]

# arbitrarily sized array of 64 bit ints
Arri64: TypeAlias = NDArray[i64]
