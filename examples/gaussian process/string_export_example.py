"""Demonstrates how to export a fitted Gaussian Process as a raw mathematical
string representation.
"""

import numpy as np
from gplite import GaussianProcess, RBFKernel


def main() -> None:
    # generate random dataset
    rng = np.random.default_rng(0)
    x_train = rng.uniform(-2, 2, size=(5, 2))
    y_train = np.sin(x_train[:, 0]) + np.cos(x_train[:, 1])

    # fit the gp
    kernel = RBFKernel(length_scale=[1.0, 1.0], isotropic=False)
    gp = GaussianProcess(kernel=kernel)
    gp.fit(x_train, y_train, optimize=True)

    # export to string
    equation = gp.to_str(variable_names=["x", "y"])

    print(equation)


if __name__ == "__main__":
    main()
