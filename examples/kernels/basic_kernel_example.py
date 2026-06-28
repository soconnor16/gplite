"""Intro Kernel demonstration. This should be sufficient for most regular usage
and exposes the core API.
"""

import numpy as np
from gplite import RBFKernel

# sample inputs for demonstration, you replace these with your real input values
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(0, 10, 50)

# most basic kernel, this can be used on most datasets
# though it will not always be the most data-efficient
kernel = RBFKernel(length_scale=1.0)

# retrieve the covariance matrix between x1 and itself
# this can also be computed as kernel(x1, x1)
cov_x1_x1 = kernel.compute(x1, x1)

# retrieve the covariance matrix and gradients between x1 and x2
cov_x1_x2, grad_x1 = kernel.compute_with_gradient(x1, x2)

# change the length scale of the kernel
kernel.set_params(2.0)

# view the new hyperparameters to check
print(kernel.get_params())
