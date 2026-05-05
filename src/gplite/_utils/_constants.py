# small numerical tolerance for jitter and floating point comparisons
EPSILON = 1e-8

# constants for optimization
GLOBAL_MAXITER = 30  # max iterations for initial, global optimization
LOCAL_MAXITER = 500  # max iterations for secondary, gradient based optimization
N_REFINE = 2  # number of top optimization candidates to refine

# constant set to hold valid nu values for the matern kernel
VALID_NU = {1.5, 2.5}
