"""
Active learning benchmarks: full learn loop with different selection strategies,
kernel types, data sizes, and dimensions.

Uses very low RMSE threshold so the learner runs until max_points is reached.
Reports time and time per point (s/point).

Run:  python benchmarks/bench_active_learning.py [-n N]

  -n N   Average time over N runs (default: 3).
"""

import argparse
import time
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
from gplite import ActiveLearner, ConstantKernel, PeriodicKernel, RBFKernel

RNG = np.random.default_rng(0)

# very low threshold so we use all max_points (run to completion)
RMSE_USE_ALL_POINTS = 1e-6


def _timer(fn, n_runs, quiet=False):
    """Run fn() n_runs times; return average time in seconds."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        if quiet:
            with redirect_stdout(StringIO()):
                fn()
        else:
            fn()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def _kernel_factory(kind, d, isotropic):
    if kind == "rbf":
        return (
            RBFKernel(1.0)
            if isotropic
            else RBFKernel(np.ones(d) * 1.0, isotropic=False)
        )
    if kind == "periodic":
        return (
            PeriodicKernel(1.0, 1.0)
            if isotropic
            else PeriodicKernel(
                np.ones(d) * 1.0, np.ones(d) * 1.0, isotropic=False
            )
        )
    if kind == "const*rbf":
        return ConstantKernel(1.0) * RBFKernel(1.0)
    if kind == "rbf+periodic":
        return RBFKernel(1.0) + PeriodicKernel(1.0, 1.0)
    raise ValueError(kind)


def _make_data(pool_size, d, noise=0.04):
    x = RNG.uniform(-2, 2, (pool_size, d)).astype(np.float64)
    y = (
        np.sin(x[:, 0])
        + 0.3 * np.sum(x**2, axis=1)
        + noise * RNG.standard_normal(pool_size)
    ).astype(np.float64)
    return x, y


def run_learn(
    x_full,
    y_full,
    kernel_factory,
    strategy,
    max_points,
    n_runs=3,
    optimize_interval=None,
):
    """
    optimize_interval=None: no hyperparameter optimization during loop (faster).
    """
    learner = None

    def run():
        nonlocal learner
        kernel = kernel_factory()
        learner = ActiveLearner(
            kernel,
            x_full,
            y_full,
            max_points=max_points,
            rmse_threshold=RMSE_USE_ALL_POINTS,
            optimize_interval=optimize_interval,
        )
        learner.learn(strategy, batch_size=1, update=False, log=False)

    t = _timer(run, n_runs, quiet=True)
    n_used = learner.x_train.shape[0]
    return t, n_used


def main():
    parser = argparse.ArgumentParser(description="Active learning benchmarks")
    parser.add_argument(
        "-n",
        "--num-runs",
        type=int,
        default=3,
        metavar="N",
        help="Average time over N runs (default: 3)",
    )
    args = parser.parse_args()
    n_runs = max(1, args.num_runs)

    #  scenarios: (pool, max_points, d, kernel_label, kind, iso, strategy).
    scenarios = [
        (600, 500, 2, "RBF (iso)", "rbf", True, "uncertainty"),
        (1100, 1000, 3, "RBF (iso)", "rbf", True, "ei_max"),
        (600, 500, 3, "RBF (aniso)", "rbf", False, "random"),
        (2100, 500, 4, "Const×RBF (iso)", "const*rbf", True, "ei_min"),
        (400, 200, 4, "RBF (aniso)", "rbf", False, "uncertainty"),
        (500, 200, 4, "Periodic (iso)", "periodic", True, "uncertainty"),
        (
            550,
            100,
            4,
            "RBF+Periodic (iso)",
            "rbf+periodic",
            True,
            "mae",
        ),  # heavier
        (450, 150, 4, "Periodic (aniso)", "periodic", False, "mae"),
        (
            900,
            150,
            5,
            "RBF+Periodic (iso)",
            "rbf+periodic",
            True,
            "uncertainty",
        ),  # heavier
    ]

    results = []
    for pool_size, max_points, d, klabel, kind, iso, strategy in scenarios:
        x_full, y_full = _make_data(pool_size, d)
        kernel_factory = lambda k=kind, dim=d, i=iso: _kernel_factory(k, dim, i)
        t, n_used = run_learn(
            x_full, y_full, kernel_factory, strategy, max_points, n_runs=n_runs
        )
        s_per_pt = t / n_used if n_used else 0
        scenario = (
            f"pool={pool_size} d={d} max_pt={max_points} {klabel} {strategy}"
        )
        results.append((scenario, t, n_used, s_per_pt))

    print(
        "Active learning benchmark  (average over %d run%s)"
        % (n_runs, "s" if n_runs != 1 else "")
    )
    print(
        "(RMSE threshold = %.0e so run uses all max_points)"
        % RMSE_USE_ALL_POINTS
    )
    print()
    print("%-52s %10s %8s %10s" % ("Scenario", "Time (s)", "Points", "s/point"))
    print("-" * 82)
    for scenario, t, n, s in results:
        print("%-52s %10.3f %8d %10.4f" % (scenario, t, n, s))
    print()


if __name__ == "__main__":
    main()
