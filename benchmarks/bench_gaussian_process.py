"""
Gaussian Process benchmarks: fit (with and without hyperparameter optimization)
and predict (mean only, with std).

Varies data size (n), dimension (d), and isotropic vs anisotropic kernels.
Run:  python benchmarks/bench_gaussian_process.py [-n N]

  -n N   Average time over N runs (default: 3).
"""

import argparse
import time
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
from gplite import ConstantKernel, GaussianProcess, PeriodicKernel, RBFKernel

RNG = np.random.default_rng(0)


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


def _make_data(n, d, noise=0.05):
    x = RNG.uniform(-2, 2, (n, d)).astype(np.float64)
    y = (
        np.sin(x[:, 0])
        + 0.3 * np.sum(x**2, axis=1)
        + noise * RNG.standard_normal(n)
    ).astype(np.float64)
    return x, y


def main():
    parser = argparse.ArgumentParser(
        description="Gaussian Process fit/predict benchmarks"
    )
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

    # (n_train, n_test, d)
    scenarios = [
        (300, 500, 2),
        (500, 800, 3),
        (600, 1000, 4),
        (800, 1200, 5),
    ]
    kernel_configs = [
        ("RBF (iso)", "rbf", True),
        ("RBF (aniso)", "rbf", False),
        ("Periodic (iso)", "periodic", True),
        ("Periodic (aniso)", "periodic", False),
        ("Const×RBF (iso)", "const*rbf", True),
        ("RBF+Periodic (iso)", "rbf+periodic", True),
    ]

    results_fit = []
    results_fit_opt = []
    results_predict = []
    results_predict_std = []

    for n_train, n_test, d in scenarios:
        x_train, y_train = _make_data(n_train, d)
        x_test, _ = _make_data(n_test, d)
        x_test = RNG.uniform(-2, 2, (n_test, d)).astype(np.float64)

        for label, kind, iso in kernel_configs:
            if kind in ("const*rbf", "rbf+periodic") and not iso:
                continue
            scenario = f"n={n_train} d={d} {label}"

            # Fit without optimization
            def do_fit():
                k = _kernel_factory(kind, d, iso)
                gp = GaussianProcess(k)
                gp.fit(x_train, y_train, optimize=False)

            t_fit = _timer(do_fit, n_runs, quiet=True)
            results_fit.append((scenario, t_fit))

            # Fit with optimization (slower)
            def do_fit_opt():
                k = _kernel_factory(kind, d, iso)
                gp = GaussianProcess(k)
                gp.fit(x_train, y_train, optimize=True)

            t_fit_opt = _timer(do_fit_opt, n_runs, quiet=True)
            results_fit_opt.append((scenario, t_fit_opt))

            # Predict mean only (reuse last gp from do_fit)
            k = _kernel_factory(kind, d, iso)
            gp = GaussianProcess(k)
            gp.fit(x_train, y_train, optimize=False)

            t_pred = _timer(lambda: gp.predict(x_test), n_runs)
            results_predict.append((scenario, t_pred))

            t_pred_std = _timer(
                lambda: gp.predict(x_test, return_std=True), n_runs
            )
            results_predict_std.append((scenario, t_pred_std))

    print(
        "Gaussian Process benchmark  (average over %d run%s)"
        % (n_runs, "s" if n_runs != 1 else "")
    )
    print()
    print("Fit (no optimization)")
    print("%-32s %12s" % ("Scenario", "Time (s)"))
    print("-" * 46)
    for s, t in results_fit:
        print("%-32s %12.4f" % (s, t))
    print()
    print("Fit (with hyperparameter optimization)")
    print("%-32s %12s" % ("Scenario", "Time (s)"))
    print("-" * 46)
    for s, t in results_fit_opt:
        print("%-32s %12.4f" % (s, t))
    print()
    print("Predict (mean only)")
    print("%-32s %12s" % ("Scenario", "Time (s)"))
    print("-" * 46)
    for s, t in results_predict:
        print("%-32s %12.4f" % (s, t))
    print()
    print("Predict (mean + std)")
    print("%-32s %12s" % ("Scenario", "Time (s)"))
    print("-" * 46)
    for s, t in results_predict_std:
        print("%-32s %12.4f" % (s, t))
    print()


if __name__ == "__main__":
    main()
