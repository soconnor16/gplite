"""
Kernel benchmarks: compute and gradient for RBF, Periodic, and composite
                   kernels.

Varies data size (n), dimension (d), and isotropic vs anisotropic.
Run:  python benchmarks/bench_kernels.py [-n N]

  -n N   Average time over N runs (default: 3).
"""

import argparse
import time

import numpy as np
from gplite import ConstantKernel, PeriodicKernel, RBFKernel

RNG = np.random.default_rng(0)


def _timer(fn, n_runs):
    """Run fn() n_runs times; return average time in seconds."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def _kernel_factory(kind, d, isotropic):
    """Return a fresh kernel. kind in ('rbf','periodic','constant','rbf+periodic','const*rbf')."""
    if kind == "rbf":
        if isotropic:
            return RBFKernel(1.0)
        return RBFKernel(np.ones(d) * 1.0, isotropic=False)
    if kind == "periodic":
        if isotropic:
            return PeriodicKernel(1.0, 1.0)
        return PeriodicKernel(
            np.ones(d) * 1.0, np.ones(d) * 1.0, isotropic=False
        )
    if kind == "constant":
        return ConstantKernel(1.0)
    if kind == "rbf+periodic":
        return RBFKernel(1.0) + PeriodicKernel(1.0, 1.0)
    if kind == "const*rbf":
        return ConstantKernel(1.0) * RBFKernel(1.0)
    raise ValueError(kind)


def run_compute(x, kind, d, isotropic):
    k = _kernel_factory(kind, d, isotropic)
    k.compute(x, x)


def run_gradient(x, kind, d, isotropic):
    k = _kernel_factory(kind, d, isotropic)
    k.gradient(x, x)


def main():
    parser = argparse.ArgumentParser(
        description="Kernel compute/gradient benchmarks"
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

    # (n, d) scenarios
    sizes = [
        (200, 1),
        (400, 2),
        (600, 3),
        (800, 4),
        (1000, 5),
    ]
    # kernel kind, (isotropic only for constant and composites)
    kernel_configs = [
        ("RBF", "rbf", True),
        ("RBF", "rbf", False),
        ("Periodic", "periodic", True),
        ("Periodic", "periodic", False),
        ("Constant", "constant", True),
        ("RBF+Periodic", "rbf+periodic", True),
        ("Const×RBF", "const*rbf", True),
    ]

    results = []
    for n, d in sizes:
        x = RNG.standard_normal((n, d)).astype(np.float64)
        for label, kind, iso in kernel_configs:
            if kind in ("rbf+periodic", "const*rbf", "constant") and not iso:
                continue
            if kind == "constant" and d != 1:
                # constant is 1D-agnostic but we still run for one d
                pass
            iso_str = "iso" if iso else "aniso"
            scenario = f"n={n} d={d} {label} ({iso_str})"
            t_comp = _timer(
                lambda k=kind, dim=d, i=iso: run_compute(x, k, dim, i), n_runs
            )
            t_grad = _timer(
                lambda k=kind, dim=d, i=iso: run_gradient(x, k, dim, i), n_runs
            )
            results.append((scenario, t_comp, t_grad))

    print(
        "Kernel benchmark  (average over %d run%s)"
        % (n_runs, "s" if n_runs != 1 else "")
    )
    print()
    print("%-36s %12s %12s" % ("Scenario", "Compute (s)", "Gradient (s)"))
    print("-" * 62)
    for scenario, t_c, t_g in results:
        print("%-36s %12.4f %12.4f" % (scenario, t_c, t_g))
    print()


if __name__ == "__main__":
    main()
