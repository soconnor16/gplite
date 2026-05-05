"""
Run all gplite benchmarks in sequence: kernels, Gaussian processes,
                                    active learning.

    python benchmarks/benchmark_gplite.py
    python benchmarks/benchmark_gplite.py -n 5   # average over 5 runs

To run a single benchmark:
    python benchmarks/bench_kernels.py [-n N]
    python benchmarks/bench_gaussian_process.py [-n N]
    python benchmarks/bench_active_learning.py [-n N]
"""

import argparse
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
SCRIPTS = [
    ("Kernels", "bench_kernels.py"),
    ("Gaussian processes", "bench_gaussian_process.py"),
    ("Active learning", "bench_active_learning.py"),
]


def main():
    parser = argparse.ArgumentParser(description="Run all gplite benchmarks")
    parser.add_argument(
        "-n",
        "--num-runs",
        type=int,
        default=3,
        metavar="N",
        help="Pass -n N to each script (average over N runs)",
    )
    args = parser.parse_args()
    extra = ["-n", str(args.num_runs)] if args.num_runs != 3 else []

    for name, script in SCRIPTS:
        path = BENCH_DIR / script
        print("\n" + "=" * 60)
        print("  %s" % name)
        print("=" * 60)
        rc = subprocess.run(
            [sys.executable, str(path)] + extra, cwd=BENCH_DIR.parent
        )
        if rc.returncode != 0:
            sys.exit(rc.returncode)
    print()


if __name__ == "__main__":
    main()
