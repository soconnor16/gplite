# gplite benchmarks

Three separate benchmark scripts for testing compute heavy sections of the package.

| File | What it benchmarks |
|------|--------------------|
| **bench_kernels.py** | Kernel `compute` and `gradient`; varies n, d, isotropic/anisotropic; RBF, Periodic, Constant, RBF+Periodic, Const×RBF. |
| **bench_gaussian_process.py** | GP `fit` (with and without hyperparameter optimization) and `predict` (mean only, mean+std); varies n, d, kernel types. |
| **bench_active_learning.py** | Full learn loop; fewer scenario combos (not all kernel×strategy×dim); a couple runs are heavier (5–10 s). Very low RMSE threshold so run uses all `max_points`. Reports time and **s/point**. |

**Run all (in order):**
```bash
python benchmarks/benchmark_gplite.py
python benchmarks/benchmark_gplite.py -n 5   # average over 5 runs
```

**Run one:**
```bash
python benchmarks/bench_kernels.py
python benchmarks/bench_gaussian_process.py
python benchmarks/bench_active_learning.py
```

**`-n N`** (all scripts): run each scenario N times and report the **average** time (default: 3).  
Example: `python benchmarks/bench_active_learning.py -n 5`
