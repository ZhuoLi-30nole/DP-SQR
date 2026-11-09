# Reproduction quick start

This repository accompanies **Differentially Private Smoothed Quantile Regression in Low and High Dimensions**, by Zhuo Li and Xiaochao Xia.

## Install

Use Python 3.12 and a C++17 compiler available as `c++`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py check --tests
```

Adapted DP-SGD and FRAPPE compile their numerical kernels automatically. CVXPY is not required. `requirements-locked.txt` records the verification environment; a fresh dependency installation has not been tested.

## Reconstruct the paper's figures and tables

```bash
python reproduce.py --mode archived
```

This uses the committed `paper_results/` measurements without refitting. Outputs appear in `outputs/figures/` and `outputs/tables/`: 12 experimental figures, nine numeric result tables, and the application-design table. The smoothing schematic is part of the manuscript's LaTeX/TikZ source.

Numeric tables are checked at the precision displayed in the manuscript. Generated table layouts are simple; manuscript layouts are provided in `reference/tables/`. The [manuscript-to-code map](REPRODUCIBILITY.md) lists the corresponding figures, tables, and configurations.

## Rerun all model fits

```bash
python reproduce.py --mode full --workers 1
```

This executes nine formal configurations (750,000 method fits) and seven screening scenarios (10,500 fits), all with 300 repetitions, then generates the figures and tables. It requires substantial computation. The complete study has not been rerun as part of repository verification; archived reconstruction and small fitting runs have been checked. Cross-platform numerical results may differ.

The low-dimensional grids include combinations beyond the manuscript's displayed subsets. Figure and table scripts select the required subsets. FRAPPE comparisons are retained in the supplement. The screening study uses `n_test=5000` and 300 repetitions.

For a quick workflow check:

```bash
python run_smoke.py
python scripts/screening_stress.py --n 60 --p 20 --test-n 80 --repeats 1 --workers 1 --output results/screening_smoke
```

Repeat an identical fitting command at the same output path to resume from checkpoints. Use a new directory when source code, configurations, or data change. Keep new results separate from `paper_results/`. Individual-group and export commands are in the [README](../README.md) and [reproduction details](REPRODUCIBILITY.md).

## Data and interpretation

The archive contains individual metrics, fit identifiers, repetition indices, seeds, and required coefficients. Privacy budgets refer to a single fit on processed synthetic/public benchmark inputs; the entire benchmarking workflow is not a single private release of confidential raw records.

See [data details](DATA.md), [experiment specifications](EXPERIMENT_PROTOCOL.md), [source attribution](../THIRD_PARTY_NOTICES.md), and [validation scope](RELEASE_AUDIT.md).
