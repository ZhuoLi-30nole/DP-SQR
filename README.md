# Differentially Private Smoothed Quantile Regression

Reproduction code for **Differentially Private Smoothed Quantile Regression in Low and High Dimensions**, by Zhuo Li and Xiaochao Xia.

This repository contains the algorithms, experiment configurations, two public benchmark datasets, and individual-fit results needed to reproduce the paper's experiments. See the [quick start](docs/QUICKSTART.md) and [manuscript-to-code map](docs/REPRODUCIBILITY.md).

## Installation

Use Python 3.12 and a C++17 compiler (`c++` on the command line). DP-SGD and FRAPPE compile their numerical kernels automatically. CVXPY is not required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py check --tests
```

`requirements-locked.txt` records the verification environment. Installation in a fresh environment has not been tested; numerical results may vary across platforms and dependency versions.

## Reproduce the paper's figures and tables

```bash
python reproduce.py --mode archived
```

This reads the individual-fit measurements in `paper_results/` and writes to `outputs/figures/` and `outputs/tables/`, without fitting models again. It reconstructs:

- All 12 experimental figures used by the main text and supplement.
- Nine numeric result tables, checked at the precision displayed in the paper, and the application-design table.

Generated tables use a simple layout; `reference/tables/` contains the manuscript layouts. The smoothing schematic is authored in the manuscript's LaTeX/TikZ source. Method-comparison and theoretical tables are described in the [manuscript-to-code map](docs/REPRODUCIBILITY.md).

## Rerun experiments from the data

```bash
python reproduce.py --mode full --workers 1
```

This runs nine formal configurations and seven screening scenarios with 300 repetitions, then rebuilds the figures and tables. The configurations comprise 750,000 method fits plus 10,500 screening fits, so a full rerun requires substantial computation. DP-SGD uses `n²−1` updates. The low-dimensional grids include settings beyond the paper's displayed subsets; plotting and table scripts select the documented subsets.

Individual groups can be run separately:

```bash
python run.py plan --config sim_high_core
python run.py run --config sim_high_core --output results/formal/sim_high_core --workers 1
python run.py batch --study low --output-root results/formal --workers 1
python run.py batch --study high --output-root results/formal --workers 1
python run.py batch --study real --output-root results/formal --workers 1
python scripts/screening_stress.py --noise t3 --test-n 5000 --repeats 300 --workers 1 --output results/screening_stress_t3_r300_test5000
```

Each fit has an atomic checkpoint. Repeat an identical command at the same output path to resume. Use a new output directory after changing source code, configurations, or data. See [reproduction details](docs/REPRODUCIBILITY.md) for exporting new results.

## Small diagnostic runs

```bash
python run_smoke.py
python scripts/screening_stress.py --n 60 --p 20 --test-n 80 --repeats 1 --workers 1 --output results/screening_smoke
```

These execute small fits to check the workflow; the paper's study uses 300 repetitions.

## Repository contents

| Path | Contents |
| --- | --- |
| `src/dpsqr/` | Algorithms, initialization, privacy accounting, baselines, data preparation, and reporting |
| `configs/` | Nine formal configurations and four diagnostic configurations |
| `data/` | Public benchmark inputs and fixed crime-variable schema |
| `paper_results/` | Compressed individual-fit measurements and provenance |
| `reference/` | Manuscript figure PDFs and table fixtures |
| `reproduce.py` | Archived reconstruction or full rerun |
| `make_figures.py`, `make_tables.py` | Figure and table generation |
| `export_results.py` | Export of completed fits into compact inputs |
| `verify_release.py`, `tests/` | Integrity, manuscript-value, and protocol checks |

## Validation

All 19 tests passed. Archived reconstruction of 12 figures and 10 tables passed, including checks of the nine numeric tables against the manuscript and checks of publication labels. Small fitting runs were also exercised. The complete 300-repetition study has not been rerun as part of this verification. See [validation details](docs/RELEASE_AUDIT.md).

## Data and method scope

The public benchmarks are California Housing (20,640 records, five selected predictors) and Communities and Crime (1,994 retained records, 98 predictors). Transformations are estimated from training observations only. See [data details](docs/DATA.md), [experiment specifications](docs/EXPERIMENT_PROTOCOL.md), and [source attribution](THIRD_PARTY_NOTICES.md).

Privacy accounting applies to one fit on the specified processed inputs. Ordinary standardization and publication of multiple fits, splits, seeds, and diagnostics do not establish a single end-to-end private release of confidential raw records.

The parameter-error theorems apply to fixed public starting vectors satisfying the stated initial-distance and regularity conditions. Finite-step entry also requires a public initial-error bound and the joint tuning, sample, noise, and iteration conditions. The experimental pipeline uses a separately budgeted private ridge-SQR initializer; its statistical localization is left for future work. SR gives sure screening under marginal utility separation and an error bound in the retained dimension. IHT has its own conditions and does not automatically resolve weak marginal signals.
