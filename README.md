# Differentially Private Smoothed Quantile Regression

Reproduction code for **Differentially Private Smoothed Quantile Regression in Low and High Dimensions**, by Zhuo Li and Xiaochao Xia.

This release follows the final manuscript and supplement snapshot of **6 November 2026**. It contains the original numerical implementations, fixed experiment configurations, two public benchmark datasets, and compact individual-fit results. See [the quick start](docs/QUICKSTART.md) and [the manuscript-to-code map](docs/REPRODUCIBILITY.md).

## Installation

Use Python 3.12 and a C++17 compiler (`c++` on the command line). The Tran-SGD and FRAPPE implementations compile their numerical kernels automatically. CVXPY is not required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py check --tests
```

`requirements-locked.txt` records the environment used to verify this release. It was recorded from installed packages; installation in a fresh environment was not tested. Numerical reruns may differ across operating systems and numerical-library versions.

## Reproduce the published figures and tables

```bash
python reproduce.py --mode archived
```

This reads the committed, compressed per-fit measurements in `paper_results/`; it performs no new model fits. Outputs are written to `outputs/figures/` and `outputs/tables/`.

- All **12 external experimental figures** used by the main text and supplement are regenerated. The schematic illustration of smoothing is authored in the manuscript's TikZ source.
- **Nine numeric tables** are recomputed and checked against the manuscript's displayed values. The application-design table is also reconstructed from configurations. Generated table layouts are simple; `reference/tables/` preserves the manuscript layouts.
- `verify_release.py` checks the release source digest, archive integrity, and table values; historical source digests remain in the provenance records. The archived numerical strings have been retained; repeated parameter, privacy-ledger, and runtime-metadata columns have been removed from the compact files.

The final main figures omit FRAPPE where specified by the manuscript; the corresponding supplemental comparisons retain it. Nonprivate fits are shared across budget panels and are not independent fits in each panel.

## Rerun all experiments from the data

```bash
python reproduce.py --mode full --workers 1
```

This runs the nine formal experiment configurations with 300 repetitions, then the seven screening-stress scenarios with 300 repetitions and five methods. It exports new measurements and generates the final figures and tables in `outputs/`.

**This is a substantial computation.** The original formal configurations specify 750,000 method fits; screening adds 10,500 fits. Tran-SGD retains its original `n²−1` updates. The low-dimensional grids include settings beyond the subset displayed in the paper; they are retained to preserve the original experiment specification and fingerprints. No historical pilot results are substituted.

Individual groups can be planned or run separately:

```bash
python run.py plan --config sim_high_core
python run.py run --config sim_high_core --output results/formal/sim_high_core --workers 1
python run.py batch --study low --output-root results/formal --workers 1
python run.py batch --study high --output-root results/formal --workers 1
python run.py batch --study real --output-root results/formal --workers 1
python scripts/screening_stress.py --noise t3 --test-n 5000 --repeats 300 --workers 1 --output results/screening_stress_t3_r300_test5000
```

Each fit has an atomic checkpoint. Rerun an identical command at the same output path to resume; changed source/configuration/data require a new directory. Keep the committed `paper_results/` archive separate from new fits. See [reproduction details](docs/REPRODUCIBILITY.md) for the export commands.

## Quick diagnostic runs

```bash
python run_smoke.py
python scripts/screening_stress.py --n 60 --p 20 --test-n 80 --repeats 1 --workers 1 --output results/screening_smoke
```

These execute actual small fits and are not substitutes for the 300-repetition study.

## Repository contents

| Directory/file | Purpose |
| --- | --- |
| `src/dpsqr/` | SQR algorithms, initialization, privacy accounting, baselines, simulation, data preparation, reporting |
| `configs/` | Nine original formal configurations and four smoke configurations |
| `data/` | Supplied raw public benchmark files and fixed crime-variable schema |
| `paper_results/` | Approximately 17 MB of compressed per-fit results and provenance |
| `reference/` | The 12 final manuscript PDFs and displayed table fixtures |
| `reproduce.py` | Archived-result reconstruction or full rerun |
| `make_figures.py`, `make_tables.py` | Final manuscript figures and numeric tables |
| `export_results.py` | Projection of new completed outputs into compact inputs |
| `verify_release.py`, `tests/` | Integrity checks, table checks, and numerical/protocol tests |

Generated `results/`, `outputs/`, environments, editor settings, pilots, old screening runs, and repeated figure exports are excluded from version control.

## Data, attribution, and interpretation

The datasets are California Housing (20,640 records, five selected predictors) and Communities and Crime (1,994 retained records, 98 predictors). Data transformations are estimated using training observations only. See [data details](docs/DATA.md), [experiment specifications](docs/EXPERIMENT_PROTOCOL.md), and [third-party notices](THIRD_PARTY_NOTICES.md). Existing upstream license text is retained under `licenses/`; this release does not assign a new blanket license to third-party code, data, or the manuscript.

These are synthetic/public-data benchmarking experiments. Privacy accounting is for one fit on the specified processed inputs. Ordinary training-set standardization and the publication of multiple fits, splits, seeds, and diagnostic losses are not a single end-to-end private release of confidential raw records. Method-specific numerical qualifications remain in the source.

## Verification performed for this release

- Twelve existing unit/protocol tests passed; both public datasets loaded successfully.
- Four small experiment grids and a small screening-stress run were executed.
- All 12 regenerated figure PNGs matched their original final-output PNGs pixel for pixel in the verification environment.
- Nine numeric tables matched the displayed manuscript values; the design table matched configurations.
- The complete 300-repetition fits were **not rerun** during packaging. Reconstruction from archived results and execution of the full experiment code are distinct checks.

See [the release audit](docs/RELEASE_AUDIT.md) for included/excluded material and validation scope.
