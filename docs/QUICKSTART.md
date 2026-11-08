# DP-SQR reproduction quick start

This release accompanies **Differentially Private Smoothed Quantile Regression in Low and High Dimensions**, by Zhuo Li and Xiaochao Xia, and is synchronized with the final main-text and supplement on 7 November 2026. The original numerical archive remains the 6 November release.

It is a separate GitHub release directory. The original experiment project is unchanged. The release includes the numerical algorithms, nine formal experiment configurations, four small diagnostic configurations, public benchmark inputs, and approximately 17 MB of compressed individual-fit measurements. Historical pilots, duplicate outputs, checkpoints, local environments, and superseded plotting entry points are excluded.

## Install

Python 3.12 is recommended. A C++17 compiler must be available as `c++`; DP-SGD and FRAPPE compile their numerical kernels automatically. CVXPY is not required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py check --tests
```

`requirements-locked.txt` records the packages actually installed in the release verification environment. A fresh installation was not tested. Numerical reruns can differ across platforms, BLAS implementations, and dependency versions.

## Reconstruct figures and tables from archived fits

```bash
python reproduce.py --mode archived
```

This reads `paper_results/` without refitting and writes to `outputs/figures/` and `outputs/tables/`. It covers all 12 external experimental figures, nine numeric result tables, and the application-design table used in the main text and supplement. The smoothing schematic is authored in LaTeX/TikZ and is not an experiment output.

In the verification environment, all 12 regenerated PNGs matched the original final-output PNGs pixel for pixel. The nine numeric tables were recalculated from individual-fit measurements and checked at the precision displayed in the manuscript. Generated LaTeX tables use a simple layout; the final manuscript layouts are preserved separately in `reference/tables/`.

## Rerun all model fits

```bash
python reproduce.py --mode full --workers 1
```

This runs the nine formal configurations, totaling 750,000 method fits in the original grids, and seven screening scenarios with 300 repetitions and five methods, totaling another 10,500 fits. It then rebuilds the final figures and tables. This is a substantial computation; the full 300-repetition study was not rerun during release preparation.

The numerical implementation and formal configurations are retained. The numerical update and noise formulas are unchanged. This release updates publication labels and rejects invalid forced-coordinate inputs; see RELEASE_UPDATE_20261108.md. The release runner differs only in its introductory docstring, which now refers to the English README; both the historical and release source digests are recorded.

The original low-dimensional grids include combinations not displayed in the manuscript. They are retained to preserve the original experimental specification. The figures and tables use their documented subsets.

See the [main README](../README.md) and [manuscript-to-code mapping](REPRODUCIBILITY.md) for individual-group commands and checkpoint/resume behavior. To execute small diagnostic fits:

```bash
python run_smoke.py
python scripts/screening_stress.py --n 60 --p 20 --test-n 80 --repeats 1 --workers 1 --output results/screening_smoke
```

## Archived measurements

`paper_results/` retains individual metrics, fit identifiers, repetition indices, seeds, and required coefficients. Readers can inspect the numerical results and reconstruct figures without completing all model fits first. Compression preserves the original numeric strings while removing repeated JSON parameter, privacy-ledger, and runtime-metadata columns.

The archived results support the default reconstruction command. They can be distributed with the repository or separately as a release asset, provided readers can obtain them at the documented paths. Checkpoints, historical outputs, `.venv/`, `.idea/`, caches, and logs do not need to be distributed. `results/` and `outputs/` are ignored by Git; `paper_results/` and `reference/` are retained.

FRAPPE is omitted from some main-text figures but remains in the supplemental comparisons, so its implementation and measurements are included. The final screening study uses `n_test=5000` with 300 repetitions; the historical `n_test=2000` version is excluded.

## Scope and attribution

Privacy budgets refer to one fit on the specified processed inputs in synthetic/public-data benchmarks. Ordinary standardization and the joint publication of multiple splits, methods, and budgets are not a single private release of raw confidential records. Numerical and method-specific qualifications remain in the code and [experiment specifications](EXPERIMENT_PROTOCOL.md).

See [data details](DATA.md), [third-party notices](../THIRD_PARTY_NOTICES.md), and `licenses/` for sources and existing license material. The release does not assign a new blanket license to all code, data, and manuscript content. File selection and validation scope are documented in the [release audit](RELEASE_AUDIT.md).
