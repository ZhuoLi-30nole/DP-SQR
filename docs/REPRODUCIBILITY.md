# Manuscript-to-code mapping

Snapshot: 6 November 2026. The authoritative original experiment source digest is recorded in `paper_results/manifest.json`. All `src/**/*.py` files and the screening runner are retained byte for byte. Only the introductory docstring in `run.py` was updated to reference the English README; its executable code is unchanged. Historical and release source digests are recorded separately. The release plotting entry point consolidates the final plotting implementation and names outputs as in the manuscript.

## Experiment coverage

| Formal configuration | Design | Used in |
| --- | --- | --- |
| `sim_low_p10` | Dense p=10; n=250/500/750/1000; Gaussian/t3/Cauchy; three quantiles; five budgets | Main low-dimensional plots and excess-loss table; supplemental FRAPPE comparisons |
| `sim_low_p5`, `sim_low_p20` | Original complete p=5/20 grids; displayed subset is median, epsilon=.5 | Supplemental dimension comparison |
| `sim_high_core` | p=12000; five active slopes; Gaussian median; n=5000/7500/10000; epsilon=.5/.75/1 | Main high-dimensional core; supplemental coverage and excess loss |
| `sim_high_gaussian_quantiles` | Gaussian nonmedian quantiles at n=10000 | Robustness panels, with median rows reused from core |
| `sim_high_tails` | t2.25/Cauchy; three quantiles; epsilon=.75/1 | Main and supplemental robustness panels |
| `sim_high_support` | Three/nine active slopes; Gaussian median | Supplemental support-size panel; five slopes reused from core |
| `real_housing` | 300 paired splits; training n=4000/8000/12000/16000; three quantiles/budgets | Housing prediction, coefficients, paired comparisons |
| `real_crime` | 300 paired splits; training n=400/640/880/1200/1520; three quantiles/budgets | Community prediction and paired comparisons |
| `scripts/screening_stress.py` | Seven dependence/signal scenarios; n=5000; p=6000; t3; test n=5000; tau=.5; epsilon=1 | Main and supplemental screening tables; marginal-rank diagnostics |

The screening runner's `diagnostic_only` field describes its synthetic stress-study scope. The archived final study has 300 repetitions per scenario/method; it is not the five-repeat default pilot. Its exact configuration is in `paper_results/provenance/screening_stress.json`.

## Final figure files

`python make_figures.py` produces exactly these experimental PDFs, with PNG, SVG, plotted source rows, panel metadata, and captions alongside them:

| Document | Output PDF | Compact plotting source |
| --- | --- | --- |
| Main | `fig2_no_frappe.pdf` | `sim_low_overview` |
| Main | `fig3_no_frappe.pdf` | `sim_low_quantiles_eps0.3`, `sim_low_quantiles_eps0.9` |
| Main | `sim_high_core.pdf` | `sim_high_core` |
| Main | `figure5_robustness_eps1.pdf` | `sim_high_quantiles_eps1` |
| Main | `figure6_housing_no_frappe.pdf` | `real_housing_quantile_grid` |
| Main | `figure7_community_log.pdf` | `real_crime_quantile_grid` |
| Supplement | `fig2_with_frappe.pdf` | `sim_low_overview` |
| Supplement | `fig3_with_frappe.pdf` | `sim_low_quantiles_eps0.3`, `sim_low_quantiles_eps0.9` |
| Supplement | `sim_low_dimensions.pdf` | `sim_low_dimensions` |
| Supplement | `figure5_robustness_eps0.75.pdf` | `sim_high_quantiles_eps0.75` |
| Supplement | `sim_high_support_size.pdf` | `sim_high_support_size` |
| Supplement | `figure6_housing.pdf` | `real_housing_quantile_grid` |

Compact files are `paper_results/plot_sources/<name>_source.csv.gz`. They can overlap because a fit can appear in several figures. Within each file `task_id` is unique. Do not concatenate overlapping figure files and count them as independent fits.

The `generated_file` entries in the archive manifest identify the original project's v2 export paths, used to verify the final manuscript PDFs. `release_generated_file` gives the current output basename.

## Final tables

`python make_tables.py` reconstructs:

| Output basename | Manuscript table label/content |
| --- | --- |
| `low_prediction_error` | `tab:sim-low-pe`; main median excess loss, no FRAPPE |
| `low_prediction_with_frappe` | `tab:supp-low-pe-full`; supplemental full median comparison |
| `housing` | `tab:real-housing-coefficients`; means in original predictor/log response units |
| `housing_with_frappe` | `tab:supp-housing-coefficients-full` |
| `coverage_combined` | `tab:coverage-combined`; Recall and full-coverage counts |
| `high_excess_loss` | `tab:sim-high-pe`; Gaussian median excess loss |
| `screening_main` | `tab:screening-stress`; four-method display |
| `screening_supp` | `tab:supp-screening-stress`; includes nonprivate SQR-IHT |
| `paired_real` | `tab:real-data-paired`; within-split loss differences and SE |
| `real_design` | `tab:real-data-design`; application sample sizes and dimensions |

Theoretical/method comparison tables and the privacy-allocation table describe the method rather than fitted results; their implementation is in `src/dpsqr/policy.py`, the source modules, and `docs/EXPERIMENT_PROTOCOL.md`.

For fit summaries SE is sample SD divided by sqrt(300). Housing coefficients are means over the 300 recorded training splits. Paired comparisons align the repetition index before subtracting losses. Main/supplemental table layouts may differ from the simple generated `.tex` layouts; numeric checks use the actual final-manuscript fixtures in `reference/tables/`.

## Exporting newly fitted results manually

After all nine formal groups and the screening run have completed:

```bash
python run.py figures --input results/formal/real_crime results/formal/real_housing results/formal/sim_high_core results/formal/sim_high_gaussian_quantiles results/formal/sim_high_support results/formal/sim_high_tails results/formal/sim_low_p10 results/formal/sim_low_p20 results/formal/sim_low_p5 --output outputs/processed
python export_results.py --processed outputs/processed --screening results/screening_stress_t3_r300_test5000 --output outputs/recomputed_results
python make_figures.py --source outputs/recomputed_results/plot_sources --output outputs/recomputed_figures
python make_tables.py --source outputs/recomputed_results --output outputs/recomputed_tables
```

The first command retains the original reporting/provenance checks and creates large intermediate CSVs; these can be kept locally and are ignored by Git. The compact export contains scalar per-fit measurements and coefficients rather than repeated solver metadata. It does not replace per-fit checkpoints for resuming fits.

`verify_release.py` verifies the shipped archive; it does not automatically assert equality of newly fitted results. Compare a new run's summaries with the archive, accounting for numerical environment differences. A source modification intentionally changes the digest and should use a new output directory.
