# Targeted screening experiments

The final study uses independent Student t3 errors, Gaussian covariates, the median quantile `tau=0.5`, and `epsilon=1`. It compares nonprivate SR, DP-SQR-SR, nonprivate SQR-IHT, DP-SQR-IHT, and DP-Huber-IHT.

Default settings are `n_train=5000`, `p=6000` including the intercept, five active slopes, working dimension 12, `n_test=5000`, and `delta=10*n_train**(-1.1)`. The t3 distribution has unit scale and variance 3. Budget allocation, initialization, and fixed parameter rules reuse `policy.parameters`, `fit_pipeline`, and `budget_check`.

## Run 300 repetitions

Execute from the project root:

```bash
python scripts/screening_stress.py \
  --noise t3 \
  --test-n 5000 \
  --repeats 300 \
  --workers 4 \
  --output results/screening_stress_t3_r300_test5000
```

Seven scenarios, 300 repetitions, and five methods produce 10,500 complete fits. Four independent processes run in parallel, with one BLAS thread per process. Training, test, and algorithm seeds follow the original deterministic rules.

The manuscript uses the final `n_test=5000` study with 300 repetitions. Its archived measurements are in `paper_results/screening_stress.csv.gz`. Historical runs with `n_test=2000` are excluded. Use a separate output directory for a new study; the script rejects checkpoint reuse when `test_n` differs.

To resume after interruption, run the same command again. The worker count can change while the script and experimental settings remain identical. Only one process group should write to a given output directory at a time.

## Summary format

The comparison metric is slope Recall, excluding the intercept. `summary.txt` reports mean Recall, standard error, and the number of repetitions. SE is the sample SD divided by the square root of the actual completed repetition count. A completed formal study has 300 repetitions per scenario/method. Sample SD and SE cannot be estimated from a single repetition and are left empty in that case.

`summary.csv` contains `recall_mean`, `recall_se`, `recall_std`, and `recall_count`. SD is retained for auditing; the manuscript reports mean (SE).

- `config.json` records the configuration, summary format, and source digest.
- `checkpoints/` records training/test sizes, selected support, parameters, seeds, and the privacy ledger for each fit.
- `raw.csv` stores individual measurements, including prediction diagnostics calculated on 5,000 independent test observations.
- `summary.csv` and `summary.txt` store Recall summaries.
- `rank_diagnostics.csv` records empirical rank utilities and ranks for active variables.

## Scenarios

| Scenario | Design and coefficients | Purpose |
| --- | --- | --- |
| `reference_ar`, rho=0.1 | Gaussian AR design with fixed Rademacher coefficient signs | Reference design |
| `strong_ar`, rho=0.8/0.95 | Same coefficients with stronger AR correlation | Screening under strong dependence |
| `cancellation`, rho=0.8/0.95 | Only the first two slopes are correlated; coefficients (1,-1), with other active coefficients (1,-1,1) | Progressively weaken the first pair's marginal signals |
| `suppressor`, rho=0.8/0.95 | Same correlated pair; coefficients (1,-rho), with other active coefficients (1,-1,1) | Give the second variable a nonzero conditional effect but marginal independence from the response |

The intercept coefficient is one. Signal variances are not normalized across scenarios. SQR-IHT and Huber-IHT retain the original screening-based initialization and iteration rules. Gaussian errors can be selected with `--noise gaussian`, and `--methods` can restrict the method set.

With no arguments, the script runs a five-repetition pilot in `results/screening_stress_t3_test5000_pilot`.

These are synthetic-data experiments. Privacy accounting and numerical qualifications follow the original project. Seeds, unperturbed utilities, and test losses are simulation diagnostics.
