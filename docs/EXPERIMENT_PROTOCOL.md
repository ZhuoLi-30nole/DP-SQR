# Experiment specifications

Protocol version: `20261101-fixed-rules-v1`. Changed settings require new fits rather than substitution of results from a different configuration.

## Simulations

**Low dimensions.** Sample sizes are `n=250/500/750/1000`, with `p=5/10/20`, no intercept, and independent `Uniform[-1,1]` covariates. All coefficients have magnitude 0.6 and alternating signs. Errors are Gaussian, Student t3, or Cauchy, with unit scale and no variance normalization. Responses satisfy `Y = X beta + xi - F_xi^{-1}(tau)`. Quantile levels are `.25/.5/.75`, and privacy budgets are `.1/.3/.5/.7/.9`. Each setting uses 300 repetitions and an independent test sample of size 5,000.

**High dimensions.** The ambient dimension is `p=12000`, including the intercept. Slopes have a Gaussian AR(1) design with correlation 0.1; coefficient signs use seed 0. The core design has five active slopes and an intercept, all of magnitude one. The working dimension is 12. The core grid uses Gaussian errors, `tau=.5`, `n=5000/7500/10000`, and `epsilon=.5/.75/1`. Robustness comparisons fix `n=10000`, use three quantiles and Gaussian/t2.25/Cauchy errors, and use `epsilon=.75/1`. Overlapping Gaussian median fits are reused from the core grid. The support-size supplement uses three or nine active slopes, `n=10000`, Gaussian errors, `tau=.5`, and `epsilon=.75/1`. There are 29 distinct high-dimensional conditions, each with 300 repetitions.

The standard low-dimensional methods are SQR-GD, its nonprivate counterpart, Adapted DP-SGD, and Adapted OPM; Huber-GD and FRAPPE are added at the median, including under Cauchy errors. The standard high-dimensional methods are SQR-SR/IHT and their nonprivate counterparts; Huber-IHT is added at the median, including under Cauchy errors. Cai-IHT is compared only at the Gaussian/t2.25 median. Huber comparisons under Cauchy errors are empirical stress tests and are not interpreted as requiring an existing finite conditional mean.

The primary low-dimensional metric is parameter L2 error. High-dimensional metrics are parameter L2 error and slope Recall. Supplemental metrics are excess quantile loss and full-support coverage. Excess loss is calculated by subtracting losses on the same test observations and then averaging; negative values are allowed. Prediction-function MSE, marginal calibration error, and separate stability/timing reports are excluded from the final presentation.

## Applications

Housing uses total sampled record counts `m=5000/10000/15000/20000`, with training sizes `n=4000/8000/12000/16000`. Crime uses `m=500/800/1100/1500/1900`, with `n=400/640/880/1200/1520`. Each repetition randomly selects `m` records and partitions them 80/20 using a common permutation: the first `n` records are training observations, and the rest are test observations. The sets are disjoint. The split is shared across methods, budgets, and quantiles. Each sample size has 300 resampling repetitions; reuse of records is not treated as 300 independent cohorts.

The grids use `epsilon=.3/.5/.9`, `tau=.25/.5/.75`, and `delta=10*n**(-1.1)`. The Crime working dimension is five, comprising an intercept and four slopes. Method eligibility follows the simulation rules. Mean-target comparators are used only as prediction benchmarks in median panels.

Preprocessing follows a fixed variable list and missing-column exclusion rule. Housing retains its log response, and Crime begins with its original response. Splitting precedes transformation fitting: predictors are standardized using training statistics only. The Housing log response is centered, and the Crime response is standardized using the training mean and SD. The same transformations are applied to test observations. Reported quantile losses are multiplied back by the response scale, giving log-price units for Housing and original crime-rate units for Crime. Ordinary standardization is not itself a DP mechanism; it does not establish a raw-record privacy guarantee for the whole workflow. This scope is recorded in raw rows and manifests.

Applications report the mean test quantile loss `QL_tau`. Housing coefficients are exported only at `m=20000`, `tau=.5`, and `epsilon=.9` for private fits; nonprivate/LAD/OLS fits have no privacy budget. The coefficient table reuses prediction fits. LAD/OLS are computed once per repetition for this table and are not included in the three-quantile prediction plots. Each coefficient vector is transformed back to the original predictor units before means and SDs are calculated. SDs are not confidence intervals.

## Privacy ledger

All entries are stage-specific `(epsilon, delta)` allocations for one complete fit. Public fixed hyperparameters do not consume privacy budget; the main protocol performs no parameter search. Here `e` and `d` denote the total epsilon and delta.

| Method | Screening | Initialization, including scale | Main iterations/solver |
| --- | --- | --- | --- |
| SQR-GD | None | e/12, d/6 | 11e/12, 5d/6 |
| Huber-GD | None | e/3, d/6 | 2e/3, 5d/6 |
| SQR-SR | 3e/4, d/3 | e/12, d/3 | e/6, d/3 |
| SQR-IHT | e/3, 0 | e/12, d/2 | 7e/12, d/2 |
| Huber-IHT | e/3, 0 | e/3, d/2 | e/3, d/2 |
| Adapted DP-SGD | None | Zero initial vector | e, min(d, 1/n) |
| Adapted OPM | None | Zero initial vector | e, 0 |
| Cai-IHT | None | Zero initial vector | e, d |
| FRAPPE | None | e/3, d/3 after amplification | 2e/3, 2d/3 |

GD uses GDP calibration for its main loop; IHT/SR screening uses the corresponding zCDP accounting. Cross-stage composition still follows the `(epsilon, delta)` allocations above; these fractions are not rho allocations. FRAPPE totals `amplified_stages`, rather than incorrectly adding unamplified base epsilon values. For the revised delta rule, Huber uses the available basic-composition branch when the advanced-composition branch is inapplicable.

One quarter of initialization epsilon is assigned to scale estimation, and three quarters to the initial-vector release. All initialization delta is assigned to the Gaussian release. The bandwidth is post-processing of the DP scale and consumes no additional budget. The stored `actual_delta` may be smaller than the total budget.

## Initialization and hyperparameters

Ordinary SQR/Huber initialization uses ridge parameter 0.2. With an intercept, L2 clipping is applied only to the slope part of each observation, at radius `sqrt(d)/6`, where `d` includes the intercept. Without an intercept, all covariates are clipped at that radius. Scale estimation first clips responses to `[-log(n), log(n)]`; the first- and second-moment estimates each receive half the scale epsilon. Nonpositive estimated variance triggers a fallback of 2. SQR uses a bounded scale clipped to `[.1, 10]` and initial bandwidth `.5*scale`. Huber uses the original private scale for its initialization threshold.

Ridge initialization uses a fixed numerical iteration budget and a uniform optimization-gap bound `gap <= 1e-8`. Its sensitivity adds `2*sqrt(2*gap/.2)` to the exact-solution bound. The exact bound is `2*B_I*c_tau/(n*.2)` for SQR and `2*B_I*a_I/(n*.2)` for standard Huber loss. Loss normalization follows standard Huber loss; noise constants from a CVXPY implementation using twice that loss cannot be transferred without adjustment. Ordinary initialization does not project onto a radius-10 ball. FRAPPE retains its own radius-3 projection.

| Method | Bandwidth/clipping | Step size | Iterations |
| --- | --- | --- | --- |
| SQR-GD | h=((p+log n)/n)^.25; gamma=sqrt(p) | .75 | ceil(3 log n) |
| SQR-SR | h=(log(pn)/n)^.25; refit gamma=sqrt(s) | .4 | ceil(3 log n) |
| SQR-IHT | Same high-dimensional h; coordinate bound 1 | .4 | ceil(3 log n) |
| Huber-GD | B=.5 sqrt(p+log n) | .2 | ceil(3 log n) |
| Huber-IHT | B=max(1,.5 sqrt(log(pn))); retain intercept | .01 | ceil(3 log n) |
| Cai-IHT | X bound 5; response bound 4 sqrt(log n); parameter radius 4 | 2/15 | ceil(3 log n) |
| Adapted DP-SGD | Logistic; h=.5; Cx=sqrt(p); reference radius 10 | Original public formula | n^2-1, with averaging |
| Adapted OPM | Logistic; low-dimensional h; Cx=sqrt(p) | Not applicable | One fixed perturbation followed by optimization |
| FRAPPE | See below | .9/p | 10 outer iterations; ceil(3 log n) inner iterations |

SQR bandwidths in high dimensions always use ambient dimension `p`, rather than selected dimension `s`. The fixed numerical step sizes in the applications are not claimed to estimate population curvature or be optimal. SQR does not automatically inherit Chang's Huber statistical error theorem; actual sensitivity depends on the step size and `c_tau`.

The main Huber threshold is `.04*private_scale*sqrt(n*epsilon/D)`, where `D=p+log(n)` in low dimensions and `D=s*log(p)+log(n)` in high dimensions. The threshold rule uses total epsilon; noise calibration uses the main-stage allocation.

FRAPPE uses 200 initialization observations, or all `n` if smaller, `C=3`, and outer iterations `v=0,...,9`. Its rules are `h_v=scale*max(.1,.5*.9^v)`, `f_min=.05/scale`, and `lambda_v=.1*sqrt(p)*scale*sqrt(log(2*p)/n)`. Initialization uses ridge parameter .2 and L1 penalty `.1*B_I*sqrt(log(2*p)/m_I)`. Every step projects onto the radius-3 ball. Half the main rho is assigned to density estimation and half to all inner gradient steps. The source implementation's qualification about finite-ADMM approximation is retained.

## Huber accounting at s=5

The original NoisyHT Laplace scale and update are retained. A check is not simply removed to claim privacy. With sensitivity `lambda` and Laplace scale `b`, each noisy-max selection is `2*lambda/b`-DP and its independently released selected-coordinate value is `lambda/b`-DP. Converting each pure-DP step to zCDP gives a combined bound `rho <= 5*s*lambda^2/(2*b^2)`. Selections and independent value releases can be viewed as interleaved steps; the original output is their post-processing.

The source scale `b=2*lambda*sqrt(5*s*log(1/delta_t))/epsilon_t` yields `rho <= epsilon_t^2/[8*log(1/delta_t)]` per iteration, without requiring `s>=10`. The code accumulates rho over `T` iterations, converts at the main-stage delta, and explicitly checks that epsilon does not exceed the main allocation. The paper's `s>=10` sufficient condition remains in `paper_budget_domain` metadata, but is not treated as a numerical prohibition at `s=5`. This accounting alone does not establish guarantees for every other stage or numerical implementation detail.

## Recording and reproducibility

Full raw rows from new runs contain actual parameters, total delta, stage ledgers, seeds, shared data keys, status, code version, and method qualifications. A nonprivate method is fitted only once on each sample and shared across epsilon panels, rather than counted repeatedly. Outliers, failures, and negative excess losses are not removed based on their values. Every atomic fit writes a checkpoint; resume requires matching source/configuration/data fingerprints. The committed compact archive projects these rows to the columns needed for manuscript reconstruction.

Each fit uses its own total budget. Publication of 300 repetitions and multiple budgets, quantiles, and methods is not claimed to have the privacy loss of one fit. Test metrics and diagnostic information are not free private releases.

## Statistical scope

The parameter-error theorems apply to fixed public starting vectors satisfying the stated initial-distance and regularity conditions. Finite-step entry also requires a public initial-error bound and the joint tuning, sample, noise, and iteration conditions. The experimental pipeline instead uses a separately budgeted private ridge-SQR initializer; its statistical localization is left for future work. SR gives sure screening under marginal utility separation and an error bound in the retained dimension, not unconditional exact support recovery. IHT has its own conditions and does not automatically resolve weak marginal signals.

## Quantile-comparator attribution and implementation

Adapted DP-SGD and objective perturbation build on Chen and Chua (ICML 2023), who provide excess-risk results and an objective-perturbation parameter-error bound. The comparator implementations use the Tran, Reimherr and Slavkovic (2024) code lineage; their smoothed KNG contribution is separate. Internal identifiers `tran_dpsgd` and `tran_opm` are retained for archive compatibility.

Adapted DP-SGD uses the public constant-step reference formula with fixed logistic bandwidth 0.5, rather than enforcing the original theorem's bandwidth/step coupling. Adapted OPM draws one isotropic radial perturbation with a Gamma-distributed radius, applies replace-one calibration, and solves the resulting ridge-regularized objective with BFGS. This differs from the Gaussian draw in Chen and Chua's Algorithm 2. These are adapted comparators, as described in Supplementary Section S8.5, rather than claims of exact replication.

Sources: https://proceedings.mlr.press/v202/chen23d.html and https://doi.org/10.1007/978-3-031-69651-0_2 .
