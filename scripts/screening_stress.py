"""Paired synthetic diagnostics for marginal screening, using existing fit policies.

Run from the project root with .venv/bin/python scripts/screening_stress.py.
Raw utilities and seeds are simulation diagnostics, not confidential-data releases.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import multiprocessing
import hashlib
import json
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(ROOT / "src"))
from dpsqr.core import check_loss
from dpsqr.pipeline import fit_pipeline
from dpsqr.policy import parameters, total_delta
from dpsqr.simulation import generate
from dpsqr.selection import supremum_rank_utilities
from dpsqr.study import atomic_json, budget_check, derived_seed, source_digest

SCENARIOS = [
    ("reference_ar", .1), ("strong_ar", .8), ("strong_ar", .95),
    ("cancellation", .8), ("cancellation", .95),
    ("suppressor", .8), ("suppressor", .95),
]
METHODS = ["sqr_sr_np", "sqr_sr", "sqr_iht_np", "sqr_iht", "huber_iht"]


def sample(n, p, kind, rho, seed, noise="gaussian"):
    """Five active slopes plus intercept, Gaussian design, symmetric errors.

    AR scenarios reproduce the existing fixed-Rademacher coefficient rule.
    Block scenarios correlate ONLY slopes 1 and 2. Cancellation uses (1,-1).
    Suppressor uses (1,-rho), giving Cov(X2,Y)=0 exactly at population level.
    Gaussian design and independent errors imply marginal independence of
    X2 and Y in the latter case, for both Gaussian and t3 errors.
    No sample-dependent scaling, centering, clipping, or signal normalization.
    """
    if noise not in ("gaussian", "t3"):
        raise ValueError("Expected gaussian or t3 errors")
    if kind in ("reference_ar", "strong_ar"):
        s = generate(n, p, 6, tau=.5, noise=noise, seed=seed,
                     signal=1., correlation=rho, intercept=True,
                     design_distribution="gaussian",
                     coefficient_pattern="fixed_rademacher", coefficient_seed=0)
        # The generator supplies beta[0]=1 under the fixed-Rademacher rule.
        return s.X, s.y, s.beta
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, 0] = 1.
    X[:, 2] = rho * X[:, 1] + np.sqrt(1 - rho**2) * X[:, 2]
    beta = np.zeros(p)
    beta[:6] = [1., 1., -1. if kind == "cancellation" else -rho, 1., -1., 1.]
    error = rng.normal(size=n) if noise == "gaussian" else rng.standard_t(3, size=n)
    return X, X @ beta + error, beta


def summarize(rows, out):
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "raw.csv", index=False)
    summary = frame.groupby(["scenario", "rho", "method"], sort=False)[["recall"]].agg(
        ["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["recall_se"] = summary["recall_std"] / np.sqrt(summary["recall_count"])
    summary = summary[["recall_mean", "recall_se", "recall_std", "recall_count"]]
    summary.to_csv(out / "summary.csv")
    display = summary[["recall_mean", "recall_se", "recall_count"]].round(4)
    (out / "summary.txt").write_text(display.to_string() + "\n")


def run_job(payload):
    args, kind, rho, rep = payload
    with threadpool_limits(limits=1):
        return fit_job(args, kind, rho, rep)


def fit_job(args, kind, rho, rep):
    out = args.output.resolve()
    rows, diagnostics = [], []
    # Common base Gaussian draws across scenarios within each repetition.
    data_seed = derived_seed(args.seed, "train", rep)
    test_seed = derived_seed(args.seed, "test", rep)
    X, y, beta = sample(args.n, args.p, kind, rho, data_seed, args.noise)
    Xt, yt, test_beta = sample(args.test_n, args.p, kind, rho, test_seed, args.noise)
    assert np.array_equal(beta, test_beta)
    truth = np.flatnonzero(beta[1:]) + 1
    oracle_loss = check_loss(yt - Xt @ beta, .5).mean()
    # Rank diagnostics only, recomputed independently of private fits.
    utilities = supremum_rank_utilities(X, y, .5)
    max_null = float(utilities[6:].max())
    for j in truth:
        diagnostics.append(dict(scenario=kind, rho=rho, rep=rep,
            variable=int(j), utility=float(utilities[j]),
            rank=int(1 + np.sum(utilities[1:] > utilities[j])),
            max_null_utility=max_null,
            empirical_gap=float(utilities[truth].min() - max_null)))
    for method in args.methods:
        path = out / "checkpoints" / f"{kind}_{rho}_{rep}_{method}.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
            continue
        if args.reuse_from:
            previous_path = args.reuse_from / "checkpoints" / path.name
            if previous_path.exists():
                row = json.loads(previous_path.read_text())
                budget_check(SimpleNamespace(privacy=row["privacy"]),
                             args.epsilon, total_delta(args.n))
                atomic_json(path, row)
                rows.append(row)
                continue
        fit_seed = derived_seed(args.seed, "fit", rep, method.removesuffix("_np"))
        params = parameters(method, n=args.n, p=args.p, regime="high",
                            s=args.size, intercept=True)
        start = time.perf_counter()
        fit = fit_pipeline(method, X, y, params, epsilon=args.epsilon,
                           delta=total_delta(args.n), tau=.5, seed=fit_seed)
        budget_check(fit, args.epsilon, total_delta(args.n))
        selected = np.asarray(fit.support)
        slopes = selected[selected != 0]
        hits = np.intersect1d(slopes, truth).size
        loss = check_loss(yt - Xt @ fit.coef, .5).mean()
        row = dict(scenario=kind, rho=rho, rep=rep, method=method, noise=args.noise,
            n_train=args.n, n_test=args.test_n,
            recall=hits/len(truth), coverage=int(hits == len(truth)),
            precision=hits/len(slopes) if len(slopes) else 0.,
            fdp=1-hits/len(slopes) if len(slopes) else 0.,
            pair_recall=np.isin([1, 2], slopes).mean(),
            weak_selected=int(2 in slopes),
            l2_error=float(np.linalg.norm(fit.coef-beta)),
            check_loss=float(loss), excess_check_loss=float(loss-oracle_loss),
            seconds=time.perf_counter()-start, selected=selected.tolist(),
            seed=fit_seed, data_seed=data_seed,
            parameters=params, privacy=fit.privacy, metadata=fit.metadata)
        atomic_json(path, row)
        rows.append(row)
        print(f"{kind} rho={rho} rep={rep+1} {method}: "
              f"recall={row['recall']:.1f}, excess={row['excess_check_loss']:.4f}",
              flush=True)
    return rows, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent worker processes; each uses one BLAS thread")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--p", type=int, default=6000)
    parser.add_argument("--test-n", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1.)
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026110601)
    parser.add_argument("--noise", choices=("gaussian", "t3"), default="t3")
    parser.add_argument("--methods", nargs="+", choices=METHODS + ["cai_iht"],
                        default=METHODS)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/screening_stress_t3_test5000_pilot")
    parser.add_argument("--reuse-from", type=Path,
                        help="Reuse matching completed fits from an earlier pilot")
    args = parser.parse_args()
    if not (args.n >= 20 and args.p > args.size >= 10 and args.test_n >= 20
            and args.repeats >= 1 and 0 < args.epsilon <= 1):
        parser.error("Require n,test-n>=20, p>size>=10, repeats>=1, 0<epsilon<=1")
    if args.workers < 1 or len(set(args.methods)) != len(args.methods):
        parser.error("Require workers>=1 and distinct methods")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output"] = str(out)
    config["reuse_from"] = str(args.reuse_from.resolve()) if args.reuse_from else None
    config.update(scenarios=SCENARIOS,
                  source_digest=source_digest(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  diagnostic_only=True, tau=.5, summary_format="mean_se",
                  delta=total_delta(args.n),
                  scope="Synthetic processed-input per-fit accounting; inherits pipeline guarantees")
    manifest = out / "config.json"
    if manifest.exists():
        previous_run = json.loads(manifest.read_text())
        current_run = json.loads(json.dumps(config))
        previous_run.pop("workers", None)
        current_run.pop("workers", None)
        if previous_run != current_run:
            raise ValueError("Output contains a different configuration or source version")
    if args.reuse_from:
        previous = json.loads((args.reuse_from / "config.json").read_text())
        if previous.get("noise", "gaussian") != args.noise:
            raise ValueError("Cannot reuse fits with a different error distribution")
        for key in ("n", "p", "test_n", "repeats", "epsilon", "size", "seed",
                    "scenarios", "source_digest", "tau", "delta"):
            if previous[key] != json.loads(json.dumps(config[key])):
                raise ValueError(f"Reuse configuration mismatch: {key}")
    atomic_json(manifest, config)
    rows, diagnostics = [], []
    jobs = [(args, kind, rho, rep) for kind, rho in SCENARIOS
            for rep in range(args.repeats)]
    pool = (ProcessPoolExecutor(max_workers=args.workers,
                               mp_context=multiprocessing.get_context("spawn"))
            if args.workers > 1 else nullcontext())
    with pool as executor:
        results = executor.map(run_job, jobs) if executor else map(run_job, jobs)
        for job_rows, job_diagnostics in results:
            rows.extend(job_rows)
            diagnostics.extend(job_diagnostics)
            summarize(rows, out)
            pd.DataFrame(diagnostics).to_csv(out / "rank_diagnostics.csv", index=False)
    print((out / "summary.txt").read_text(), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
