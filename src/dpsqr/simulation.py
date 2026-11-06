"""Synthetic data and metrics. All outputs here are for simulated/public data.

Simulation random seeds and losses MUST NOT be published for a confidential-data
fit under the estimator's privacy account. This module is not a DP evaluator.
"""
from dataclasses import dataclass
import numpy as np
from scipy import stats
from .core import integer, quantile


@dataclass
class SyntheticSample:
    X: np.ndarray
    y: np.ndarray
    beta: np.ndarray
    noise: str
    tau: float


def generate(n, p, sparsity, *, tau=0.5, noise="gaussian", seed=0,
             signal=0.6, correlation=0.0, intercept=False,
             design_distribution="uniform_copula", coefficient_pattern="alternating",
             coefficient_seed=0, noise_scale=1.0, public_coordinate_clip=None):
    """Known conditional quantile with an explicit public design distribution.

    Gaussian-copula Uniform[-1,1] design has a public coordinate bound 1.
    correlation controls latent AR(1) Gaussian correlation, not Pearson corr(X).
    Errors subtract the DISTRIBUTION's tau quantile, never the sample quantile.
    p includes the intercept, if requested. sparsity includes its nonzero value.
    ``gaussian`` uses AR(1) normal slopes, matching Chang's design distribution.
    A coordinate cap, when supplied, is applied BEFORE constructing Y, and is
    therefore a separately named bounded-design adaptation, not exact Gaussian.
    Defaults preserve the historical low-dimensional experiments byte for byte.
    """
    for name, value in [("n", n), ("p", p), ("sparsity", sparsity)]:
        integer(name, value)
    quantile(tau)
    if sparsity > p or not 0 <= correlation < 1:
        raise ValueError("Need sparsity<=p and 0<=correlation<1")
    if design_distribution not in ("uniform_copula", "gaussian"):
        raise ValueError("Unknown design_distribution")
    if coefficient_pattern not in ("alternating", "fixed_rademacher"):
        raise ValueError("Unknown coefficient_pattern")
    if not np.isfinite(signal) or signal <= 0 or not np.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError("signal and noise_scale must be finite and positive")
    if public_coordinate_clip is not None and (
            not np.isfinite(public_coordinate_clip) or public_coordinate_clip <= 0):
        raise ValueError("public_coordinate_clip must be finite and positive")
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, p))
    if correlation:
        for j in range(1, p):
            z[:, j] = correlation*z[:, j-1] + np.sqrt(1-correlation**2)*z[:, j]
    X = 2*stats.norm.cdf(z)-1 if design_distribution == "uniform_copula" else z
    if public_coordinate_clip is not None:
        X = np.clip(X, -public_coordinate_clip, public_coordinate_clip)
    if intercept:
        X[:, 0] = 1
    beta = np.zeros(p)
    signs = np.where(np.arange(sparsity)%2 == 0, 1., -1.)
    if coefficient_pattern == "fixed_rademacher":
        # RandomState reproduces the author's np.random.seed(0)/binomial signs.
        # This RNG is separate from training/test data and fixed across repeats.
        signs = 2*np.random.RandomState(coefficient_seed).binomial(1, .5, sparsity)-1
    beta[:sparsity] = signal * signs
    families = {"gaussian": stats.norm, "t3": stats.t(df=3), "t2.25": stats.t(df=2.25),
                "cauchy": stats.cauchy, "exponential": stats.expon}
    if noise not in families:
        raise ValueError(f"Unknown noise family {noise}")
    family = families[noise]
    error = noise_scale*(family.rvs(size=n, random_state=rng)-family.ppf(tau))
    return SyntheticSample(X, X@beta+error, beta, noise, tau)


