"""Rank-deviation screening, historical squared CQS, and Laplace selection."""
import numpy as np
from .core import data_xy, quantile, integer, positive


def cqs_utilities(X, y, tau):
    """O(np log n), strict predictor inequality and public row-position tie break.

    Raw utilities are internal/nonprivate diagnostics, NOT a DP release.
    """
    X, y = data_xy(X, y)
    quantile(tau)
    n, p = X.shape
    if n < 2:
        raise ValueError("CQS requires n>=2")
    K = int(np.ceil(n*tau))-1
    order_y = np.argsort(y, kind="stable")
    a = np.full(n, tau)
    a[order_y[:K]] -= 1
    utilities = np.empty(p)
    for j in range(p):
        order = np.argsort(X[:, j], kind="stable")
        values = X[order, j]
        starts = np.r_[0, 1+np.flatnonzero(values[1:] != values[:-1])]
        ends = np.r_[starts[1:], n]
        prefix = np.r_[0.0, np.cumsum(a[order])]
        utilities[j] = np.sum((ends-starts)*(prefix[starts]/n)**2)/n
    return utilities


def supremum_rank_utilities(X, y, tau):
    """Single-quantile supremum over ALL real thresholds; sensitivity <=1/n.

    Stable response ranks use the fixed public record position for ties.
    Only prefixes ending at complete predictor tie groups are attainable.
    The zero prefix and the threshold beyond the largest X are included.
    Unperturbed scores are internal statistics, not a private release.
    """
    X, y = data_xy(X, y)
    quantile(tau)
    n, p = X.shape
    if n < 2:
        raise ValueError("Rank screening requires n>=2")
    a = np.full(n, tau)
    a[np.argsort(y, kind="stable")[:int(np.ceil(n*tau))-1]] -= 1
    utilities = np.empty(p)
    for j in range(p):
        order = np.argsort(X[:, j], kind="stable")
        values = X[order, j]
        ends = np.r_[1+np.flatnonzero(values[1:] != values[:-1]), n]
        prefix = np.r_[0., np.cumsum(a[order])]
        utilities[j] = np.max(np.abs(prefix[np.r_[0, ends]]))/n
    return utilities


def peeling(scores, size, scale, rng):
    """Select max(score+fresh Laplace) each round; smallest index wins ties."""
    u = np.asarray(scores, dtype=float)
    integer("size", size)
    if u.ndim != 1 or size > len(u) or not np.isfinite(u).all():
        raise ValueError("Invalid scores or size")
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and nonnegative")
    available = np.ones(len(u), dtype=bool)
    selected = []
    for _ in range(size):
        idx = np.flatnonzero(available)
        noise = rng.laplace(0, scale, len(idx)) if scale else np.zeros(len(idx))
        winner = idx[np.argmax(u[idx]+noise)]
        selected.append(winner)
        available[winner] = False
    return np.array(selected, dtype=int)


def hard_threshold(v, size):
    v = np.asarray(v, dtype=float)
    idx = peeling(np.abs(v), size, 0, np.random.default_rng(0))
    out = np.zeros_like(v)
    out[idx] = v[idx]
    return out, np.sort(idx)


def noisy_ht(v, size, sensitivity, rho, rng):
    positive("sensitivity", sensitivity)
    positive("rho", rho)
    scale = sensitivity*np.sqrt(5*size/(2*rho))
    v = np.asarray(v, dtype=float)
    idx = peeling(np.abs(v), size, scale, rng)
    # Independent p-dimensional noise as explicitly written in S6.
    value_noise = rng.laplace(0, scale, len(v))
    out = np.zeros_like(v)
    out[idx] = v[idx]+value_noise[idx]
    return out, np.sort(idx)
