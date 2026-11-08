"""Algorithms 3.1, 3.3 and S7.2 of the supplied main/supplement manuscript.

All tuning and beta0 must be public or separately privatized for privacy.
Statistical bounds require the stated fixed-public-start distance conditions. These routines
never fit a scaler or choose parameters from the confidential sample.
private=False is a paired noise-removal ablation; clip_nonprivate=False gives
the ordinary, unclipped SQR algorithm. No raw objective/gradient is returned.
"""
import numpy as np
from .core import FitResult, data_xy, initial, integer, positive, quantile, score
from .privacy import rho_from_eps_delta, mu_from_eps_delta
from .selection import (supremum_rank_utilities, peeling, hard_threshold, noisy_ht,
                        validate_forced_indices)


def _setup(X, y, tau, h, eta, steps, beta0):
    X, y = data_xy(X, y)
    if beta0 is None:
        raise ValueError("beta0 is required; use pipeline.fit_pipeline for a budgeted initializer")
    quantile(tau)
    positive("h", h)
    positive("eta", eta)
    integer("steps", steps)
    return X, y, initial(beta0, X.shape[1])


def sqr_gd(X, y, *, tau=0.5, h, gamma, eta, steps, mu=None,
           epsilon=None, delta=None, beta0=None, private=True,
           clip_nonprivate=True, kernel="gaussian", seed=None):
    """Main Algorithm 3.1. Specify mu OR (epsilon,delta) for private GD."""
    X, y, b = _setup(X, y, tau, h, eta, steps, beta0)
    positive("gamma", gamma)
    rng = np.random.default_rng(seed)
    c = max(tau, 1-tau)
    if private:
        if mu is not None and (epsilon is not None or delta is not None):
            raise ValueError("Supply mu OR (epsilon,delta), not both")
        if mu is None:
            if epsilon is None or delta is None:
                raise ValueError("Private GD needs mu or epsilon and delta")
            mu = mu_from_eps_delta(epsilon, delta)
        positive("mu", mu)
        sigma = 2*gamma*c*np.sqrt(steps)/(len(y)*mu)
        ledger = {"type": "GDP", "mu": mu, "rho": mu*mu/2,
                  "epsilon": epsilon, "delta": delta, "gradient_sd": sigma}
    else:
        sigma, ledger = 0.0, {"type": "nonprivate"}
    for _ in range(steps):
        g = score(X, y, b, tau=tau, h=h, bound=gamma,
                  clipping="l2" if private or clip_nonprivate else "none", kernel=kernel)
        b -= eta*(g + (rng.normal(0, sigma, X.shape[1]) if private else 0))
    return FitResult(b, np.flatnonzero(b), ledger,
                     {"method": "DP-SQR-GD" if private else "nonprivate-SQR-GD",
                      "steps": steps, "kernel": kernel, "clip_nonprivate": clip_nonprivate})


def sqr_sr(X, y, *, tau=0.5, h, gamma, eta, steps, size,
           epsilon=None, delta=None, screen_fraction=0.5, forced=(), beta0=None,
           private=True, clip_nonprivate=True, kernel="gaussian", seed=None):
    """Rank-supremum screening then same-sample restricted refit; public/DP beta0 required."""
    X, y, b0 = _setup(X, y, tau, h, eta, steps, beta0)
    positive("gamma", gamma)
    integer("size", size)
    n, p = X.shape
    forced = validate_forced_indices(forced, p)
    if not len(forced) < size < p:
        raise ValueError("Paper requires len(forced)<size<p")
    if not 0 < screen_fraction < 1:
        raise ValueError("screen_fraction must be in (0,1)")
    candidates = np.setdiff1d(np.arange(p), forced)
    k = size-len(forced)
    rng = np.random.default_rng(seed)
    c = max(tau, 1-tau)
    sensitivity = 1./n
    if private:
        if epsilon is None or delta is None:
            raise ValueError("Private SR needs epsilon and delta")
        rho = rho_from_eps_delta(epsilon, delta)
        rho_scr, rho_ref = rho*screen_fraction, rho*(1-screen_fraction)
        scale = sensitivity*np.sqrt(2*k/rho_scr)
        mu_ref = np.sqrt(2*rho_ref)
        ledger = {"type": "zCDP", "rho": rho, "rho_screen": rho_scr,
                  "rho_refit": rho_ref, "epsilon": epsilon, "delta": delta,
                  "screen_laplace_scale": scale, "screen_sensitivity": sensitivity,
                  "screen_utility": "supremum_rank", "screen_rounds": k}
    else:
        scale, mu_ref, ledger = 0.0, None, {"type": "nonprivate"}
    utilities = supremum_rank_utilities(X, y, tau)
    chosen = candidates[peeling(utilities[candidates], k, scale, rng)]
    support = np.sort(np.r_[forced, chosen])
    refit = sqr_gd(X[:, support], y, tau=tau, h=h, gamma=gamma, eta=eta,
                   steps=steps, mu=mu_ref, beta0=b0[support], private=private,
                   clip_nonprivate=clip_nonprivate, kernel=kernel, seed=rng)
    b = np.zeros(p)
    b[support] = refit.coef
    if private:
        ledger["gradient_sd"] = refit.privacy["gradient_sd"]
    return FitResult(b, support, ledger,
                     {"method": "DP-SQR-SR" if private else "nonprivate-SQR-SR",
                      "steps": steps, "kernel": kernel, "forced": forced.tolist(),
                      "screen_utility": "supremum_rank",
                      "clip_nonprivate": clip_nonprivate})


def sqr_iht(X, y, *, tau=0.5, h, gamma, eta, steps, size,
            epsilon=None, delta=None, beta0=None, private=True,
            clip_nonprivate=True, kernel="gaussian", seed=None):
    """Supplement S7 Algorithm DP-SQR-IHT, manuscript zCDP calibration."""
    X, y, b = _setup(X, y, tau, h, eta, steps, beta0)
    positive("gamma", gamma)
    integer("size", size)
    if size > X.shape[1] or np.count_nonzero(b) > size:
        raise ValueError("Need size<=p and an at-most-size-sparse initializer")
    rng = np.random.default_rng(seed)
    sensitivity = 2*eta*gamma*max(tau, 1-tau)/len(y)
    if private:
        if epsilon is None or delta is None:
            raise ValueError("Private IHT needs epsilon and delta")
        rho = rho_from_eps_delta(epsilon, delta)
        ledger = {"type": "zCDP", "rho": rho, "epsilon": epsilon, "delta": delta,
                  "laplace_scale": sensitivity*np.sqrt(5*size*steps/(2*rho))}
    else:
        ledger = {"type": "nonprivate"}
    for _ in range(steps):
        g = score(X, y, b, tau=tau, h=h, bound=gamma,
                  clipping="linf" if private or clip_nonprivate else "none", kernel=kernel)
        v = b-eta*g
        if private:
            b, support = noisy_ht(v, size, sensitivity, rho/steps, rng)
        else:
            b, support = hard_threshold(v, size)
    return FitResult(b, support, ledger,
                     {"method": "DP-SQR-IHT" if private else "nonprivate-SQR-IHT",
                      "steps": steps, "kernel": kernel, "clip_nonprivate": clip_nonprivate,
                      "theorem_working_sparsity_ge_10": bool(size >= 10)})
