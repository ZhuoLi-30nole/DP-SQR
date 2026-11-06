"""Main-text conversion equations; no subsampling accountant is implied."""
import numpy as np
from scipy.special import log_ndtr
from scipy.optimize import brentq
from .core import positive


def validate_budget(epsilon, delta):
    positive("epsilon", epsilon)
    if not np.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must be in (0,1)")


def rho_from_eps_delta(epsilon, delta):
    validate_budget(epsilon, delta)
    L = np.log(1/delta)
    # Rationalized Eq. rhostar avoids cancellation for small epsilon.
    return (epsilon / (np.sqrt(L+epsilon)+np.sqrt(L)))**2


def eps_from_rho_delta(rho, delta):
    positive("rho", rho)
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0,1)")
    return rho + 2*np.sqrt(rho*np.log(1/delta))


def gdp_delta(epsilon, mu):
    positive("mu", mu)
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    log_a = log_ndtr(-epsilon/mu+mu/2)
    log_b = epsilon + log_ndtr(-epsilon/mu-mu/2)
    return float(np.exp(log_a)*(-np.expm1(min(0.0, log_b-log_a))))


def mu_from_eps_delta(epsilon, delta):
    """Exact GDP-to-(epsilon,delta) inversion for GD only."""
    validate_budget(epsilon, delta)
    upper = 1.0
    while gdp_delta(epsilon, upper) < delta:
        upper *= 2
    return brentq(lambda mu: gdp_delta(epsilon, mu)-delta, 1e-12, upper, xtol=1e-13)
