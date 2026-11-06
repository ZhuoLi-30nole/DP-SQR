"""Chang et al. DP-Huber Algorithms 1--3, official-source port.

X is a complete design matrix; add an intercept explicitly if desired.
No private-data standardization or automatic robustification tuning is done.
Official source: 10896c61ab249f7491536d2503822c252fff1c9a.
"""
import numpy as np
from ..core import FitResult, data_xy, initial, integer, positive
from ..privacy import eps_from_rho_delta


def _start(beta0, p, legacy_zero_init):
    if beta0 is None and not legacy_zero_init:
        raise ValueError('beta0 is required; pass legacy_zero_init=True only for source comparison')
    return initial(beta0, p)


def _budget(epsilon, delta):
    positive('epsilon', epsilon)
    if not 0 < delta < 1:
        raise ValueError('delta must be in (0,1)')


def _clip(X, bound, order):
    if order == np.inf:
        norms=np.empty(len(X))
        for start in range(0,len(X),128):
            norms[start:start+128]=np.max(np.abs(X[start:start+128]),axis=1)
    else:
        norms = np.linalg.norm(X, ord=order, axis=1)
    return X * np.minimum(1., bound / np.maximum(norms, np.finfo(float).tiny))[:, None]


def _clip_with_intercept(X,bound):
    # Retain the intercept and write clipped slopes directly into one buffer.
    if X.shape[1]==1:return X.copy()
    norms=np.empty(len(X))
    for start in range(0,len(X),128):
        norms[start:start+128]=np.max(np.abs(X[start:start+128,1:]),axis=1)
    factor=np.minimum(1.,bound/np.maximum(norms,np.finfo(float).tiny))
    Z=np.empty_like(X);Z[:,0]=X[:,0]
    np.multiply(X[:,1:],factor[:,None],out=Z[:,1:])
    return Z


def compute_sigma_scale(T, B, epsilon, delta):
    """Chang low-dimensional Gaussian calibration, only using valid branches.

    Basic composition uses the paper's Gaussian mechanism at per-step
    ``(epsilon/T, delta/T)`` and is admitted here only for epsilon/T <= 1.
    The advanced-composition alternative is available only on its stated
    total-budget domain epsilon <= 1 and delta <= .01. The smaller scale is
    selected among branches whose theorem conditions hold.
    """
    integer('T', T); positive('B', B); _budget(epsilon, delta)
    scales = []
    if epsilon / T <= 1:
        scales.append(2 * B * T * np.sqrt(2 * np.log(1.25 * T / delta)) / epsilon)
    if epsilon <= 1 and delta <= .01:
        scales.append(2 * B * np.sqrt(5 * T * np.log(2 / delta) *
                                       np.log(2.5 * T / delta)) / epsilon)
    if not scales:
        raise ValueError("No certified Chang low-dimensional calibration applies: require epsilon/T<=1 for basic composition, or epsilon<=1 and delta<=.01 for advanced composition")
    return min(scales)


def highdim_step_budget(T, epsilon, delta):
    """Return the lower-noise *valid* Chang NoisyHT per-step allocation.

    The paper's NoisyHT lemma requires per-call epsilon <= .5 and delta <=
    .011. Advanced composition additionally requires total epsilon <= 1 and
    delta <= .01. The selected allocation is chosen by comparing the actual
    corresponding NoisyHT scales, as in the paper.
    """
    integer('T', T); _budget(epsilon, delta)
    candidates = []
    eps_basic, del_basic = epsilon / T, delta / T
    if eps_basic <= .5 and del_basic <= .011:
        basic_scale = np.sqrt(np.log(1 / del_basic)) / eps_basic
        candidates.append((basic_scale, eps_basic, del_basic))
    if epsilon <= 1 and delta <= .01:
        eps_adv = epsilon / np.sqrt(2.5 * T * np.log(2 / delta))
        del_adv = delta / (2 * T)
        if eps_adv <= .5 and del_adv <= .011:
            adv_scale = np.sqrt(np.log(1 / del_adv)) / eps_adv
            candidates.append((adv_scale, eps_adv, del_adv))
    if not candidates:
        raise ValueError("No certified Chang NoisyHT composition applies: require s>=10 and per-call epsilon<=.5, delta<=.011; advanced composition also requires total epsilon<=1, delta<=.01")
    _, eps_t, del_t = min(candidates, key=lambda item: item[0])
    return eps_t, del_t


def noisy_hard_threshold(v, s, epsilon, delta, sensitivity, rng):
    integer('s', s); _budget(epsilon, delta)
    if s > len(v):
        raise ValueError('s exceeds dimension')
    scale = sensitivity * 2 * np.sqrt(5 * s * np.log(1 / delta)) / epsilon
    chosen = np.zeros(len(v), dtype=bool)
    for _ in range(s):
        scores = np.abs(v) + rng.laplace(0, scale, size=len(v))
        scores[chosen] = -np.inf
        chosen[np.argmax(scores)] = True
    noise = rng.laplace(0, scale, size=len(v))
    return np.where(chosen, v + noise, 0.)


def huber_gd(X, y, *, tau, B, T=50, lr=0.2, epsilon=1., delta=1e-6,
             beta0=None, private=True, rng=None, sigma_scale=None,
             legacy_zero_init=False, clip_nonprivate=False):
    """Algorithm 1; fixed public tau/B/T/lr and optional public or DP beta0.

    sigma_scale overrides calibration for exact upstream numerical comparison.
    Such an override is explicitly marked unverified in returned metadata.
    Nonprivate defaults to official gd; clip_nonprivate=True gives a
    matched-clipping, noise-removal diagnostic without changing private updates.
    """
    X, y = data_xy(X, y); positive('tau', tau); positive('B', B)
    positive('lr', lr); integer('T', T)
    b = _start(beta0, X.shape[1], legacy_zero_init); rng = np.random.default_rng() if rng is None else rng
    custom_scale = sigma_scale is not None
    if private:
        _budget(epsilon, delta)
        sigma_scale = compute_sigma_scale(T, B, epsilon, delta) if sigma_scale is None else sigma_scale
        if sigma_scale < 0 or not np.isfinite(sigma_scale):
            raise ValueError('sigma_scale must be nonnegative finite')
        Z = _clip(X, B, 2)
    else:
        Z, sigma_scale = (_clip(X, B, 2) if clip_nonprivate else X), 0.
    for _ in range(T):
        score = np.clip(y - X @ b, -tau, tau)
        noise = tau * sigma_scale * rng.standard_normal(X.shape[1]) if private else 0.
        b += (lr / len(y)) * (Z.T @ score + noise)
    return FitResult(b, np.flatnonzero(b),
        {'private': private, 'epsilon': epsilon if private else None, 'delta': delta if private else None,
         'paper_budget_domain': (epsilon <= 1 and delta <= .01) if private else None,
         'guarantee_verified': False,
         'finite_precision_dp_proved': False if private else None},
        {'algorithm': 'Chang Algorithm 1', 'T': T, 'sigma_scale': sigma_scale,
         'learning_rate':lr, 'clip_nonprivate':clip_nonprivate,
         'custom_noise_scale_unverified': custom_scale, 'initialization_and_tuning_budget_included': False,
         'initialization': 'explicit beta0' if beta0 is not None else 'legacy zero',
         'privacy_verified': False})


def huber_iht(X, y, *, s, tau, B, T=50, lr=0.01, epsilon=1., delta=1e-6,
              beta0=None, private=True, rng=None, intercept=False,
              legacy_zero_init=False, clip_nonprivate=False):
    """Algorithm 3 with Algorithm 2 noisy peeling. X already includes intercept.

    intercept=True excludes column zero from radial infinity clipping as upstream.
    Require B>=1 then, preserving the declared score sensitivity bound.
    """
    X, y = data_xy(X, y); positive('tau', tau); positive('B', B)
    positive('lr', lr); integer('T', T); integer('s', s)
    if s > X.shape[1]: raise ValueError('s exceeds dimension')
    if intercept and (B < 1 or not np.allclose(X[:, 0], 1)):
        raise ValueError('intercept=True requires first column ones and B>=1')
    rng = np.random.default_rng() if rng is None else rng
    b = _start(beta0, X.shape[1], legacy_zero_init)
    if private:
        eps_t, del_t = highdim_step_budget(T, epsilon, delta)
        # Independent accounting of the SAME Laplace noise, valid for s>=1:
        # noisy-max + independent selected-value release has rho<=5s*l^2/(2b^2).
        # Chang's b=2*l*sqrt(5s*log(1/del_t))/eps_t gives the bound below.
        rho_main = T * eps_t**2 / (8*np.log(1/del_t))
        if eps_from_rho_delta(rho_main, delta) > epsilon*(1+1e-12):
            raise ValueError('Independent NoisyHT composition exceeds main budget')
        Z = _clip_with_intercept(X,B) if intercept else _clip(X,B,np.inf)
    else:
        Z = (_clip_with_intercept(X,B) if intercept else _clip(X,B,np.inf)) if clip_nonprivate else X
    for _ in range(T):
        b += (lr / len(y)) * (Z.T @ np.clip(y - X @ b, -tau, tau))
        if private:
            b = noisy_hard_threshold(b, s, eps_t, del_t, 2 * lr * B * tau / len(y), rng)
        else:
            support = np.argsort(-np.abs(b))[:s]
            out = np.zeros_like(b); out[support] = b[support]; b = out
    return FitResult(b, np.flatnonzero(b),
        {'private': private, 'epsilon': epsilon if private else None, 'delta': delta if private else None,
         'paper_budget_domain': (epsilon <= 1 and delta <= .01 and s >= 10 and eps_t <= .5 and del_t <= .011) if private else None,
         'accountant': 'Laplace selection/value zCDP bound; original noise scale unchanged',
         'rho_main_bound': rho_main if private else 0.,
         'epsilon_bound_at_main_delta': eps_from_rho_delta(rho_main, delta) if private else 0.,
         'guarantee_verified': False,
         'finite_precision_dp_proved': False if private else None},
        {'algorithm': 'Chang Algorithm 3', 'T': T,
         'learning_rate':lr, 'clip_nonprivate':clip_nonprivate,
         'initialization_and_tuning_budget_included': False,
         'initialization': 'explicit beta0' if beta0 is not None else 'legacy zero',
         'privacy_verified': False})
