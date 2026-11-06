"""Cai, Wang, Zhang (AoS 2021), Algorithms 3.2 and 4.2, published pp2834/2839.

Published Peeling uses sqrt(3*s*log(1/delta)), not Huber's sqrt(5*...).
No residual clipping or sample splitting is substituted for response clipping.
"""
import numpy as np
from ..core import FitResult, data_xy, initial, positive, integer
from ..privacy import validate_budget


def _start(beta0, p, legacy_zero_init):
    if beta0 is None and not legacy_zero_init:
        raise ValueError('beta0 is required; pass legacy_zero_init=True only for source comparison')
    return initial(beta0, p)


def cai_peeling(v, sparsity, epsilon, delta, sensitivity, rng):
    """Published Algorithm 3.2, independent selection and value noise."""
    integer('sparsity', sparsity)
    validate_budget(epsilon, delta)
    positive('sensitivity', sensitivity)
    v = np.asarray(v, dtype=float)
    if v.ndim != 1 or sparsity > len(v) or not np.isfinite(v).all():
        raise ValueError('Invalid vector/sparsity')
    scale = sensitivity*2*np.sqrt(3*sparsity*np.log(1/delta))/epsilon
    selected = np.zeros(len(v), dtype=bool)
    for _ in range(sparsity):
        scores = np.abs(v)+rng.laplace(0, scale, len(v))
        scores[selected] = -np.inf
        selected[np.argmax(scores)] = True
    out = np.where(selected, v+rng.laplace(0, scale, len(v)), 0.)
    return out


def cai_source_step(sparsity, covariance_lower, covariance_upper):
    """Cai Theorem 4.4: eta=s/(6L), using public sparse moment bounds.

    D2' requires 1/L <= s*lambda_min <= s*lambda_max <= L.
    This supplies a step reference; it does not verify all D1'/D2' assumptions.
    """
    integer('sparsity', sparsity)
    positive('covariance_lower',covariance_lower)
    positive('covariance_upper',covariance_upper)
    if covariance_lower > covariance_upper:
        raise ValueError('covariance_lower must not exceed covariance_upper')
    L = max(sparsity*covariance_upper, 1/(sparsity*covariance_lower))
    return float(sparsity/(6*L)), float(L)


def fit_cai_iht(X, y, *, epsilon, delta, sparsity, step_size, steps,
                response_clip, x_bound, beta_radius=1., beta0=None,
                private=True, seed=None, legacy_zero_init=False,
                public_covariance_lower=.8, public_covariance_upper=1.25, noise_scale=1.):
    """Algorithm 4.2 with explicit public coordinate bound x_bound.

    Set c_x=sqrt(s)*x_bound for the fixed working-s support sensitivity;
    Lemma 4.4 gives B=4*(R+C*c_x)*c_x/sqrt(s). This is a conservative
    calibration for |X_ij|<=x_bound and ||beta||2<=C, ||beta||0<=s.
    This does NOT assert the dimension-uniform D1'/D2' statistical assumptions.
    Design outside the declared domain is rejected, not silently rescaled.
    """
    X, y = data_xy(X, y)
    source_step = isinstance(step_size,str) and step_size == 'source_public_covariance'
    source_L = None
    if source_step:
        step_size, source_L = cai_source_step(sparsity, public_covariance_lower, public_covariance_upper)
    source_response = isinstance(response_clip,str) and response_clip == 'source_simulation'
    if source_response:
        positive('noise_scale',noise_scale)
        response_clip = 4*noise_scale*np.sqrt(np.log(len(y)))  # Section 5.1
    for key, val in [('step_size',step_size),('response_clip',response_clip),
                     ('x_bound',x_bound),('beta_radius',beta_radius)]:
        positive(key,val)
    integer('sparsity',sparsity); integer('steps',steps)
    n,p = X.shape
    max_abs=max(np.max(np.abs(X[start:start+128])) for start in range(0,n,128))
    if sparsity>p or max_abs>x_bound*(1+1e-12):
        raise ValueError('Need sparsity<=p and public domain |X_ij|<=x_bound')
    b = _start(beta0,p,legacy_zero_init)
    if np.count_nonzero(b)>sparsity or np.linalg.norm(b)>beta_radius*(1+1e-12):
        raise ValueError('Initializer must be s-sparse and inside the public L2 ball')
    rng = np.random.default_rng(seed)
    cx = np.sqrt(sparsity)*x_bound
    B = 4*(response_clip+beta_radius*cx)*cx/np.sqrt(sparsity)
    sensitivity = step_size*B/n
    if private:
        validate_budget(epsilon,delta)
        eps_t,delta_t = epsilon/steps,delta/steps
        scale = sensitivity*2*np.sqrt(3*sparsity*np.log(1/delta_t))/eps_t
    else:
        scale=0.
    y_clip = np.clip(y,-response_clip,response_clip)
    for _ in range(steps):
        v = b-step_size/n*(X.T@(X@b-y_clip))
        if private:
            b = cai_peeling(v,sparsity,eps_t,delta_t,sensitivity,rng)
        else:
            idx = np.argsort(-np.abs(v),kind='stable')[:sparsity]
            b = np.zeros(p); b[idx]=v[idx]
        b *= min(1.,beta_radius/max(np.linalg.norm(b),np.finfo(float).tiny))
    return FitResult(b,np.flatnonzero(b),
        {'type':'approximate_DP' if private else 'nonprivate',
         'epsilon':epsilon if private else None,'delta':delta if private else None,
         'laplace_scale':scale,'composition':'basic epsilon/T,delta/T',
         'guarantee_verified':False,
         'finite_precision_dp_proved':False if private else None},
        {'algorithm':'Cai et al. 2021 published Algorithm 4.2',
         'peeling_algorithm':'published 3.2; sqrt(3s), not sqrt(5s)',
         'B':B,'c_x_working_support':cx,'coordinate_bound':x_bound,
         'per_step_peeling_sensitivity':sensitivity,
         'statistical_D1_D2_uniform_conditions_verified':False,
         'response_clipping':response_clip,'beta_radius':beta_radius,
         'initialization':'explicit beta0' if beta0 is not None else 'legacy zero',
         'privacy_assumption_note': 'Cai Lemma 4.4 relies on (P1\u2032),(D1\u2032); the coordinate bound used here is a stronger public domain condition, but the complete numerical mechanism audit remains incomplete',
         'steps':steps,'step_size':step_size,
         'step_policy':'Cai Thm4.4 public covariance reference' if source_step else 'explicit public step',
         'public_covariance_L':source_L,
         'response_policy':'Cai Sec5.1 4 sigma sqrt(log n)' if source_response else 'explicit public bound',
         'iteration_policy':'explicit T; unknown theorem rho not asserted',
         'source_statistical_rate_claimed':False})
