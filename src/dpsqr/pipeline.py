"""Consolidated-specification complete fits: private initialization plus remaining main budget.

Research interface with fixed public configuration and stage-budget checks.
The private initializer is budgeted; its statistical localization is not
established by the fixed-public-start estimation theorems.
Low-level core routines are not complete end-to-end mechanisms on their own.
"""
import numpy as np
from .core import FitResult, data_xy
from .initialization import (initialize_sqr, initialize_huber, initialize_ridge_ls,
                             chang_screen)
from .methods import sqr_gd, sqr_iht
from .selection import supremum_rank_utilities, peeling, validate_forced_indices
from .privacy import rho_from_eps_delta
from .step_policy import resolve_step
from .policy import allocation, VERSION


def _initializer(X, y, family, eps, delta, tau, radius, mode, rng, private,
                 intercept_index=None):
    # Module adds an intercept. Strip/reorder the already supplied column and
    # map the resulting coefficients back; no intercept is inferred from data.
    order = np.arange(X.shape[1])
    if intercept_index is not None:
        order = np.r_[intercept_index, order[order != intercept_index]]
        if not np.all(X[:, intercept_index] == 1):
            raise ValueError('Declared intercept column must contain ones')
        slopes = X[:, order[1:]]
    else:
        slopes = X
    kw = dict(epsilon_initialization=eps, delta_initialization=delta,
              output_radius=radius, privacy_mode=mode, rng=rng,
              private=private, intercept=intercept_index is not None)
    if family.startswith('sqr'):
        init = initialize_sqr(slopes, y, q=tau,
                              kernel='logistic' if family == 'sqr_logistic' else 'gaussian', **kw)
    elif family == 'huber':
        init = initialize_huber(slopes, y, **kw)
    else:
        init = initialize_ridge_ls(slopes, y, **kw)
    beta = np.empty(X.shape[1]); beta[order] = init.beta
    return beta, init


def fit_pipeline(name, X, y, params, *, epsilon, delta, tau, seed):
    X, y = data_xy(X, y)
    options, step_metadata = resolve_step(params)
    private = not name.endswith('_np')
    base = name.removesuffix('_np')
    rng = np.random.default_rng(seed)
    radius = options.pop('initializer_radius', None)
    mode = options.pop('initializer_mode', 'uniform_gap')
    intercept = options.pop('intercept_index', None)
    if intercept is not None:
        if (isinstance(intercept, (bool, np.bool_)) or not isinstance(intercept, (int, np.integer))
                or not 0 <= intercept < X.shape[1]):
            raise ValueError('intercept_index must be a nonnegative integer column index')
        if not np.all(X[:, intercept] == 1):
            raise ValueError('Declared intercept column must contain ones')
    if base in ('frappe', 'frappe_matched'):
        if tau != .5:
            raise ValueError('FRAPPE is a median-only comparison (tau=.5)')
        protocol = options.pop('protocol', 'original')
        if base == 'frappe' and protocol == 'liu200':
            from .baselines.frappe import fit_frappe_liu200
            fit = fit_frappe_liu200(
                X, y, epsilon=epsilon, delta=delta,
                intercept_index=intercept, private=private, seed=seed, **options)
            fit.metadata.update(
                protocol='FRAPPE-Liu200-SWP-v1',
                protocol_version='FRAPPE-Liu200-SWP-v1',
                method_variant='FRAPPE Liu200 subsample initializer with amplified initialization budget',
                comparison_role='research_approximation_revised_frappe',
                formal_dp_ready=False,
                finite_precision_dp_proved=False)
            fit.metadata.update(step_metadata)
            return fit
        if protocol != 'original':
            raise ValueError(f'Unknown FRAPPE protocol: {protocol}')
        from .baselines.frappe import fit_frappe_v1, frappe_budget_ledger
        fixed_scheme = 'matched_init' if base == 'frappe_matched' else 'original'
        configured_scheme = options.pop('budget_scheme', fixed_scheme)
        if configured_scheme != fixed_scheme:
            raise ValueError(f'{base} fixes budget_scheme={fixed_scheme}')
        fit = fit_frappe_v1(X, y, epsilon=epsilon, delta=delta,
                            intercept_index=intercept, private=private, seed=rng,
                            budget_scheme=fixed_scheme, **options)
        ledger = frappe_budget_ledger(epsilon, delta, fixed_scheme)
        fit.privacy.update(stages=ledger['stages'] if private else [],
            actual_epsilon=epsilon if private else 0.,
            actual_delta=delta if private else 0., unused_delta=0. if private else None)
        fit.metadata.update(protocol='README-v1.1',
            method_variant=('FRAPPE matched-init post-hoc ablation'
                            if base == 'frappe_matched' else
                            'FRAPPE + full-sample Su-Wang initializer + projection'),
            comparison_role=('post_hoc_budget_sensitivity_and_algorithm_contribution'
                             if base == 'frappe_matched' else
                             'original_end_to_end_comparison'),
            parameter_sources='public configuration and postprocessing of budgeted DP scale',
            p_over_n=X.shape[1]/len(y),
            dimension_scope='p<n' if X.shape[1]<len(y) else 'p>=n empirical extension')
        fit.metadata.update(step_metadata)
        return fit
    known = {'sqr_gd', 'sqr_sr', 'sqr_iht', 'huber_gd', 'huber_iht', 'cai_iht',
             'tran_dpsgd', 'tran_opm'}
    if base == 'cai_iht' and 'coordinate_clip' in options:
        cap = options.pop('coordinate_clip')
        if not np.isfinite(cap) or cap <= 0:
            raise ValueError('coordinate_clip must be positive and public')
        X = np.clip(X, -cap, cap)
    if base not in known:
        raise ValueError(f'Unknown method: {name}')
    if 'beta0' in options or 'legacy_zero_init' in options:
        raise ValueError('The complete pipeline constructs and accounts its own initializer')
    n, p = X.shape
    budget = allocation(base, epsilon, delta, n)
    eps_s, del_s = budget['eps_s'], budget['del_s']
    eps_i, del_i = budget['eps_i'], budget['del_i']
    eps_m, del_m = budget['eps_m'], budget['del_m']
    sparse_key = {'sqr_sr':'size', 'sqr_iht':'size', 'huber_iht':'s', 'cai_iht':'sparsity'}.get(base)
    if sparse_key:
        options.setdefault(sparse_key, min(p, 10))
    working_sparsity = options.get(sparse_key) if sparse_key else None
    stages = []
    support = np.arange(p)
    high = base in ('sqr_iht', 'huber_iht')
    if base == 'sqr_sr':
        size = options.pop('size')
        forced = validate_forced_indices(options.pop('forced', ()), p)
        if intercept is not None and intercept not in forced:
            forced = np.r_[forced, intercept]
        if len(set(forced)) != len(forced) or np.any((forced < 0) | (forced >= p)):
            raise ValueError('Invalid forced support')
        if not len(forced) < size <= p:
            raise ValueError('SR requires len(forced)<size<=p')
        if 'screen_fraction' in options:
            raise ValueError('v1 SR fixes three stages; remove legacy screen_fraction')
        candidates = np.setdiff1d(np.arange(p), forced)
        k = size-len(forced)
        streams = np.random.SeedSequence(seed).spawn(3)
        screen_rng, init_rng, main_rng = [np.random.default_rng(s) for s in streams]
        scale = (1./n*np.sqrt(2*k/rho_from_eps_delta(eps_s, del_s)) if private else 0.)
        selected = candidates[peeling(supremum_rank_utilities(X, y, tau)[candidates],
                                      k, scale, screen_rng)]
        support = np.sort(np.r_[forced, selected])
        stages.append(dict(stage='rank_supremum_screen', epsilon=eps_s, delta=del_s,
                           sensitivity=1./n, laplace_scale=scale, rounds=k))
        rng = init_rng
    elif high:
        key = {'sqr_iht': 'size', 'huber_iht': 's', 'cai_iht': 'sparsity'}[base]
        support, _ = chang_screen(X, y, options[key], eps_s,
                                  intercept_index=intercept, rng=rng, private=private)
        stages.append(dict(stage='Chang_conservative_screen', epsilon=eps_s, delta=del_s))
    family = ('huber' if base.startswith('huber') else 'ls' if base == 'cai_iht'
              else 'sqr_logistic' if base.startswith('tran') else 'sqr')
    local_intercept = (int(np.flatnonzero(support == intercept)[0])
                       if intercept is not None else None)
    zero_initialization = base in ('cai_iht', 'tran_dpsgd', 'tran_opm')
    if zero_initialization:
        from types import SimpleNamespace
        local_beta = np.zeros(len(support))
        init = SimpleNamespace(method='public_zero', scale=None, sensitivity_bound=0.,
                               gaussian_sigma=0., metadata={'uniform_gap_bound': 0.})
    else:
        local_beta, init = _initializer(X[:, support], y, family, eps_i, del_i, tau,
                                        radius, mode, rng, private, local_intercept)
        stages.extend([dict(stage='scale', epsilon=eps_i/4, delta=0.),
                       dict(stage='initial_output', epsilon=3*eps_i/4, delta=del_i)])
    b0 = np.zeros(p); b0[support] = local_beta
    stages.append(dict(stage='main', epsilon=eps_m, delta=0. if base == 'tran_opm' else del_m))
    if base == 'sqr_sr':
        rng = main_rng
    if base in ('sqr_gd', 'sqr_sr'):
        fit = sqr_gd(X[:, support], y, tau=tau, beta0=local_beta,
                     epsilon=eps_m if private else None, delta=del_m if private else None,
                     private=private, seed=rng, **options)
        full = np.zeros(p); full[support] = fit.coef
        fit.coef = full; fit.support = support if base == 'sqr_sr' else np.flatnonzero(full)
    elif base == 'sqr_iht':
        fit = sqr_iht(X, y, tau=tau, beta0=b0, epsilon=eps_m if private else None,
                      delta=del_m if private else None, private=private, seed=rng, **options)
    elif base.startswith('huber'):
        from .baselines.huber import huber_gd, huber_iht
        work_denom = (options['s']*np.log(p)+np.log(n)) if high else (p+np.log(n))
        options.setdefault('tau', .04*init.scale*np.sqrt(n*epsilon/work_denom))
        huber_order = np.arange(p)
        if high and intercept is not None:
            huber_order = np.r_[intercept, huber_order[huber_order != intercept]]
            options['intercept'] = True
        # Avoid a full advanced-index copy when the public intercept is already first.
        huber_X = X if np.array_equal(huber_order,np.arange(p)) else X[:,huber_order]
        fit = (huber_iht if high else huber_gd)(huber_X, y, beta0=b0[huber_order], epsilon=eps_m,
                     delta=del_m, private=private, rng=rng, **options)
        restored = np.empty(p); restored[huber_order] = fit.coef
        fit.coef = restored; fit.support = np.flatnonzero(restored)
    elif base == 'cai_iht':
        from .baselines.cai_iht import fit_cai_iht
        options.setdefault('response_clip', np.sqrt(2*np.log(n)))
        options.setdefault('beta_radius', radius)
        if not zero_initialization and options['beta_radius'] != radius:
            raise ValueError('Cai initial/main radii must match')
        fit = fit_cai_iht(X, y, beta0=b0, epsilon=eps_m, delta=del_m,
                          private=private, seed=rng, **options)
    else:
        from .baselines.tran import tran_dpsgd, tran_opm
        if not private:
            raise ValueError('DP-SGD/OPM nonprivate counterpart is not part of this adapter')
        if base == 'tran_dpsgd':
            options.setdefault('step_schedule', 'paper_constant')
            options.setdefault('parameter_radius', 10.)
            options['delta'] = del_m
        fit = (tran_dpsgd if base == 'tran_dpsgd' else tran_opm)(
            X, y, tau=tau, beta0=b0, epsilon=eps_m, rng=rng, **options)
    main = dict(fit.privacy)
    fit.privacy = dict(type='sequential_complete_fit' if private else 'nonprivate',
                      epsilon=epsilon if private else None, delta=delta if private else None,
                      stages=stages if private else [], main=main,
                      actual_epsilon=sum(s['epsilon'] for s in stages) if private else 0.,
                      actual_delta=sum(s['delta'] for s in stages) if private else 0.,
                      unused_delta=delta-sum(s['delta'] for s in stages) if private else None,
                      guarantee_verified=False if private else None,
                      finite_precision_dp_proved=False if private else None)
    fit.metadata.update(protocol='SR-rank-supremum-20261019', initialization=init.method,
                        initialization_mode='public_zero' if zero_initialization else mode if private else 'nonprivate',
                        initialization_solver_mode=mode,
                        initialization_sensitivity=init.sensitivity_bound,
                        initialization_noise_sd=init.gaussian_sigma,
                        initialization_gap_bound=init.metadata.get('uniform_gap_bound'),
                        initialization_support=support.tolist(),
                        formal_dp_ready=False,
                        admission_scope='configuration admission is assessed by experiment runner',
                        method_variant=('OPM + public zero initialization; replacement calibration'
                                        if base=='tran_opm' else base+' + '+init.method),
                        parameter_sources='public configuration and postprocessing of budgeted DP scale',
                        working_sparsity=working_sparsity,
                        budget_policy=VERSION,
                        screen_utility='supremum_rank' if base=='sqr_sr' else None,
                        initialization_budget_epsilon=eps_i,
                        main_budget_epsilon=eps_m)
    fit.metadata.update(step_metadata)
    return fit
