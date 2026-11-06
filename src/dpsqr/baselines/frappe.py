"""FRAPPE implementations.

``fit_frappe`` below is a legacy arXiv-calibration reproduction retained for
backwards compatibility; it is unsupported by the project v1 protocol and is
not DP-qualified. ``fit_frappe_v1`` implements README v1.
"""
import numpy as np
import ctypes
import hashlib
import os
import subprocess
import tempfile
from scipy.optimize import minimize
from ..core import FitResult, data_xy, positive, integer, radial_clip
from ..privacy import validate_budget
from ..privacy import rho_from_eps_delta


_FRAPPE_CPP = r'''#include <algorithm>
#include <cmath>
extern "C" long long coordinate_descent(const double* Z, const double* penalty,
    const double* column_norm2, double* beta, double* work_resid,
    long long n, long long p, double lambda_l2_over_rho, double rho,
    long long max_sweeps) {
  for (long long sweep=0; sweep<max_sweeps; ++sweep) {
    double max_change=0.0;
    for (long long j=0; j<p; ++j) {
      const double old=beta[j];
      double dot=0.0;
      for (long long i=0; i<n; ++i) dot += Z[j*n+i]*work_resid[i];
      const double v=dot+column_norm2[j]*old;
      const double threshold=penalty[j]/rho;
      const double updated=std::copysign(std::max(std::abs(v)-threshold,0.0),v)/
          (column_norm2[j]+lambda_l2_over_rho);
      beta[j]=updated;
      const double change=updated-old;
      for (long long i=0; i<n; ++i) work_resid[i]-=Z[j*n+i]*change;
      max_change=std::max(max_change,std::abs(change));
    }
    double beta_max=0.0;
    for (long long j=0; j<p; ++j) beta_max=std::max(beta_max,std::abs(beta[j]));
    if (max_change<=1e-8*(1.0+beta_max)) return sweep+1;
  }
  return max_sweeps;
}
'''


def _compiled_coordinate_kernel():
    digest=hashlib.sha256(_FRAPPE_CPP.encode()).hexdigest()[:16]
    directory=os.path.join(tempfile.gettempdir(),"dpsqr_frappe_kernel")
    os.makedirs(directory,exist_ok=True)
    source=os.path.join(directory,f"frappe_{digest}.cpp")
    library=os.path.join(directory,f"frappe_{digest}.so")
    if not os.path.exists(library):
        with open(source,"w",encoding="utf-8") as handle: handle.write(_FRAPPE_CPP)
        tmp=library+f".{os.getpid()}.tmp"
        subprocess.run(["c++","-O3","-std=c++17","-shared","-fPIC",source,"-o",tmp],
                       check=True,capture_output=True,text=True)
        os.replace(tmp,library)
    fn=ctypes.CDLL(library).coordinate_descent
    ptr=ctypes.POINTER(ctypes.c_double)
    fn.argtypes=[ptr,ptr,ptr,ptr,ptr,ctypes.c_longlong,ctypes.c_longlong,
                 ctypes.c_double,ctypes.c_double,ctypes.c_longlong]
    fn.restype=ctypes.c_longlong
    return fn,library


def soft_threshold(v, threshold):
    return np.sign(v)*np.maximum(np.abs(v)-threshold,0.)


def frappe_kernel(u, kernel='paper_sextic'):
    """Paper simulation's signed sextic (called 'biweight' there) by default."""
    u=np.asarray(u,dtype=float)
    x=np.clip(u,-1,1)
    if kernel=='paper_sextic':
        values=(105-525*x*x+735*x**4-315*x**6)/64
    elif kernel=='biweight':
        values=15/16*(1-x*x)**2
    else:
        raise ValueError('kernel must be paper_sextic or biweight (explicit alternative)')
    return np.where(np.abs(u)<1,values,0.)


def elastic_net_lad(X,y,l1,l2,*,gap_tolerance=1e-6,maxiter=10000):
    """Solve Eq(9) using its smooth box-constrained dual (no loss smoothing).

    Dual: max y'u-||soft(X'u,l1)||²/(2*l2), |u_i|<=1/m.
    beta=soft(X'u,l1)/l2. A primal-dual gap tests numerical accuracy against
    the EXACT nonsmooth objective. Returns beta and diagnostic gap, not DP.
    """
    X,y=data_xy(X,y); positive('l2',l2); positive('gap_tolerance',gap_tolerance)
    if not np.isfinite(l1) or l1<0: raise ValueError('l1 must be finite >=0')
    integer('maxiter',maxiter)
    m=len(y)
    def objective(u):
        z=soft_threshold(X.T@u,l1)
        b=z/l2
        return .5*(z@z)/l2-y@u, X@b-y
    result=minimize(objective,np.zeros(m),jac=True,method='L-BFGS-B',
                    bounds=[(-1/m,1/m)]*m,
                    options={'maxiter':maxiter,'maxls':60,'ftol':1e-15,'gtol':1e-11,'maxcor':30})
    u=np.clip(result.x,-1/m,1/m)
    z=soft_threshold(X.T@u,l1); b=z/l2
    primal=np.mean(np.abs(y-X@b))+l1*np.sum(np.abs(b))+.5*l2*(b@b)
    dual=y@u-.5*(z@z)/l2
    gap=max(0.,float(primal-dual))
    if gap>gap_tolerance:
        raise RuntimeError(f'FRAPPE LAD initializer unresolved: primal-dual gap {gap:.3g} exceeds {gap_tolerance:.3g}; increase init_maxiter or explicitly change tolerance')
    return b,gap


def paper_noise_scales(*,N,init_size,p,epsilon,delta,x_bound,beta_radius,
                       density_floor,init_l2,bandwidth,outer_steps,inner_steps,kernel='paper_sextic'):
    """Theorem 1 variances, square-rooted once. p only documents dimension."""
    validate_budget(epsilon,delta)
    if init_size<=N*delta:
        raise ValueError('Theorem1 needs log(init_size/(N*delta))>0')
    B=105/64 if kernel=='paper_sextic' else 15/16
    cf=1/density_floor  # Eq(5) floor = 1/c_f; see conflicting notation in Assumption2.
    G=4*x_bound*x_bound*beta_radius+cf
    return (np.sqrt(24*x_bound*x_bound*np.log(init_size/(N*delta)))/(epsilon*init_l2*N),
            np.sqrt(24*B*B*np.log(1/delta)*outer_steps)/(epsilon*N*np.asarray(bandwidth)),
            np.sqrt(6*G*G*np.log(1/delta)*inner_steps*outer_steps)/(epsilon*N))


def _schedule(value, length, name, allow_zero=False):
    a=np.asarray(value,dtype=float)
    if a.ndim==0: a=np.full(length,float(a))
    if a.shape!=(length,) or not np.isfinite(a).all() or np.any(a<0 if allow_zero else a<=0):
        raise ValueError(f'{name} must be a valid scalar or length-outer_steps sequence')
    return a


def fit_frappe(X,y,*,epsilon,delta,bandwidth,l1_penalty,step_size,
               outer_steps,inner_steps,x_bound,beta_radius,density_floor,
               init_l1,init_l2,init_size,private=True,seed=None,
               kernel='paper_sextic',project_initial=False,
               init_gap_tolerance=1e-6,init_maxiter=10000):
    """LEGACY: arXiv-calibrated FRAPPE, unsupported by project protocol v1.

    Uses noisy LAD start, noisy KDE, and noisy proximal LS updates.

    x_bound is a PUBLIC row L2 bound. density_floor specifies Eq(5)'s floor;
    c_f=1/density_floor in Theorem1's printed G. project_initial=False follows
    the literal algorithm; True is an explicitly labelled stabilizing extension.
    Statistical theory assumes p<N. Private tuning/BIC is NOT silently added.
    """
    X,y=data_xy(X,y);N,p=X.shape
    for key,val in [('step_size',step_size),('x_bound',x_bound),('beta_radius',beta_radius),
                    ('density_floor',density_floor),('init_l2',init_l2)]: positive(key,val)
    for key,val in [('outer_steps',outer_steps),('inner_steps',inner_steps),('init_size',init_size)]: integer(key,val)
    if init_size>N: raise ValueError('init_size must be <=N')
    if np.max(np.linalg.norm(X,axis=1))>x_bound*(1+1e-12):
        raise ValueError('Design violates the declared public L2 bound; no silent preprocessing')
    h=_schedule(bandwidth,outer_steps,'bandwidth')
    lam=_schedule(l1_penalty,outer_steps,'l1_penalty',allow_zero=True)
    frappe_kernel(np.array([0.]),kernel)
    rng=np.random.default_rng(seed)
    idx=rng.choice(N,size=init_size,replace=False)
    b,_=elastic_net_lad(X[idx],y[idx],init_l1,init_l2,
                        gap_tolerance=init_gap_tolerance,maxiter=init_maxiter)
    if private:
        sd0,sdf,sdg=paper_noise_scales(N=N,init_size=init_size,p=p,epsilon=epsilon,delta=delta,
            x_bound=x_bound,beta_radius=beta_radius,density_floor=density_floor,
            init_l2=init_l2,bandwidth=h,outer_steps=outer_steps,inner_steps=inner_steps,kernel=kernel)
        b=b+rng.normal(0,sd0,p)
    else: sd0,sdf,sdg=0.,np.zeros(outer_steps),0.
    if project_initial:
        b=b*min(1.,beta_radius/max(np.linalg.norm(b),np.finfo(float).tiny))
    for v in range(outer_steps):
        outer_b=b.copy()
        density=np.mean(frappe_kernel((y-X@outer_b)/h[v],kernel))/h[v]
        if private: density+=rng.normal(0,sdf[v])
        density=max(density,density_floor)
        fitted=X@outer_b
        pseudo=fitted-((y<=fitted).astype(float)-.5)/density
        for _ in range(inner_steps):
            grad=X.T@(X@b-pseudo)/N
            if private: grad=grad+rng.normal(0,sdg,p)
            b=soft_threshold(b-step_size*grad,step_size*lam[v])
            b=b*min(1.,beta_radius/max(np.linalg.norm(b),np.finfo(float).tiny))
    return FitResult(b,np.flatnonzero(b),
        {'type':'paper_literal_calibration' if private else 'nonprivate',
         'epsilon':epsilon if private else None,'delta':delta if private else None,
         'guarantee_verified':False if private else None,
         'initial_noise_sd':float(sd0),'density_noise_sd':np.asarray(sdf).tolist(),'gradient_noise_sd':float(sdg)},
        {'algorithm':'FRAPPE Algorithm1 + Theorem1 (arXiv2401.01294)',
         'paper_fidelity':'literal equations with numerical Eq9 initializer',
         'kernel':kernel,'project_initial_extension':project_initial,
         'statistical_dimension_regime_p_lt_N':bool(p<N),
         'initializer_gap_tolerance':init_gap_tolerance,
         'initializer_l2_error_upper_bound':float(np.sqrt(2*init_gap_tolerance/init_l2)),
         'density_constant_convention':'floor=1/c_f; conflicts with literal Assumption2 notation',
         'optimizer_success':True,
         'privacy_caveats':'printed constants; noisy initial vector unbounded unless explicit extension; finite solver error not charged'})


# ---------------------------------------------------------------------------
# Project v1 implementation. The functions above preserve the old arXiv-style
# calibration for reproducibility; they are legacy and are not DP-qualified.

def _soft_threshold_scalar(value, threshold):
    return float(np.sign(value) * max(abs(value) - threshold, 0.0))


def _frappe_primal_value(Z, y, beta, lambda_l1, lambda_l2, intercept_index):
    slopes = np.ones(beta.size, dtype=bool)
    if intercept_index is not None:
        slopes[intercept_index] = False
    return float(0.5 * np.mean(np.abs(y - Z @ beta)) +
                 lambda_l1 * np.sum(np.abs(beta[slopes])) +
                 0.5 * lambda_l2 * np.dot(beta, beta))


def _frappe_dual_value(Z, y, dual, lambda_l1, lambda_l2, intercept_index):
    """Objective of a feasible dual point for mean median check loss."""
    v = Z.T @ dual
    slopes = np.ones(v.size, dtype=bool)
    if intercept_index is not None:
        slopes[intercept_index] = False
    conjugate = 0.5 / lambda_l2 * np.dot(v[~slopes], v[~slopes])
    conjugate += 0.5 / lambda_l2 * np.sum(
        np.maximum(np.abs(v[slopes]) - lambda_l1, 0.0) ** 2)
    return float(y @ dual - conjugate)


def _frappe_polish_dual_certificate(
        Z, y, dual, lambda_l1, lambda_l2, intercept_index):
    """Improve a feasible dual certificate without changing the primal fit."""
    n, p = Z.shape
    slopes = np.ones(p, dtype=bool)
    if intercept_index is not None:
        slopes[intercept_index] = False

    def objective(candidate):
        v = Z.T @ candidate
        beta_from_dual = np.empty(p)
        beta_from_dual[~slopes] = v[~slopes] / lambda_l2
        beta_from_dual[slopes] = soft_threshold(v[slopes], lambda_l1) / lambda_l2
        return (-_frappe_dual_value(
                    Z, y, candidate, lambda_l1, lambda_l2, intercept_index),
                Z @ beta_from_dual - y)

    bound = 0.5 / n
    initial = np.clip(np.asarray(dual, dtype=float), -bound, bound)
    result = minimize(
        objective, initial, jac=True, method="L-BFGS-B",
        bounds=[(-bound, bound)] * n,
        options={"maxiter": 2000, "maxls": 60, "ftol": 1e-15,
                 "gtol": 1e-12, "maxcor": 30})
    candidate = np.clip(result.x, -bound, bound)
    if (_frappe_dual_value(Z, y, candidate, lambda_l1, lambda_l2, intercept_index) <
            _frappe_dual_value(Z, y, initial, lambda_l1, lambda_l2, intercept_index)):
        candidate = initial
    return candidate, {
        "attempted": True,
        "optimizer_success": bool(result.success),
        "optimizer_iterations": int(result.nit),
    }


def elastic_net_qr_admm(Z, y, lambda_l1, lambda_l2, *, intercept_index=None,
                        rho=1.0, abs_tol=1e-8, rel_tol=1e-8,
                        gap_tolerance=1e-8, maxiter=10000,
                        coordinate_max_sweeps=1000, backend="compiled",
                        gap_check_interval=0):
    """Su--Wang ADMM for mean median check loss plus elastic-net penalty.

    The intercept, when supplied, is excluded from the L1 penalty and remains
    included in the L2 ridge. Returns ``(beta, diagnostics)``. The reported
    primal--dual gap is a floating-point diagnostic, not a finite-precision
    privacy certificate.
    """
    Z, y = data_xy(Z, y)
    Z=np.ascontiguousarray(Z,dtype=np.float64); y=np.ascontiguousarray(y,dtype=np.float64)
    n, p = Z.shape
    for name, value in (("lambda_l2", lambda_l2), ("rho", rho),
                        ("abs_tol", abs_tol), ("rel_tol", rel_tol),
                        ("gap_tolerance", gap_tolerance)):
        positive(name, value)
    if not np.isfinite(lambda_l1) or lambda_l1 < 0:
        raise ValueError("lambda_l1 must be finite and nonnegative")
    if intercept_index is not None:
        if isinstance(intercept_index, (bool, np.bool_)) or int(intercept_index) != intercept_index or not 0 <= intercept_index < p:
            raise ValueError("intercept_index must be None or a valid column index")
        intercept_index = int(intercept_index)
    integer("maxiter", maxiter)
    integer("coordinate_max_sweeps", coordinate_max_sweeps)
    if backend not in {"compiled", "python"}:
        raise ValueError("backend must be compiled or python")

    # Fixed deterministic start, then warm-started cyclic coordinate descent
    # for each elastic-net quadratic subproblem.
    beta = np.full(p, 1e-3 / np.sqrt(p))
    residual = y - Z @ beta
    dual_admm = np.zeros(n)
    penalty = np.full(p, lambda_l1)
    if intercept_index is not None:
        penalty[intercept_index] = 0.0
    column_norm2 = np.sum(Z * Z, axis=0)
    coordinate_kernel = _compiled_coordinate_kernel()[0] if backend == "compiled" else None
    # Column-contiguous storage preserves the cyclic updates and summation
    # order while removing strided memory loads inside the C++ kernel.
    Z_kernel = np.asfortranarray(Z) if coordinate_kernel is not None else None
    if isinstance(gap_check_interval, bool) or int(gap_check_interval) != gap_check_interval or gap_check_interval < 0:
        raise ValueError("gap_check_interval must be a nonnegative integer")
    stop_reason = "iteration_limit"
    converged = False
    primal_residual_norm = np.inf
    dual_residual_norm = np.inf
    for iteration in range(1, int(maxiter) + 1):
        beta_old = beta.copy()
        target = y - residual + dual_admm / rho
        work_resid = target - Z @ beta
        if coordinate_kernel is not None:
            coordinate_kernel(Z_kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                penalty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                column_norm2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                beta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                work_resid.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,p,
                lambda_l2/rho,rho,int(coordinate_max_sweeps))
        else:
            for _ in range(int(coordinate_max_sweeps)):
                max_change = 0.0
                for j in range(p):
                    old = beta[j]
                    denominator = column_norm2[j] + lambda_l2 / rho
                    updated = _soft_threshold_scalar(
                        float(Z[:, j] @ work_resid + column_norm2[j] * old),
                        penalty[j] / rho) / denominator
                    beta[j] = updated
                    work_resid -= Z[:, j] * (updated - old)
                    max_change = max(max_change, abs(updated - old))
                if max_change <= 1e-8 * (1.0 + np.max(np.abs(beta))):
                    break

        old_residual = residual.copy()
        prox_input = y - Z @ beta + dual_admm / rho
        threshold = 1.0 / (2.0 * n * rho)
        residual = np.sign(prox_input) * np.maximum(np.abs(prox_input) - threshold, 0.0)
        primal_vec = Z @ beta + residual - y
        dual_admm -= rho * primal_vec
        dual_vec = rho * Z.T @ (residual - old_residual)
        primal_residual_norm = float(np.linalg.norm(primal_vec))
        dual_residual_norm = float(np.linalg.norm(dual_vec))
        eps_primal = np.sqrt(n) * abs_tol + rel_tol * max(
            np.linalg.norm(Z @ beta), np.linalg.norm(residual), np.linalg.norm(y))
        eps_dual = np.sqrt(p) * abs_tol + rel_tol * np.linalg.norm(Z.T @ dual_admm)
        if primal_residual_norm <= eps_primal and dual_residual_norm <= eps_dual:
            converged = True
            stop_reason = "residual_tolerances"
            break
        if gap_check_interval and iteration % gap_check_interval == 0:
            # A feasible dual lower bound certifies objective accuracy even
            # when ADMM splitting residuals converge more slowly. This is a
            # numerical check, never a finite-solver privacy certificate.
            trial_dual = np.clip(dual_admm, -0.5 / n, 0.5 / n)
            trial_primal = _frappe_primal_value(Z, y, beta, lambda_l1, lambda_l2, intercept_index)
            trial_lower = _frappe_dual_value(Z, y, trial_dual, lambda_l1, lambda_l2, intercept_index)
            trial_gap = trial_primal - trial_lower
            if (np.isfinite(trial_gap) and
                    -1e-12 * max(1.0, abs(trial_primal)) <= trial_gap <= gap_tolerance):
                converged = True
                stop_reason = "primal_dual_gap"
                break

    # The update sign in this ADMM parameterization makes dual_admm the
    # subgradient of the check loss at convergence. Clipping keeps the point
    # feasible even when the finite iterate has not converged.
    dual = np.clip(dual_admm, -0.5 / n, 0.5 / n)
    primal = _frappe_primal_value(Z, y, beta, lambda_l1, lambda_l2, intercept_index)
    dual_value = _frappe_dual_value(Z, y, dual, lambda_l1, lambda_l2, intercept_index)
    gap = max(0.0, float(primal - dual_value))
    gap_before_polish = gap
    certificate = {"attempted": False}
    if converged and gap > gap_tolerance:
        # The ADMM primal can be accurate even when its finite dual state is a
        # loose certificate. Polish only the feasible certificate; beta and
        # therefore the fitted mechanism remain unchanged.
        dual, certificate = _frappe_polish_dual_certificate(
            Z, y, dual, lambda_l1, lambda_l2, intercept_index)
        dual_value = _frappe_dual_value(
            Z, y, dual, lambda_l1, lambda_l2, intercept_index)
        gap = max(0.0, float(primal - dual_value))
    return beta, {
        "converged": converged,
        "stopping_reason": stop_reason,
        "gap_check_interval": gap_check_interval,
        "iterations": iteration,
        "primal_residual_norm": primal_residual_norm,
        "dual_residual_norm": dual_residual_norm,
        "primal_objective": primal,
        "dual_objective": dual_value,
        "primal_dual_gap": gap,
        "primal_dual_gap_before_polish": gap_before_polish,
        "gap_tolerance": gap_tolerance,
        "dual_certificate_polish": certificate,
        "numerical_accuracy_met": bool(converged and gap <= gap_tolerance),
        "gap_is_finite_precision_certified": False,
    }


def frappe_kernel_v1(u):
    """Published signed compact-support kernel, with a conservative abs bound."""
    u = np.asarray(u, dtype=float)
    x = np.clip(u, -1.0, 1.0)
    value = (105.0 - 525.0*x*x + 735.0*x**4 - 315.0*x**6) / 64.0
    return np.where(np.abs(u) < 1.0, value, 0.0)


def frappe_kernel_abs_bound_v1():
    """Exact real-arithmetic sup |K|, checked at the polynomial's extrema."""
    points = np.array([0.0, np.sqrt(5.0 / 9.0), 1.0])
    vals = (105.0 - 525.0*points**2 + 735.0*points**4 -
            315.0*points**6) / 64.0
    return float(np.max(np.abs(vals)))


FRAPPE_BUDGET_SCHEMES = {
    "original": {
        "epsilon_scale_fraction": 1/12,
        "epsilon_initial_output_fraction": 1/4,
        "delta_initial_output_fraction": 1/3,
        "epsilon_main_fraction": 2/3,
        "delta_main_fraction": 2/3,
        "label": "original_end_to_end",
    },
    "matched_init": {
        "epsilon_scale_fraction": 1/12,
        "epsilon_initial_output_fraction": 1/12,
        "delta_initial_output_fraction": 1/6,
        "epsilon_main_fraction": 5/6,
        "delta_main_fraction": 5/6,
        "label": "post_hoc_matched_initialization_budget_ablation",
    },
}


def frappe_budget_ledger(epsilon, delta, budget_scheme="original"):
    """Return one of the two fixed, auditable FRAPPE budget ledgers."""
    validate_budget(epsilon, delta)
    if budget_scheme not in FRAPPE_BUDGET_SCHEMES:
        raise ValueError("budget_scheme must be original or matched_init")
    spec = FRAPPE_BUDGET_SCHEMES[budget_scheme]
    stages = [
        {"stage": "scale", "epsilon": epsilon*spec["epsilon_scale_fraction"], "delta": 0.0},
        {"stage": "initial_output", "epsilon": epsilon*spec["epsilon_initial_output_fraction"],
         "delta": delta*spec["delta_initial_output_fraction"]},
        {"stage": "main", "epsilon": epsilon*spec["epsilon_main_fraction"],
         "delta": delta*spec["delta_main_fraction"]},
    ]
    if not np.isclose(sum(s["epsilon"] for s in stages), epsilon):
        raise RuntimeError("FRAPPE epsilon ledger does not compose to the total")
    if not np.isclose(sum(s["delta"] for s in stages), delta):
        raise RuntimeError("FRAPPE delta ledger does not compose to the total")
    return {"scheme": budget_scheme, "label": spec["label"], "stages": stages}


def frappe_liu200_budget_ledger(epsilon, delta, *, q):
    """Budget the shared scale/initializer block using fixed-size amplification.

    The amplified block receives (epsilon/3, delta/3). This inverts the
    replace-one formula epsilon_amp=log1p(q*expm1(epsilon_base)); all block
    delta is assigned to the noisy initializer output.
    """
    validate_budget(epsilon, delta)
    if not np.isfinite(q) or not 0 < q <= 1:
        raise ValueError("q must be in (0, 1]")
    epsilon_block = epsilon / 3.0
    delta_block = delta / (3.0 * q)
    if delta_block >= 1.0:
        raise ValueError("amplified initializer base delta must be < 1")
    epsilon_base = float(np.log1p(np.expm1(epsilon_block) / q))
    stages = [
        {"stage": "scale_base_subsample", "epsilon": epsilon_base / 4.0, "delta": 0.0},
        {"stage": "initial_output_base_subsample", "epsilon": 3.0 * epsilon_base / 4.0,
         "delta": delta_block},
        {"stage": "main_global", "epsilon": 2.0 * epsilon / 3.0,
         "delta": 2.0 * delta / 3.0},
    ]
    amplified_stages = [
        {"stage": "shared_initialization_block_amplified", "epsilon": epsilon_block,
         "delta": delta / 3.0},
        {"stage": "main_global", "epsilon": 2.0 * epsilon / 3.0,
         "delta": 2.0 * delta / 3.0},
    ]
    return {"scheme": "liu200", "label": "fixed_size_replace_one_amplified_init",
            "q": float(q), "epsilon_base_block": epsilon_base,
            "delta_base_block": delta_block, "stages": stages,
            "amplified_stages": amplified_stages}


def frappe_v1_noise_scales(*, n, epsilon, delta, bandwidths, outer_steps,
                           inner_steps, x_bound, beta_radius, density_floor,
                           init_sensitivity, budget_scheme="original",
                           _ledger_override=None):
    """Return ideal real-arithmetic Gaussian scales for the README v1 ledger."""
    validate_budget(epsilon, delta)
    integer("n", n); integer("outer_steps", outer_steps); integer("inner_steps", inner_steps)
    for name, value in (("x_bound", x_bound), ("beta_radius", beta_radius),
                        ("density_floor", density_floor),
                        ("init_sensitivity", init_sensitivity)):
        positive(name, value)
    h = _schedule(bandwidths, outer_steps, "bandwidths")
    ledger = (_ledger_override if _ledger_override is not None else
              frappe_budget_ledger(epsilon, delta, budget_scheme))
    initial_stage = ledger["stages"][1]
    main_stage = ledger["stages"][2]
    rho_init = rho_from_eps_delta(initial_stage["epsilon"], initial_stage["delta"])
    rho_main = rho_from_eps_delta(main_stage["epsilon"], main_stage["delta"])
    sigma_init = init_sensitivity / np.sqrt(2.0 * rho_init)
    delta_density = 2.0 * frappe_kernel_abs_bound_v1() / (n * h)
    # Conditional on the preceding DP transcript, outer/inner iterates are fixed.
    # Each record contributes xx^T(beta-beta_outer) + x*(indicator-.5)/f.
    # Replacement gives 4*B^2*C + B/f_min (the sign term retains the X bound).
    delta_gradient = (4.0*x_bound*x_bound*beta_radius + x_bound/density_floor) / n
    sigma_density = delta_density * np.sqrt(outer_steps / rho_main)
    sigma_gradient = delta_gradient * np.sqrt(outer_steps * inner_steps / rho_main)
    return {
        "rho_initialization": float(rho_init),
        "rho_main": float(rho_main),
        "initial_sigma": float(sigma_init),
        "density_sensitivity": delta_density,
        "density_sigma": sigma_density,
        "gradient_sensitivity": float(delta_gradient),
        "gradient_sigma": float(sigma_gradient),
        "budget_scheme": budget_scheme,
    }


def _project_l2(beta, radius):
    beta = np.asarray(beta, dtype=float)
    scale = float(np.max(np.abs(beta)))
    if scale == 0.0:
        return beta.copy()
    direction = beta / scale
    direction_norm = float(np.linalg.norm(direction))
    if scale <= radius / direction_norm:
        return beta.copy()
    return direction * (radius / direction_norm)


def _frappe_v1_initial_design(X, intercept_index):
    """README 3.3 initializer clipping for an already assembled design."""
    n, p = X.shape
    d = p
    slopes = np.ones(p, dtype=bool)
    if intercept_index is not None:
        if not 0 <= intercept_index < p:
            raise ValueError("intercept_index is out of range")
        if not np.allclose(X[:, intercept_index], 1.0):
            raise ValueError("intercept_index must identify a column of ones")
        slopes[intercept_index] = False
    clipped = X.copy()
    bound = np.sqrt(d) / 6.0
    norm = np.linalg.norm(clipped[:, slopes], axis=1)
    clipped[:, slopes] *= np.minimum(1.0, bound / np.maximum(norm, np.finfo(float).tiny))[:, None]
    row_bound = float(np.sqrt(1.0 + d/36.0) if intercept_index is not None else bound)
    return clipped, row_bound


def _frappe_nonprivate_scale(y):
    """Chang-style clipped-moment scale rule without the privacy noise."""
    n = len(y)
    cap = float(np.log(n)) if n > 1 else 1.0
    clipped = np.clip(y, -cap, cap)
    mean = float(np.mean(clipped))
    second_moment = float(np.mean(clipped * clipped))
    variance = second_moment - mean * mean
    tau0 = float(np.sqrt(variance)) if variance > 0.0 else 2.0
    return tau0, {
        "method": "clipped first and second moments without noise",
        "response_clip": cap,
        "mean": mean,
        "second_moment": second_moment,
    }


def fit_frappe_v1(X, y, *, epsilon, delta, x_bound, beta_radius=10.0,
                  density_floor=None, step_size=None, bandwidths=None,
                  l1_penalties=None, outer_steps=10, inner_steps=50,
                  init_l1=None, init_l2=0.2, intercept_index=None,
                  admm_rho=1.0, admm_abs_tol=1e-8, admm_rel_tol=1e-8,
                  admm_maxiter=10000, coordinate_max_sweeps=1000,
                  admm_backend="compiled",
                  sensitivity_mode="theoretical_output", budget_scheme="original",
                  synthetic_diagnostics=False, private=True, seed=None,
                  _initialization_indices=None, _ledger_override=None,
                  _initializer_solver_options=None, _protocol_label=None):
    """Fit FRAPPE-SuWang-init-projected (project v1 protocol).

    ``theoretical_output`` perturbs the exact full-sample minimizer using its
    sensitivity bound B/(n*lambda_l2), then projects the noisy output.
    Finite ADMM output is a high-accuracy approximation, not an exact-solver
    DP proof. Its fixed 1e-8 primal--dual gap target is a numerical check only.
    The explicit ``bounded_output`` alternative projects before noise and
    uses sensitivity 2*beta_radius, independent of solver accuracy. NumPy's
    floating-point Gaussian sampler is not a finite-precision DP certificate.
    """
    X, y = data_xy(X, y)
    n, p = X.shape
    validate_budget(epsilon, delta)
    ledger = (_ledger_override if _ledger_override is not None else
              frappe_budget_ledger(epsilon, delta, budget_scheme))
    if synthetic_diagnostics not in (True, False):
        raise ValueError("synthetic_diagnostics must be boolean")
    positive("x_bound", x_bound); positive("beta_radius", beta_radius); positive("init_l2", init_l2)
    integer("outer_steps", outer_steps); integer("inner_steps", inner_steps)
    positive("admm_rho", admm_rho); positive("admm_abs_tol", admm_abs_tol)
    positive("admm_rel_tol", admm_rel_tol)
    integer("admm_maxiter", admm_maxiter); integer("coordinate_max_sweeps", coordinate_max_sweeps)
    if intercept_index is not None:
        if isinstance(intercept_index, (bool, np.bool_)) or int(intercept_index) != intercept_index or not 0 <= intercept_index < p:
            raise ValueError("intercept_index must be None or a valid column index")
        intercept_index = int(intercept_index)
    if sensitivity_mode not in {"theoretical_output", "bounded_output"}:
        raise ValueError("sensitivity_mode must be theoretical_output or bounded_output")
    Z, init_row_bound = _frappe_v1_initial_design(X, intercept_index)
    init_sample_n = (n if _initialization_indices is None else
                     int(np.asarray(_initialization_indices).size))
    # This fixed radial preprocessing is part of the public mechanism and is
    # used consistently by the main loop's KDE and gradient calculations.
    X = radial_clip(X, x_bound)
    if init_l1 is None:
        init_l1 = 0.1 * init_row_bound * np.sqrt(np.log(2.0*p) / init_sample_n)
    if not np.isfinite(init_l1) or init_l1 < 0:
        raise ValueError("init_l1 must be finite and nonnegative")

    # The scale query spends epsilon/12. This utility is shared with the
    # project initializer implementation and already performs public clipping.
    from ..initialization import private_scale_estimate
    rng = np.random.default_rng(seed)
    if _initialization_indices is None:
        init_idx = np.arange(n)
    else:
        init_idx = np.asarray(_initialization_indices, dtype=int)
        if init_idx.ndim != 1 or init_idx.size == 0 or np.any(init_idx < 0) or np.any(init_idx >= n):
            raise ValueError("_initialization_indices must be a nonempty valid index vector")
    init_n = int(init_idx.size)
    if private:
        tau0, scale_meta = private_scale_estimate(
            y[init_idx], ledger["stages"][0]["epsilon"], rng=rng)
    else:
        tau0, scale_meta = _frappe_nonprivate_scale(y[init_idx])
    scale = float(np.clip(tau0, 0.1, 10.0))
    if bandwidths is None:
        bandwidths = scale * np.maximum(0.1, 0.5 * 0.9**np.arange(outer_steps))
    bandwidths = _schedule(bandwidths, outer_steps, "bandwidths")
    if density_floor is None:
        density_floor = 0.05 / scale
    positive("density_floor", density_floor)
    if step_size is None:
        step_size = 0.9 / (x_bound*x_bound)
    positive("step_size", step_size)
    if l1_penalties is None:
        l1_penalties = 0.1 * x_bound * scale * np.sqrt(np.log(2.0*p) / n)
    l1_penalties = _schedule(l1_penalties, outer_steps, "l1_penalties", allow_zero=True)

    try:
        solver_options = {} if _initializer_solver_options is None else dict(_initializer_solver_options)
        beta_fit, diagnostics = elastic_net_qr_admm(
            Z[init_idx], y[init_idx], init_l1, init_l2, intercept_index=intercept_index,
            rho=admm_rho, abs_tol=admm_abs_tol, rel_tol=admm_rel_tol,
            maxiter=admm_maxiter, coordinate_max_sweeps=coordinate_max_sweeps,
            backend=admm_backend, **solver_options)
        beta_fit = np.asarray(beta_fit, dtype=float)
        if beta_fit.shape != (p,) or not np.isfinite(beta_fit).all():
            raise FloatingPointError("FRAPPE initializer returned an invalid vector")
    except Exception as exc:
        if not private:
            raise
        if _protocol_label is not None:
            raise RuntimeError("FRAPPE initializer numerical failure; no result released")
        if sensitivity_mode == "theoretical_output":
            raise RuntimeError("FRAPPE initializer numerical failure; no theoretical-output result released") from exc
        # Preserve the legacy v1 bounded-output fallback for compatibility.
        beta_fit = np.zeros(p)
        diagnostics = {"converged": False, "numerical_accuracy_met": False,
                       "fallback": "public_zero_vector"}
    if (private and sensitivity_mode == "theoretical_output" and
            _protocol_label is None and not diagnostics.get("numerical_accuracy_met", False)):
        raise RuntimeError("FRAPPE initializer did not meet the fixed 1e-8 numerical gap target")
    if private:
        if sensitivity_mode == "bounded_output":
            # The pre-noise projection is the sensitivity bound.
            beta = _project_l2(beta_fit, beta_radius)
            init_sensitivity = 2.0 * beta_radius
        else:
            # Exact minimizer sensitivity for mean median check loss plus a
            # lambda_l2-strongly-convex ridge term; projection is postprocessing.
            beta = beta_fit
            init_sensitivity = init_row_bound / (init_n * init_l2)
        scales = frappe_v1_noise_scales(
            n=n, epsilon=epsilon, delta=delta, bandwidths=bandwidths,
            outer_steps=outer_steps, inner_steps=inner_steps, x_bound=x_bound,
            beta_radius=beta_radius, density_floor=density_floor,
            init_sensitivity=init_sensitivity, budget_scheme=budget_scheme,
            _ledger_override=ledger)
        beta = _project_l2(beta + rng.normal(0.0, scales["initial_sigma"], p), beta_radius)
    else:
        scales = {"initial_sigma": 0.0,
                  "density_sigma": np.zeros(outer_steps),
                  "gradient_sigma": 0.0,
                  "rho_initialization": 0.0, "rho_main": 0.0}
        beta = _project_l2(beta_fit, beta_radius)

    # This value is available only under an explicit synthetic-evaluation flag.
    # It is not part of the private mechanism's releasable output.
    synthetic_initial_beta = beta.copy() if synthetic_diagnostics else None

    density_sigma = np.asarray(scales["density_sigma"])
    for v in range(outer_steps):
        outer_beta = beta.copy()
        fitted = X @ outer_beta
        density = np.mean(frappe_kernel_v1((y - fitted) / bandwidths[v])) / bandwidths[v]
        if private:
            density += rng.normal(0.0, density_sigma[v])
        density = max(float(density), density_floor)
        pseudo = fitted - ((y <= fitted).astype(float) - 0.5) / density
        for _ in range(inner_steps):
            grad = X.T @ (X @ beta - pseudo) / n
            if private:
                grad += rng.normal(0.0, scales["gradient_sigma"], p)
            beta = soft_threshold(beta - step_size*grad,
                                  step_size*l1_penalties[v])
            beta = _project_l2(beta, beta_radius)

    privacy = {
        "type": ("ideal_exact_optimizer_gaussian_approximate_solver"
                 if private and sensitivity_mode == "theoretical_output" else
                 "ideal_real_arithmetic_gaussian_bounded_output"
                 if private else "nonprivate"),
        "method": (_protocol_label or (("FRAPPE-SuWang-init-projected-matched-init-v1"
                    if budget_scheme == "matched_init" else
                    "FRAPPE-SuWang-init-projected-v1")) if private else "nonprivate"),
        "epsilon": epsilon if private else None,
        "delta": delta if private else None,
        "epsilon_scale": ledger["stages"][0]["epsilon"] if private else None,
        "epsilon_initial_output": ledger["stages"][1]["epsilon"] if private else None,
        "delta_initial_output": ledger["stages"][1]["delta"] if private else None,
        "epsilon_main": ledger["stages"][2]["epsilon"] if private else None,
        "delta_main": ledger["stages"][2]["delta"] if private else None,
        "budget_scheme": budget_scheme if private else "nonprivate",
        "budget_scheme_label": ledger["label"] if private else "nonprivate",
        "stages": ledger["stages"] if private else [],
        "sensitivity_mode": sensitivity_mode if private else "nonprivate",
        "initial_output_sensitivity": init_sensitivity if private else None,
        "initial_output_sigma": float(scales["initial_sigma"]) if private else None,
        "main_density_sensitivity": (np.asarray(scales["density_sensitivity"]).tolist()
                                      if private else None),
        "main_gradient_sensitivity": scales["gradient_sensitivity"] if private else None,
        "theory_applies_to": "exact optimizer; finite ADMM approximation is not proved DP" if private else None,
        "finite_admm_dp_guaranteed": False if private else None,
        "floating_point_dp_certified": False,
        "guarantee_verified": False if private else None,
        "formal_dp_ready": False,
        "privacy_status": "research_approximation" if _protocol_label else None,
        "initialization_subsample_rate": ledger.get("q") if private else None,
        "initialization_base_epsilon": ledger.get("epsilon_base_block") if private else None,
        "initialization_base_delta": ledger.get("delta_base_block") if private else None,
        "amplified_stages": ledger.get("amplified_stages", []) if private else [],
    }
    metadata = {
        "algorithm": (_protocol_label or ("FRAPPE-SuWang-init-projected-matched-init-v1"
                      if budget_scheme == "matched_init" else
                      "FRAPPE-SuWang-init-projected-v1")),
        "budget_scheme": budget_scheme,
        "budget_scheme_label": ledger["label"],
        "privacy_status": ("research_approximation" if _protocol_label else "project_v1"),
        "amplified_stages": ledger.get("amplified_stages", []),
        "literature_basis": "IEEE TIFS 19 (2024), DOI 10.1109/TIFS.2023.3349054; Su and Wang (2021), DOI 10.1016/j.cam.2021.113462",
        "published_source_pdf_sha256": {
            "ieee_tifs_article": "c950fc91f7e1a037ed2a0b7efc21ecfa4cac1832cfa14777c1bf882651cf9081",
            "ieee_tifs_supplement": "618d0ca3ebd1be24692d770f7d51fb1bbcf68a61807aa56c27ce1f0302de4c0e",
        },
        "initializer": (f"{init_n}-row Su-Wang ADMM median elastic-net QR"),
        "initialization_sample_size": init_n,
        "main_sample_size": n,
        "admm_abs_tol": admm_abs_tol,
        "admm_rel_tol": admm_rel_tol,
        "admm_gap_tolerance": 1e-5 if _initializer_solver_options else None,
        "admm_maxiter": admm_maxiter,
        "coordinate_max_sweeps": coordinate_max_sweeps,
        "admm_backend": admm_backend,
        "initializer_row_bound": init_row_bound,
        "main_design_clipping": "radial L2 to public x_bound",
        "initial_projection_radius": beta_radius,
        "initial_projection_order": ("nonprivate post-fit" if not private else
                                      "after Gaussian perturbation" if sensitivity_mode == "theoretical_output" else
                                      "before and after Gaussian perturbation"),
        "outer_steps": outer_steps,
        "inner_steps": inner_steps,
        "bandwidths": np.asarray(bandwidths).tolist(),
        "density_floor": density_floor,
        "step_size": step_size,
        "l1_penalties": l1_penalties.tolist(),
        "kernel_abs_bound": frappe_kernel_abs_bound_v1(),
        "diagnostics_available_only_in_nonprivate_mode": True,
        "admm_diagnostics_available_only_in_nonprivate_or_synthetic_mode": True,
        "synthetic_diagnostics_enabled": bool(synthetic_diagnostics),
        "synthetic_diagnostics_release_scope": (
            "synthetic_only_not_a_DP_release" if synthetic_diagnostics else "disabled"),
    }
    if synthetic_diagnostics:
        metadata["_synthetic_initial_beta_not_for_release"] = synthetic_initial_beta
    # Data-dependent solver status, scale estimate, and gap are never placed in
    # the private result object. They remain available for synthetic debugging.
    if not private or synthetic_diagnostics:
        metadata["admm_diagnostics"] = diagnostics
        metadata["numerical_status"] = ("converged" if diagnostics.get("numerical_accuracy_met", False)
                                        else "nonconverged")
        metadata["admm_diagnostics_scope"] = (
            "synthetic_only" if synthetic_diagnostics else "nonprivate_debug")
    if not private:
        metadata["scale_debug"] = {**scale_meta, "tau0": tau0, "scale_used": scale}
    return FitResult(beta, np.flatnonzero(beta), privacy, metadata)


def fit_frappe_liu200(X, y, *, epsilon, delta, x_bound, beta_radius=10.0,
                      density_floor=None, step_size=None, bandwidths=None,
                      l1_penalties=None, outer_steps=10, inner_steps=50,
                      init_l1=None, init_l2=0.2, intercept_index=None,
                      admm_rho=1.0, admm_backend="compiled",
                      synthetic_diagnostics=False, private=True, seed=None):
    """Research opt-in: amplify a shared scale and Su--Wang initializer block.

    A fixed-size uniform sample of at most 200 rows is shared by the scale and
    initializer. Main-loop queries continue to use all rows. The reported
    initializer sensitivity is for the exact optimizer; finite ADMM and
    floating-point sampling are explicitly not certified as DP.
    """
    X, y = data_xy(X, y)
    n = len(y)
    m = min(200, n)
    q = m / n
    ledger = frappe_liu200_budget_ledger(epsilon, delta, q=q)
    if isinstance(seed, np.random.Generator):
        entropy = seed.integers(0, 2**32, size=4, dtype=np.uint32)
        seed_sequence = np.random.SeedSequence(entropy)
    else:
        seed_sequence = np.random.SeedSequence(seed)
    sample_seed, fit_seed = seed_sequence.spawn(2)
    sample_rng = np.random.default_rng(sample_seed)
    init_idx = sample_rng.choice(n, size=m, replace=False)
    # Keep the same sampled-block experimental protocol in both privacy modes.
    return fit_frappe_v1(
        X, y, epsilon=epsilon, delta=delta, x_bound=x_bound,
        beta_radius=beta_radius, density_floor=density_floor,
        step_size=step_size, bandwidths=bandwidths,
        l1_penalties=l1_penalties, outer_steps=outer_steps,
        inner_steps=inner_steps, init_l1=init_l1, init_l2=init_l2,
        intercept_index=intercept_index, admm_rho=admm_rho,
        admm_abs_tol=1e-5, admm_rel_tol=1e-5, admm_maxiter=2000,
        coordinate_max_sweeps=200, admm_backend=admm_backend,
        sensitivity_mode="theoretical_output", budget_scheme="liu200",
        synthetic_diagnostics=synthetic_diagnostics, private=private,
        seed=fit_seed,
        _initialization_indices=init_idx, _ledger_override=ledger,
        _initializer_solver_options={"gap_tolerance": 1e-5, "gap_check_interval": 25},
        _protocol_label="FRAPPE-Liu200-SWP-v1")
