"""Tran et al. replication baselines; upstream bugs/deviations are explicit.

The public repository implements Chen & Chua convolution-smoothed SGD/OPM.
These ports do NOT assert a verified end-to-end DP guarantee. See audit notes.
"""
import numpy as np
import ctypes
import hashlib
import os
import subprocess
import tempfile
from scipy.special import ndtr, expit
from scipy.optimize import minimize
from ..core import FitResult, data_xy, initial, positive, integer, quantile


_TRAN_CPP = r'''#include <cmath>
#include <cstddef>
extern "C" void tran_chunk(double* X, const double* y, double* b, double* total,
                            const long long* indices, const double* noise,
                            long long count, long long n, long long p,
                            double tau, double h, double sigma, long long offset,
                            double fixed_rate, long long constant_step) {
  for (long long t=0; t<count; ++t) {
    const long long j=indices[t];
    double z=-y[j];
    for (long long k=0; k<p; ++k) z += X[j*p+k]*b[k];
    const double u=z/h;
    double prob;
    if (u >= 0) { const double e=std::exp(-u); prob=1.0/(1.0+e); }
    else { const double e=std::exp(u); prob=e/(1.0+e); }
    const double rate=constant_step ? fixed_rate : 1.0/std::sqrt(double(offset+t+1));
    const double* nt=noise+t*p;
    for (long long k=0; k<p; ++k) {
      b[k] -= ((prob-tau)*X[j*p+k] + sigma*nt[k])*rate;
      total[k] += b[k];
    }
  }
}
'''


def _compiled_tran_kernel():
    """Build the exact recurrence kernel once; all randomness remains NumPy-generated."""
    digest = hashlib.sha256(_TRAN_CPP.encode()).hexdigest()[:16]
    directory = os.path.join(tempfile.gettempdir(), "dpsqr_tran_kernel")
    os.makedirs(directory, exist_ok=True)
    source = os.path.join(directory, f"tran_{digest}.cpp")
    library = os.path.join(directory, f"tran_{digest}.so")
    if not os.path.exists(library):
        with open(source, "w", encoding="utf-8") as handle:
            handle.write(_TRAN_CPP)
        tmp_library = library + f".{os.getpid()}.tmp"
        subprocess.run(["c++", "-O3", "-std=c++17", "-shared", "-fPIC", source,
                        "-o", tmp_library], check=True, capture_output=True, text=True)
        os.replace(tmp_library, library)
    fn = ctypes.CDLL(library).tran_chunk
    ptr = ctypes.POINTER(ctypes.c_double)
    fn.argtypes = [ptr, ptr, ptr, ptr, ctypes.POINTER(ctypes.c_longlong), ptr,
                   ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
                   ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_longlong,
                   ctypes.c_double, ctypes.c_longlong]
    fn.restype = None
    return fn, library


def _start(beta0, p, legacy_zero_init):
    if beta0 is None and not legacy_zero_init:
        raise ValueError('beta0 is required; pass legacy_zero_init=True only for source comparison')
    return initial(beta0, p)


def _kernel(u, h, kernel):
    z = np.asarray(u) / h
    if kernel == 'gaussian':
        cdf = ndtr(z)
        integral = np.sqrt(2 / np.pi) * np.exp(-z*z/2) + z * (2*cdf-1)
        return 1 / np.sqrt(2*np.pi), integral, cdf
    if kernel == 'logistic':
        return .25, 2*np.logaddexp(z/2, -z/2), expit(z)
    if kernel == 'uniform':
        return .5, np.where(abs(z)<=1, z*z/2+.5, abs(z)), np.clip((z+1)/2, 0, 1)
    raise ValueError('kernel must be gaussian, logistic, or uniform')


def _prepare(X, y, Cx, preprocessing):
    X, y = data_xy(X, y); positive('Cx', Cx)
    if preprocessing == 'upstream':
        X = np.minimum(X, Cx)  # Exact dp_qr3 one-sided elementwise rule.
    elif preprocessing == 'l2':
        norm = np.linalg.norm(X, axis=1)
        X = X * np.minimum(1, Cx / np.maximum(norm, np.finfo(float).tiny))[:,None]
    elif preprocessing != 'none':
        raise ValueError('preprocessing must be upstream, l2, or none')
    return X, y


def tran_public_step(n, p, *, tau, h, Cx, epsilon, delta,
                     parameter_radius=10., kernel="logistic", private=True):
    """Chen--Chua (2023), Theorem 3.4 eta; fixed h remains an adaptation.

    This rule uses only public parameters, never an observed Hessian or truth.
    The theorem couples h to eta; using Tran's simulation h=.5 does NOT
    assert that the coupled asymptotic theorem applies unchanged.
    """
    positive('parameter_radius', parameter_radius)
    positive('h', h); positive('Cx', Cx)
    L = max(tau, 1-tau)*Cx
    kbar = {'logistic': .25, 'gaussian': 1/np.sqrt(2*np.pi), 'uniform': .5}[kernel]
    moment = {'logistic': 2*np.log(2), 'gaussian': np.sqrt(2/np.pi), 'uniform': .5}[kernel]
    sigma2 = 8*L*L*np.log(1/delta)/epsilon**2 if private else 0.
    denominator = (2*L*L*n**3 + (p*sigma2+L*L+moment*kbar*Cx*Cx/2)*n**2
                   + 2*L*L*n)
    reference = parameter_radius/np.sqrt(denominator)
    return float(min(reference, h/(kbar*Cx*Cx)))


def tran_dpsgd(X, y, *, tau=.5, h=.5, epsilon=1., Cx=1., steps=None,
                delta=1e-6, kernel='logistic', beta0=None, private=True,
                rng=None, preprocessing='l2', legacy_zero_init=False,
                step_schedule='upstream_decay', eta=None, parameter_radius=10.):
    """Tran Eq.(8) constant-step option; upstream_decay preserves old R code.

    Complete pipeline uses paper_constant and a public reference eta.
    steps=None preserves all n²−1 updates and averages all n² states.
    """
    X,y = _prepare(X,y,Cx,preprocessing); quantile(tau); positive('h',h)
    _kernel(0.,h,kernel)
    n,p = X.shape
    if private:
        positive('epsilon',epsilon)
        if epsilon > 1:
            raise ValueError('the cited DP-SGD calibration requires epsilon <= 1')
        if not 0 < delta <= 1/n:
            raise ValueError('the cited DP-SGD calibration requires 0 < delta <= 1/n')
        if preprocessing != 'l2':
            raise ValueError("private DP-SGD requires preprocessing='l2' for the feature-norm bound")
    steps = n*n-1 if steps is None else steps
    integer('steps',steps,minimum=0)
    if private and steps > n*n-1:
        raise ValueError('the cited DP-SGD calibration covers at most n^2-1 updates')
    b = _start(beta0,p,legacy_zero_init); total=b.copy()
    rng = np.random.default_rng() if rng is None else rng
    L = max(tau,1-tau)*Cx
    sigma = np.sqrt(8*L*L*np.log(1/delta))/epsilon if private else 0.
    if step_schedule not in ('paper_constant', 'upstream_decay'):
        raise ValueError('step_schedule must be paper_constant or upstream_decay')
    if step_schedule == 'upstream_decay' and eta is not None:
        raise ValueError('upstream_decay fixes 1/sqrt(t); eta would be ignored')
    if step_schedule == 'paper_constant':
        eta = tran_public_step(n,p,tau=tau,h=h,Cx=Cx,epsilon=epsilon,delta=delta,
                               parameter_radius=parameter_radius,kernel=kernel,private=private) if eta is None else eta
        positive('eta',eta)
        kbar = _kernel(0.,h,kernel)[0]
        if eta > 2*h/(kbar*Cx*Cx):
            raise ValueError('constant eta exceeds Chen--Chua smoothness range 2/beta')
    fixed_rate = float(eta) if eta is not None else 0.
    if kernel == 'logistic' and steps:
        # Chunking bounds temporary noise memory while preserving every update,
        # its 1/sqrt(i) step, independent Gaussian draw and running-state average.
        fn, _ = _compiled_tran_kernel()
        chunk_size = 100_000
        Xc=np.ascontiguousarray(X, dtype=np.float64); yc=np.ascontiguousarray(y, dtype=np.float64)
        for offset in range(0, steps, chunk_size):
            count=min(chunk_size, steps-offset)
            indices=np.asarray(rng.integers(n, size=count), dtype=np.int64)
            noise=(np.asarray(rng.standard_normal((count,p)), dtype=np.float64)
                   if private else np.zeros((count,p), dtype=np.float64))
            fn(Xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
               yc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
               b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
               total.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
               indices.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
               noise.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), count, n, p,
               tau, h, sigma, offset, fixed_rate, int(step_schedule=='paper_constant'))
    else:
        for i in range(1,steps+1):
            j = rng.integers(n) if hasattr(rng,'integers') else rng.randint(n)
            grad=(_kernel(X[j] @ b-y[j],h,kernel)[2]-tau)*X[j]
            noise = sigma*rng.standard_normal(p) if private else 0.
            rate = fixed_rate if step_schedule=='paper_constant' else 1/np.sqrt(i)
            b -= rate*(grad+noise)
            total += b
    out=total/(steps+1)
    return FitResult(out,np.flatnonzero(out),
        {'private':private,'epsilon_input':epsilon,'delta_input':delta,'guarantee_verified':False,
         'finite_precision_dp_proved':False if private else None,
         'theoretical_parameter_domain_satisfied':bool(private)},
        {'algorithm':'Tran repository sgd_convsm','steps':steps,'full_upstream_iterations':steps==n*n-1,
         'execution_backend':'compiled C++17 recurrence with NumPy-generated randomness' if kernel == 'logistic' else 'Python',
         'preprocessing':preprocessing,'noise_sd':sigma,
         'step_schedule':step_schedule,'eta':fixed_rate if step_schedule=='paper_constant' else None,
         'parameter_radius':parameter_radius,
         'source_eta_unused':step_schedule=='upstream_decay',
         'step_source':'Tran2024 Eq8; ChenChua2023 Thm3.4 public eta' if step_schedule=='paper_constant' else 'Tran R source 1/sqrt(t)',
         'theorem_bandwidth_coupling_enforced':False})


def tran_opm(X,y, *, tau=.5,h=.5,epsilon=1.,Cx=1.,kernel='logistic',
             beta0=None,rng=None,mode='fixed',preprocessing='l2',maxiter=2000,
             legacy_zero_init=False,calibration='replace_one',
             optimization_gap_tolerance=1e-8):
    """OPM with one fixed perturbation by default; upstream bug mode explicit.

    mode='upstream_resample' mimics stochastic objective evaluations (Nelder Mead).
    mode='fixed' uses one perturbation and a BFGS solve.  The default
    calibration='replace_one' adjusts Kifer et al.'s add/remove calibration for
    fixed-size replacement adjacency by running that theorem at epsilon/2.
    calibration='paper_literal' retains the paper's add/remove parameterization.
    Neither option certifies the finite-precision optimizer as DP.
    """
    X,y=_prepare(X,y,Cx,preprocessing); quantile(tau); positive('h',h)
    positive('epsilon',epsilon); integer('maxiter',maxiter)
    positive('optimization_gap_tolerance',optimization_gap_tolerance)
    if preprocessing != 'l2':
        raise ValueError("OPM requires preprocessing='l2' to enforce the feature-norm bound")
    if mode not in ('fixed','upstream_resample'): raise ValueError('unknown mode')
    if calibration not in ('replace_one','paper_literal'):
        raise ValueError("calibration must be 'replace_one' or 'paper_literal'")
    n,p=X.shape; b0=_start(beta0,p,legacy_zero_init)
    rng=np.random.default_rng() if rng is None else rng
    kbar=_kernel(0.,h,kernel)[0]
    xi=n*max(tau,1-tau)*Cx
    # Equation (7) uses an upper Hessian bound lambda and a *lower*
    # strong-convexity bound gamma.  The sample Hessian is
    #   sum_i k_h(y_i - x_i^T b) x_i x_i^T,
    # so n*kbar/h is an upper scalar bound only when ||x_i|| <= 1; it
    # cannot be used as a global lower bound.  With no full-rank design
    # assumption, gamma=0 is the valid global lower bound.
    lambda_upper=n*kbar*Cx*Cx/h
    gamma_lower=0.
    # Kifer et al. calibrate objective perturbation for |D△D'|=1
    # (add/remove).  A replacement changes two records, so invoke that
    # guarantee at epsilon/2 (group privacy).  In their notation this doubles
    # both the ridge Hessian and radial noise scale relative to the literal
    # add/remove calibration.
    calibration_multiplier=2. if calibration=='replace_one' else 1.
    ridge=max(calibration_multiplier*2*lambda_upper/epsilon-gamma_lower,0.)/n
    def draw():
        radius=rng.gamma(shape=p,scale=calibration_multiplier*2*xi/epsilon)
        direction=rng.standard_normal(p)
        return radius*direction/np.linalg.norm(direction)
    perturb=draw() if mode=='fixed' else None
    def loss_sum(b):
        u=y-X@b
        # The even integral is the stable logistic positive-part sum:
        # z + 2 log(1 + exp(-z)) = 2 log(2 cosh(z/2)).
        return np.sum(h/2*_kernel(u,h,kernel)[1]+(tau-.5)*u)
    def fun(b):
        z=perturb if mode=='fixed' else draw()
        return loss_sum(b)+ridge*np.dot(b,b)/2+z@b/n
    def jac(b):
        return X.T@(_kernel(X@b-y,h,kernel)[2]-tau)+ridge*b+perturb/n
    gradient_tolerance=np.sqrt(2*ridge*optimization_gap_tolerance/p)*.5
    result=minimize(fun,b0,jac=jac if mode=='fixed' else None,
                    method='BFGS' if mode=='fixed' else 'Nelder-Mead',
                    options={'maxiter':maxiter,'gtol':gradient_tolerance} if mode=='fixed' else {'maxiter':maxiter})
    residual_norm=float(np.linalg.norm(jac(result.x))) if mode=='fixed' else float('nan')
    gap_upper=(residual_norm**2/(2*ridge) if mode=='fixed' and ridge>0 else float('nan'))
    gap_met=bool(mode=='fixed' and gap_upper<=optimization_gap_tolerance)
    return FitResult(result.x,np.flatnonzero(result.x),
        {'private':True,'epsilon_input':epsilon,'guarantee_verified':False,
         'finite_precision_dp_proved':False},
        {'algorithm':'Tran repository opm_convsm','mode':mode,'preprocessing':preprocessing,
         'optimizer_success':gap_met if mode=='fixed' else bool(result.success),
         'scipy_success':bool(result.success),
         'optimization_gap_tolerance_met':gap_met if mode=='fixed' else None,
         'optimization_gap_tolerance':optimization_gap_tolerance,
         'optimizer_message':str(result.message),
         'ridge':ridge,'lambda_upper':lambda_upper,'gamma_lower':gamma_lower,
         'privacy_calibration':calibration,
         'optimizer_gradient_norm':residual_norm,
         'strong_convexity_modulus':ridge,
         'optimization_gap_upper_bound':gap_upper,
         'kernel':kernel,'initialization':'explicit beta0 or legacy zero',
         'paper_fidelity':'source-corrected; paper pending',
         'finite_precision_privacy_argument':'incomplete'})
