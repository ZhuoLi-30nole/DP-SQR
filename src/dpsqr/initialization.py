"""Private initialization; allocation is centralized in policy.py.

The routines implement the *ideal real-arithmetic* Laplace/Gaussian mechanisms.
NumPy's floating-point random samplers are not an end-to-end DP-certified
implementation. The default uniform-gap route uses fixed iteration counts and
an error-aware sensitivity bound. The explicit bounded_output_2c alternative
projects before noise and uses sensitivity 2 * output_radius. No function
claims a floating-software DP certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, ndtr


@dataclass
class InitializationResult:
    beta: np.ndarray
    method: str
    epsilon: float
    delta: float
    scale: float
    bandwidth: float | None
    output_radius: float | None
    sensitivity_bound: float
    gaussian_sigma: float
    metadata: dict[str, Any]


def _valid_eps_delta(epsilon: float, delta: float) -> None:
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    if not np.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")


def _rng_or_default(rng: np.random.Generator | None) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


def _validate_xy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size or X.shape[0] == 0:
        raise ValueError("X must be a nonempty 2D matrix aligned with y")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must contain only finite values")
    return X, y


def clip_design_rows(X: np.ndarray, *, intercept: bool = True) -> tuple[np.ndarray, float]:
    """Apply README 3.3 radial clipping to slopes, then optionally add intercept.

    The public slope radius is ``sqrt(d)/6``, where d includes the intercept
    when requested. Returns the resulting design and its public row bound B.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not np.all(np.isfinite(X)):
        raise ValueError("X must be a finite 2D matrix")
    n, p = X.shape
    d = p + int(intercept)
    radius = np.sqrt(d) / 6
    norms = np.linalg.norm(X, axis=1)
    scale = np.minimum(1., radius / np.maximum(norms, np.finfo(float).tiny))
    slopes = X * scale[:, None]
    Z = np.column_stack((np.ones(n), slopes)) if intercept else slopes
    return Z, float(np.sqrt(1 + d / 36) if intercept else np.sqrt(d) / 6)


def private_scale_estimate(y: np.ndarray, epsilon_scale: float, *,
                           rng: np.random.Generator | None = None) -> tuple[float, dict[str, float]]:
    """Chang-style noisy moments, spending ``epsilon_scale`` in two Laplace calls.

    y is clipped to [-log(n), log(n)]. Each moment gets epsilon_scale/2,
    yielding scales 4 log(n)/(n eps_scale) and 2 log(n)^2/(n eps_scale).
    This is the equivalent scale formulation of README 3.1's allocation
    epsilon_I/4 after allocating epsilon_I/8 to each moment. A value within a
    small floating roundoff tolerance of zero variance uses the same fallback
    as a nonpositive value.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size == 0 or not np.all(np.isfinite(y)):
        raise ValueError("y must be nonempty and finite")
    if not np.isfinite(epsilon_scale) or epsilon_scale <= 0:
        raise ValueError("epsilon_scale must be positive and finite")
    n = y.size
    cap = float(np.log(n)) if n > 1 else 1.
    yc = np.clip(y, -cap, cap)
    # Replacing one record changes the mean by at most 2*cap/n and the
    # second moment by at most cap^2/n (range [0, cap^2]).
    rng = _rng_or_default(rng)
    eps_each = epsilon_scale / 2
    mean_scale = 2 * cap / (n * eps_each)
    second_scale = cap**2 / (n * eps_each)
    mean_noisy = float(np.mean(yc) + rng.laplace(0., mean_scale))
    second_noisy = float(np.mean(yc**2) + rng.laplace(0., second_scale))
    variance = second_noisy - mean_noisy**2
    roundoff = 16 * np.finfo(float).eps * max(abs(second_noisy), mean_noisy**2, 1.)
    tau = float(np.sqrt(variance)) if variance > roundoff else 2.
    return tau, {"mean_laplace_scale": mean_scale,
                 "second_moment_laplace_scale": second_scale,
                 "mean_noisy": mean_noisy,
                 "second_moment_noisy": second_noisy,
                 "response_clip": cap}


def _clipped_product_scores(X,y,xi):
    # Each column is reduced over all n rows, in the original row order.
    # Column tiles avoid simultaneous n-by-p product and clipping buffers.
    n,p=X.shape
    score=np.empty(p)
    for start in range(0,p,256):
        end=min(p,start+256)
        products=y[:,None]*X[:,start:end]
        np.clip(products,-xi,xi,out=products)
        score[start:end]=np.abs(np.mean(products,axis=0))
    return score


def chang_screen(X: np.ndarray, y: np.ndarray, s: int, epsilon_screen: float | None = None, *,
                 intercept_index: int | None = None,
                 rng: np.random.Generator | None = None,
                 private: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """Chang product-score peeling with the README's conservative noise scale.

    Scores are |mean(clip(y*x_j, -xi, xi))|, xi=sqrt(log(n*p)); each peeling
    round draws fresh Laplace noise of scale 2(s-1)Delta/epsilon_screen.
    If intercept_index is supplied it is retained and s-1 other columns are
    selected. Selection returns sorted original column indices.
    """
    X, y = _validate_xy(X, y)
    n, p = X.shape
    if not isinstance(s, (int, np.integer)) or s < 1 or s > p:
        raise ValueError("s must be an integer in [1, number of columns]")
    if private and (epsilon_screen is None or not np.isfinite(epsilon_screen) or epsilon_screen <= 0):
        raise ValueError("private screening requires positive epsilon_screen")
    if intercept_index is not None and not 0 <= intercept_index < p:
        raise ValueError("intercept_index is out of range")
    wanted = s - int(intercept_index is not None)
    candidates = np.array([j for j in range(p) if j != intercept_index], dtype=int)
    if wanted < 0 or wanted > candidates.size:
        raise ValueError("s is incompatible with the fixed intercept")
    xi = float(np.sqrt(np.log(n * p)))
    score = _clipped_product_scores(X,y,xi)
    delta_score = 2 * xi / n
    rounds = wanted
    laplace_scale = (2 * rounds * delta_score / epsilon_screen) if private and rounds else 0.
    rng = _rng_or_default(rng)
    remaining = list(map(int, candidates))
    selected: list[int] = []
    for _ in range(rounds):
        noise = rng.laplace(0., laplace_scale, size=len(remaining)) if private else 0.
        perturbed = score[remaining] + noise
        idx = int(np.argmax(perturbed))
        selected.append(remaining.pop(idx))
    if intercept_index is not None:
        selected.append(intercept_index)
    return np.array(sorted(selected), dtype=int), {
        "method": "Chang product-score peeling",
        "epsilon": epsilon_screen if private else 0.,
        "delta": 0.,
        "xi": xi,
        "score_sensitivity": delta_score,
        "laplace_scale": laplace_scale,
        "rounds": rounds,
        "conservative_factor": 2,
    }


def _project_ball(beta: np.ndarray, radius: float) -> np.ndarray:
    norm = float(np.linalg.norm(beta))
    if norm <= radius:
        return beta.copy()
    return beta * (radius / norm)


def _gaussian_sigma(sensitivity: float, epsilon: float, delta: float) -> float:
    """Conservative Gaussian calibration via exact zCDP inversion, any eps>0."""
    from .privacy import rho_from_eps_delta
    return float(sensitivity / np.sqrt(2*rho_from_eps_delta(epsilon, delta)))


def _uniform_gd_iterations(L: float, ridge: float, gradient_at_zero_bound: float,
                           gap_target: float) -> int:
    """Iterations giving a data-independent objective-gap bound in exact math."""
    distance_bound = gradient_at_zero_bound / ridge
    initial_gap_bound = .5 * L * distance_bound**2
    if initial_gap_bound <= gap_target:
        return 0
    contraction = 1 - ridge / L
    if not 0 < contraction < 1:
        return 1
    return int(np.ceil(np.log(gap_target / initial_gap_bound) /
                       (2 * np.log(contraction))))


def _uniform_gradient_fit(objective, dimension: int, *, L: float, ridge: float,
                          gradient_at_zero_bound: float,
                          gap_target: float = 1e-8) -> tuple[np.ndarray, int, float]:
    """Fixed-step gradient descent with a uniform strongly-convex gap bound.

    For an L-smooth, ridge-strongly-convex objective, x*=argmin f satisfies
    ||x*|| <= ||grad f(0)||/ridge. A 1/L step contracts distance by
    (1-ridge/L); L-smoothness then bounds the objective gap. All inputs to the
    iteration count are public bounds, never observed convergence diagnostics.
    """
    L = max(float(L), float(ridge))
    count = _uniform_gd_iterations(L, ridge, gradient_at_zero_bound, gap_target)
    beta = np.zeros(dimension)
    for _ in range(count):
        _, grad = objective(beta)
        beta -= grad / L
    distance = gradient_at_zero_bound / ridge * (1 - ridge / L) ** count
    gap_bound = .5 * L * distance**2
    return beta, count, float(gap_bound)


def _smooth_sqr_objective(beta: np.ndarray, Z: np.ndarray, y: np.ndarray,
                          q: float, h: float, ridge: float,
                          kernel: str = "gaussian") -> tuple[float, np.ndarray]:
    residual = y - Z @ beta
    if kernel == "logistic":
        score = q + expit(residual / h) - 1.
        loss = q * residual + h * np.logaddexp(0., residual / h) - residual
    elif kernel == "gaussian":
        z = residual / h
        cdf = ndtr(z)
        density = np.exp(-.5 * z**2) / np.sqrt(2 * np.pi)
        score = q + cdf - 1.
        loss = q * residual + residual * cdf + h * density - residual
    else:
        raise ValueError("kernel must be 'gaussian' or 'logistic'")
    value = float(np.mean(loss) + ridge * np.dot(beta, beta) / 2)
    grad = -(Z.T @ score) / y.size + ridge * beta
    return value, grad


def _huber_objective(beta: np.ndarray, Z: np.ndarray, y: np.ndarray,
                     tau: float, ridge: float) -> tuple[float, np.ndarray]:
    residual = y - Z @ beta
    abs_r = np.abs(residual)
    loss = np.where(abs_r <= tau, .5 * residual**2,
                    tau * (abs_r - .5 * tau))
    score = np.clip(residual, -tau, tau)
    return (float(np.mean(loss) + ridge * np.dot(beta, beta) / 2),
            -(Z.T @ score) / y.size + ridge * beta)


def _ridge_ls_objective(beta: np.ndarray, Z: np.ndarray, y: np.ndarray,
                        ridge: float) -> tuple[float, np.ndarray]:
    residual = Z @ beta - y
    return (float(.5 * np.mean(residual**2) + ridge * np.dot(beta, beta) / 2),
            Z.T @ residual / y.size + ridge * beta)


def _fit_and_release(X: np.ndarray, y: np.ndarray, *, method: str,
                     epsilon_initialization: float, delta_initialization: float,
                     output_radius: float, q: float, ridge: float,
                     response_clip: float | None, rng: np.random.Generator | None,
                     maxiter: int, privacy_mode: str, intercept: bool,
                     private: bool, kernel: str) -> InitializationResult:
    _valid_eps_delta(epsilon_initialization, delta_initialization)
    if output_radius is not None and (not np.isfinite(output_radius) or output_radius <= 0):
        raise ValueError("output_radius must be None or positive and finite")
    if output_radius is None and (privacy_mode != "uniform_gap" or method == "ridge_ls"):
        raise ValueError("Only uniform-gap SQR/Huber initialization supports no projection")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be positive and finite")
    X, y = _validate_xy(X, y)
    Z, B = clip_design_rows(X, intercept=intercept)
    eps_scale = epsilon_initialization / 4
    eps_fit = 3 * epsilon_initialization / 4
    if private:
        tau0, scale_meta = private_scale_estimate(y, eps_scale, rng=rng)
    else:
        # Keep the same clipped-moment estimator and fallback as the paired
        # private method; only the two Laplace draws and their budget are absent.
        y0 = np.asarray(y, dtype=float).reshape(-1)
        n0 = y0.size
        cap0 = float(np.log(n0)) if n0 > 1 else 1.
        yc0 = np.clip(y0, -cap0, cap0)
        mean0 = float(np.mean(yc0))
        second0 = float(np.mean(yc0**2))
        variance0 = second0 - mean0**2
        roundoff0 = 16 * np.finfo(float).eps * max(abs(second0), mean0**2, 1.)
        tau0 = float(np.sqrt(variance0)) if variance0 > roundoff0 else 2.
        scale_meta = {"mean_laplace_scale": 0., "second_moment_laplace_scale": 0.,
                      "mean_noisy": mean0, "second_moment_noisy": second0,
                      "response_clip": cap0}
    s_dp = float(np.clip(tau0, .1, 10.))
    bandwidth = .5 * s_dp if method == "ridge_sqr" else None
    huber_threshold = float(tau0)
    if method == "ridge_ls" and response_clip is None:
        response_clip = s_dp * np.sqrt(2 * np.log(max(2, y.size)))
    y_fit = np.clip(y, -response_clip, response_clip) if response_clip is not None else y
    if method == "ridge_sqr":
        if not 0 < q < 1:
            raise ValueError("q must be in (0,1)")
        objective = lambda beta: _smooth_sqr_objective(beta, Z, y_fit, q, bandwidth, ridge, kernel)
    elif method == "ridge_huber":
        objective = lambda beta: _huber_objective(beta, Z, y_fit, huber_threshold, ridge)
    elif method == "ridge_ls":
        objective = lambda beta: _ridge_ls_objective(beta, Z, y_fit, ridge)
    else:
        raise ValueError(f"unsupported initialization method: {method}")
    if privacy_mode not in {"bounded_output_2c", "uniform_gap"}:
        raise ValueError("privacy_mode must be 'bounded_output_2c' or 'uniform_gap'")
    gap_bound = None
    iterations = None
    if privacy_mode == "uniform_gap":
        # ||Z_i|| <= B. Public smoothness and zero-gradient bounds are specific
        # to the three normalized objectives and never inspect solver success.
        B2 = B**2
        if method == "ridge_sqr":
            kernel_lipschitz = (1 / np.sqrt(2 * np.pi) if kernel == "gaussian" else .25)
            L = B2 * kernel_lipschitz / bandwidth + ridge
            grad0 = B * max(q, 1 - q)
            exact_sensitivity = 2 * B * max(q, 1 - q) / (y.size * ridge)
        elif method == "ridge_huber":
            L = B2 + ridge
            grad0 = B * huber_threshold
            exact_sensitivity = 2 * B * huber_threshold / (y.size * ridge)
        else:
            # Constrained problem in ||beta|| <= C. The public gradient bound
            # at zero is B*R; projected gradient contracts for this strongly
            # convex smooth objective on the convex ball.
            L = B2 + ridge
            grad0 = B * float(response_clip)
            exact_sensitivity = 2 * B * (float(response_clip) + B * output_radius) / (y.size * ridge)
        if method == "ridge_ls":
            # Projected GD on the public ball; this is the specified constrained
            # ridge-LS objective, not an unconstrained fit followed by clipping.
            opt_grad_bound = B * (float(response_clip) + B * output_radius) + ridge * output_radius
            # The constrained optimum and zero are both in the public C-ball.
            distance_bound = output_radius
            count = 0
            while (opt_grad_bound * distance_bound + .5 * L * distance_bound**2) > 1e-8:
                distance_bound *= 1 - ridge / L
                count += 1
                if count > 1_000_000:
                    raise RuntimeError("could not establish the requested uniform gap bound")
            candidate = np.zeros(Z.shape[1])
            for _ in range(count):
                _, grad = objective(candidate)
                candidate = _project_ball(candidate - grad / L, output_radius)
            gap_bound = float(opt_grad_bound * distance_bound + .5 * L * distance_bound**2)
            iterations = count
        else:
            candidate, iterations, gap_bound = _uniform_gradient_fit(
                objective, Z.shape[1], L=L, ridge=ridge,
                gradient_at_zero_bound=grad0, gap_target=1e-8)
        bounded = candidate if output_radius is None else _project_ball(candidate, output_radius)
        opt_error = float(np.sqrt(2 * gap_bound / ridge))
        sensitivity = float(exact_sensitivity + 2 * opt_error)
    else:
        # For the 2C route the optimizer's quality is immaterial. A failed or
        # nonfinite numeric result maps to the fixed zero vector, then every
        # possible output is projected to the public radius before adding noise.
        if method == "ridge_ls":
            fit = minimize(objective, np.zeros(Z.shape[1]), jac=True, method="SLSQP",
                           constraints={"type": "ineq", "fun": lambda b: output_radius**2 - np.dot(b, b),
                                        "jac": lambda b: -2 * b},
                           options={"maxiter": int(maxiter), "ftol": 1e-12})
        else:
            fit = minimize(objective, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-9})
        candidate = np.asarray(fit.x, dtype=float)
        if candidate.shape != (Z.shape[1],) or not np.all(np.isfinite(candidate)):
            candidate = np.zeros(Z.shape[1])
        bounded = _project_ball(candidate, output_radius)
        sensitivity = 2 * output_radius
    sigma = (_gaussian_sigma(sensitivity, eps_fit, delta_initialization)
             if private else 0.)
    rng = _rng_or_default(rng)
    released = bounded + rng.normal(0., sigma, size=bounded.size)
    if output_radius is not None:
        released = _project_ball(released, output_radius)
    return InitializationResult(
        beta=released, method=method,
        epsilon=epsilon_initialization if private else 0.,
        delta=delta_initialization if private else 0.,
        scale=huber_threshold if method == "ridge_huber" else s_dp,
        bandwidth=bandwidth,
        output_radius=output_radius, sensitivity_bound=sensitivity,
        gaussian_sigma=sigma,
        metadata={
            "privacy_mode": privacy_mode if private else "nonprivate",
            "privacy_interpretation": "ideal real-arithmetic Gaussian/Laplace mechanisms only",
            "floating_point_dp_certified": False if private else None,
            "optimization_certificate_used": privacy_mode == "uniform_gap",
            "uniform_gap_bound": gap_bound,
            "uniform_gap_target": 1e-8 if privacy_mode == "uniform_gap" else None,
            "optimization_iterations": iterations,
            "scale_epsilon": eps_scale if private else 0.,
            "output_epsilon": eps_fit if private else 0.,
            "output_delta": delta_initialization if private else 0.,
            "ridge": ridge,
            "raw_noisy_scale": tau0,
            "s_dp": s_dp,
            "design_row_bound": B,
            "response_clip": response_clip,
            **scale_meta,
        })


def initialize_sqr(X: np.ndarray, y: np.ndarray, *, q: float,
                   epsilon_initialization: float, delta_initialization: float,
                   output_radius: float = 10., ridge: float = .2,
                   rng: np.random.Generator | None = None,
                   maxiter: int = 2000,
                   privacy_mode: str = "uniform_gap",
                   kernel: str = "gaussian", intercept: bool = True,
                   private: bool = True) -> InitializationResult:
    """Row-clipped, ridge-penalized smoothed SQR initialization (README 3.3)."""
    return _fit_and_release(X, y, method="ridge_sqr",
                            epsilon_initialization=epsilon_initialization,
                            delta_initialization=delta_initialization,
                            output_radius=output_radius, q=q, ridge=ridge,
                            response_clip=None, rng=rng, maxiter=maxiter,
                            privacy_mode=privacy_mode, intercept=intercept,
                            private=private, kernel=kernel)


def initialize_huber(X: np.ndarray, y: np.ndarray, *,
                     epsilon_initialization: float, delta_initialization: float,
                     output_radius: float = 10., ridge: float = .2,
                     rng: np.random.Generator | None = None,
                     maxiter: int = 2000,
                     privacy_mode: str = "uniform_gap",
                     intercept: bool = True, private: bool = True) -> InitializationResult:
    """Standard-Huber ridge initialization with Chang's noisy scale (README 3.1)."""
    return _fit_and_release(X, y, method="ridge_huber",
                            epsilon_initialization=epsilon_initialization,
                            delta_initialization=delta_initialization,
                            output_radius=output_radius, q=.5, ridge=ridge,
                            response_clip=None, rng=rng, maxiter=maxiter,
                            privacy_mode=privacy_mode, intercept=intercept,
                            private=private, kernel="logistic")


def initialize_ridge_ls(X: np.ndarray, y: np.ndarray, *,
                         epsilon_initialization: float, delta_initialization: float,
                         response_clip: float | None = None,
                         output_radius: float = 10., ridge: float = .2,
                         rng: np.random.Generator | None = None,
                         maxiter: int = 2000,
                         privacy_mode: str = "uniform_gap",
                         intercept: bool = True, private: bool = True) -> InitializationResult:
    """Constrained-design ridge LS initialization with clipped response (README 3.4).

    The fitting output is restricted to the same public L2 ball used for the
    2C mechanism, and the noisy output is projected back to that ball.
    """
    if response_clip is not None and (not np.isfinite(response_clip) or response_clip <= 0):
        raise ValueError("response_clip must be positive and finite")
    return _fit_and_release(X, y, method="ridge_ls",
                            epsilon_initialization=epsilon_initialization,
                            delta_initialization=delta_initialization,
                            output_radius=output_radius, q=.5, ridge=ridge,
                            response_clip=response_clip, rng=rng, maxiter=maxiter,
                            privacy_mode=privacy_mode, intercept=intercept,
                            private=private, kernel="gaussian")

