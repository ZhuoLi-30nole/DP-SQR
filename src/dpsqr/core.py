"""Main text Section 2: kernels, quantile loss, clipped scores, validation."""
from dataclasses import dataclass, field
import numpy as np
from scipy.special import ndtr, expit


@dataclass
class FitResult:
    coef: np.ndarray
    support: np.ndarray
    privacy: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def positive(name, value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def integer(name, value, minimum=1):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def data_xy(X, y):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or len(y) != X.shape[0] or min(X.shape) < 1:
        raise ValueError("X must be nonempty (n,p), y must have shape (n,)")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must be finite")
    return X, y


def initial(beta0, p):
    b = np.zeros(p) if beta0 is None else np.array(beta0, dtype=float, copy=True)
    if b.shape != (p,) or not np.isfinite(b).all():
        raise ValueError("beta0 must be a finite vector of length p")
    return b


def quantile(tau):
    if not np.isfinite(tau) or not 0 < tau < 1:
        raise ValueError("tau must be in (0,1)")


def kernel_cdf(z, kernel="gaussian"):
    z = np.asarray(z, dtype=float)
    if kernel == "gaussian":
        return ndtr(z)
    if kernel == "logistic":
        return expit(z)
    u = np.clip(z, -1, 1)
    if kernel == "epanechnikov":
        return 0.5 + 0.75 * (u - u**3 / 3)
    if kernel == "biweight":
        return 0.5 + 15 / 16 * (u - 2 * u**3 / 3 + u**5 / 5)
    raise ValueError(f"Unknown kernel: {kernel}")


def kernel_pdf(z, kernel="gaussian"):
    z = np.asarray(z, dtype=float)
    if kernel == "gaussian":
        return np.exp(-0.5*z*z) / np.sqrt(2*np.pi)
    if kernel == "logistic":
        q = expit(z)
        return q * (1-q)
    u = np.clip(z, -1, 1)
    if kernel == "epanechnikov":
        return 0.75*(1-u*u)
    if kernel == "biweight":
        return 15/16*(1-u*u)**2
    raise ValueError(f"Unknown kernel: {kernel}")


def radial_clip(X, bound):
    positive("bound", bound)
    # Scaling first avoids overflow in the norm for large finite coordinates.
    scale = np.max(np.abs(X), axis=1, keepdims=True)
    unit = X / np.maximum(scale, np.finfo(float).tiny)
    norm_unit = np.linalg.norm(unit, axis=1, keepdims=True)
    factor = np.minimum(1.0, (bound / np.maximum(scale, np.finfo(float).tiny)) /
                        np.maximum(norm_unit, np.finfo(float).tiny))
    return X * factor


def score(X, y, beta, *, tau, h, bound=None, clipping="none", kernel="gaussian"):
    """Eq. clippedgradients: residual uses ORIGINAL X, only multiplier is clipped."""
    quantile(tau)
    positive("h", h)
    if clipping == "l2":
        multiplier = radial_clip(X, bound)
    elif clipping == "linf":
        positive("bound", bound)
        multiplier = np.clip(X, -bound, bound)
    elif clipping == "none":
        multiplier = X
    else:
        raise ValueError("clipping must be none, l2 or linf")
    return multiplier.T @ (kernel_cdf((X @ beta-y)/h, kernel)-tau) / len(y)


def check_loss(residual, tau):
    quantile(tau)
    r = np.asarray(residual)
    return np.maximum(tau*r, (tau-1)*r)


def gaussian_smoothed_loss(residual, tau, h):
    """Exact Gaussian convolution, useful for finite-difference checks."""
    quantile(tau)
    positive("h", h)
    u = np.asarray(residual)
    z = u/h
    return u*(ndtr(z)+tau-1) + h*kernel_pdf(z)
