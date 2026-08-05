"""
Fast JIT-compiled filter kernels for all scoring criteria.

Each kernel runs the full forward recursion and returns (lam_path, total_score).
All kernels are numba @njit for ~10-20x speedup over pure Python.

# Custom — Context7 found no library equivalent for these paper-specific filters
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.stats import norm as _scipy_norm

# ---------------------------------------------------------------------------
# Shared norm.cdf / norm.pdf via lookup — numba cannot call scipy directly
# ---------------------------------------------------------------------------

@njit(cache=True)
def _phi(x: float) -> float:
    """Standard normal PDF, numba-compatible."""
    return 0.3989422804014327 * np.exp(-0.5 * x * x)

@njit(cache=True)
def _Phi(x: float) -> float:
    """Standard normal CDF via erfc, numba-compatible."""
    return 0.5 * (1.0 + _erf(x * 0.7071067811865476))  # x/sqrt(2)

@njit(cache=True)
def _erf(x: float) -> float:
    """Pure-Python erf approximation (Abramowitz & Stegun, max err 1.5e-7)."""
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
               + t * (-1.453152027 + t * 1.061405429))))
    result = 1.0 - poly * np.exp(-x * x)
    return result if x >= 0.0 else -result


# ---------------------------------------------------------------------------
# Numba-JIT filter kernels: return (lam_path T+1, total_score)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _filter_gaussian_log(y: np.ndarray, omega: float, phi: float, alpha: float) -> tuple:
    T = len(y)
    lam = np.empty(T + 1)
    lam[0] = omega / (1.0 - phi) if abs(phi) < 1.0 else 0.0
    total = 0.0
    for t in range(T):
        lt = lam[t]
        z = y[t] * np.exp(-0.5 * lt)
        z2 = z * z
        u = z2 - 1.0          # G=2, psi=0.5*(z^2-1) => G*psi = z^2-1
        s = 0.5 * lt + 0.5 * z2
        lam[t + 1] = omega + phi * lt + alpha * u
        if not np.isfinite(s) or not np.isfinite(lam[t + 1]):
            return lam, 1e10
        total += s
    return lam, total / T


@njit(cache=True)
def _filter_density_power(y: np.ndarray, omega: float, phi: float, alpha: float,
                           c_beta: float, kb: float, c1p: float,
                           c1p_inv: float, c11b: float, beta: float) -> tuple:
    T = len(y)
    lam = np.empty(T + 1)
    lam[0] = omega / (1.0 - phi) if abs(phi) < 1.0 else 0.0
    total = 0.0
    for t in range(T):
        lt = lam[t]
        z = y[t] * np.exp(-0.5 * lt)
        z2 = z * z
        ebz2 = np.exp(-beta * z2 * 0.5)
        # g = (1+beta)*c_beta*exp(-beta*z^2/2)*(z^2-1) + kb
        g = c1p * ebz2 * (z2 - 1.0) + kb
        # G = 2*exp(beta*lam/2), so u = G*psi = G*(g/G) = g
        u = g
        s = c_beta * np.exp(-beta * lt * 0.5) * (c1p_inv - c11b * ebz2)
        lam[t + 1] = omega + phi * lt + alpha * u
        if not np.isfinite(s) or not np.isfinite(lam[t + 1]):
            return lam, 1e10
        total += s
    return lam, total / T


@njit(cache=True)
def _filter_crps(y: np.ndarray, omega: float, phi: float, alpha: float) -> tuple:
    """CRPS filter: u = g_CRPS(z) = 1/sqrt(pi) - sqrt(2/pi)*exp(-z^2/2)."""
    T = len(y)
    lam = np.empty(T + 1)
    lam[0] = omega / (1.0 - phi) if abs(phi) < 1.0 else 0.0
    INV_SQPI = 0.5641895835477563   # 1/sqrt(pi)
    SQRT2OPI = 0.7978845608028654   # sqrt(2/pi)
    total = 0.0
    for t in range(T):
        lt = lam[t]
        z = y[t] * np.exp(-0.5 * lt)
        sg = np.exp(0.5 * lt)
        g = INV_SQPI - SQRT2OPI * np.exp(-z * z * 0.5)
        u = g                       # G=2*exp(-lam/2), psi=g/G => u=G*psi=g (verified)
        s = sg * (z * (2.0 * _Phi(z) - 1.0) + 2.0 * _phi(z) - INV_SQPI)
        lam[t + 1] = omega + phi * lt + alpha * u
        if not np.isfinite(s) or not np.isfinite(lam[t + 1]):
            return lam, 1e10
        total += s
    return lam, total / T


@njit(cache=True)
def _filter_mmd(y: np.ndarray, omega: float, phi: float, alpha: float, ell: float) -> tuple:
    """MMD-RBF filter with fixed bandwidth ell."""
    T = len(y)
    lam = np.empty(T + 1)
    lam[0] = omega / (1.0 - phi) if abs(phi) < 1.0 else 0.0
    ell2 = ell * ell
    ell_ell2p2_m3h = 2.0 * ell / (ell2 + 2.0) ** 1.5   # 2*ell/(ell^2+2)^(3/2)
    total = 0.0
    for t in range(T):
        lt = lam[t]
        yt = y[t]
        exp_neg_half_lt = np.exp(-0.5 * lt)
        z = yt * exp_neg_half_lt
        r = ell * exp_neg_half_lt
        D = r * r + 1.0
        sqD = np.sqrt(D)
        z2 = z * z
        u = (2.0 * (r / sqD) * np.exp(-z2 / (2.0 * D))
             * (z2 / (D * D) - 1.0 / D)
             + ell_ell2p2_m3h)
        v = np.exp(lt)
        denom = ell2 + v
        s = (1.0 - 2.0 * ell / np.sqrt(denom) * np.exp(-yt * yt / (2.0 * denom))
             + ell / np.sqrt(ell2 + 2.0 * v))
        lam[t + 1] = omega + phi * lt + alpha * u
        if not np.isfinite(s) or not np.isfinite(lam[t + 1]):
            return lam, 1e10
        total += s
    return lam, total / T


@njit(cache=True)
def _filter_student_log(y: np.ndarray, omega: float, phi: float, alpha: float,
                         nu: float) -> tuple:
    """Student-t log score filter (variance-standardised)."""
    T = len(y)
    lam = np.empty(T + 1)
    lam[0] = omega / (1.0 - phi) if abs(phi) < 1.0 else 0.0
    nu2 = nu - 2.0
    total = 0.0
    for t in range(T):
        lt = lam[t]
        z = y[t] * np.exp(-0.5 * lt)
        z2 = z * z
        # psi = 0.5*((nu+1)*z^2/(nu-2+z^2) - 1), u = G*psi = 2*psi
        u = (nu + 1.0) * z2 / (nu2 + z2) - 1.0
        s = 0.5 * lt + 0.5 * (nu + 1.0) * np.log(1.0 + z2 / nu2)
        lam[t + 1] = omega + phi * lt + alpha * u
        if not np.isfinite(s) or not np.isfinite(lam[t + 1]):
            return lam, 1e10
        total += s
    return lam, total / T


# ---------------------------------------------------------------------------
# Public API: objective and path functions
# ---------------------------------------------------------------------------

def _make_dp_consts(beta: float) -> tuple:
    """Pre-compute density-power constants."""
    c_beta = (2.0 * np.pi) ** (-beta / 2.0)
    kb = beta * c_beta * (1.0 + beta) ** (-0.5)
    c1p = (1.0 + beta) * c_beta
    c1p_inv = (1.0 + beta) ** (-0.5)
    c11b = (1.0 + 1.0 / beta) * c_beta
    return c_beta, kb, c1p, c1p_inv, c11b


def filter_score(
    params: np.ndarray,
    y: np.ndarray,
    criterion: str,
    ell: float | None = None,
    beta: float = 0.15,
    nu: float = 6.0,
) -> float:
    """
    Compute average scoring risk for given params via JIT kernel.

    Parameters
    ----------
    params : (omega, phi, alpha)
    y : 1-D observation array
    criterion : 'gaussian_log' | 'density_power' | 'crps' | 'mmd' | 'student_log'
    ell : MMD bandwidth (required for 'mmd')
    beta : density-power exponent (default 0.15)
    nu : Student degrees of freedom (default 6.0)

    Returns
    -------
    float : average scoring risk (1e10 if numerically invalid)
    """
    omega, phi, alpha = float(params[0]), float(params[1]), float(params[2])
    if abs(phi) >= 0.999:
        return 1e10

    if criterion == 'gaussian_log':
        _, score = _filter_gaussian_log(y, omega, phi, alpha)
    elif criterion == 'density_power':
        c_beta, kb, c1p, c1p_inv, c11b = _make_dp_consts(beta)
        _, score = _filter_density_power(y, omega, phi, alpha, c_beta, kb, c1p, c1p_inv, c11b, beta)
    elif criterion == 'crps':
        _, score = _filter_crps(y, omega, phi, alpha)
    elif criterion == 'mmd':
        ell_ = float(ell) if ell is not None else 2.0
        _, score = _filter_mmd(y, omega, phi, alpha, ell_)
    elif criterion == 'student_log':
        _, score = _filter_student_log(y, omega, phi, alpha, float(nu))
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    return float(score)


def filter_path(
    params: np.ndarray,
    y: np.ndarray,
    criterion: str,
    ell: float | None = None,
    beta: float = 0.15,
    nu: float = 6.0,
) -> np.ndarray:
    """
    Run forward filter and return the full lam path (T+1,).

    Parameters
    ----------
    Same as filter_score.

    Returns
    -------
    np.ndarray shape (T+1,) : log-variance state path
    """
    omega, phi, alpha = float(params[0]), float(params[1]), float(params[2])

    if criterion == 'gaussian_log':
        lam, _ = _filter_gaussian_log(y, omega, phi, alpha)
    elif criterion == 'density_power':
        c_beta, kb, c1p, c1p_inv, c11b = _make_dp_consts(beta)
        lam, _ = _filter_density_power(y, omega, phi, alpha, c_beta, kb, c1p, c1p_inv, c11b, beta)
    elif criterion == 'crps':
        lam, _ = _filter_crps(y, omega, phi, alpha)
    elif criterion == 'mmd':
        ell_ = float(ell) if ell is not None else 2.0
        lam, _ = _filter_mmd(y, omega, phi, alpha, ell_)
    elif criterion == 'student_log':
        lam, _ = _filter_student_log(y, omega, phi, alpha, float(nu))
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    return np.asarray(lam)


def warm_jit(beta: float = 0.15, nu: float = 6.0, ell: float = 2.0) -> None:
    """
    Pre-compile all JIT kernels with tiny dummy arrays to avoid cold-start
    latency during timed experiments.
    """
    dummy = np.array([0.1, -0.2, 0.05, 0.3], dtype=np.float64)
    p = np.array([0.0, 0.95, 0.05])
    for crit in ('gaussian_log', 'density_power', 'crps', 'mmd', 'student_log'):
        filter_score(p, dummy, crit, ell=ell, beta=beta, nu=nu)
