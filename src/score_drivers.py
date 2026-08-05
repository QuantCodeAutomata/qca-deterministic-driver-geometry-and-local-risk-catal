"""
Score functions, raw innovation drivers, and implemented updates for
observation-driven volatility filters.

Scale-state parameterisation: v = exp(lambda), residual z = y / sqrt(v).
Raw innovation (score derivative): psi_S(y, lambda) = -d/dlambda S(P_lambda, y)
Implemented update: u_S(y, lambda) = G_S(lambda) * psi_S(y, lambda)

All formulas follow the paper exactly.
# Custom — Context7 found no library equivalent for these paper-specific score drivers
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from typing import Union

ArrayLike = Union[np.ndarray, float]


# ---------------------------------------------------------------------------
# Gaussian working density utilities
# ---------------------------------------------------------------------------

def gaussian_log_score(lam: ArrayLike, y: ArrayLike) -> ArrayLike:
    """
    Gaussian log score (negative log-likelihood):
        S_log = 0.5 * lambda + 0.5 * exp(-lambda) * y^2

    Parameters
    ----------
    lam : log-scale state lambda
    y   : observation

    Returns
    -------
    Score value S_log(lambda, y)
    """
    lam = np.asarray(lam, dtype=float)
    y = np.asarray(y, dtype=float)
    return 0.5 * lam + 0.5 * np.exp(-lam) * y**2


def gaussian_log_driver(z: ArrayLike) -> ArrayLike:
    """
    Gaussian log-score residual driver: g_log(z) = z^2 - 1
    Scaling constant: G_log = 2

    Parameters
    ----------
    z : standardised residual y / sqrt(exp(lambda))

    Returns
    -------
    Driver value g_log(z)
    """
    z = np.asarray(z, dtype=float)
    return z**2 - 1.0


def gaussian_log_psi(y: ArrayLike, lam: ArrayLike) -> ArrayLike:
    """
    Gaussian log-score raw innovation: psi_log = 0.5 * (exp(-lambda)*y^2 - 1)

    Parameters
    ----------
    y   : observation
    lam : log-scale state

    Returns
    -------
    psi_log(y, lambda)
    """
    y = np.asarray(y, dtype=float)
    lam = np.asarray(lam, dtype=float)
    return 0.5 * (np.exp(-lam) * y**2 - 1.0)


G_LOG = 2.0  # Scaling constant for Gaussian log driver


# ---------------------------------------------------------------------------
# Variance-standardised Student-t log score
# ---------------------------------------------------------------------------

def student_log_driver(z: ArrayLike, nu: float = 6.0) -> ArrayLike:
    """
    Variance-standardised Student-t_nu log-score residual driver:
        g(z) = ((nu+1) * z^2) / ((nu-2) + z^2) - 1

    Parameters
    ----------
    z  : standardised residual (standardised to unit variance)
    nu : degrees of freedom (default 6)

    Returns
    -------
    Driver value
    """
    z = np.asarray(z, dtype=float)
    assert nu > 2, "nu must be > 2"
    return ((nu + 1.0) * z**2) / ((nu - 2.0) + z**2) - 1.0


def student_log_psi(y: ArrayLike, lam: ArrayLike, nu: float = 6.0) -> ArrayLike:
    """
    Variance-standardised Student-t_nu log-score raw innovation.

    Parameters
    ----------
    y   : observation
    lam : log-scale state
    nu  : degrees of freedom

    Returns
    -------
    psi_student_log(y, lambda)
    """
    y = np.asarray(y, dtype=float)
    lam = np.asarray(lam, dtype=float)
    v = np.exp(lam)
    z2 = y**2 / (v * (nu - 2.0) / nu)  # standardised to unit variance
    # z = y / sqrt(v*(nu-2)/nu), so z^2 = y^2 * nu / (v*(nu-2))
    z2_adj = y**2 / v  # z^2 where z = y/sqrt(v)
    # psi = d/dlambda log f_t(y|lambda) with scaling
    # Use driver form: psi = 0.5 * (g(z) * G_S_inv) where g is residual driver
    # For Student-t_nu log score, psi = 0.5 * [((nu+1)*z^2)/((nu-2)+z^2) - 1]
    z = y / np.sqrt(v)
    return 0.5 * (((nu + 1.0) * z**2) / ((nu - 2.0) + z**2) - 1.0)


G_STUDENT = 2.0  # Scaling constant for Student log driver


# ---------------------------------------------------------------------------
# Density-power score (beta-score) with Gaussian working density
# ---------------------------------------------------------------------------

def density_power_constants(beta: float) -> tuple[float, float]:
    """
    Compute density-power score constants for Gaussian working density.

    Returns
    -------
    c_beta : (2*pi)^(-beta/2)
    kappa_beta : beta * c_beta * (1+beta)^(-1/2)
    """
    c_beta = (2.0 * np.pi) ** (-beta / 2.0)
    kappa_beta = beta * c_beta * (1.0 + beta) ** (-0.5)
    return c_beta, kappa_beta


def density_power_score(lam: ArrayLike, y: ArrayLike, beta: float = 0.15) -> ArrayLike:
    """
    Density-power score for Gaussian working density:
        S_beta = c_beta * exp(-beta*lambda/2) * [(1+beta)^(-1/2) - (1+1/beta)*exp(-beta*z^2/2)]

    Parameters
    ----------
    lam  : log-scale state
    y    : observation
    beta : power exponent (default 0.15)

    Returns
    -------
    Score value S_beta(lambda, y)
    """
    lam = np.asarray(lam, dtype=float)
    y = np.asarray(y, dtype=float)
    c_beta, _ = density_power_constants(beta)
    v = np.exp(lam)
    z = y / np.sqrt(v)
    return c_beta * np.exp(-beta * lam / 2.0) * (
        (1.0 + beta) ** (-0.5) - (1.0 + 1.0 / beta) * np.exp(-beta * z**2 / 2.0)
    )


def density_power_driver(z: ArrayLike, beta: float = 0.15) -> ArrayLike:
    """
    Density-power residual driver:
        g_beta(z) = (1+beta) * c_beta * exp(-beta*z^2/2) * (z^2 - 1) + kappa_beta

    Parameters
    ----------
    z    : standardised residual
    beta : power exponent

    Returns
    -------
    Driver value g_beta(z)
    """
    z = np.asarray(z, dtype=float)
    c_beta, kappa_beta = density_power_constants(beta)
    return (1.0 + beta) * c_beta * np.exp(-beta * z**2 / 2.0) * (z**2 - 1.0) + kappa_beta


def density_power_psi(y: ArrayLike, lam: ArrayLike, beta: float = 0.15) -> ArrayLike:
    """
    Density-power raw innovation psi_beta(y, lambda) = -d/dlambda S_beta.

    Derived as: psi_beta = 0.5 * c_beta * exp(-beta*lam/2) *
                            [(1+beta)^(-1/2)*beta - (1+1/beta)*exp(-beta*z^2/2)*(beta - beta*z^2 + beta)]
    Using the chain rule on S_beta with respect to lambda:
    psi_beta = G_beta(lambda)^{-1} * g_beta(z) when G_beta = 2*exp(beta*lam/2)
    So psi_beta = g_beta(z) / (2*exp(beta*lam/2))

    Parameters
    ----------
    y    : observation
    lam  : log-scale state
    beta : power exponent

    Returns
    -------
    psi_beta(y, lambda)
    """
    y = np.asarray(y, dtype=float)
    lam = np.asarray(lam, dtype=float)
    z = y / np.sqrt(np.exp(lam))
    g = density_power_driver(z, beta)
    return g / (2.0 * np.exp(beta * lam / 2.0))


def density_power_G(lam: ArrayLike, beta: float = 0.15) -> ArrayLike:
    """Scaling function G_beta(lambda) = 2 * exp(beta * lambda / 2)."""
    lam = np.asarray(lam, dtype=float)
    return 2.0 * np.exp(beta * lam / 2.0)


# ---------------------------------------------------------------------------
# Gaussian CRPS score
# ---------------------------------------------------------------------------

def crps_score(lam: ArrayLike, y: ArrayLike) -> ArrayLike:
    """
    Gaussian CRPS score:
        S_CRPS = exp(lambda/2) * [z*(2*Phi(z)-1) + 2*phi(z) - pi^(-1/2)]

    where z = y/sqrt(v), v = exp(lambda).

    Parameters
    ----------
    lam : log-scale state
    y   : observation

    Returns
    -------
    Score value S_CRPS(lambda, y)
    """
    lam = np.asarray(lam, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = np.exp(0.5 * lam)
    z = y / sigma
    return sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - np.pi**(-0.5))


def crps_driver(z: ArrayLike) -> ArrayLike:
    """
    Gaussian CRPS residual driver:
        g_CRPS(z) = pi^(-1/2) - sqrt(2/pi) * exp(-z^2/2)

    Note: This driver has bounded range (tail-saturating behavior).

    Parameters
    ----------
    z : standardised residual

    Returns
    -------
    Driver value g_CRPS(z)
    """
    z = np.asarray(z, dtype=float)
    return np.pi**(-0.5) - np.sqrt(2.0 / np.pi) * np.exp(-z**2 / 2.0)


def crps_psi(y: ArrayLike, lam: ArrayLike) -> ArrayLike:
    """
    Gaussian CRPS raw innovation psi_CRPS = g_CRPS(z) / G_CRPS(lambda).
    G_CRPS(lambda) = 2 * exp(-lambda/2)

    Parameters
    ----------
    y   : observation
    lam : log-scale state

    Returns
    -------
    psi_CRPS(y, lambda)
    """
    y = np.asarray(y, dtype=float)
    lam = np.asarray(lam, dtype=float)
    z = y / np.exp(0.5 * lam)
    g = crps_driver(z)
    G = 2.0 * np.exp(-0.5 * lam)
    return g / G


def crps_G(lam: ArrayLike) -> ArrayLike:
    """Scaling function G_CRPS(lambda) = 2 * exp(-lambda/2)."""
    lam = np.asarray(lam, dtype=float)
    return 2.0 * np.exp(-0.5 * lam)


# ---------------------------------------------------------------------------
# Gaussian MMD-RBF score (fixed bandwidth)
# ---------------------------------------------------------------------------

def mmd_score(lam: ArrayLike, y: ArrayLike, ell: float = 2.0) -> ArrayLike:
    """
    Fixed-bandwidth Gaussian MMD-RBF score:
        S_MMD = 1 - 2*ell/sqrt(ell^2+v) * exp(-y^2/(2*(ell^2+v)))
                + ell/sqrt(ell^2+2*v)

    where v = exp(lambda).

    Parameters
    ----------
    lam : log-scale state
    y   : observation
    ell : bandwidth (default 2.0)

    Returns
    -------
    Score value S_MMD(lambda, y)
    """
    lam = np.asarray(lam, dtype=float)
    y = np.asarray(y, dtype=float)
    v = np.exp(lam)
    term1 = 2.0 * ell / np.sqrt(ell**2 + v) * np.exp(-y**2 / (2.0 * (ell**2 + v)))
    term2 = ell / np.sqrt(ell**2 + 2.0 * v)
    return 1.0 - term1 + term2


def mmd_update(lam: ArrayLike, z: ArrayLike, ell: float = 2.0) -> ArrayLike:
    """
    Implemented state-dependent MMD-RBF update (driver):
        u_MMD(lambda, z) = 2*(r/sqrt(D))*exp(-z^2/(2D)) * [z^2/D^2 - 1/D]
                           + 2*r/(r^2+2)^(3/2)

    where r_lambda = ell * exp(-lambda/2), D_lambda = r_lambda^2 + 1.
    The G_MMD = 2 scaling is already incorporated in this form.

    Parameters
    ----------
    lam : log-scale state
    z   : standardised residual y / sqrt(v)
    ell : bandwidth (default 2.0)

    Returns
    -------
    MMD implemented update u_MMD(lambda, z)
    """
    lam = np.asarray(lam, dtype=float)
    z = np.asarray(z, dtype=float)
    r = ell * np.exp(-0.5 * lam)          # r_lambda
    D = r**2 + 1.0                         # D_lambda
    term1 = 2.0 * (r / np.sqrt(D)) * np.exp(-z**2 / (2.0 * D)) * (z**2 / D**2 - 1.0 / D)
    term2 = 2.0 * r / (r**2 + 2.0) ** 1.5
    return term1 + term2


def mmd_psi(y: ArrayLike, lam: ArrayLike, ell: float = 2.0) -> ArrayLike:
    """
    MMD-RBF raw innovation psi_MMD = u_MMD / G_MMD, G_MMD = 2.

    Parameters
    ----------
    y   : observation
    lam : log-scale state
    ell : bandwidth

    Returns
    -------
    psi_MMD(y, lambda)
    """
    y = np.asarray(y, dtype=float)
    lam = np.asarray(lam, dtype=float)
    z = y / np.exp(0.5 * lam)
    u = mmd_update(lam, z, ell)
    return u / 2.0


G_MMD = 2.0  # Scaling constant for MMD driver


# ---------------------------------------------------------------------------
# Criterion registry: unified interface
# ---------------------------------------------------------------------------

CRITERION_REGISTRY = {
    "gaussian_log": {
        "driver_fn": gaussian_log_driver,  # g(z) - residual only
        "psi_fn": gaussian_log_psi,
        "G_const": G_LOG,
        "G_fn": lambda lam, **kw: G_LOG * np.ones_like(np.asarray(lam, dtype=float)),
        "state_dependent": False,
        "score_fn": gaussian_log_score,
        "label": "G-L",
    },
    "student_log": {
        "driver_fn": lambda z, nu=6.0: student_log_driver(z, nu),
        "psi_fn": lambda y, lam, nu=6.0: student_log_psi(y, lam, nu),
        "G_const": G_STUDENT,
        "G_fn": lambda lam, **kw: G_STUDENT * np.ones_like(np.asarray(lam, dtype=float)),
        "state_dependent": False,
        "score_fn": None,  # Not needed for deterministic experiments
        "label": "T-L",
    },
    "density_power": {
        "driver_fn": lambda z, beta=0.15: density_power_driver(z, beta),
        "psi_fn": lambda y, lam, beta=0.15: density_power_psi(y, lam, beta),
        "G_const": None,  # State-dependent
        "G_fn": lambda lam, beta=0.15, **kw: density_power_G(lam, beta),
        "state_dependent": True,
        "score_fn": density_power_score,
        "label": "G-DP",
    },
    "crps": {
        "driver_fn": crps_driver,
        "psi_fn": crps_psi,
        "G_const": None,  # State-dependent
        "G_fn": lambda lam, **kw: crps_G(lam),
        "state_dependent": True,
        "score_fn": crps_score,
        "label": "G-C",
    },
    "mmd": {
        "driver_fn": None,  # State-dependent, use mmd_update directly
        "psi_fn": mmd_psi,
        "G_const": G_MMD,
        "G_fn": lambda lam, **kw: G_MMD * np.ones_like(np.asarray(lam, dtype=float)),
        "state_dependent": True,
        "score_fn": mmd_score,
        "label": "G-M",
    },
}


def get_driver_value(criterion: str, z: np.ndarray, lam: float = 0.0, **kwargs) -> np.ndarray:
    """
    Evaluate the residual driver for a given criterion over a grid of z values.

    Parameters
    ----------
    criterion : str
        One of 'gaussian_log', 'student_log', 'density_power', 'crps', 'mmd'.
    z : np.ndarray
        Array of standardised residual values.
    lam : float
        Log-scale state (relevant for state-dependent drivers like MMD).
    **kwargs
        Additional parameters (nu, beta, ell).

    Returns
    -------
    np.ndarray : driver values at each z.
    """
    if criterion == "mmd":
        ell = kwargs.get("ell", 2.0)
        return mmd_update(lam, z, ell)
    elif criterion == "gaussian_log":
        return gaussian_log_driver(z)
    elif criterion == "student_log":
        nu = kwargs.get("nu", 6.0)
        return student_log_driver(z, nu)
    elif criterion == "density_power":
        beta = kwargs.get("beta", 0.15)
        return density_power_driver(z, beta)
    elif criterion == "crps":
        return crps_driver(z)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


def get_psi_value(criterion: str, y: np.ndarray, lam: np.ndarray, **kwargs) -> np.ndarray:
    """
    Evaluate the raw innovation psi_S(y, lambda) for a given criterion.

    Parameters
    ----------
    criterion : str
    y : np.ndarray
    lam : np.ndarray
    **kwargs

    Returns
    -------
    np.ndarray : psi values.
    """
    if criterion == "gaussian_log":
        return gaussian_log_psi(y, lam)
    elif criterion == "student_log":
        nu = kwargs.get("nu", 6.0)
        return student_log_psi(y, lam, nu)
    elif criterion == "density_power":
        beta = kwargs.get("beta", 0.15)
        return density_power_psi(y, lam, beta)
    elif criterion == "crps":
        return crps_psi(y, lam)
    elif criterion == "mmd":
        ell = kwargs.get("ell", 2.0)
        return mmd_psi(y, lam, ell)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
