"""
Predictive diagnostics for scoring-rule-driven volatility filters.

Implements:
    - QLIKE (forecast-dependent) loss
    - VaR coverage rates
    - PIT computation and uniformity tests
    - Dynamic PIT adequacy via midrank normal scores and portmanteau statistics

# Custom — Context7 found no library equivalent for paper-specific diagnostics
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, kstest, uniform
from typing import Optional


# ---------------------------------------------------------------------------
# QLIKE loss
# ---------------------------------------------------------------------------

def qlike_fd(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Forecast-dependent QLIKE:
        QLIKE_fd(q, v) = log(v) + q / v

    where q is the observed squared return proxy and v is the predicted variance.

    Parameters
    ----------
    q : np.ndarray - observed proxy (e.g., squared return)
    v : np.ndarray - predicted variance

    Returns
    -------
    np.ndarray : elementwise QLIKE contributions
    """
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    return np.log(v) + q / v


def mean_qlike(q: np.ndarray, v: np.ndarray) -> float:
    """Mean QLIKE loss."""
    return float(np.mean(qlike_fd(q, v)))


# ---------------------------------------------------------------------------
# VaR coverage
# ---------------------------------------------------------------------------

def var_forecast(lam: np.ndarray, p: float, working_density: str = "gaussian",
                 nu: float = 6.0) -> np.ndarray:
    """
    Compute VaR_t(p) = q_Phi(p) * sqrt(exp(lambda_t)).

    Parameters
    ----------
    lam : np.ndarray - log-scale state path
    p : float - tail probability (e.g., 0.01, 0.05)
    working_density : str - 'gaussian' or 'student'
    nu : float - degrees of freedom for Student working density

    Returns
    -------
    np.ndarray : VaR values
    """
    sigma = np.sqrt(np.exp(lam))
    if working_density == "gaussian":
        q = norm.ppf(p)
    elif working_density == "student":
        from scipy.stats import t
        scale = np.sqrt((nu - 2.0) / nu)
        q = t.ppf(p, df=nu) * scale
    else:
        raise ValueError(f"Unknown working density: {working_density}")
    return q * sigma


def coverage_rate(y: np.ndarray, var: np.ndarray) -> float:
    """
    Compute empirical VaR coverage: fraction of y_t < VaR_t.

    Parameters
    ----------
    y : np.ndarray - returns
    var : np.ndarray - VaR values (negative, since VaR is the lower quantile)

    Returns
    -------
    float : empirical coverage rate
    """
    hits = (y < var).astype(float)
    return float(np.mean(hits))


# ---------------------------------------------------------------------------
# PIT computation
# ---------------------------------------------------------------------------

def compute_pit(y: np.ndarray, lam: np.ndarray, working_density: str = "gaussian",
                nu: float = 6.0) -> np.ndarray:
    """
    Compute probability integral transforms u_t = F_Phi(y_t / sqrt(v_t)).

    Parameters
    ----------
    y : np.ndarray - returns
    lam : np.ndarray - log-scale states (same length as y)
    working_density : str
    nu : float

    Returns
    -------
    np.ndarray : PIT values in (0, 1)
    """
    sigma = np.sqrt(np.exp(lam))
    z = y / sigma
    if working_density == "gaussian":
        u = norm.cdf(z)
    elif working_density == "student":
        from scipy.stats import t
        scale = np.sqrt((nu - 2.0) / nu)
        u = t.cdf(z / scale, df=nu)
    else:
        raise ValueError(f"Unknown working density: {working_density}")
    return np.clip(u, 1e-10, 1 - 1e-10)


def pit_uniformity_dmax(u: np.ndarray, n_grid: int = 100) -> float:
    """
    Compute D_max = max absolute deviation of empirical PIT CDF from uniform.

    Parameters
    ----------
    u : np.ndarray - PIT values in (0, 1)
    n_grid : int - number of grid points

    Returns
    -------
    float : D_max
    """
    grid = np.linspace(0.0, 1.0, n_grid)
    ecdf = np.array([np.mean(u <= g) for g in grid])
    return float(np.max(np.abs(ecdf - grid)))


# ---------------------------------------------------------------------------
# Dynamic PIT adequacy (portmanteau statistic)
# ---------------------------------------------------------------------------

def midrank_normal_scores(u: np.ndarray) -> np.ndarray:
    """
    Transform PIT values into midrank normal scores.
    Midrank: rank_i = (rank_in_sorted - 0.5) / n, then apply norm.ppf.

    Parameters
    ----------
    u : np.ndarray - PIT values

    Returns
    -------
    np.ndarray : normal scores
    """
    n = len(u)
    # Compute midranks
    ranks = np.argsort(np.argsort(u)) + 1  # ranks from 1 to n
    midranks = (ranks - 0.5) / n
    return norm.ppf(midranks)


def pit_portmanteau(u: np.ndarray, max_lag: int = 10) -> float:
    """
    Compute linear-and-quadratic portmanteau statistic for dynamic PIT adequacy.

    Uses midrank normal scores xi_t = norm.ppf((rank(u_t) - 0.5) / n).
    Statistic: sum_{j=1}^{max_lag} T * [r_j(xi)^2 + r_j(xi^2)^2] / n
    where r_j is the lag-j autocorrelation.

    Parameters
    ----------
    u : np.ndarray - PIT values
    max_lag : int - maximum lag (default 10)

    Returns
    -------
    float : portmanteau test statistic
    """
    xi = midrank_normal_scores(u)
    xi2 = xi**2
    n = len(xi)

    stat = 0.0
    xi_mean = np.mean(xi)
    xi2_mean = np.mean(xi2)
    xi_demeaned = xi - xi_mean
    xi2_demeaned = xi2 - xi2_mean

    xi_var = np.var(xi)
    xi2_var = np.var(xi2)

    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        # Lag-lag autocorrelation of xi
        r_lin = np.mean(xi_demeaned[lag:] * xi_demeaned[:-lag]) / (xi_var + 1e-15)
        # Lag-lag autocorrelation of xi^2
        r_quad = np.mean(xi2_demeaned[lag:] * xi2_demeaned[:-lag]) / (xi2_var + 1e-15)
        stat += n * (r_lin**2 + r_quad**2)

    return float(stat)


def pit_permutation_pvalue(
    u: np.ndarray, stat_obs: float, n_perm: int = 1999, seed: int = 42
) -> float:
    """
    Calibrate portmanteau statistic via time permutations.

    Parameters
    ----------
    u : np.ndarray - PIT values
    stat_obs : float - observed portmanteau statistic
    n_perm : int - number of permutations
    seed : int

    Returns
    -------
    float : permutation p-value
    """
    rng = np.random.default_rng(seed)
    count_exceed = 0
    for _ in range(n_perm):
        u_perm = rng.permutation(u)
        stat_perm = pit_portmanteau(u_perm)
        if stat_perm >= stat_obs:
            count_exceed += 1
    return (count_exceed + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Holm multiple testing correction
# ---------------------------------------------------------------------------

def holm_correction(pvalues: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Holm-Bonferroni correction to an array of p-values.

    Parameters
    ----------
    pvalues : np.ndarray
    alpha : float - family-wise error rate

    Returns
    -------
    reject : np.ndarray of bool
    adjusted_pvalues : np.ndarray
    """
    from statsmodels.stats.multitest import multipletests
    reject, pvals_adj, _, _ = multipletests(pvalues, alpha=alpha, method="holm")
    return pvals_adj, reject
