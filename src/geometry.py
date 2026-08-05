"""
Local geometry of scoring-rule-driven volatility filters.

Computes:
    J_S = R''_S(lambda*) = local risk curvature at pseudo-true target
    K_S = E[psi_S(y, lambda*)^2] = innovation second moment
    Xi_S = E[u_S(y, lambda*)^2] = implemented update second moment
    C_S = Cov(Z, u_S) = covariance of residual with implemented update

under a declared reference residual law (Gaussian or variance-standardised Student-t).

# Custom — Context7 found no library equivalent for paper-specific J_S, K_S formulas
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional

from src.quadrature import expectation_normal, expectation_student_standardised
from src.score_drivers import (
    gaussian_log_driver, student_log_driver,
    density_power_driver, density_power_constants,
    crps_driver, mmd_update,
    gaussian_log_psi, student_log_psi,
    density_power_psi, crps_psi, mmd_psi,
    G_LOG, G_STUDENT, G_MMD,
)


# ---------------------------------------------------------------------------
# Gaussian-reference J_S calculations (analytic / numerical)
# ---------------------------------------------------------------------------

def J_gaussian_log(n_quad: int = 64) -> float:
    """
    J_S for Gaussian log score under Gaussian reference.
    Analytic result: J_log = 0.5.
    Verified numerically for consistency.

    Returns
    -------
    float : J_log = 0.5
    """
    # Analytic: J_log = E[(z^2-1)^2 / 4] * ... actually derived differently
    # J_S = -d^2/dlambda^2 R_S(lambda) at lambda*=0 under Z~N(0,1)
    # R_log(lambda) = E[0.5*lambda + 0.5*exp(-lambda)*y^2]
    # dR/dlambda = 0.5 - 0.5*exp(-lambda)*E[y^2]
    # d^2R/dlambda^2 = 0.5*exp(-lambda)*E[y^2] at lambda*=0, E[y^2]=1 -> J=0.5
    # Numerical verification:
    h = 1e-5
    lam_star = 0.0

    def R(lam):
        # E[S_log(lambda, y)] under y = sqrt(exp(lam*))*z, z~N(0,1)
        # At lam*=0: y=z
        return expectation_normal(lambda z: 0.5 * lam + 0.5 * np.exp(-lam) * z**2, n=n_quad)

    J_num = (R(lam_star + h) - 2 * R(lam_star) + R(lam_star - h)) / h**2
    return float(J_num)


def K_gaussian_log(n_quad: int = 64) -> float:
    """
    K_S for Gaussian log score under Gaussian reference.
    K_log = E[psi_log(z,0)^2] where psi_log = 0.5*(z^2-1).
    Analytic: K_log = 0.5 * Var(z^2)/4 ... = E[(z^2-1)^2]/4 = 2/4 = 0.5

    Returns
    -------
    float : K_log = 0.5
    """
    def f(z):
        psi = 0.5 * (z**2 - 1.0)
        return psi**2
    return expectation_normal(f, n=n_quad)


def J_density_power(beta: float = 0.15, n_quad: int = 64) -> float:
    """
    J_S for density-power score under Gaussian reference at lambda*=0.
    Computed by second numerical derivative of E_Z[S_beta(lambda, sqrt(exp(lam*))*z)].

    Parameters
    ----------
    beta : power exponent
    n_quad : number of quadrature nodes

    Returns
    -------
    float : J_beta
    """
    h = 1e-5
    lam_star = 0.0

    def R(lam):
        from src.score_drivers import density_power_score
        return expectation_normal(
            lambda z: density_power_score(lam, np.exp(0.5 * lam_star) * z, beta), n=n_quad
        )

    J_num = (R(lam_star + h) - 2 * R(lam_star) + R(lam_star - h)) / h**2
    return float(J_num)


def K_density_power(beta: float = 0.15, n_quad: int = 64) -> float:
    """
    K_S for density-power score under Gaussian reference at lambda*=0.

    Parameters
    ----------
    beta : power exponent
    n_quad : quadrature nodes

    Returns
    -------
    float : K_beta = E[psi_beta(y, 0)^2] under Z~N(0,1)
    """
    def f(z):
        psi = density_power_psi(z, 0.0, beta)  # y=z at lam*=0, exp(0)=1
        return psi**2
    return expectation_normal(f, n=n_quad)


def J_crps(n_quad: int = 64) -> float:
    """
    J_S for Gaussian CRPS under Gaussian reference at lambda*=0.

    Returns
    -------
    float : J_CRPS
    """
    h = 1e-5
    lam_star = 0.0

    def R(lam):
        from src.score_drivers import crps_score
        return expectation_normal(
            lambda z: crps_score(lam, np.exp(0.5 * lam_star) * z), n=n_quad
        )

    J_num = (R(lam_star + h) - 2 * R(lam_star) + R(lam_star - h)) / h**2
    return float(J_num)


def K_crps(n_quad: int = 64) -> float:
    """
    K_S for Gaussian CRPS under Gaussian reference at lambda*=0.

    Returns
    -------
    float : K_CRPS = E[psi_CRPS(z, 0)^2] under Z~N(0,1)
    """
    def f(z):
        psi = crps_psi(z, 0.0)
        return psi**2
    return expectation_normal(f, n=n_quad)


def J_mmd(ell: float = 2.0, n_quad: int = 64) -> float:
    """
    J_S for Gaussian MMD-RBF under Gaussian reference at lambda*=0.

    Parameters
    ----------
    ell : bandwidth
    n_quad : quadrature nodes

    Returns
    -------
    float : J_MMD
    """
    h = 1e-5
    lam_star = 0.0

    def R(lam):
        from src.score_drivers import mmd_score
        return expectation_normal(
            lambda z: mmd_score(lam, np.exp(0.5 * lam_star) * z, ell), n=n_quad
        )

    J_num = (R(lam_star + h) - 2 * R(lam_star) + R(lam_star - h)) / h**2
    return float(J_num)


def K_mmd(ell: float = 2.0, n_quad: int = 64) -> float:
    """
    K_S for Gaussian MMD-RBF under Gaussian reference at lambda*=0.

    Parameters
    ----------
    ell : bandwidth
    n_quad : quadrature nodes

    Returns
    -------
    float : K_MMD = E[psi_MMD(z, 0)^2] under Z~N(0,1)
    """
    def f(z):
        psi = mmd_psi(z, 0.0, ell)
        return psi**2
    return expectation_normal(f, n=n_quad)


# ---------------------------------------------------------------------------
# Student-reference J_S, K_S
# ---------------------------------------------------------------------------

def J_student_log(nu: float = 6.0, n_quad: int = 64) -> float:
    """
    J_S for variance-standardised Student-t_nu log score under Student reference.

    Parameters
    ----------
    nu : degrees of freedom
    n_quad : quadrature nodes

    Returns
    -------
    float : J_student_log
    """
    h = 1e-5
    lam_star = 0.0

    # Under variance-standardised t_nu: Y = Z * sqrt(exp(lam*)) where Z has var 1
    # At lam*=0: Y = Z where Z has unit variance under standardised t_nu
    def R(lam):
        from src.score_drivers import student_log_psi
        # S_student_log = 0.5*lam + 0.5*(nu+1)*log(1+z^2/(nu-2)) where z=y/sqrt(v)
        # Y = Z * sqrt(exp(lam*)) = Z at lam*=0
        def f(z):
            y = np.exp(0.5 * lam_star) * z
            v = np.exp(lam)
            z_at_lam = y / np.sqrt(v)
            return 0.5 * lam + 0.5 * (nu + 1.0) * np.log(1.0 + z_at_lam**2 / (nu - 2.0))
        return expectation_student_standardised(f, nu, n_quad)

    J_num = (R(lam_star + h) - 2 * R(lam_star) + R(lam_star - h)) / h**2
    return float(J_num)


def K_student_log(nu: float = 6.0, n_quad: int = 64) -> float:
    """
    K_S for variance-standardised Student-t_nu log score under Student reference.

    Parameters
    ----------
    nu : degrees of freedom
    n_quad : quadrature nodes

    Returns
    -------
    float : K_student_log
    """
    def f(z):
        psi = 0.5 * (((nu + 1.0) * z**2) / ((nu - 2.0) + z**2) - 1.0)
        return psi**2
    return expectation_student_standardised(f, nu, n_quad)


# ---------------------------------------------------------------------------
# Unified catalogue function
# ---------------------------------------------------------------------------

CATALOGUE_SPECS = {
    "gaussian_log": {
        "J_fn": J_gaussian_log,
        "K_fn": K_gaussian_log,
        "reference": "gaussian",
        "label": "G-L",
    },
    "density_power": {
        "J_fn": lambda n: J_density_power(beta=0.15, n_quad=n),
        "K_fn": lambda n: K_density_power(beta=0.15, n_quad=n),
        "reference": "gaussian",
        "label": "G-DP",
    },
    "crps": {
        "J_fn": J_crps,
        "K_fn": K_crps,
        "reference": "gaussian",
        "label": "G-C",
    },
    "mmd": {
        "J_fn": lambda n: J_mmd(ell=2.0, n_quad=n),
        "K_fn": lambda n: K_mmd(ell=2.0, n_quad=n),
        "reference": "gaussian",
        "label": "G-M",
    },
    "student_log": {
        "J_fn": lambda n: J_student_log(nu=6.0, n_quad=n),
        "K_fn": lambda n: K_student_log(nu=6.0, n_quad=n),
        "reference": "student_t6",
        "label": "T-L",
    },
}


def compute_geometry_catalogue(n_quad: int = 64) -> dict:
    """
    Compute the full J_S, K_S catalogue for all criteria.

    Parameters
    ----------
    n_quad : int
        Number of quadrature nodes for numerical integration.

    Returns
    -------
    dict mapping criterion name to {'J': ..., 'K': ..., 'ratio': sqrt(K)/|J|}
    """
    results = {}
    for name, spec in CATALOGUE_SPECS.items():
        J = spec["J_fn"](n_quad)
        K = spec["K_fn"](n_quad)
        ratio = np.sqrt(K) / abs(J) if abs(J) > 1e-15 else float("nan")
        results[name] = {
            "J": J,
            "K": K,
            "ratio": ratio,
            "label": spec["label"],
            "reference": spec["reference"],
        }
    return results
