"""
Minimum-scoring-risk batch estimator for scoring-rule-driven volatility filters.

Estimates vartheta = (omega, phi, alpha) by minimising the in-sample average
scoring risk, using the same rule that drives the recursion.

# Custom — Context7 found no library equivalent for this paper-specific estimator
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from typing import Optional, Callable

from src.score_drivers import (
    gaussian_log_score, density_power_score, crps_score, mmd_score,
    get_psi_value,
)
from src.recursion import run_recursion


# ---------------------------------------------------------------------------
# Training objective
# ---------------------------------------------------------------------------

def _build_score_function(criterion: str, **kwargs) -> Callable:
    """Return the appropriate score function for criterion."""
    if criterion == "gaussian_log":
        return lambda lam, y: gaussian_log_score(lam, y)
    elif criterion == "density_power":
        beta = kwargs.get("beta", 0.15)
        return lambda lam, y: density_power_score(lam, y, beta)
    elif criterion == "crps":
        return lambda lam, y: crps_score(lam, y)
    elif criterion == "mmd":
        ell = kwargs.get("ell", 2.0)
        return lambda lam, y: mmd_score(lam, y, ell)
    elif criterion == "student_log":
        nu = kwargs.get("nu", 6.0)
        def student_log_score_fn(lam, y):
            v = np.exp(lam)
            z = y / np.sqrt(v)
            return 0.5 * lam + 0.5 * (nu + 1.0) * np.log(1.0 + z**2 / (nu - 2.0))
        return student_log_score_fn
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


def average_scoring_risk(
    params: np.ndarray,
    y: np.ndarray,
    criterion: str,
    **kwargs,
) -> float:
    """
    Compute T_tr^{-1} sum_{t=1}^{T_tr} S(P_{lambda_t}, y_t).

    Parameters
    ----------
    params : np.ndarray, shape (3,) = [omega, phi, alpha]
    y : np.ndarray, shape (T_tr,)
    criterion : str
    **kwargs

    Returns
    -------
    float : average scoring risk
    """
    omega, phi, alpha = params

    # Stationarity check
    if abs(phi) >= 0.999:
        return 1e10

    lam0 = omega / (1.0 - phi)
    lam = np.empty(len(y) + 1)
    lam[0] = lam0
    score_fn = _build_score_function(criterion, **kwargs)

    total_score = 0.0
    for t in range(len(y)):
        # Compute update
        psi_vals = get_psi_value(criterion, np.array([y[t]]), np.array([lam[t]]), **kwargs)
        psi = psi_vals[0]

        # Apply G_S scaling
        if criterion == "gaussian_log":
            u = 2.0 * psi
        elif criterion == "student_log":
            u = 2.0 * psi
        elif criterion == "density_power":
            from src.score_drivers import density_power_G
            beta = kwargs.get("beta", 0.15)
            G = density_power_G(lam[t], beta)
            u = G * psi
        elif criterion == "crps":
            from src.score_drivers import crps_G
            G = crps_G(lam[t])
            u = G * psi
        elif criterion == "mmd":
            u = 2.0 * psi
        else:
            u = psi

        lam[t + 1] = omega + phi * lam[t] + alpha * u
        score_val = float(score_fn(lam[t], y[t]))
        if not np.isfinite(score_val):
            return 1e10
        total_score += score_val

    return total_score / len(y)


# ---------------------------------------------------------------------------
# Admissibility gates
# ---------------------------------------------------------------------------

def check_gates(
    params: np.ndarray,
    y: np.ndarray,
    criterion: str,
    obj_val: float,
    grad_norm: float,
    hess_min_eig: float,
    hess_cond: float,
    lam_path: Optional[np.ndarray] = None,
    **kwargs,
) -> dict:
    """
    Apply all admissibility gates and return a dict of pass/fail for each gate.

    Gates:
    - gradient_gate: grad_norm <= 1e-3
    - hessian_gate: hess_min_eig > 1e-8
    - conditioning_gate: hess_cond <= 1e10
    - convergence_gate: obj_val is finite
    - parameter_gate: |phi| < 0.999
    - path_finiteness_gate: all state values finite

    Parameters
    ----------
    params : np.ndarray [omega, phi, alpha]
    y : observation series
    criterion : str
    obj_val : optimisation objective value
    grad_norm : gradient norm at solution
    hess_min_eig : minimum eigenvalue of Hessian
    hess_cond : condition number of Hessian
    lam_path : optional, precomputed state path

    Returns
    -------
    dict : {gate_name: bool} and 'all_pass': bool
    """
    omega, phi, alpha = params
    gates = {}

    gates["gradient_gate"] = grad_norm <= 1e-3
    gates["hessian_gate"] = hess_min_eig > 1e-8
    gates["conditioning_gate"] = hess_cond <= 1e10
    gates["convergence_gate"] = np.isfinite(obj_val) and obj_val < 1e9
    gates["parameter_gate"] = abs(phi) < 0.999
    gates["path_finiteness_gate"] = (
        np.all(np.isfinite(lam_path)) if lam_path is not None else True
    )

    gates["all_pass"] = all(gates.values())
    return gates


# ---------------------------------------------------------------------------
# Multi-start estimator
# ---------------------------------------------------------------------------

DEFAULT_STARTS = [
    [0.0, 0.95, 0.05],
    [0.0, 0.90, 0.10],
    [-0.5, 0.97, 0.03],
    [0.5, 0.98, 0.02],
    [0.0, 0.85, 0.15],
    [0.0, 0.80, 0.20],
    [-1.0, 0.95, 0.05],
    [1.0, 0.95, 0.05],
]


def estimate_parameters(
    y: np.ndarray,
    criterion: str,
    n_starts: int = 8,
    bounds: Optional[list] = None,
    **kwargs,
) -> dict:
    """
    Estimate recursion parameters (omega, phi, alpha) by minimum-scoring-risk.

    Uses 8 predeclared multistarts and retains the lowest finite objective
    among fits passing all gates.

    Parameters
    ----------
    y : np.ndarray
        Training observations.
    criterion : str
    n_starts : int
        Number of multistarts.
    bounds : list of (min, max) tuples for [omega, phi, alpha]
    **kwargs

    Returns
    -------
    dict with keys: params, obj, gates, grad_norm, hess_min_eig, hess_cond, success
    """
    if bounds is None:
        bounds = [(-5.0, 5.0), (-0.999, 0.999), (-1.0, 1.0)]

    best_obj = np.inf
    best_result = None

    starts = DEFAULT_STARTS[:n_starts]

    for x0 in starts:
        try:
            res = minimize(
                average_scoring_risk,
                x0=x0,
                args=(y, criterion),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-6},
                kwargs=kwargs if kwargs else {},
            )
        except Exception:
            continue

        if not res.success and not np.isfinite(res.fun):
            continue

        if res.fun < best_obj:
            best_obj = res.fun
            best_result = res

    if best_result is None:
        return {
            "params": None, "obj": np.inf, "success": False,
            "gates": {"all_pass": False},
        }

    params = best_result.x
    obj = best_result.fun

    # Compute gradient norm
    grad_norm = float(np.linalg.norm(best_result.jac)) if hasattr(best_result, "jac") and best_result.jac is not None else np.inf

    # Numerical Hessian via finite differences
    h_fd = 1e-4
    n_params = len(params)
    H = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            pp = params.copy(); pp[i] += h_fd; pp[j] += h_fd
            pm = params.copy(); pp[i] += h_fd; pp[j] -= h_fd
            mp = params.copy(); mp[i] -= h_fd; mp[j] += h_fd
            mm = params.copy(); mm[i] -= h_fd; mm[j] -= h_fd
            H[i, j] = (
                average_scoring_risk(pp, y, criterion, **kwargs)
                - average_scoring_risk(pm, y, criterion, **kwargs)
                - average_scoring_risk(mp, y, criterion, **kwargs)
                + average_scoring_risk(mm, y, criterion, **kwargs)
            ) / (4.0 * h_fd**2)

    eigvals = np.linalg.eigvalsh(H)
    hess_min_eig = float(np.min(eigvals))
    hess_cond = float(np.linalg.cond(H)) if np.all(eigvals > 0) else np.inf

    # Compute state path for gate checks
    try:
        lam_path = run_recursion(y, *params, criterion, **kwargs)
        path_ok = np.all(np.isfinite(lam_path))
    except Exception:
        lam_path = None
        path_ok = False

    gates = check_gates(
        params, y, criterion, obj, grad_norm, hess_min_eig, hess_cond, lam_path, **kwargs
    )
    gates["path_finiteness_gate"] = path_ok

    return {
        "params": params,
        "obj": obj,
        "grad_norm": grad_norm,
        "hess_min_eig": hess_min_eig,
        "hess_cond": hess_cond,
        "gates": gates,
        "success": gates["all_pass"],
        "lam_path": lam_path,
    }
