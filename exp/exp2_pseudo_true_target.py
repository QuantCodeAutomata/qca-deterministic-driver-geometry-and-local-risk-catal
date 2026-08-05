"""
Experiment 2: Pseudo-True Target Shift Under Variance Contamination

Demonstrates that changing the scoring criterion changes the pseudo-true
variance target under contaminated Gaussian mixture true law.

True law: Q = (1-epsilon)*N(0,v0) + epsilon*N(0,tau*v0)
epsilon=0.05, v0=1, tau grid from 1 to 100.

# Custom — Context7 found no library equivalent for pseudo-true target computation
"""
import sys, os
sys.path.insert(0, '/workspace/project')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermgauss
from scipy.stats import norm

from src.score_drivers import (
    density_power_score, crps_score, mmd_score,
)

RESULTS_DIR = '/workspace/project/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
EPSILON = 0.05
V0 = 1.0
BETA = 0.15
ELL = 2.0
N_QUAD = 96
N_LAM_SCAN = 6000
TAU_GRID = np.logspace(0, 2, 200)  # tau from 1 to 100

# Pre-compute GH nodes/weights once (96-node)
_GH_NODES, _GH_WEIGHTS = hermgauss(N_QUAD)          # nodes t, weights w: E[f]=1/√π Σ w f(√2 t)
_GH_Y_SCALE = np.sqrt(2.0)                           # y = sigma * sqrt(2) * t


# ---------------------------------------------------------------------------
# Vectorised expected-score computation
# ---------------------------------------------------------------------------

def mixture_risk_grid(score_fn_vec, lam_grid: np.ndarray, v0: float, tau: float,
                      epsilon: float = EPSILON) -> np.ndarray:
    """
    Compute E_Q[S(lambda, y)] for every lambda in lam_grid simultaneously.

    score_fn_vec(lam_col, y_row) must broadcast over shape (N_LAM, N_QUAD).
    Returns array of shape (N_LAM,).

    Uses 96-node Gauss-Hermite; E_N(0,sig2)[f] = 1/√π Σ_j w_j f(sig*√2*t_j).
    """
    # lam_col shape (N_LAM,1), y-points are transformed per component
    lam_col = lam_grid[:, None]                       # (N_LAM, 1) for broadcasting

    risk = np.zeros(len(lam_grid))
    for sig2, mix_w in [(v0, 1.0 - epsilon), (tau * v0, epsilon)]:
        sig = np.sqrt(sig2)
        y_pts = sig * _GH_Y_SCALE * _GH_NODES        # (N_QUAD,)
        # score evaluated at all (lam, y) pairs: shape (N_LAM, N_QUAD)
        scores = score_fn_vec(lam_col, y_pts[None, :])
        # GH integral: E = 1/√π Σ w_j f(√2 t_j)
        component_risk = (scores @ _GH_WEIGHTS) / np.sqrt(np.pi)
        risk += mix_w * component_risk
    return risk


def find_pseudo_true_variance(criterion: str, tau: float, v0: float = V0,
                               epsilon: float = EPSILON) -> float:
    """
    Find pseudo-true variance v_S* = exp(lambda_S*) by 6000-point dense scan.

    The entire lambda grid is evaluated in one vectorised call per mixture
    component, then minimised.  Returns v*/v0 ratio.
    """
    lam_grid = np.linspace(-3.0, 6.0, N_LAM_SCAN)

    if criterion == 'density_power':
        score_fn_vec = lambda lam_, y: density_power_score(lam_, y, BETA)
    elif criterion == 'crps':
        score_fn_vec = crps_score
    elif criterion == 'mmd':
        score_fn_vec = lambda lam_, y: mmd_score(lam_, y, ELL)
    else:
        raise ValueError(criterion)

    risk_vals = mixture_risk_grid(score_fn_vec, lam_grid, v0, tau, epsilon)

    idx_min = np.argmin(risk_vals)
    v_star = np.exp(lam_grid[idx_min])
    return v_star / v0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Experiment 2: Pseudo-True Target Shift Under Variance Contamination")
    print(f"  epsilon={EPSILON}, v0={V0}, tau grid: {TAU_GRID[0]:.1f} to {TAU_GRID[-1]:.1f}")

    # Exact KL target for Gaussian log score
    v_star_log = 1.0 + EPSILON * (TAU_GRID - 1.0)

    # Robust criteria
    results = {'tau': TAU_GRID, 'v_star_log': v_star_log}

    for criterion in ['density_power', 'crps', 'mmd']:
        print(f"  Computing pseudo-true targets for {criterion}...")
        v_stars = np.zeros(len(TAU_GRID))
        for i, tau in enumerate(TAU_GRID):
            try:
                v_stars[i] = find_pseudo_true_variance(criterion, tau)
            except Exception as e:
                print(f"    Error at tau={tau:.2f}: {e}")
                v_stars[i] = float('nan')
            if (i + 1) % 20 == 0:
                print(f"    Progress: {i+1}/{len(TAU_GRID)}, tau={tau:.2f}, v*/v0={v_stars[i]:.4f}")
        results[f'v_star_{criterion}'] = v_stars

    # Validation checks
    assert abs(results['v_star_log'][0] - 1.0) < 1e-10, "Log target at tau=1 should be 1.0"
    for crit in ['density_power', 'crps', 'mmd']:
        assert abs(results[f'v_star_{crit}'][0] - 1.0) < 0.02, f"{crit} target at tau=1 should be ~1.0"
    print("  Validation: v*/v0 at tau=1 ≈ 1.0 for all criteria ✓")

    # Check saturation vs linear growth
    v_log_large = results['v_star_log'][-1]
    for crit in ['density_power', 'crps', 'mmd']:
        v_crit_large = results[f'v_star_{crit}'][-1]
        if v_crit_large < v_log_large * 0.5:
            print(f"  {crit}: v*/v0={v_crit_large:.3f} at tau={TAU_GRID[-1]:.0f} (log={v_log_large:.3f}) - saturation confirmed ✓")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(TAU_GRID, v_star_log, 'b-', linewidth=2, label='Gaussian Log (KL, exact)')
    colors = {'density_power': 'red', 'crps': 'purple', 'mmd': 'orange'}
    labels = {'density_power': 'Density-Power (β=0.15)', 'crps': 'Gaussian CRPS', 'mmd': 'Gaussian MMD-RBF'}
    for crit in ['density_power', 'crps', 'mmd']:
        ax.plot(TAU_GRID, results[f'v_star_{crit}'], color=colors[crit],
                linewidth=1.5, label=labels[crit])
    ax.set_xscale('log')
    ax.set_xlabel('Contamination variance ratio τ')
    ax.set_ylabel('Pseudo-true ratio v*/v₀')
    ax.set_title('Pseudo-True Target Shift Under Variance Contamination\n(ε=0.05, Gaussian mixture true law)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{RESULTS_DIR}/exp2_pseudo_true.png', dpi=150)
    plt.close()
    print(f"  Plot saved to {RESULTS_DIR}/exp2_pseudo_true.png")

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(f'{RESULTS_DIR}/exp2_results.csv', index=False)
    print(f"  Results saved to {RESULTS_DIR}/exp2_results.csv")

    return results


if __name__ == '__main__':
    results = main()
    print('Experiment 2 completed successfully')
