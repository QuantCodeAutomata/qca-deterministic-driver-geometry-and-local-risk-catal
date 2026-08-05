"""
Experiment 1: Deterministic Driver Geometry and Local Risk Catalogue

Reproduces the score-driver shape plots, tail envelope displays, and
J_S/K_S numerical catalogue for Gaussian and Student-t working densities.

# Custom — Context7 found no library equivalent for paper-specific geometry
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
    gaussian_log_driver, student_log_driver, density_power_driver,
    crps_driver, mmd_update, mmd_psi, crps_psi, density_power_psi, gaussian_log_psi,
    G_LOG, G_MMD,
)
from src.quadrature import expectation_normal
from src.geometry import (
    J_gaussian_log, K_gaussian_log,
    J_density_power, K_density_power,
    J_crps, K_crps, J_mmd, K_mmd,
    J_student_log, K_student_log,
)

RESULTS_DIR = '/workspace/project/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
BETA = 0.15
NU = 6.0
ELL = 2.0
N_Z = 401
N_R = 401

z_grid = np.linspace(-5.0, 5.0, N_Z)
r_grid = np.linspace(0.0, 10.0, N_R)

CRITERIA = {
    'gaussian_log': ('Gaussian Log', 'blue'),
    'student_log':  ('Student-t₆ Log', 'green'),
    'density_power': ('Density-Power (β=0.15)', 'red'),
    'crps': ('Gaussian CRPS', 'purple'),
    'mmd': ('Gaussian MMD-RBF (ℓ=2)', 'orange'),
}


# ---------------------------------------------------------------------------
# Driver functions at lam=0
# ---------------------------------------------------------------------------
def get_driver(criterion, z):
    if criterion == 'gaussian_log':
        return gaussian_log_driver(z)
    elif criterion == 'student_log':
        return student_log_driver(z, NU)
    elif criterion == 'density_power':
        return density_power_driver(z, BETA)
    elif criterion == 'crps':
        return crps_driver(z)
    elif criterion == 'mmd':
        return mmd_update(0.0, z, ELL)
    raise ValueError(criterion)


# ---------------------------------------------------------------------------
# Tail envelope T_S(r) = sup_{|z|>=r} |g_S(z)|
# ---------------------------------------------------------------------------
def compute_tail_envelope(criterion, r_grid):
    envelope = np.zeros_like(r_grid)
    # Fine z grid for accurate sup computation
    z_fine = np.linspace(-15.0, 15.0, 5001)
    g_fine = np.abs(get_driver(criterion, z_fine))
    for i, r in enumerate(r_grid):
        mask = np.abs(z_fine) >= r
        envelope[i] = np.max(g_fine[mask]) if np.any(mask) else 0.0
    return envelope


# ---------------------------------------------------------------------------
# J_S, K_S computation
# ---------------------------------------------------------------------------
def compute_JK(criterion, n_quad):
    """Return (J, K) for a criterion using n_quad Gauss-Hermite nodes."""
    if criterion == 'gaussian_log':
        J = J_gaussian_log(n_quad)
        K = K_gaussian_log(n_quad)
    elif criterion == 'density_power':
        J = J_density_power(BETA, n_quad)
        K = K_density_power(BETA, n_quad)
    elif criterion == 'crps':
        J = J_crps(n_quad)
        K = K_crps(n_quad)
    elif criterion == 'mmd':
        J = J_mmd(ELL, n_quad)
        K = K_mmd(ELL, n_quad)
    elif criterion == 'student_log':
        J = J_student_log(NU, n_quad)
        K = K_student_log(NU, n_quad)
    return J, K


# ---------------------------------------------------------------------------
# Numerical derivative check
# ---------------------------------------------------------------------------
def numerical_derivative_check():
    """Compare closed-form psi_log vs finite-difference derivative of S_log."""
    from src.score_drivers import gaussian_log_score
    z_test, lam_test = 1.5, 0.3
    y_test = z_test * np.exp(0.5 * lam_test)
    h = 1e-6
    psi_fd = -(gaussian_log_score(lam_test + h, y_test) - gaussian_log_score(lam_test - h, y_test)) / (2 * h)
    psi_cf = gaussian_log_psi(y_test, lam_test)
    return float(psi_cf), float(psi_fd), abs(float(psi_cf) - float(psi_fd))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 1. Driver shape plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for crit, (label, color) in CRITERIA.items():
        g_vals = get_driver(crit, z_grid)
        ax.plot(z_grid, g_vals, label=label, color=color)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Standardised residual z')
    ax.set_ylabel('Driver g_S(z)')
    ax.set_title('Score-Driven Update Drivers')
    ax.legend(fontsize=8)
    ax.set_ylim(-5, 30)
    ax.grid(True, alpha=0.3)

    # 2. Tail envelope
    ax2 = axes[1]
    for crit, (label, color) in CRITERIA.items():
        env = compute_tail_envelope(crit, r_grid)
        ax2.plot(r_grid, env, label=label, color=color)
    ax2.set_xlabel('Threshold r')
    ax2.set_ylabel('T_S(r) = sup_{|z|≥r} |g_S(z)|')
    ax2.set_title('Tail Envelope of Drivers')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'{RESULTS_DIR}/exp1_drivers.png', dpi=150)
    plt.close()

    # Separate tail envelope plot
    fig2, ax3 = plt.subplots(figsize=(8, 5))
    for crit, (label, color) in CRITERIA.items():
        env = compute_tail_envelope(crit, r_grid)
        ax3.plot(r_grid, env, label=label, color=color)
    ax3.set_xlabel('Threshold r')
    ax3.set_ylabel('T_S(r)')
    ax3.set_title('Tail Envelope: Gaussian Log Diverges, Others Saturate')
    ax3.legend(fontsize=8)
    ax3.set_ylim(0, 80)
    ax3.grid(True, alpha=0.3)
    fig2.savefig(f'{RESULTS_DIR}/exp1_tail_envelope.png', dpi=150)
    plt.close()

    print("Driver and tail envelope plots saved.")

    # 3. Geometry catalogue
    rows = []
    QUAD_NODES = [64, 128]
    catalogue_criteria = ['gaussian_log', 'density_power', 'crps', 'mmd', 'student_log']

    for crit in catalogue_criteria:
        label = CRITERIA[crit][0]
        row = {'criterion': crit, 'label': label}
        J_vals, K_vals = {}, {}
        for n in QUAD_NODES:
            J, K = compute_JK(crit, n)
            J_vals[n] = J
            K_vals[n] = K
            ratio = np.sqrt(K) / abs(J) if abs(J) > 1e-15 else float('nan')
            row[f'J_{n}'] = J
            row[f'K_{n}'] = K
            row[f'ratio_{n}'] = ratio
            print(f"  {crit:20s} n={n:3d}: J={J:.8f}, K={K:.8f}, √K/|J|={ratio:.4f}")

        # Quadrature stability check
        delta_J = abs(J_vals[64] - J_vals[128])
        delta_K = abs(K_vals[64] - K_vals[128])
        row['delta_J_64_128'] = delta_J
        row['delta_K_64_128'] = delta_K
        # J uses finite-differences (h=1e-5) so node-count stability is ~1e-5;
        # K is a direct quadrature integral so 1e-8 is achievable.
        stability_ok = (delta_J < 1e-4) and (delta_K < 1e-8)
        row['stability_ok'] = stability_ok
        if not stability_ok:
            print(f"  WARNING: {crit} stability check failed: ΔJ={delta_J:.2e}, ΔK={delta_K:.2e}")
        else:
            print(f"  {crit}: quadrature agrees to ΔJ={delta_J:.2e}, ΔK={delta_K:.2e} ✓")
        rows.append(row)

    # 4. Numerical derivative check
    psi_cf, psi_fd, err = numerical_derivative_check()
    print(f"\nNumerical derivative check (z=1.5, lam=0.3):")
    print(f"  Closed-form psi_log = {psi_cf:.10f}")
    print(f"  Finite-difference  = {psi_fd:.10f}")
    print(f"  Absolute error     = {err:.2e}")
    assert err < 1e-6, f"Derivative check failed: error {err:.2e}"

    # 5. Validate expected values
    df = pd.DataFrame(rows)
    print("\n=== Geometry Catalogue ===")
    print(df[['label', 'J_64', 'K_64', 'ratio_64']].to_string(index=False, float_format=lambda x: f'{x:.8f}'))

    # Check approximate expected values
    expected = {
        'gaussian_log': (0.5, 0.5),
        'density_power': (0.3572056243, 0.268),
        'crps': (0.07052369794, 0.012),
        'mmd': (0.034, 0.002),
    }
    for crit, (J_exp, K_exp) in expected.items():
        row_d = df[df['criterion'] == crit].iloc[0]
        J_got = row_d['J_64']
        K_got = row_d['K_64']
        J_ok = abs(J_got - J_exp) < 0.01
        K_ok = abs(K_got - K_exp) < 0.02
        print(f"  {crit}: J={J_got:.6f} (expected≈{J_exp}) {'✓' if J_ok else 'WARN'} | K={K_got:.6f} (expected≈{K_exp}) {'✓' if K_ok else 'WARN'}")

    df.to_csv(f'{RESULTS_DIR}/exp1_catalogue.csv', index=False)
    print(f"\nCatalogue saved to {RESULTS_DIR}/exp1_catalogue.csv")

    return {'catalogue': df, 'psi_check_error': err}


if __name__ == '__main__':
    results = main()
    print('Experiment 1 completed successfully')
