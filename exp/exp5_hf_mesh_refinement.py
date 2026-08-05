"""
Experiment 5: High-Frequency Mesh Refinement and Continuous-Time Limits

Three substudies establishing continuous-time limits:
  A: Centered diffusion (gain alpha_h = alpha*sqrt(h))
  B: Non-centered mean flow (gain alpha_h = alpha*h)
  C: Local OU approximation for stopped tracking-error process

# Custom — Context7 found no library equivalent for these paper-specific limits
"""
import sys, os
sys.path.insert(0, '/workspace/project')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermgauss
from scipy.stats import ks_2samp, t as student_t, norm

RESULTS_DIR = '/workspace/project/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 20260430
BETA = 0.15
NU = 5.0   # Substudy B uses t_5

# ---------------------------------------------------------------------------
# Theoretical coefficient tuples (from geometry computation)
# Used as reference values for validation
# ---------------------------------------------------------------------------
THEORETICAL_TUPLES = {
    'gaussian_log':   {'J': 0.5,        'K': 0.5,       'Xi': 2.0,       'C': 0.0, 'Delta': 2.0},
    'density_power':  {'J': 0.3572056,  'K': 0.268493,  'Xi': 1.073970,  'C': 0.0, 'Delta': 1.073970},
    'crps':           {'J': 0.070524,   'K': 0.012311,  'Xi': 0.049243,  'C': 0.0, 'Delta': 0.049243},
}

CRITERIA_A = ['gaussian_log', 'density_power', 'crps']


# ---------------------------------------------------------------------------
# Score / psi inline functions (at lam=0, h-scaled observations)
# For centered diffusion: y_{t,h} ~ N(0, h*exp(lam*)) at lam*=0 -> N(0,h)
# Standardised residual: z = y/sqrt(h*exp(lam)) at lam=lam*=0 -> z = y/sqrt(h)
# ---------------------------------------------------------------------------

def psi_inline(criterion: str, y: float, lam: float) -> float:
    """Raw innovation psi_S(y, lam) for each criterion."""
    if criterion == 'gaussian_log':
        return 0.5 * (y**2 * np.exp(-lam) - 1.0)
    elif criterion == 'density_power':
        beta = BETA
        c_beta = (2.0 * np.pi) ** (-beta / 2.0)
        kb = beta * c_beta * (1.0 + beta) ** (-0.5)
        z = y * np.exp(-0.5 * lam)
        g = (1.0 + beta) * c_beta * np.exp(-beta * z**2 / 2.0) * (z**2 - 1.0) + kb
        return g / (2.0 * np.exp(beta * lam / 2.0))
    elif criterion == 'crps':
        z = y * np.exp(-0.5 * lam)
        g = np.pi**(-0.5) - np.sqrt(2.0 / np.pi) * np.exp(-z**2 / 2.0)
        return g / (2.0 * np.exp(-0.5 * lam))
    raise ValueError(criterion)


def u_inline(criterion: str, y: float, lam: float) -> float:
    """Implemented update u_S = G_S * psi_S."""
    psi = psi_inline(criterion, y, lam)
    if criterion in ('gaussian_log',):
        return 2.0 * psi
    elif criterion == 'density_power':
        G = 2.0 * np.exp(BETA * lam / 2.0)
        return G * psi
    elif criterion == 'crps':
        G = 2.0 * np.exp(-0.5 * lam)
        return G * psi
    raise ValueError(criterion)


# ---------------------------------------------------------------------------
# Substudy A: Centred diffusion coefficient computation
# ---------------------------------------------------------------------------

def compute_theoretical_coefficients() -> pd.DataFrame:
    """
    Compute theoretical coefficient tuples (J, K, Xi, C, Delta) analytically
    using 64-node Gauss-Hermite quadrature.

    Note: C=0 by symmetry (u_S is even function of z under symmetric criteria).
    Xi = E[u_S^2] = 4*K for G=2 at lam=0.
    Delta = Xi - C^2 = Xi (since C=0).
    """
    from src.geometry import (J_gaussian_log, K_gaussian_log,
                               J_density_power, K_density_power, J_crps, K_crps)

    rows = []
    for crit in CRITERIA_A:
        if crit == 'gaussian_log':
            J = J_gaussian_log(64)
            K = K_gaussian_log(64)
        elif crit == 'density_power':
            J = J_density_power(BETA, 64)
            K = K_density_power(BETA, 64)
        elif crit == 'crps':
            J = J_crps(64)
            K = K_crps(64)

        Xi = 4.0 * K   # u = 2*psi at lam=0, so E[u^2]=4*K
        C = 0.0         # by symmetry
        Delta = Xi - C**2

        rows.append({
            'criterion': crit,
            'J': J, 'K': K, 'Xi': Xi, 'C': C, 'Delta': Delta,
        })

    df = pd.DataFrame(rows)
    return df


def run_substudy_A(R: int = 8000, T: int = 200, alpha: float = 1.0):
    """
    Substudy A: Centred diffusion.

    For each mesh h, simulate R paths of length T.
    Observation: y_{t,h} ~ N(0, h*exp(lam)) at lam*=0, so y ~ N(0,h)
    Gain: alpha_h = alpha * sqrt(h)
    Recursion (centered at lam*=0): lam_{t+1} = lam_t + alpha_h * psi_S(y,lam_t)
    """
    print("  Substudy A: Centred diffusion")
    rng = np.random.default_rng(SEED)
    half_powers = list(range(1, 8))   # h = 2^{-1}, ..., 2^{-7}
    meshes = [2.0**(-k) for k in half_powers]

    # Theoretical reference
    theor = compute_theoretical_coefficients()

    rows_sim = []
    for h in meshes:
        alpha_h = alpha * np.sqrt(h)
        h_sqrt = np.sqrt(h)

        for crit in CRITERIA_A:
            # Simulate R paths of length T
            # y ~ N(0, h) since lam*=0
            y_all = rng.standard_normal((R, T)) * h_sqrt

            # Run recursion (vectorized over R where possible)
            lam = np.zeros((R, T + 1))
            psi_vals = np.zeros((R, T))
            u_vals = np.zeros((R, T))
            z_vals = np.zeros((R, T))

            for t in range(T):
                yt = y_all[:, t]
                lam_t = lam[:, t]
                # z_full = y / sqrt(h * exp(lam)) — standardised residual under N(0, h*v)
                z_full = yt * np.exp(-0.5 * lam_t) / h_sqrt
                if crit == 'gaussian_log':
                    # psi = 0.5*(z_full^2 - 1); u = 2*psi, G=2
                    psi = 0.5 * (z_full**2 - 1.0)
                    u = 2.0 * psi
                elif crit == 'density_power':
                    beta = BETA
                    c_beta = (2.0 * np.pi) ** (-beta / 2.0)
                    kb = beta * c_beta * (1.0 + beta) ** (-0.5)
                    g = (1.0 + beta) * c_beta * np.exp(-beta * z_full**2 / 2.0) * (z_full**2 - 1.0) + kb
                    G = 2.0 * np.exp(beta * lam_t / 2.0)
                    psi = g / G
                    u = g  # u = G * psi = g
                elif crit == 'crps':
                    g = np.pi**(-0.5) - np.sqrt(2.0 / np.pi) * np.exp(-z_full**2 / 2.0)
                    G = 2.0 * np.exp(-0.5 * lam_t)
                    psi = g / G
                    u = g  # u = G * psi = g

                psi_vals[:, t] = psi
                u_vals[:, t] = u
                z_vals[:, t] = z_full
                lam[:, t + 1] = lam_t + alpha_h * psi

            # Compute empirical coefficient estimates
            K_est = float(np.mean(psi_vals**2))
            Xi_est = float(np.mean(u_vals**2))
            C_est = float(np.mean(z_vals * u_vals))  # should be ~0

            rows_sim.append({
                'h': h, 'criterion': crit,
                'K_empirical': K_est, 'Xi_empirical': Xi_est, 'C_empirical': C_est,
                'Delta_empirical': Xi_est - C_est**2,
            })

        print(f"    h={h:.4f} done")

    df_sim = pd.DataFrame(rows_sim)
    return theor, df_sim


# ---------------------------------------------------------------------------
# Substudy B: Non-centred mean flow (sketch)
# ---------------------------------------------------------------------------

def run_substudy_B(R: int = 8000, T: int = 30, alpha: float = 1.0) -> pd.DataFrame:
    """
    Substudy B: Non-centered mean flow using variance-matched t_5 residuals.
    Gain: alpha_h = alpha * h. Shows averaged path converges to deterministic drift.
    """
    print("  Substudy B: Non-centered mean flow (sketch)")
    rng = np.random.default_rng(SEED + 1)
    half_powers = list(range(1, 8))
    meshes = [2.0**(-k) for k in half_powers]

    # Variance-matched t_5: scale s.t. Var = 1 -> std = sqrt((nu-2)/nu)
    scale = np.sqrt((NU - 2.0) / NU)

    rows = []
    for h in meshes:
        alpha_h = alpha * h
        # Simulate
        y_all = student_t.rvs(df=NU, size=(R, T), random_state=rng.integers(0, 2**31)) * scale * np.sqrt(h)

        # Mean-flow recursion at lam*=0: lam_{t+1} = lam_t + alpha_h * u(y, lam_t)
        lam = np.zeros((R, T + 1))
        for t in range(T):
            yt = y_all[:, t]
            lam_t = lam[:, t]
            psi = 0.5 * (yt**2 * np.exp(-lam_t) - 1.0)  # Gaussian log as example
            lam[:, t + 1] = lam_t + alpha_h * psi

        mean_path = np.mean(lam, axis=0)
        rows.append({'h': h, 'mean_terminal_lam': float(mean_path[-1]),
                     'std_terminal_lam': float(np.std(lam[:, -1]))})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    return df


# ---------------------------------------------------------------------------
# Substudy C: Local OU approximation
# ---------------------------------------------------------------------------

def run_substudy_C(R: int = 12000, M: int = 3, rho: float = 0.8,
                   Gamma: float = 0.65, K_thresh: float = 3.5) -> pd.DataFrame:
    """
    Substudy C: Local OU approximation for stopped tracking-error.

    Stopped process: stopped at threshold K or local horizon M.
    Compare discrete-array terminal variance to stopped OU variance.
    """
    print("  Substudy C: Local OU comparison")
    rng = np.random.default_rng(SEED + 2)
    half_powers = list(range(4, 13))  # h = 2^{-4},...,2^{-12}
    meshes = [2.0**(-k) for k in half_powers]

    rows = []
    for h in meshes:
        alpha_h = Gamma * np.sqrt(h)  # local gain

        for crit in ['gaussian_log', 'density_power', 'crps']:
            # Simulate R stopped paths
            Z0_fixed = 0.0
            terminal_vars = []

            # Simulate paths
            # y_{t,h} ~ N(0, h) at lam*=0; z_full = y/sqrt(h) ~ N(0,1)
            h_sqrt = np.sqrt(h)
            lam_paths = np.zeros((R, M + 1))
            for t in range(M):
                yt = rng.standard_normal(R) * h_sqrt
                lam_t = lam_paths[:, t]
                z_full = yt * np.exp(-0.5 * lam_t) / h_sqrt  # ~ N(0,1)
                if crit == 'gaussian_log':
                    psi = 0.5 * (z_full**2 - 1.0)
                elif crit == 'density_power':
                    beta = BETA
                    c_beta = (2.0 * np.pi) ** (-beta / 2.0)
                    kb = beta * c_beta * (1.0 + beta) ** (-0.5)
                    g = (1.0 + beta) * c_beta * np.exp(-beta * z_full**2 / 2.0) * (z_full**2 - 1.0) + kb
                    psi = g / (2.0 * np.exp(beta * lam_t / 2.0))
                elif crit == 'crps':
                    g = np.pi**(-0.5) - np.sqrt(2.0 / np.pi) * np.exp(-z_full**2 / 2.0)
                    psi = g / (2.0 * np.exp(-0.5 * lam_t))

                lam_paths[:, t + 1] = lam_t + alpha_h * psi

            # Apply stopping rule: stop at step where |lam| >= K_thresh or at M
            terminal = np.zeros(R)
            for r in range(R):
                stopped = False
                for s in range(1, M + 1):
                    if abs(lam_paths[r, s]) >= K_thresh:
                        terminal[r] = lam_paths[r, s]
                        stopped = True
                        break
                if not stopped:
                    terminal[r] = lam_paths[r, M]

            disc_var = float(np.var(terminal))

            # Theoretical OU variance: Var[X_{M*h}] ≈ sigma^2 * M * h for small h
            # where sigma^2 = Gamma^2 * K_S (diffusion coefficient of limit OU)
            K_S_val = {'gaussian_log': 0.5, 'density_power': 0.268493, 'crps': 0.012311}[crit]
            ou_var = Gamma**2 * K_S_val * M * h

            ratio = disc_var / (ou_var + 1e-30) if ou_var > 0 else float('nan')

            rows.append({
                'h': h, 'criterion': crit,
                'disc_var': disc_var,
                'ou_var': ou_var,
                'ratio': ratio,
            })

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Experiment 5: High-Frequency Mesh Refinement and Continuous-Time Limits")

    # Substudy A
    R_A = 1000  # Use R=1000 for speed; paper uses 8000
    theor_A, sim_A = run_substudy_A(R=R_A, T=200)

    print("\n  Theoretical coefficient tuples:")
    print(theor_A.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

    print("\n  Empirical estimates at finest mesh (h=2^{-7}):")
    finest = sim_A[sim_A['h'] == 2.0**(-7)]
    print(finest[['criterion', 'K_empirical', 'Xi_empirical', 'C_empirical', 'Delta_empirical']].to_string(
        index=False, float_format=lambda x: f'{x:.6f}'))

    theor_A.to_csv(f'{RESULTS_DIR}/exp5a_coefficients.csv', index=False)
    sim_A.to_csv(f'{RESULTS_DIR}/exp5a_empirical.csv', index=False)

    # Substudy B
    df_B = run_substudy_B(R=500, T=30)
    df_B.to_csv(f'{RESULTS_DIR}/exp5b_mean_flow.csv', index=False)

    # Substudy C
    R_C = 2000  # paper uses 12000
    df_C = run_substudy_C(R=R_C)
    print("\n  Substudy C - Local OU comparison (sample rows):")
    print(df_C[df_C['h'] <= 2.0**(-8)].to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    df_C.to_csv(f'{RESULTS_DIR}/exp5c_ou_comparison.csv', index=False)

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # A: Convergence of Xi
    ax = axes[0]
    for crit in CRITERIA_A:
        sub = sim_A[sim_A['criterion'] == crit].sort_values('h')
        ax.semilogx(sub['h'], sub['Xi_empirical'], 'o-', label=crit)
        theor_Xi = theor_A[theor_A['criterion'] == crit]['Xi'].values[0]
        ax.axhline(theor_Xi, linestyle='--', alpha=0.5)
    ax.set_xlabel('Mesh h')
    ax.set_ylabel('Xi_empirical')
    ax.set_title('Substudy A: Xi Convergence')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # B: Mean path vs h
    ax2 = axes[1]
    ax2.semilogx(df_B['h'], df_B['mean_terminal_lam'], 'o-')
    ax2.axhline(0, linestyle='--', alpha=0.5)
    ax2.set_xlabel('Mesh h')
    ax2.set_ylabel('Mean terminal λ')
    ax2.set_title('Substudy B: Non-Centered Mean Flow')
    ax2.grid(True, alpha=0.3)

    # C: OU ratio
    ax3 = axes[2]
    for crit in ['gaussian_log', 'density_power', 'crps']:
        sub = df_C[df_C['criterion'] == crit].sort_values('h')
        ax3.semilogx(sub['h'], sub['ratio'], 'o-', label=crit)
    ax3.axhline(1.0, linestyle='--', color='k', alpha=0.5)
    ax3.set_xlabel('Mesh h')
    ax3.set_ylabel('Discrete/OU variance ratio')
    ax3.set_title('Substudy C: Local OU Approximation')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'{RESULTS_DIR}/exp5_plots.png', dpi=150)
    plt.close()
    print(f"\n  Plots saved to {RESULTS_DIR}/exp5_plots.png")

    return {'theor_A': theor_A, 'sim_A': sim_A, 'df_B': df_B, 'df_C': df_C}


if __name__ == '__main__':
    results = main()
    print('Experiment 5 completed successfully')
