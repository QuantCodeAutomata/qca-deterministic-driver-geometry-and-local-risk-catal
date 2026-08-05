"""
Experiment 10: Heavy-Tail Stress Test with Tail-Index Sweep

APPROXIMATE REPRODUCTION: Exact latent-state DGP is archive-dependent.
Fallback AR log-volatility DGP used, clearly labeled as such.

Sweeps innovation tail index nu in {1, 1.5, 2, 3, 5, inf} (inf=Gaussian).
All laws normalized to median absolute innovation = 0.67449.

# Custom — Context7 found no library equivalent for this paper-specific tail stress
"""
import sys, os
sys.path.insert(0, '/workspace/project')

import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student_t, norm
from scipy.optimize import minimize

RESULTS_DIR = '/workspace/project/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 20260430
T_TOTAL = 2000    # paper uses T=10000; use 2000 for feasibility
TRAIN_FRAC = 0.60
BOOT_REPS = 100    # paper uses 500; use 100 for feasibility
MEDIAN_TARGET = 0.67449

BETA = 0.15
NU_FILTER = 6.0

# Tail index grid: nu in {1, 1.5, 2, 3, 5, inf}
NU_GRID = [1.0, 1.5, 2.0, 3.0, 5.0, float('inf')]

CRITERIA = ['gaussian_log', 'density_power', 'crps']
BOUNDS = [(-3.0, 3.0), (0.80, 0.999), (0.001, 0.5)]
MULTISTARTS = [[0.0, 0.95, 0.05], [0.0, 0.90, 0.10]]


# ---------------------------------------------------------------------------
# Innovation law normalization
# ---------------------------------------------------------------------------

def generate_innovations(nu: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate n innovations from law with tail index nu, normalized to MAI = 0.67449.

    nu=inf: Gaussian. Else Student-t with df=nu, variance-scaled.
    Normalization: scale s.t. population median of |eps| = 0.67449.
    """
    if np.isinf(nu):
        # N(0,1): MAI = MAD(N(0,1)) = sqrt(2)*erfinv(0.5) ≈ 0.67449 already
        scale = 1.0  # N(0,1) median absolute value = 0.67449
        return rng.standard_normal(n) * scale
    else:
        # Student-t_nu: standardize to unit variance if nu > 2, else just scale
        raw = rng.standard_normal(n) / np.sqrt(rng.chisquare(nu, n) / nu)
        # Compute empirical MAI scale factor analytically
        # For t_nu, median of |t| = t_{nu}(0.75) (3/4 quantile of |t|)
        from scipy.stats import t
        med_abs = float(t.ppf(0.75, df=nu))  # median of |t_nu|
        scale = MEDIAN_TARGET / med_abs
        return raw * scale


# ---------------------------------------------------------------------------
# Inline filter functions
# ---------------------------------------------------------------------------

def run_filter_obj(params, y, criterion):
    """Average scoring risk (from exp6 logic)."""
    omega, phi, alpha = params
    if abs(phi) >= 0.999:
        return 1e10
    T = len(y)
    lam = np.zeros(T + 1)
    lam[0] = omega / (1.0 - phi) if abs(phi) < 1.0 else 0.0
    total = 0.0
    for t in range(T):
        z = y[t] * np.exp(-0.5 * lam[t])
        if criterion == 'gaussian_log':
            psi = 0.5 * (z**2 - 1.0); G = 2.0
            s = 0.5 * lam[t] + 0.5 * np.exp(-lam[t]) * y[t]**2
        elif criterion == 'density_power':
            beta = BETA
            c_beta = (2.0 * np.pi) ** (-beta / 2.0)
            kb = beta * c_beta * (1.0 + beta) ** (-0.5)
            g = (1.0 + beta) * c_beta * np.exp(-beta * z**2 / 2.0) * (z**2 - 1.0) + kb
            psi = g / (2.0 * np.exp(beta * lam[t] / 2.0)); G = 2.0 * np.exp(beta * lam[t] / 2.0)
            s = c_beta * np.exp(-beta * lam[t] / 2.0) * (
                (1.0 + beta)**(-0.5) - (1.0 + 1.0 / beta) * np.exp(-beta * z**2 / 2.0))
        elif criterion == 'crps':
            sg = np.exp(0.5 * lam[t])
            g = np.pi**(-0.5) - np.sqrt(2.0 / np.pi) * np.exp(-z**2 / 2.0)
            psi = g / (2.0 * np.exp(-0.5 * lam[t])); G = 2.0 * np.exp(-0.5 * lam[t])
            s = sg * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - np.pi**(-0.5))
        else:
            raise ValueError(criterion)
        u = G * psi
        lam[t + 1] = omega + phi * lam[t] + alpha * u
        if not np.isfinite(s) or not np.isfinite(lam[t + 1]):
            return 1e10
        total += s
    return total / T


def run_filter_path(params, y, criterion):
    """Return full lam path."""
    omega, phi, alpha = params
    T = len(y)
    lam = np.zeros(T + 1)
    lam[0] = omega / (1.0 - phi) if abs(phi) < 1.0 else 0.0
    for t in range(T):
        z = y[t] * np.exp(-0.5 * lam[t])
        if criterion == 'gaussian_log':
            psi = 0.5 * (z**2 - 1.0); G = 2.0
        elif criterion == 'density_power':
            beta = BETA
            c_beta = (2.0 * np.pi) ** (-beta / 2.0)
            kb = beta * c_beta * (1.0 + beta) ** (-0.5)
            g = (1.0 + beta) * c_beta * np.exp(-beta * z**2 / 2.0) * (z**2 - 1.0) + kb
            psi = g / (2.0 * np.exp(beta * lam[t] / 2.0)); G = 2.0 * np.exp(beta * lam[t] / 2.0)
        elif criterion == 'crps':
            g = np.pi**(-0.5) - np.sqrt(2.0 / np.pi) * np.exp(-z**2 / 2.0)
            psi = g / (2.0 * np.exp(-0.5 * lam[t])); G = 2.0 * np.exp(-0.5 * lam[t])
        else:
            raise ValueError(criterion)
        lam[t + 1] = omega + phi * lam[t] + alpha * G * psi
    return lam


def estimate(y_train, criterion):
    """Multi-start L-BFGS-B."""
    best_obj, best_params = np.inf, np.array([0.0, 0.95, 0.05])
    for x0 in MULTISTARTS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(run_filter_obj, x0=x0, args=(y_train, criterion),
                               method='L-BFGS-B', bounds=BOUNDS,
                               options={'maxiter': 200, 'ftol': 1e-9})
            if res.fun < best_obj and np.isfinite(res.fun):
                best_obj, best_params = res.fun, res.x
        except Exception:
            pass
    return best_params, best_obj


# ---------------------------------------------------------------------------
# Tracking correlation computation
# ---------------------------------------------------------------------------

def tracking_correlation(lam_filtered: np.ndarray, lam_true: np.ndarray) -> float:
    """Pearson correlation between filtered and true log-variance path."""
    if np.std(lam_filtered) < 1e-12 or np.std(lam_true) < 1e-12:
        return float('nan')
    return float(np.corrcoef(lam_filtered, lam_true)[0, 1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Experiment 10: Heavy-Tail Stress Test with Tail-Index Sweep")
    print("  NOTE: APPROXIMATE REPRODUCTION - Fallback DGP used.")
    print(f"  T={T_TOTAL}, train_frac={TRAIN_FRAC}, boot_reps={BOOT_REPS}")

    rng = np.random.default_rng(SEED)
    T_train = int(T_TOTAL * TRAIN_FRAC)
    T_eval = T_TOTAL - T_train

    rows = []

    for nu in NU_GRID:
        nu_label = 'Gaussian' if np.isinf(nu) else f'Student-t(nu={nu})'
        print(f"\n  Innovation law: {nu_label}")

        # Generate latent log-variance path (same DGP across nu)
        lam_true = np.zeros(T_TOTAL + 1)
        eta = rng.standard_normal(T_TOTAL)
        for t in range(T_TOTAL):
            lam_true[t + 1] = 0.05 + 0.97 * lam_true[t] + 0.05 * eta[t]
        lam_target = lam_true[1:]

        # Generate innovations from specified law
        eps = generate_innovations(nu, T_TOTAL, rng)
        y = np.exp(0.5 * lam_true[:T_TOTAL]) * eps

        y_train = y[:T_train]
        y_eval = y[T_train:]
        lam_eval_true = lam_target[T_train:]

        for criterion in CRITERIA:
            params, obj = estimate(y_train, criterion)
            if not np.isfinite(obj):
                rows.append({'nu': nu if not np.isinf(nu) else 'inf', 'criterion': criterion,
                              'corr': float('nan'), 'corr_boot_se': float('nan'),
                              'admissible': False})
                continue

            # Get eval-period filter path
            lam_train_path = run_filter_path(params, y_train, criterion)
            lam_eval_path = run_filter_path(params, y_eval, criterion,
                                             )[1:]  # skip initial
            # Actually rerun from end of training
            omega, phi, alpha = params
            lam_eval = np.zeros(T_eval + 1)
            lam_eval[0] = lam_train_path[-1]
            for t in range(T_eval):
                z = y_eval[t] * np.exp(-0.5 * lam_eval[t])
                if criterion == 'gaussian_log':
                    psi = 0.5 * (z**2 - 1.0); G = 2.0
                elif criterion == 'density_power':
                    beta = BETA
                    c_beta = (2.0 * np.pi) ** (-beta / 2.0)
                    kb = beta * c_beta * (1.0 + beta) ** (-0.5)
                    g = (1.0 + beta) * c_beta * np.exp(-beta * z**2 / 2.0) * (z**2 - 1.0) + kb
                    psi = g / (2.0 * np.exp(beta * lam_eval[t] / 2.0)); G = 2.0 * np.exp(beta * lam_eval[t] / 2.0)
                elif criterion == 'crps':
                    g = np.pi**(-0.5) - np.sqrt(2.0 / np.pi) * np.exp(-z**2 / 2.0)
                    psi = g / (2.0 * np.exp(-0.5 * lam_eval[t])); G = 2.0 * np.exp(-0.5 * lam_eval[t])
                lam_eval[t + 1] = omega + phi * lam_eval[t] + alpha * G * psi

            lam_eval_f = lam_eval[1:]
            corr = tracking_correlation(lam_eval_f, lam_eval_true)

            # Bootstrap SE
            boot_corrs = np.zeros(BOOT_REPS)
            for b in range(BOOT_REPS):
                idx = rng.integers(0, T_eval, T_eval)
                boot_corrs[b] = tracking_correlation(lam_eval_f[idx], lam_eval_true[idx])
            boot_se = float(np.nanstd(boot_corrs))

            rows.append({
                'nu': nu if not np.isinf(nu) else 'inf',
                'criterion': criterion,
                'corr': corr,
                'corr_boot_se': boot_se,
                'admissible': True,
            })
            print(f"    {criterion:20s}: corr={corr:.4f} ± {boot_se:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(f'{RESULTS_DIR}/exp10_tracking.csv', index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'gaussian_log': 'blue', 'density_power': 'red', 'crps': 'purple'}
    nu_labels = [str(nu) if not np.isinf(nu) else 'inf' for nu in NU_GRID]

    for criterion in CRITERIA:
        sub = df[df['criterion'] == criterion]
        corrs = sub['corr'].values
        ses = sub['corr_boot_se'].values
        x_pos = list(range(len(corrs)))
        ax.plot(x_pos, corrs, 'o-', color=colors[criterion], label=criterion)
        ax.fill_between(x_pos, corrs - 2*ses, corrs + 2*ses, color=colors[criterion], alpha=0.2)

    ax.set_xticks(range(len(NU_GRID)))
    ax.set_xticklabels(nu_labels)
    ax.set_xlabel('Innovation tail index ν')
    ax.set_ylabel('Tracking correlation with latent λ')
    ax.set_title('Heavy-Tail Stress: Tracking Correlation vs Tail Index\n(APPROXIMATE - Fallback DGP)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{RESULTS_DIR}/exp10_tracking.png', dpi=150)
    plt.close()
    print(f"\n  Plot saved to {RESULTS_DIR}/exp10_tracking.png")

    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    return {'tracking': df}


if __name__ == '__main__':
    results = main()
    print('Experiment 10 completed successfully')
