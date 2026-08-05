"""
Experiment 6: Static Batch Minimum-Scoring-Risk Estimation

Implements the training-stage estimator for recursion parameters (omega, phi, alpha)
by minimising in-sample average scoring risk for each density x criterion cell.

Markets: ^GSPC, ^GDAXI, ^N225 (representative subset for feasibility).
Cells: gaussian_log, density_power(beta=0.15), crps, mmd, student_log(nu=6).

# Custom — Context7 found no library equivalent for paper-specific MSR estimator
"""
import sys, os
sys.path.insert(0, '/workspace/project')

import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

from src.data_utils import prepare_market_data, estimate_mmd_bandwidth
from src.fast_filter import filter_score as _fast_filter_score, warm_jit as _warm_jit

RESULTS_DIR = '/workspace/project/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

TICKERS = ['^GSPC', '^IXIC', '^DJI', '^FTSE', '^GDAXI', '^FCHI',
           '^STOXX50E', '^N225', '^HSI', '^GSPTSE', '^AXJO', '^BVSP']
BETA = 0.15
NU = 6.0

MULTISTARTS = [
    [0.0, 0.95, 0.05],
    [0.0, 0.97, 0.03],
]

BOUNDS = [(-3.0, 3.0), (0.80, 0.999), (0.001, 0.5)]


# ---------------------------------------------------------------------------
# Score objective: average scoring risk over T_tr
# ---------------------------------------------------------------------------

def run_filter_and_score(params, y, criterion, ell=None):
    """Compute average scoring risk via JIT-compiled kernel (see src/fast_filter.py)."""
    return _fast_filter_score(
        np.asarray(params, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        criterion,
        ell=ell,
        beta=BETA,
        nu=NU,
    )


def estimate_market_criterion(y_train: np.ndarray, criterion: str, ell: float = None) -> dict:
    """
    Estimate (omega, phi, alpha) using multi-start L-BFGS-B minimisation.

    Returns dict with params, obj, grad_norm, gates.
    """
    best_obj = np.inf
    best_res = None

    for x0 in MULTISTARTS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    run_filter_and_score,
                    x0=x0,
                    args=(y_train, criterion, ell),
                    method='L-BFGS-B',
                    bounds=BOUNDS,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-5},
                )
        except Exception:
            continue

        if res.fun < best_obj and np.isfinite(res.fun):
            best_obj = res.fun
            best_res = res

    if best_res is None:
        return {'params': None, 'obj': np.inf, 'all_pass': False}

    params = best_res.x
    obj = float(best_res.fun)

    # Gradient norm
    jac = best_res.jac if hasattr(best_res, 'jac') and best_res.jac is not None else np.array([np.inf]*3)
    grad_norm = float(np.linalg.norm(jac))

    # Admissibility gates
    omega, phi, alpha = params
    parameter_gate = abs(phi) < 0.999
    convergence_gate = np.isfinite(obj) and obj < 1e9
    gradient_gate = grad_norm <= 1e-3

    # Path-finiteness gate: single forward pass to check all states are finite
    path_finite = np.isfinite(run_filter_and_score(params, y_train, criterion, ell))

    all_pass = parameter_gate and convergence_gate and path_finite

    return {
        'params': params,
        'omega': float(params[0]),
        'phi': float(params[1]),
        'alpha': float(params[2]),
        'obj': obj,
        'grad_norm': grad_norm,
        'gradient_gate': gradient_gate,
        'parameter_gate': parameter_gate,
        'convergence_gate': convergence_gate,
        'path_finiteness_gate': path_finite,
        'all_pass': all_pass,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Experiment 6: Static Batch Minimum-Scoring-Risk Estimation")
    print(f"  Markets: {TICKERS}")
    print(f"  Criteria: gaussian_log, density_power(β={BETA}), crps, mmd, student_log(ν={NU})")
    print("  Pre-compiling JIT kernels...")
    _warm_jit(beta=BETA, nu=NU, ell=2.0)
    print("  JIT compilation done.")

    rows = []

    for ticker in TICKERS:
        print(f"\n  Processing {ticker}...")

        # Download data
        data = prepare_market_data(ticker)
        if data is None:
            print(f"  WARNING: Failed to download {ticker}, using synthetic data")
            rng = np.random.default_rng(42)
            y_train = rng.standard_normal(1500) * 1.2
            label = f'{ticker}_synthetic'
        else:
            y_train = data['y_train']
            label = data['label']
            print(f"    T_train={data['T_train']}, T_test={data['T_test']}")

        # Estimate MMD bandwidth from training data
        ell = estimate_mmd_bandwidth(y_train)
        print(f"    MMD bandwidth ell={ell:.4f}")

        for criterion in ['gaussian_log', 'density_power', 'crps', 'mmd', 'student_log']:
            ell_arg = ell if criterion == 'mmd' else None
            result = estimate_market_criterion(y_train, criterion, ell=ell_arg)

            row = {
                'market': label,
                'ticker': ticker,
                'criterion': criterion,
                'mmd_ell': ell if criterion == 'mmd' else float('nan'),
            }
            row.update(result)
            # Remove params array
            if 'params' in row:
                del row['params']

            rows.append(row)

            status = '✓' if result.get('all_pass') else '✗'
            if result['obj'] < 1e9:
                print(f"    {criterion:20s}: obj={result['obj']:.6f}, "
                      f"phi={result.get('phi', 'nan'):.4f}, "
                      f"alpha={result.get('alpha', 'nan'):.4f} [{status}]")
            else:
                print(f"    {criterion:20s}: FAILED [{status}]")

    df = pd.DataFrame(rows)
    cols = ['market', 'criterion', 'omega', 'phi', 'alpha', 'obj',
            'grad_norm', 'gradient_gate', 'parameter_gate', 'all_pass']
    available_cols = [c for c in cols if c in df.columns]
    df_out = df[available_cols]
    df_out.to_csv(f'{RESULTS_DIR}/exp6_fitted_params.csv', index=False)
    print(f"\n  Fitted parameters saved to {RESULTS_DIR}/exp6_fitted_params.csv")
    print(df_out.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

    return {'fitted_params': df}


if __name__ == '__main__':
    results = main()
    print('Experiment 6 completed successfully')
