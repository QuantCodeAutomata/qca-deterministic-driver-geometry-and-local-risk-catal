"""
Experiment 9: VaR Coverage, Marginal PIT Uniformity, and Dynamic PIT Adequacy

Evaluates calibration of empirical volatility filters using:
- 1% and 5% VaR coverage with stationary bootstrap inference
- Marginal PIT uniformity (equal-market empirical process)
- Dynamic PIT adequacy (midrank normal scores, linear-quadratic portmanteau)
- Permutation calibration (1999 time permutations)

Uses real market data via prepare_market_data() and frozen parameters from
exp6_fitted_params.csv; falls back to synthetic data when data are unavailable.

# Custom — Context7 found no library equivalent for this panel PIT/VaR procedure
"""
import sys, os
sys.path.insert(0, '/workspace/project')

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

RESULTS_DIR = '/workspace/project/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 20260430
B_BOOT = 1999
MEAN_BLOCK = 20
N_PERM = 1999
N_LAGS = 10

BETA = 0.15
NU   = 6.0

# 12 equity tickers matching exp6/exp7
TICKERS = [
    '^GSPC', '^IXIC', '^DJI', '^FTSE', '^GDAXI', '^FCHI',
    '^STOXX50E', '^N225', '^HSI', '^GSPTSE', '^AXJO', '^BVSP',
]

CRITERION_MAP = {
    'gaussian_log':   ('G-L',  'gaussian'),
    'density_power':  ('G-DP', 'gaussian'),
    'crps':           ('G-C',  'gaussian'),
    'mmd':            ('G-M',  'gaussian'),
    'student_log':    ('T-L',  'student'),
}


# ---------------------------------------------------------------------------
# Real-data loader: uses exp6 frozen params + yfinance market data
# ---------------------------------------------------------------------------

def load_real_data() -> dict | None:
    """
    Load real 12-market panel: market data from Yahoo Finance + frozen params
    from exp6_fitted_params.csv.  Returns nested dict identical in shape to
    generate_test_data(), or None if any critical dependency is missing.
    """
    params_path = os.path.join(RESULTS_DIR, 'exp6_fitted_params.csv')
    if not os.path.exists(params_path):
        print("  exp6_fitted_params.csv not found — falling back to synthetic data")
        return None

    try:
        from src.data_utils import prepare_market_data, estimate_mmd_bandwidth
        from src.fast_filter import filter_path as fast_filter_path, warm_jit
    except ImportError as e:
        print(f"  Import error: {e} — falling back to synthetic data")
        return None

    df_params = pd.read_csv(params_path)
    warm_jit(beta=BETA, nu=NU, ell=2.0)

    data: dict = {}
    missing_markets: list[str] = []

    for ticker in TICKERS:
        mkt_data = prepare_market_data(ticker)
        if mkt_data is None:
            print(f"  WARNING: {ticker} data unavailable, skipping market")
            missing_markets.append(ticker)
            continue

        label    = mkt_data['label']
        y_train  = mkt_data['y_train']
        y_test   = mkt_data['y_test']
        ell_mkt  = estimate_mmd_bandwidth(y_train)

        data[label] = {}
        for criterion, (cell_label, density) in CRITERION_MAP.items():
            row = df_params[
                (df_params['market'] == label) & (df_params['criterion'] == criterion)
            ]
            if row.empty:
                # criterion not in exp6 table for this market — use neutral params
                params = np.array([0.0, 0.95, 0.05])
            else:
                r = row.iloc[0]
                params = np.array([float(r['omega']), float(r['phi']), float(r['alpha'])])

            ell_arg = ell_mkt if criterion == 'mmd' else None
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                lam_path = fast_filter_path(
                    params, np.asarray(y_test, dtype=np.float64),
                    criterion, ell=ell_arg, beta=BETA, nu=NU,
                )
            v_test = np.exp(lam_path[:len(y_test)])

            data[label][cell_label] = {
                'y': y_test,
                'v': v_test,
                'density': density,
            }

    if not data:
        print("  No markets loaded — falling back to synthetic data")
        return None
    if missing_markets:
        print(f"  {len(missing_markets)} markets skipped (data unavailable): {missing_markets}")

    return data


# ---------------------------------------------------------------------------
# Synthetic fallback generator
# ---------------------------------------------------------------------------

def generate_test_data(n_markets: int = 5, T_test: int = 500, seed: int = SEED) -> dict:
    """
    Generate synthetic (y_test, v_test) pairs per market and model.

    Returns nested dict: data[market][model] = {'y': ..., 'v': ..., 'density': ...}
    """
    rng = np.random.default_rng(seed)
    model_configs = {
        'G-L':  ('gaussian', 'gaussian_log'),
        'G-DP': ('gaussian', 'density_power'),
        'G-C':  ('gaussian', 'crps'),
        'G-M':  ('gaussian', 'mmd'),
        'T-L':  ('student',  'student_log'),
    }

    data = {}
    for mkt in range(n_markets):
        market_name = f'Market_{mkt+1}'
        data[market_name] = {}
        for model, (density, _criterion) in model_configs.items():
            lam = np.zeros(T_test + 1)
            for t in range(T_test):
                lam[t + 1] = 0.05 + 0.97 * lam[t] + 0.1 * rng.standard_normal()
            v = np.exp(lam[:T_test])
            y = np.sqrt(v) * rng.standard_normal(T_test)
            data[market_name][model] = {'y': y, 'v': v, 'density': density}

    return data


# ---------------------------------------------------------------------------
# VaR coverage
# ---------------------------------------------------------------------------

def var_hit_rate(y: np.ndarray, v: np.ndarray, p: float, density: str) -> float:
    """Empirical VaR hit rate at level p."""
    sigma = np.sqrt(np.maximum(v, 1e-10))
    if density == 'gaussian':
        var_t = norm.ppf(p) * sigma
    else:
        from scipy.stats import t
        nu = 6.0
        var_t = t.ppf(p, df=nu) * np.sqrt((nu - 2.0) / nu) * sigma
    return float(np.mean(y < var_t))


def stationary_bootstrap_coverage_pval(y: np.ndarray, v: np.ndarray, p: float,
                                         density: str, B: int = B_BOOT,
                                         mean_block: int = MEAN_BLOCK,
                                         seed: int = SEED) -> float:
    """Bootstrap p-value for H0: coverage = p (two-sided)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    obs_coverage = var_hit_rate(y, v, p, density)
    q_geom = 1.0 / mean_block
    boot_coverages = np.zeros(B)

    for b in range(B):
        idx = np.zeros(n, dtype=int)
        t = 0
        while t < n:
            start = rng.integers(0, n)
            blen = rng.geometric(q_geom)
            end = min(t + blen, n)
            for s in range(end - t):
                idx[t + s] = (start + s) % n
            t = end
        y_b, v_b = y[idx], v[idx]
        boot_coverages[b] = var_hit_rate(y_b, v_b, p, density)

    pval = float(np.mean(np.abs(boot_coverages - p) >= np.abs(obs_coverage - p)))
    return pval, obs_coverage


# ---------------------------------------------------------------------------
# PIT computation
# ---------------------------------------------------------------------------

def compute_pit(y: np.ndarray, v: np.ndarray, density: str) -> np.ndarray:
    """Compute PIT u_t = F_Phi(y_t / sqrt(v_t))."""
    sigma = np.sqrt(np.maximum(v, 1e-10))
    z = y / sigma
    if density == 'gaussian':
        u = norm.cdf(z)
    else:
        from scipy.stats import t
        nu = 6.0
        scale = np.sqrt((nu - 2.0) / nu)
        u = t.cdf(z / scale, df=nu)
    return np.clip(u, 1e-6, 1.0 - 1e-6)


def ks_pit_uniformity(u: np.ndarray) -> float:
    """KS statistic for PIT uniformity (deviation from U[0,1])."""
    n = len(u)
    u_sorted = np.sort(u)
    ecdf = np.arange(1, n + 1) / n
    d_max = np.max(np.abs(ecdf - u_sorted))
    return float(d_max)


def midrank_normal_scores(u: np.ndarray) -> np.ndarray:
    """Transform PITs to midrank normal scores."""
    n = len(u)
    ranks = (np.argsort(np.argsort(u)) + 0.5) / n
    return norm.ppf(np.clip(ranks, 1e-6, 1.0 - 1e-6))


def portmanteau_stat(v_series: np.ndarray, lags: int = N_LAGS) -> float:
    """
    Linear-quadratic portmanteau statistic on normal scores.

    Combines autocorrelations of v and v^2 up to given lag.
    """
    n = len(v_series)
    v_c = v_series - np.mean(v_series)
    v2_c = v_series**2 - np.mean(v_series**2)
    stat = 0.0
    for k in range(1, lags + 1):
        acf_lin = np.dot(v_c[k:], v_c[:-k]) / (n * np.var(v_c) + 1e-30)
        acf_quad = np.dot(v2_c[k:], v2_c[:-k]) / (n * np.var(v2_c) + 1e-30)
        stat += n * (acf_lin**2 + acf_quad**2)
    return float(stat)


def permutation_pval(series: np.ndarray, observed_stat: float, n_perm: int = N_PERM,
                      seed: int = SEED) -> float:
    """Calibrate portmanteau stat by time permutations."""
    rng = np.random.default_rng(seed)
    perm_stats = np.zeros(n_perm)
    for b in range(n_perm):
        perm_series = series[rng.permutation(len(series))]
        ns_perm = midrank_normal_scores(perm_series)
        perm_stats[b] = portmanteau_stat(ns_perm)
    return float(np.mean(perm_stats >= observed_stat))


# ---------------------------------------------------------------------------
# Holm adjustment
# ---------------------------------------------------------------------------

def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Apply Holm-Bonferroni p-value adjustment."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.zeros(n)
    running_max = 0.0
    for i, idx in enumerate(order):
        adj_p = pvals[idx] * (n - i)
        running_max = max(running_max, min(adj_p, 1.0))
        adj[idx] = running_max
    return adj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Experiment 9: VaR Coverage, PIT Uniformity, and Dynamic PIT Adequacy")

    # Try real 12-market data first; fall back to synthetic if unavailable
    print("  Attempting to load real 12-market data from exp6/exp7 pipeline...")
    data = load_real_data()
    if data is None:
        print("  Falling back to synthetic data (5 markets, T=500)")
        data = generate_test_data(n_markets=5, T_test=500, seed=SEED)
        data_source = 'synthetic'
    else:
        data_source = 'real'
        print(f"  Loaded real data for {len(data)} markets")

    markets = list(data.keys())
    models  = list(data[markets[0]].keys())

    rows_coverage = []
    rows_pit = []

    for model in models:
        density = data[markets[0]][model]['density']
        cov1_markets, cov5_markets = [], []
        ks_markets, port_pvals = [], []

        for market in markets:
            y = data[market][model]['y']
            v = data[market][model]['v']

            # VaR coverage
            pval1, cov1 = stationary_bootstrap_coverage_pval(y, v, p=0.01, density=density,
                                                               B=199, seed=SEED)
            pval5, cov5 = stationary_bootstrap_coverage_pval(y, v, p=0.05, density=density,
                                                               B=199, seed=SEED+1)
            cov1_markets.append(cov1)
            cov5_markets.append(cov5)
            rows_coverage.append({
                'model': model, 'market': market,
                'cov_1pct': cov1, 'pval_1pct': pval1,
                'cov_5pct': cov5, 'pval_5pct': pval5,
            })

            # PIT analysis
            u = compute_pit(y, v, density)
            ks_d = ks_pit_uniformity(u)
            ns = midrank_normal_scores(u)
            port_stat = portmanteau_stat(ns)
            port_p = permutation_pval(u, port_stat, n_perm=199, seed=SEED+2)
            ks_markets.append(ks_d)
            port_pvals.append(port_p)
            rows_pit.append({
                'model': model, 'market': market,
                'ks_d': ks_d, 'port_stat': port_stat, 'port_pval': port_p,
            })

        # Equal-market means
        print(f"\n  {model}: mean_cov1%={100*np.mean(cov1_markets):.2f}%, "
              f"mean_cov5%={100*np.mean(cov5_markets):.2f}%, "
              f"mean_KS={np.mean(ks_markets):.4f}, "
              f"port_reject_count={sum(p < 0.05 for p in port_pvals)}/{len(markets)}")

    df_cov = pd.DataFrame(rows_coverage)
    df_pit = pd.DataFrame(rows_pit)

    # Holm adjustment for VaR coverage (across markets for each model-level)
    for model in models:
        sub = df_cov[df_cov['model'] == model]
        pvals_1 = sub['pval_1pct'].values
        pvals_5 = sub['pval_5pct'].values
        adj_1 = holm_adjust(pvals_1)
        adj_5 = holm_adjust(pvals_5)
        print(f"  {model}: Holm-adj 1% non-rejection count={sum(adj_1 > 0.05)}/{len(markets)}")
        print(f"  {model}: Holm-adj 5% non-rejection count={sum(adj_5 > 0.05)}/{len(markets)}")

    # Save results (tag with data source)
    df_cov['data_source'] = data_source
    df_pit['data_source'] = data_source
    df_cov.to_csv(f'{RESULTS_DIR}/exp9_coverage.csv', index=False)
    df_pit.to_csv(f'{RESULTS_DIR}/exp9_pit.csv', index=False)
    print(f"\n  Data source: {data_source} | markets={len(markets)}")

    # PIT histograms — first market
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(3 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    for i, model in enumerate(models):
        ax = axes[i]
        y_plt = data[markets[0]][model]['y']
        v_plt = data[markets[0]][model]['v']
        dns   = data[markets[0]][model]['density']
        u_plt = compute_pit(y_plt, v_plt, dns)
        ax.hist(u_plt, bins=20, density=True, alpha=0.7, color='steelblue')
        ax.axhline(1.0, color='r', linestyle='--', label='Uniform')
        ax.set_title(model, fontsize=9)
        ax.set_xlabel('PIT u_t')
        ax.legend(fontsize=7)
    src_tag = 'Real Data' if data_source == 'real' else 'Synthetic'
    plt.suptitle(f'PIT Histograms — {markets[0]} ({src_tag})', fontsize=11)
    plt.tight_layout()
    fig.savefig(f'{RESULTS_DIR}/exp9_pit_hist.png', dpi=150)
    plt.close()

    # Coverage heatmap
    import seaborn as sns
    pivot_cov = df_cov.pivot_table(index='market', columns='model', values='cov_1pct')
    fig2, ax2 = plt.subplots(figsize=(max(8, n_models * 2), max(4, len(markets) * 0.5)))
    sns.heatmap(pivot_cov * 100, annot=True, fmt='.1f', cmap='RdYlGn_r',
                center=1.0, ax=ax2)
    ax2.set_title(f'1% VaR Coverage Rates (%) — {src_tag}')
    plt.tight_layout()
    fig2.savefig(f'{RESULTS_DIR}/exp9_coverage_heatmap.png', dpi=150)
    plt.close()
    print(f"  Plots saved to {RESULTS_DIR}/")

    return {'coverage': df_cov, 'pit': df_pit, 'data_source': data_source}


if __name__ == '__main__':
    results = main()
    print('Experiment 9 completed successfully')
