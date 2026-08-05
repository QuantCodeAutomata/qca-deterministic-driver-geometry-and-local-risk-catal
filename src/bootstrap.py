"""
Panel-level stationary bootstrap and Model Confidence Set (MCS) utilities.

Preserves cross-market date dependence through common-date resampling.
Implements Hansen-Lunde-Nason 90% MCS via bootstrap p-values.

# Custom — Context7 found no library equivalent for this paper-specific panel bootstrap
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Geometric stationary bootstrap
# ---------------------------------------------------------------------------

def geometric_block_bootstrap(
    T: int,
    mean_block_length: float,
    B: int,
    seed: int = 20260430,
) -> np.ndarray:
    """
    Generate B bootstrap index arrays using the geometric stationary bootstrap.

    Each bootstrap draw samples a length-T sequence of indices by repeatedly
    drawing geometric block lengths (parameter 1/mean_block_length) and cycling.

    Parameters
    ----------
    T : int
        Length of original time series.
    mean_block_length : float
        Mean block length (p = 1/mean_block_length for geometric distribution).
    B : int
        Number of bootstrap replications.
    seed : int

    Returns
    -------
    np.ndarray, shape (B, T) : index arrays for each bootstrap draw.
    """
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block_length
    indices = np.empty((B, T), dtype=int)

    for b in range(B):
        idx = []
        start = rng.integers(0, T)
        while len(idx) < T:
            block_len = rng.geometric(p)
            for k in range(block_len):
                if len(idx) >= T:
                    break
                idx.append((start + k) % T)
            start = rng.integers(0, T)
        indices[b] = idx[:T]

    return indices


# ---------------------------------------------------------------------------
# Panel bootstrap for equal-market statistics
# ---------------------------------------------------------------------------

def panel_bootstrap_mean(
    market_losses: dict,
    B: int = 800,
    mean_block_length: float = 20,
    seed: int = 20260430,
) -> tuple[np.ndarray, float, float]:
    """
    Compute bootstrap distribution of equal-market mean loss.

    Parameters
    ----------
    market_losses : dict
        {market_name: pd.Series} of test-sample losses indexed by date.
    B : int
        Number of bootstrap replications.
    mean_block_length : float
        Mean block length for stationary bootstrap.
    seed : int

    Returns
    -------
    bootstrap_means : np.ndarray, shape (B,)
    point_mean : float - equal-market average loss
    se : float - bootstrap standard error
    """
    markets = list(market_losses.keys())
    n_markets = len(markets)

    # Align on common date index
    all_dates = sorted(set().union(*[set(market_losses[m].index) for m in markets]))
    T = len(all_dates)
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    # Build matrix (T x n_markets) with NaN for missing dates
    loss_matrix = np.full((T, n_markets), np.nan)
    for j, market in enumerate(markets):
        series = market_losses[market]
        for date, val in series.items():
            if date in date_to_idx:
                loss_matrix[date_to_idx[date], j] = val

    # Bootstrap
    boot_indices = geometric_block_bootstrap(T, mean_block_length, B, seed)
    bootstrap_means = np.empty(B)

    for b in range(B):
        sampled = loss_matrix[boot_indices[b]]  # (T, n_markets)
        # Market-level mean, then equal-market average
        market_means = np.nanmean(sampled, axis=0)  # (n_markets,)
        bootstrap_means[b] = np.mean(market_means)

    # Point estimate
    market_means_point = np.nanmean(loss_matrix, axis=0)
    point_mean = float(np.mean(market_means_point))
    se = float(np.std(bootstrap_means, ddof=1))

    return bootstrap_means, point_mean, se


# ---------------------------------------------------------------------------
# Model Confidence Set (Hansen-Lunde-Nason 90% MCS)
# ---------------------------------------------------------------------------

def mcs_bootstrap(
    loss_matrix: np.ndarray,
    model_names: list,
    alpha: float = 0.10,
    B: int = 800,
    mean_block_length: float = 20,
    seed: int = 20260430,
) -> dict:
    """
    Hansen-Lunde-Nason Model Confidence Set at level (1-alpha).

    Uses equal-market averaged losses per model. Eliminates the worst model
    iteratively until the max bootstrap p-value falls below alpha.

    Parameters
    ----------
    loss_matrix : np.ndarray, shape (T, n_models)
        Equal-market average losses per time period per model.
    model_names : list
        Names corresponding to columns of loss_matrix.
    alpha : float
        Significance level (default 0.10 for 90% MCS).
    B : int
    mean_block_length : float
    seed : int

    Returns
    -------
    dict with keys:
        'mcs_set': list of model names in MCS
        'eliminated': list of eliminated models
        'pvalues': dict {model: terminal p-value}
    """
    T, n_models = loss_matrix.shape
    assert len(model_names) == n_models

    boot_idx = geometric_block_bootstrap(T, mean_block_length, B, seed)
    active = list(range(n_models))
    eliminated = []
    pvalues = {}

    while len(active) > 1:
        # Compute pairwise loss differences relative to mean
        active_losses = loss_matrix[:, active]
        mean_losses = np.mean(active_losses, axis=0)  # (n_active,)

        # T-bar statistic: max studentized mean loss difference
        n_active = len(active)
        d_ij = np.zeros((n_active, n_active))
        se_ij = np.zeros((n_active, n_active))

        for i in range(n_active):
            for j in range(n_active):
                if i != j:
                    diff = active_losses[:, i] - active_losses[:, j]
                    d_ij[i, j] = np.mean(diff)
                    # Bootstrap SE
                    boot_diffs = np.array([np.mean(diff[boot_idx[b]]) for b in range(B)])
                    se_ij[i, j] = np.std(boot_diffs, ddof=1) + 1e-15

        # T_bar = max_i mean_i * (n_active / mean_se)
        mean_d = np.mean(d_ij, axis=1)  # mean loss difference for each model
        mean_se = np.sqrt(np.mean(se_ij**2, axis=1) + 1e-15)
        T_bar_obs = np.max(mean_d / mean_se)

        # Bootstrap distribution of T_bar
        boot_Tbar = np.empty(B)
        for b in range(B):
            boot_active = active_losses[boot_idx[b]]
            boot_mean = np.mean(boot_active, axis=0)
            boot_d_ij = np.zeros((n_active, n_active))
            for i in range(n_active):
                for j in range(n_active):
                    if i != j:
                        boot_d_ij[i, j] = boot_mean[i] - boot_mean[j]
            boot_mean_d = np.mean(boot_d_ij, axis=1)
            boot_Tbar[b] = np.max(boot_mean_d / (mean_se + 1e-15))

        p_val = np.mean(boot_Tbar >= T_bar_obs)

        if p_val >= alpha:
            # All active models survive
            for idx in active:
                if model_names[idx] not in pvalues:
                    pvalues[model_names[idx]] = p_val
            break

        # Eliminate worst model (highest mean loss difference)
        worst_local = np.argmax(mean_d)
        worst_global = active[worst_local]
        pvalues[model_names[worst_global]] = p_val
        eliminated.append(model_names[worst_global])
        active.pop(worst_local)

    mcs_set = [model_names[i] for i in active]
    for m in mcs_set:
        if m not in pvalues:
            pvalues[m] = 1.0

    return {
        "mcs_set": mcs_set,
        "eliminated": eliminated,
        "pvalues": pvalues,
    }
