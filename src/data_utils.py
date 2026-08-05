"""
Data loading and preprocessing utilities for the empirical panel study.

Downloads daily OHLC data from Yahoo Finance for 12 equity indices,
computes log returns in percent, demeaning using training mean only,
and OHLC-based variance proxies (Parkinson, Garman-Klass).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
import warnings

# 12 equity index tickers
EQUITY_TICKERS = [
    "^GSPC",    # S&P 500
    "^IXIC",    # NASDAQ Composite
    "^DJI",     # Dow Jones
    "^FTSE",    # FTSE 100
    "^GDAXI",   # DAX
    "^FCHI",    # CAC 40
    "^STOXX50E",# EURO STOXX 50
    "^N225",    # Nikkei 225
    "^HSI",     # Hang Seng
    "^GSPTSE",  # S&P/TSX
    "^AXJO",    # ASX 200
    "^BVSP",    # Bovespa
]

MARKET_LABELS = {
    "^GSPC": "S&P500",
    "^IXIC": "NASDAQ",
    "^DJI": "DJIA",
    "^FTSE": "FTSE100",
    "^GDAXI": "DAX",
    "^FCHI": "CAC40",
    "^STOXX50E": "STOXX50",
    "^N225": "Nikkei225",
    "^HSI": "HangSeng",
    "^GSPTSE": "TSX",
    "^AXJO": "ASX200",
    "^BVSP": "Bovespa",
}

DATA_START = "2000-01-01"
DATA_END = "2024-12-31"
TRAIN_FRAC = 0.70


def download_market_data(
    ticker: str,
    start: str = DATA_START,
    end: str = DATA_END,
) -> Optional[pd.DataFrame]:
    """
    Download daily OHLC data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
    start : str
    end : str

    Returns
    -------
    pd.DataFrame with columns Open, High, Low, Close indexed by Date,
    or None if download fails.
    """
    try:
        import yfinance as yf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return None
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df = df[["Open", "High", "Low", "Close"]].dropna()
        return df
    except Exception as e:
        print(f"Warning: failed to download {ticker}: {e}")
        return None


def compute_log_returns_pct(close: pd.Series) -> pd.Series:
    """
    Compute close-to-close log returns in percent.

    Parameters
    ----------
    close : pd.Series - close prices

    Returns
    -------
    pd.Series - log returns in percent (first observation dropped)
    """
    return 100.0 * np.log(close / close.shift(1)).dropna()


def parkinson_variance(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Parkinson range-based variance proxy:
        sigma^2_P = (1 / (4 * ln(2))) * (ln(H/L))^2

    Parameters
    ----------
    high, low : pd.Series - OHLC high and low prices

    Returns
    -------
    pd.Series : daily Parkinson variance estimate (in return^2 units, scaled by 100^2)
    """
    return (100.0**2 / (4.0 * np.log(2.0))) * (np.log(high / low))**2


def garman_klass_variance(open_: pd.Series, high: pd.Series,
                          low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Garman-Klass range-based variance proxy:
        sigma^2_GK = 0.5*(ln(H/L))^2 - (2*ln(2)-1)*(ln(C/O))^2

    Parameters
    ----------
    open_, high, low, close : pd.Series

    Returns
    -------
    pd.Series : daily Garman-Klass variance estimate (in return^2 units, scaled by 100^2)
    """
    return (100.0**2) * (
        0.5 * (np.log(high / low))**2
        - (2.0 * np.log(2.0) - 1.0) * (np.log(close / open_))**2
    )


def prepare_market_data(
    ticker: str,
    start: str = DATA_START,
    end: str = DATA_END,
    train_frac: float = TRAIN_FRAC,
) -> Optional[dict]:
    """
    Download and preprocess one market's data for the empirical panel study.

    Steps:
    1. Download OHLC
    2. Remove rows with any missing quote fields
    3. Compute log returns in percent
    4. Compute training sample mean and demean using training mean only
    5. Split into training and test samples
    6. Compute Parkinson and Garman-Klass proxies

    Parameters
    ----------
    ticker : str
    start : str
    end : str
    train_frac : float

    Returns
    -------
    dict with keys: returns, returns_train, returns_test, dates_train, dates_test,
                   train_mean, park_var, gk_var, park_test, gk_test, T_train, T_test
    or None if download fails.
    """
    df = download_market_data(ticker, start, end)
    if df is None or len(df) < 100:
        return None

    # Compute log returns
    rets = compute_log_returns_pct(df["Close"])
    df_rets = df.loc[rets.index].copy()
    df_rets["returns"] = rets

    # Remove rows with any missing OHLC fields
    df_rets = df_rets.dropna(subset=["Open", "High", "Low", "Close", "returns"])

    # Split
    T = len(df_rets)
    T_train = int(T * train_frac)
    T_test = T - T_train

    df_train = df_rets.iloc[:T_train]
    df_test = df_rets.iloc[T_train:]

    # Demean using training mean
    train_mean = float(df_train["returns"].mean())
    y_train = (df_train["returns"] - train_mean).values
    y_test = (df_test["returns"] - train_mean).values

    # Variance proxies (not demeaned)
    park_train = parkinson_variance(df_train["High"], df_train["Low"]).values
    gk_train = garman_klass_variance(
        df_train["Open"], df_train["High"], df_train["Low"], df_train["Close"]
    ).values
    park_test = parkinson_variance(df_test["High"], df_test["Low"]).values
    gk_test = garman_klass_variance(
        df_test["Open"], df_test["High"], df_test["Low"], df_test["Close"]
    ).values

    return {
        "ticker": ticker,
        "label": MARKET_LABELS.get(ticker, ticker),
        "returns": df_rets["returns"].values - train_mean,
        "dates": df_rets.index,
        "y_train": y_train,
        "y_test": y_test,
        "dates_train": df_train.index,
        "dates_test": df_test.index,
        "train_mean": train_mean,
        "park_train": park_train,
        "gk_train": gk_train,
        "park_test": park_test,
        "gk_test": gk_test,
        "T_train": T_train,
        "T_test": T_test,
        "T": T,
    }


def estimate_mmd_bandwidth(y_train: np.ndarray, max_subsample: int = 1024) -> float:
    """
    Estimate MMD-RBF bandwidth from training data using median pairwise absolute distance
    on a deterministic evenly-spaced subsample of at most max_subsample observations.

    Parameters
    ----------
    y_train : np.ndarray
    max_subsample : int

    Returns
    -------
    float : estimated bandwidth ell
    """
    n = len(y_train)
    if n > max_subsample:
        # Deterministic evenly-spaced subsample
        idx = np.round(np.linspace(0, n - 1, max_subsample)).astype(int)
        subsample = y_train[idx]
    else:
        subsample = y_train

    # Pairwise absolute distances
    diff = np.abs(subsample[:, None] - subsample[None, :])
    upper = diff[np.triu_indices(len(subsample), k=1)]
    ell = float(np.median(upper))
    return max(ell, 1e-3)  # prevent degenerate bandwidth
