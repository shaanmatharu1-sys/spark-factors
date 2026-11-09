"""Finite-memory building blocks.

THE RULE: a factor's value at row t depends only on rows t-lookback+1 .. t.
Textbook EMA, Wilder smoothing (RSI/ADX/ATR) and cumulative OBV have INFINITE
memory -- the value depends on every bar since the series began -- so the engine
(a ~260-row window) and SPARK (10 years) would compute different numbers for the
"same" factor. Every recursive smoother here is therefore a truncated FIR kernel
of fixed length with weights renormalised to sum to 1, and `lookback` declares
exactly how far back each factor reaches. tests/test_parity.py proves it.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Kernel length: the smallest n whose truncated tail weight (1-a)^n is below
# TRUNCATION_TOL. Sized by dropped weight, not by a fixed multiple of the span:
# Wilder smoothing (a = 1/n) decays far slower than an EMA of the same period, and
# a flat 5x rule left ~0.6% of RSI's weight outside the window (0.25 RSI points).
TRUNCATION_TOL = 1e-4


def _kernel_len(alpha: float) -> int:
    return math.ceil(math.log(TRUNCATION_TOL) / math.log(1.0 - alpha))


def ema_len(span: int) -> int:
    return _kernel_len(2.0 / (span + 1.0))


def wilder_len(period: int) -> int:
    return _kernel_len(1.0 / period)


def _fir(df: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """out[t] = sum_k w[k] * x[t-k], k = 0..n-1; NaN if any input in the window is NaN
    or fewer than n rows exist."""
    n = len(weights)
    x = df.to_numpy(dtype=float)
    T = x.shape[0]
    out = np.full_like(x, np.nan)
    if T >= n:
        for j in range(x.shape[1]):
            out[n - 1:, j] = np.convolve(x[:, j], weights, mode="valid")
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def ema_fir(df: pd.DataFrame, span: int) -> pd.DataFrame:
    alpha = 2.0 / (span + 1.0)
    k = np.arange(ema_len(span))
    w = alpha * (1.0 - alpha) ** k
    return _fir(df, w / w.sum())


def wilder_fir(df: pd.DataFrame, period: int) -> pd.DataFrame:
    alpha = 1.0 / period
    k = np.arange(wilder_len(period))
    w = alpha * (1.0 - alpha) ** k
    return _fir(df, w / w.sum())


def sma(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).mean()


def rstd(df: pd.DataFrame, n: int, ddof: int = 1) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).std(ddof=ddof)


def log_ret(close: pd.DataFrame) -> pd.DataFrame:
    return np.log(close / close.shift(1))


def true_range(high, low, close) -> pd.DataFrame:
    """max(H-L, |H-C_prev|, |L-C_prev|); NaN when the previous close is missing."""
    pc = close.shift(1)
    a = (high - low).to_numpy(dtype=float)
    b = (high - pc).abs().to_numpy(dtype=float)
    c = (low - pc).abs().to_numpy(dtype=float)
    return pd.DataFrame(np.maximum(np.maximum(a, b), c), index=close.index, columns=close.columns)


def rolling_corr_with_time(y: pd.DataFrame, n: int) -> pd.DataFrame:
    """Pearson corr of each column with time (0..n-1) over a trailing n-window."""
    t = pd.DataFrame(np.broadcast_to(np.arange(len(y), dtype=float)[:, None], y.shape),
                     index=y.index, columns=y.columns)
    return y.rolling(n, min_periods=n).corr(t)


def rolling_slope_with_time(y: pd.DataFrame, n: int) -> pd.DataFrame:
    """OLS slope of each column on time over a trailing n-window (translation-invariant in y)."""
    t = pd.DataFrame(np.broadcast_to(np.arange(len(y), dtype=float)[:, None], y.shape),
                     index=y.index, columns=y.columns)
    cov = y.rolling(n, min_periods=n).cov(t, ddof=0)
    var_t = (n * n - 1) / 12.0                    # population variance of 0..n-1
    return cov / var_t


def safe_div(a, b):
    return a / b.where(b != 0)
