"""Short-horizon reversal and oscillators.

No rev_1d: log(C_t/C_t-1) == gap_1d + intraday_ret exactly, and an exactly
collinear factor makes the IC covariance singular. No rev_21d (== -mom_1m) and
no Williams %R (== stoch_k - 100) for the same reason.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spark_factors import kernels as K
from spark_factors.panel import BarPanel


def rev_5d(p: BarPanel) -> pd.DataFrame:
    return -np.log(p.close / p.close.shift(5))


def gap_1d(p: BarPanel) -> pd.DataFrame:
    """Overnight gap: today's open vs yesterday's close."""
    return np.log(p.open / p.close.shift(1))


def intraday_ret(p: BarPanel) -> pd.DataFrame:
    """Open-to-now return of the current (possibly partial) session."""
    return np.log(p.close / p.open)


def rsi_14(p: BarPanel) -> pd.DataFrame:
    d = p.close - p.close.shift(1)
    gain = d.clip(lower=0.0)
    loss = (-d).clip(lower=0.0)
    ag, al = K.wilder_fir(gain, 14), K.wilder_fir(loss, 14)
    rs = ag / al
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.where(al > 0, 100.0).where(ag.notna() & al.notna())


def bb_pctb_20(p: BarPanel) -> pd.DataFrame:
    """Bollinger %B(20, 2): 0 at the lower band, 1 at the upper."""
    m, s = K.sma(p.close, 20), K.rstd(p.close, 20, ddof=0)
    return K.safe_div(p.close - (m - 2.0 * s), 4.0 * s)


def stoch_k_14(p: BarPanel) -> pd.DataFrame:
    lo = p.low.rolling(14, min_periods=14).min()
    hi = p.high.rolling(14, min_periods=14).max()
    return 100.0 * K.safe_div(p.close - lo, hi - lo)


def vwap_dist_20(p: BarPanel) -> pd.DataFrame:
    """Close vs the 20-session volume-weighted average price."""
    pv = (p.vwap * p.volume).rolling(20, min_periods=20).sum()
    v = p.volume.rolling(20, min_periods=20).sum()
    return p.close / K.safe_div(pv, v) - 1.0
