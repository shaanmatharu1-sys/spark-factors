"""Momentum / trend. Raw factor values; sign is NOT imposed -- the optimizer
learns it (signed weights), so e.g. a negative weight on mom_1m is reversal."""
from __future__ import annotations

import numpy as np
import pandas as pd

from spark_factors import kernels as K
from spark_factors.panel import BarPanel


def mom_1m(p: BarPanel) -> pd.DataFrame:
    return np.log(p.close / p.close.shift(21))


def mom_3m(p: BarPanel) -> pd.DataFrame:
    return np.log(p.close / p.close.shift(63))


def mom_6m(p: BarPanel) -> pd.DataFrame:
    return np.log(p.close / p.close.shift(126))


def mom_12_1(p: BarPanel) -> pd.DataFrame:
    """12-month return skipping the most recent month (Jegadeesh-Titman)."""
    return np.log(p.close.shift(21) / p.close.shift(252))


def high_52w_prox(p: BarPanel) -> pd.DataFrame:
    """Close relative to the 252-day high (George-Hwang); 0 at the high, negative below."""
    return p.close / p.high.rolling(252, min_periods=252).max() - 1.0


def ma_50_200(p: BarPanel) -> pd.DataFrame:
    return K.sma(p.close, 50) / K.sma(p.close, 200) - 1.0


def macd_hist(p: BarPanel) -> pd.DataFrame:
    """MACD(12,26,9) histogram, scaled by price so it's comparable across names."""
    macd = K.ema_fir(p.close, 12) - K.ema_fir(p.close, 26)
    return (macd - K.ema_fir(macd, 9)) / p.close


def adx_14(p: BarPanel) -> pd.DataFrame:
    """Average Directional Index: trend STRENGTH (direction-free), 0..100."""
    up = p.high - p.high.shift(1)
    dn = p.low.shift(1) - p.low
    plus_dm = up.where((up > dn) & (up > 0), 0.0).where(up.notna() & dn.notna())
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0).where(up.notna() & dn.notna())
    tr = K.true_range(p.high, p.low, p.close)
    atr = K.wilder_fir(tr, 14)
    plus_di = 100.0 * K.wilder_fir(plus_dm, 14) / atr
    minus_di = 100.0 * K.wilder_fir(minus_dm, 14) / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return K.wilder_fir(dx, 14)


def trend_r2_63(p: BarPanel) -> pd.DataFrame:
    """Signed R^2 of log price on time over 63 days: smooth trends score near +/-1."""
    r = K.rolling_corr_with_time(np.log(p.close), 63)
    return r * r.abs()
