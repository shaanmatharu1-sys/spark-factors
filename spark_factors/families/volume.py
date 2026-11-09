"""Volume / liquidity.

On a partial (intraday) session the current row's volume is a fraction of a full
day. That is a time-of-day effect common to every name, so it washes out in the
cross-sectional rank -- and research samples the same snapshot times as live, so
the factor is trained on exactly the partial volumes it will be served.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spark_factors import kernels as K
from spark_factors.panel import BarPanel


def volume_z_63(p: BarPanel) -> pd.DataFrame:
    """Today's log volume vs the PRIOR 63 sessions (today excluded from the baseline)."""
    lv = np.log(p.volume.where(p.volume > 0))
    base = lv.shift(1)
    return K.safe_div(lv - K.sma(base, 63), K.rstd(base, 63))


def _obv_slope_weights(n: int) -> np.ndarray:
    """slope(cumsum(sv) over an n-point window) == sum_m sv_m * c_m. Returns c by LAG
    (lag 0 = today). Exact, finite memory: no global running OBV."""
    k = np.arange(n, dtype=float)
    kc = k - k.mean()
    s = (kc * kc).sum()
    # Y_k = sum_{m=1..k} sv_m  ->  coefficient of sv_m is sum_{k>=m} kc_k / s
    c_m = np.array([kc[m:].sum() / s for m in range(1, n)])   # m = 1..n-1
    return c_m[::-1]                                           # lag = (n-1) - m


def obv_slope_21(p: BarPanel) -> pd.DataFrame:
    """Slope of on-balance volume over 21 sessions, in units of average daily volume."""
    sv = p.volume * np.sign(p.close - p.close.shift(1))
    slope = K._fir(sv, _obv_slope_weights(21))
    return K.safe_div(slope, K.sma(p.volume, 20))


def amihud_21(p: BarPanel) -> pd.DataFrame:
    """log of Amihud illiquidity: mean |return| per dollar traded (x1e9)."""
    illiq = K.log_ret(p.close).abs() / (p.close * p.volume).where(p.volume > 0) * 1e9
    return np.log(K.sma(illiq, 21) + 1e-12)


def turnover_trend_5_63(p: BarPanel) -> pd.DataFrame:
    return np.log(K.safe_div(K.sma(p.volume, 5), K.sma(p.volume, 63)))


def updown_vol_21(p: BarPanel) -> pd.DataFrame:
    """log(volume on up days / volume on down days) over 21 sessions."""
    d = p.close - p.close.shift(1)
    up = p.volume.where(d > 0, 0.0).where(d.notna())
    dn = p.volume.where(d < 0, 0.0).where(d.notna())
    return np.log((up.rolling(21, min_periods=21).sum() + 1.0) / (dn.rolling(21, min_periods=21).sum() + 1.0))


def dollar_vol_21(p: BarPanel) -> pd.DataFrame:
    """log average dollar volume (size / liquidity)."""
    return np.log(K.sma(p.close * p.volume, 21).where(lambda x: x > 0))
