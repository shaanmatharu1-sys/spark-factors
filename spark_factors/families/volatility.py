"""Volatility / risk. The benchmark (BAR_INPUT_SPEC.benchmark, SPY) must be a
column of the panel for beta and idiosyncratic vol."""
from __future__ import annotations

import numpy as np
import pandas as pd

from spark_factors import kernels as K
from spark_factors.panel import BarPanel
from spark_factors.spec import BAR_INPUT_SPEC

ANN = np.sqrt(252.0)


def _bench_ret(p: BarPanel) -> pd.DataFrame:
    b = BAR_INPUT_SPEC.benchmark
    if b not in p.tickers:
        raise KeyError(f"panel must include the benchmark column {b!r}")
    r = K.log_ret(p.close[[b]])[b]
    return pd.DataFrame(np.broadcast_to(r.to_numpy()[:, None], p.close.shape),
                        index=p.index, columns=p.tickers)


def vol_21(p: BarPanel) -> pd.DataFrame:
    return K.rstd(K.log_ret(p.close), 21) * ANN


def vol_63(p: BarPanel) -> pd.DataFrame:
    return K.rstd(K.log_ret(p.close), 63) * ANN


def vol_ratio_21_63(p: BarPanel) -> pd.DataFrame:
    """Vol term structure: >1 means recent vol is elevated."""
    r = K.log_ret(p.close)
    return K.safe_div(K.rstd(r, 21), K.rstd(r, 63))


def _beta_parts(p: BarPanel, n: int):
    r, m = K.log_ret(p.close), _bench_ret(p)
    mr, mm = K.sma(r, n), K.sma(m, n)
    cov = K.sma(r * m, n) - mr * mm
    var_m = K.sma(m * m, n) - mm * mm
    var_r = K.sma(r * r, n) - mr * mr
    return cov, var_m, var_r


def beta_126(p: BarPanel) -> pd.DataFrame:
    cov, var_m, _ = _beta_parts(p, 126)
    return K.safe_div(cov, var_m)


def idio_vol_63(p: BarPanel) -> pd.DataFrame:
    """Std of the residual from a 63-day market regression (Ang et al.)."""
    cov, var_m, var_r = _beta_parts(p, 63)
    beta = K.safe_div(cov, var_m)
    resid = (var_r - beta * beta * var_m).clip(lower=0.0)
    return np.sqrt(resid) * ANN


def max_ret_21(p: BarPanel) -> pd.DataFrame:
    """Largest single-day return in 21 days (lottery / MAX effect, Bali et al.)."""
    return K.log_ret(p.close).rolling(21, min_periods=21).max()


def skew_63(p: BarPanel) -> pd.DataFrame:
    return K.log_ret(p.close).rolling(63, min_periods=63).skew()


def downside_vol_63(p: BarPanel) -> pd.DataFrame:
    r = K.log_ret(p.close)
    neg = r.clip(upper=0.0).where(r.notna())
    return np.sqrt(K.sma(neg * neg, 63)) * ANN


def atr_pct_14(p: BarPanel) -> pd.DataFrame:
    return K.wilder_fir(K.true_range(p.high, p.low, p.close), 14) / p.close
