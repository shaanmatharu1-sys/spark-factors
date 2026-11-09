import numpy as np
import pandas as pd
import pytest

from spark_factors.panel import BarPanel


def make_panel(n_rows=420, tickers=("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "SPY"), seed=7) -> BarPanel:
    """Synthetic but internally consistent OHLCV (test fixture only)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_rows)
    out = {f: {} for f in ("open", "high", "low", "close", "volume", "vwap")}
    mkt = rng.normal(0.0003, 0.011, n_rows)
    for i, t in enumerate(tickers):
        beta = 0.6 + 0.15 * i
        r = beta * mkt + rng.normal(0, 0.012 + 0.002 * i, n_rows)
        if t == "SPY":
            r = mkt
        close = 50.0 * (1 + i) * np.exp(np.cumsum(r))
        prev = np.concatenate([[close[0]], close[:-1]])
        open_ = prev * np.exp(rng.normal(0, 0.004, n_rows))
        hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.006, n_rows)))
        lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.006, n_rows)))
        vol = np.exp(rng.normal(14 + 0.3 * i, 0.4, n_rows))
        vw = lo + (hi - lo) * rng.uniform(0.3, 0.7, n_rows)
        for f, v in zip(out, (open_, hi, lo, close, vol, vw)):
            out[f][t] = v
    return BarPanel(**{f: pd.DataFrame(v, index=idx) for f, v in out.items()})


@pytest.fixture(scope="session")
def panel():
    return make_panel()
