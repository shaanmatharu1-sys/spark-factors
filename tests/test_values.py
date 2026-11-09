"""Factor values vs independent reference implementations."""
import numpy as np
import pandas as pd

from spark_factors.families import momentum as M
from spark_factors.families import reversal as R
from spark_factors.families import volatility as V
from spark_factors.families import volume as Q


def test_mom_and_reversal_definitions(panel):
    c = panel.close
    t = -1
    np.testing.assert_allclose(M.mom_12_1(panel).iloc[t], np.log(c.iloc[t - 21] / c.iloc[t - 252]))
    np.testing.assert_allclose(R.rev_5d(panel).iloc[t], -np.log(c.iloc[t] / c.iloc[t - 5]))
    # gap + intraday == close-to-close (why rev_1d is excluded)
    np.testing.assert_allclose((R.gap_1d(panel) + R.intraday_ret(panel)).iloc[t],
                               np.log(c.iloc[t] / c.iloc[t - 1]))


def test_obv_slope_matches_explicit_regression(panel):
    n = 21
    sv = panel.volume * np.sign(panel.close.diff())
    y = sv.iloc[-n:].fillna(0).cumsum()           # OBV inside the window (any offset)
    k = np.arange(n, dtype=float)
    ref = {c: np.polyfit(k, y[c].to_numpy(), 1)[0] for c in y.columns}
    adv = panel.volume.iloc[-20:].mean()
    got = Q.obv_slope_21(panel).iloc[-1]
    for c in y.columns:
        np.testing.assert_allclose(got[c], ref[c] / adv[c], rtol=1e-9)


def test_beta_and_idio_vol_match_ols(panel):
    r = np.log(panel.close / panel.close.shift(1))
    for n, fn in ((126, V.beta_126),):
        x = r["SPY"].iloc[-n:].to_numpy()
        for c in ("AAA", "DDD"):
            y = r[c].iloc[-n:].to_numpy()
            b = np.polyfit(x, y, 1)[0]
            np.testing.assert_allclose(fn(panel)[c].iloc[-1], b, rtol=1e-8)
    n = 63
    x = r["SPY"].iloc[-n:].to_numpy()
    y = r["EEE"].iloc[-n:].to_numpy()
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    np.testing.assert_allclose(V.idio_vol_63(panel)["EEE"].iloc[-1], resid.std(ddof=0) * np.sqrt(252), rtol=1e-8)


def _wilder_full(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def test_fir_rsi_close_to_textbook_wilder(panel):
    d = panel.close.diff()
    ag = _wilder_full(d.clip(lower=0).fillna(0), 14)
    al = _wilder_full((-d).clip(lower=0).fillna(0), 14)
    textbook = 100 - 100 / (1 + ag / al)
    got = R.rsi_14(panel)
    # truncated tail weight < 1e-4: agreement to well under 0.05 RSI points
    assert (got.iloc[-50:] - textbook.iloc[-50:]).abs().max().max() < 0.05


def test_fir_macd_close_to_textbook(panel):
    c = panel.close
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    ref = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    got = M.macd_hist(panel)
    assert (got.iloc[-50:] - ref.iloc[-50:]).abs().max().max() < 1e-4


def test_adx_in_range_and_close_to_textbook(panel):
    got = M.adx_14(panel).iloc[-50:]
    assert ((got >= 0) & (got <= 100)).all().all()
    h, l, c = panel.high, panel.low, panel.close
    up, dn = h.diff(), -l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    mdm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()]).groupby(level=0).max()
    atr = _wilder_full(tr.fillna(0), 14)
    pdi, mdi = 100 * _wilder_full(pdm.fillna(0), 14) / atr, 100 * _wilder_full(mdm.fillna(0), 14) / atr
    ref = _wilder_full((100 * (pdi - mdi).abs() / (pdi + mdi)).fillna(0), 14)
    assert (got - ref.iloc[-50:]).abs().max().max() < 0.5


def test_bounded_oscillators(panel):
    for fn, lo, hi in ((R.stoch_k_14, 0, 100), (R.rsi_14, 0, 100), (M.trend_r2_63, -1, 1)):
        v = fn(panel).iloc[-100:]
        assert ((v >= lo - 1e-9) & (v <= hi + 1e-9)).all().all(), fn.__name__
