"""Public entry points. Research and live call the SAME functions.

Live (engine), each hourly tick or on a forced recompute:
    panel  = assemble_asof(daily, intraday, now, lookback=MAX_LOOKBACK)
    news   = score_articles(news_last_30d, panel.tickers)
    snap   = compute_snapshot(panel, news, now)          # raw [tickers x factors]
    scores = cross_sectional_z(snap.T, universe).T ...   # via snapshot_scores()
    alpha  = composite(scores, weights)

Research (SPARK), for each historical snapshot time t: the identical calls with t.
"""
from __future__ import annotations

import pandas as pd

from spark_factors.families.sentiment import sentiment_features
from spark_factors.panel import BarPanel, session_close_asof
from spark_factors.registry import FACTOR_NAMES, REGISTRY
from spark_factors.transform import cross_sectional_z


def compute_raw(panel: BarPanel, scored_news: pd.DataFrame | None = None,
                asofs: pd.DatetimeIndex | None = None) -> dict:
    """Raw values of every factor on every panel row. `asofs` (tz-aware, one per row)
    defaults to each session's close; only the sentiment family uses it."""
    if asofs is None:
        asofs = pd.DatetimeIndex([session_close_asof(d) for d in panel.index])
    if len(asofs) != len(panel.index):
        raise ValueError("asofs must have one timestamp per panel row")
    out = {}
    for spec in REGISTRY:
        if not spec.uses_news:
            out[spec.name] = spec.fn(panel)
    sent = sentiment_features(scored_news if scored_news is not None else pd.DataFrame(),
                              panel.tickers, asofs)
    for name, df in sent.items():
        out[name] = df.set_axis(panel.index, axis=0)
    return {n: out[n] for n in FACTOR_NAMES}


def compute_snapshot(panel: BarPanel, scored_news: pd.DataFrame | None, asof: pd.Timestamp) -> pd.DataFrame:
    """Raw factor values at the panel's LAST row (as-of `asof`): [tickers x factors]."""
    asofs = pd.DatetimeIndex([session_close_asof(d) for d in panel.index[:-1]] + [asof])
    raw = compute_raw(panel, scored_news, asofs)
    return pd.DataFrame({n: raw[n].iloc[-1] for n in FACTOR_NAMES}, index=panel.tickers)


def snapshot_scores(snapshot: pd.DataFrame, universe=None) -> pd.DataFrame:
    """Raw snapshot [tickers x factors] -> standardised scores [universe x factors]."""
    return cross_sectional_z(snapshot.T, universe).T
