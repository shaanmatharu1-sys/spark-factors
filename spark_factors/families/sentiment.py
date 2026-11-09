"""News sentiment from Alpaca News, scored with the finance lexicon.

Point-in-time rule: an article counts at as-of t iff created_at <= t. Windows are
(t - W, t]. Each (article, ticker) pair contributes
    score = lexicon.score(headline + " " + summary)
    weight = relevance(ticker, text, n_symbols_tagged)
so a sector sweep tagged to nine names barely moves each of them.

Names with no relevant news in a window get NaN -> neutral 0 after the
cross-sectional transform (absence of news is not negative news).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spark_factors.text import lexicon
from spark_factors.text.relevance import relevance

WINDOWS = {"1d": pd.Timedelta(hours=24), "7d": pd.Timedelta(days=7), "30d": pd.Timedelta(days=30)}


def score_articles(news: pd.DataFrame, tickers) -> pd.DataFrame:
    """news [created_at (tz-aware), symbols (list|str), headline, summary] ->
    long frame [ticker, t (UTC), score, weight], one row per relevant (article, ticker)."""
    cols = ["ticker", "t", "score", "weight"]
    if news is None or len(news) == 0:
        return pd.DataFrame(columns=cols)
    want = {str(t).upper() for t in tickers}
    rows = []
    for created, syms, head, summ in zip(news["created_at"], news["symbols"],
                                         news["headline"], news.get("summary", pd.Series([""] * len(news)))):
        if isinstance(syms, str):
            syms = [s for s in syms.split(",") if s]
        syms = [str(s).upper() for s in (syms or [])]
        hit = [s for s in syms if s in want]
        if not hit:
            continue
        text = f"{head or ''} {summ or ''}".strip()
        sc = float(lexicon.score(text))
        t = pd.Timestamp(created)
        if t.tzinfo is None:
            raise ValueError("news created_at must be tz-aware")
        t = t.tz_convert("UTC")
        for s in hit:
            w = relevance(s, text, len(syms))
            if w > 0:
                rows.append((s, t, sc, w))
    return pd.DataFrame(rows, columns=cols)


def sentiment_features(scored: pd.DataFrame, tickers, asofs: pd.DatetimeIndex) -> dict:
    """Window aggregates at each as-of time. Returns {factor_name: DataFrame[asofs x tickers]}."""
    tickers = list(tickers)
    # Explicit ns: pandas >= 3 infers per-input resolution (s/us/ns), and comparing
    # raw .asi8 integers across resolutions silently matches nothing.
    asofs_utc = pd.DatetimeIndex(asofs).tz_convert("UTC").as_unit("ns")
    a = asofs_utc.asi8
    shape = (len(asofs_utc), len(tickers))
    wsum = {k: np.zeros(shape) for k in WINDOWS}
    ssum = {k: np.zeros(shape) for k in WINDOWS}
    if len(scored):
        for j, tk in enumerate(tickers):
            d = scored[scored["ticker"] == tk]
            if d.empty:
                continue
            d = d.sort_values("t")
            tt = pd.DatetimeIndex(d["t"]).tz_convert("UTC").as_unit("ns").asi8
            cw = np.concatenate([[0.0], np.cumsum(d["weight"].to_numpy())])
            cs = np.concatenate([[0.0], np.cumsum((d["weight"] * d["score"]).to_numpy())])
            hi = np.searchsorted(tt, a, side="right")
            for k, W in WINDOWS.items():
                lo = np.searchsorted(tt, a - W.as_unit("ns").value, side="right")
                wsum[k][:, j] = cw[hi] - cw[lo]
                ssum[k][:, j] = cs[hi] - cs[lo]

    def frame(x):
        return pd.DataFrame(x, index=asofs, columns=tickers)

    mean = {k: frame(np.where(wsum[k] > 1e-12, ssum[k] / np.where(wsum[k] > 1e-12, wsum[k], 1.0), np.nan))
            for k in WINDOWS}
    return {
        "sent_1d": mean["1d"],
        "sent_7d": mean["7d"],
        "news_count_7d": frame(np.log1p(wsum["7d"])),
        "sent_shock_30d": mean["1d"] - mean["30d"],
    }
