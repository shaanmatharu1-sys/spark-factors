"""BarPanel and the as-of assembly that gives research and live ONE code path.

A factor value is defined at an as-of timestamp t:
    * every session BEFORE t's session comes from daily bars;
    * t's own session is a PARTIAL bar aggregated from intraday bars that have
      fully closed by t (start + bar_minutes <= t), per BAR_INPUT_SPEC.

The engine calls assemble_asof(daily, intraday, now). SPARK research calls the
same function with a historical t. Identical inputs -> identical factors; there
is no "research uses the official daily close, live uses a partial bar" skew.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from spark_factors.spec import BAR_INPUT_SPEC

FIELDS = ("open", "high", "low", "close", "volume", "vwap")


@dataclass(frozen=True)
class BarPanel:
    """Wide frames, index = session date (tz-naive, normalized), columns = tickers."""
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    vwap: pd.DataFrame

    def __post_init__(self):
        idx, cols = self.close.index, self.close.columns
        for f in FIELDS:
            df = getattr(self, f)
            if not (df.index.equals(idx) and df.columns.equals(cols)):
                raise ValueError(f"BarPanel.{f} is not aligned with close")
        if not idx.is_monotonic_increasing or idx.has_duplicates:
            raise ValueError("BarPanel index must be strictly increasing session dates")

    @property
    def tickers(self) -> pd.Index:
        return self.close.columns

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    def tail(self, n: int) -> "BarPanel":
        return BarPanel(**{f: getattr(self, f).iloc[-n:] for f in FIELDS})

    def upto(self, session: pd.Timestamp) -> "BarPanel":
        return BarPanel(**{f: getattr(self, f).loc[:session] for f in FIELDS})

    @classmethod
    def from_long(cls, df: pd.DataFrame) -> "BarPanel":
        """Long frame [date, ticker, open, high, low, close, volume, vwap] -> panel."""
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
        wide = {f: d.pivot(index="date", columns="ticker", values=f).sort_index() for f in FIELDS}
        cols = wide["close"].columns
        return cls(**{f: wide[f].reindex(columns=cols).astype(float) for f in FIELDS})


def _hhmm(s: str) -> pd.Timedelta:
    h, m = s.split(":")
    return pd.Timedelta(hours=int(h), minutes=int(m))


def partial_session_bar(intraday: pd.DataFrame, asof: pd.Timestamp, spec=BAR_INPUT_SPEC) -> pd.DataFrame:
    """Per-ticker partial bar for asof's session from closed intraday bars.

    intraday: long frame [ticker, t (bar START, tz-aware), open, high, low, close, volume, vwap].
    Returns frame indexed by ticker with FIELDS; tickers with no closed bar are absent.
    """
    if asof.tzinfo is None:
        raise ValueError("asof must be tz-aware")
    tz = spec.session_tz
    asof_l = asof.tz_convert(tz)
    day = asof_l.normalize()
    start_lo = day + _hhmm(spec.first_bar_start)
    start_hi = day + _hhmm(spec.last_bar_start)
    bar_len = pd.Timedelta(minutes=spec.intraday_bar_minutes)

    t = pd.to_datetime(intraday["t"])
    if t.dt.tz is None:
        raise ValueError("intraday 't' must be tz-aware")
    t = t.dt.tz_convert(tz)
    keep = (t >= start_lo) & (t <= start_hi) & (t + bar_len <= asof_l)
    d = intraday.loc[keep].assign(_t=t[keep]).sort_values(["ticker", "_t"])
    if d.empty:
        return pd.DataFrame(columns=list(FIELDS), dtype=float)
    d = d.assign(_pv=d["vwap"] * d["volume"])
    g = d.groupby("ticker", sort=True)
    vol = g["volume"].sum()
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": vol,
        "vwap": (g["_pv"].sum() / vol.where(vol > 0)),
    })
    return out.astype(float)


def assemble_asof(daily: BarPanel, intraday: pd.DataFrame, asof: pd.Timestamp,
                  lookback: int | None = None, spec=BAR_INPUT_SPEC) -> BarPanel:
    """Completed sessions (< asof's session) from `daily` + partial current session row.

    `lookback` trims to the last N rows INCLUDING the partial one (MAX_LOOKBACK for live).
    """
    session = asof.tz_convert(spec.session_tz).tz_localize(None).normalize()
    hist = BarPanel(**{f: getattr(daily, f).loc[daily.index < session] for f in FIELDS})
    part = partial_session_bar(intraday, asof, spec).reindex(daily.tickers)
    rows = {}
    for f in FIELDS:
        row = pd.DataFrame([part[f].to_numpy(dtype=float)], index=pd.DatetimeIndex([session]),
                           columns=daily.tickers)
        rows[f] = pd.concat([getattr(hist, f), row])
    panel = BarPanel(**rows)
    return panel.tail(lookback) if lookback else panel


def session_close_asof(session: pd.Timestamp, spec=BAR_INPUT_SPEC) -> pd.Timestamp:
    """As-of timestamp for 'after the last bar of this session' (16:00 local)."""
    s = pd.Timestamp(session).normalize()
    end = s + _hhmm(spec.last_bar_start) + pd.Timedelta(minutes=spec.intraday_bar_minutes)
    return end.tz_localize(spec.session_tz)


__all__ = ["FIELDS", "BarPanel", "assemble_asof", "partial_session_bar", "session_close_asof"]
