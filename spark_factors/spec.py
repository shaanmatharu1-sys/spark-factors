"""Specs: what a factor is, and exactly which input data it is defined on.

A factor's definition is its formula AND its inputs. The same formula on
split-only-adjusted bars and on fully adjusted bars gives different numbers, so
the bar request spec is part of the library and part of LIBRARY_VERSION_HASH.
Both SPARK and the engine must fetch bars exactly as BAR_INPUT_SPEC says.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

FAMILIES = ("momentum", "reversal", "volatility", "volume", "sentiment")


@dataclass(frozen=True)
class BarInputSpec:
    adjustment: str = "all"            # Alpaca Adjustment.ALL: splits AND dividends
    feed: str = "sip"
    daily_timeframe: str = "1Day"      # completed sessions
    intraday_timeframe: str = "1Hour"  # builds the CURRENT session's partial bar
    intraday_bar_minutes: int = 60
    session_tz: str = "America/New_York"
    # First intraday bar (by START time) that belongs to a session. Alpaca's 60-min
    # bars are hour-aligned, so the 09:00 bar carries 09:00-10:00 incl. 30 min of
    # pre-market. Fixed convention on both sides -> identical partial bars.
    first_bar_start: str = "09:00"
    # Last intraday bar start that belongs to the session (15:00 -> ends 16:00).
    last_bar_start: str = "15:00"
    benchmark: str = "SPY"
    fields: tuple = ("open", "high", "low", "close", "volume", "vwap")

    def as_dict(self) -> dict:
        return asdict(self)


BAR_INPUT_SPEC = BarInputSpec()


@dataclass(frozen=True)
class NewsInputSpec:
    source: str = "alpaca-news"
    text: str = "headline + ' ' + summary"
    time_field: str = "created_at"     # UTC; an article counts at asof iff created_at <= asof

    def as_dict(self) -> dict:
        return asdict(self)


NEWS_INPUT_SPEC = NewsInputSpec()


@dataclass(frozen=True)
class FactorSpec:
    name: str
    family: str
    lookback: int                      # bar rows needed INCLUDING the current one
    description: str
    fn: Callable = field(repr=False, compare=False)
    uses_news: bool = False

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"{self.name}: unknown family {self.family!r}")
        if self.lookback < 1:
            raise ValueError(f"{self.name}: lookback must be >= 1")

    def identity(self) -> dict:
        """What goes into the version hash (the fn's source is hashed separately)."""
        return {"name": self.name, "family": self.family, "lookback": self.lookback,
                "uses_news": self.uses_news}
