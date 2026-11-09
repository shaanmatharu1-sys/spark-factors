import numpy as np
import pandas as pd
import pytest

import spark_factors as sf
from spark_factors import version
from spark_factors.panel import partial_session_bar

ET = "America/New_York"


def _intraday(day="2023-08-14"):
    rows = []
    for tk, base in (("AAA", 100.0), ("SPY", 400.0)):
        for i, hh in enumerate(range(8, 17)):            # 08:00 .. 16:00 starts
            t = pd.Timestamp(f"{day} {hh:02d}:00", tz=ET)
            o = base + i
            rows.append(dict(ticker=tk, t=t, open=o, high=o + 2, low=o - 1, close=o + 1,
                             volume=1000.0 * (i + 1), vwap=o + 0.5))
    return pd.DataFrame(rows)


def test_partial_bar_uses_only_closed_session_bars():
    intr = _intraday()
    asof = pd.Timestamp("2023-08-14 11:30", tz=ET)
    bar = partial_session_bar(intr, asof).loc["AAA"]
    # session bars start 09:00; closed by 11:30 -> starts 09:00 and 10:00 (11:00 ends 12:00)
    assert bar["open"] == 101.0 and bar["close"] == 103.0
    assert bar["high"] == 104.0 and bar["low"] == 100.0
    assert bar["volume"] == 2000.0 + 3000.0
    np.testing.assert_allclose(bar["vwap"], (101.5 * 2000 + 102.5 * 3000) / 5000)


def test_partial_bar_excludes_premarket_8am_and_post_close():
    intr = _intraday()
    bar = partial_session_bar(intr, pd.Timestamp("2023-08-14 18:00", tz=ET)).loc["AAA"]
    assert bar["open"] == 101.0          # 08:00 bar excluded
    assert bar["close"] == 108.0         # last = 15:00 bar; 16:00 bar excluded


def test_assemble_asof_replaces_current_session(panel):
    day = panel.index[-1]
    intr = _intraday(day.strftime("%Y-%m-%d"))
    asof = pd.Timestamp(f"{day.date()} 12:05", tz=ET)
    p = sf.assemble_asof(panel, intr, asof, lookback=sf.MAX_LOOKBACK)
    assert len(p.index) == sf.MAX_LOOKBACK and p.index[-1] == day
    assert p.close.loc[day, "AAA"] == 104.0                       # from intraday, not daily
    assert np.isnan(p.close.loc[day, "BBB"])                      # no intraday bars -> NaN
    assert p.close.index[-2] == panel.index[-2]                   # prior sessions from daily


def test_sentiment_point_in_time_and_relevance():
    news = pd.DataFrame({
        "created_at": [pd.Timestamp("2023-08-14 10:00", tz=ET), pd.Timestamp("2023-08-14 15:00", tz=ET),
                       pd.Timestamp("2023-08-14 10:30", tz=ET)],
        "symbols": [["NVDA"], ["NVDA"], ["NVDA", "AMD", "INTC", "MU"]],
        "headline": ["Nvidia shares surge on record results", "Nvidia stock plunges", "Chip stocks rally"],
        "summary": ["", "", ""],
    })
    scored = sf.score_articles(news, ["NVDA", "AMD"])
    asofs = pd.DatetimeIndex([pd.Timestamp("2023-08-14 12:00", tz=ET)])
    from spark_factors.families.sentiment import sentiment_features
    f = sentiment_features(scored, ["NVDA", "AMD"], asofs)
    # the 15:00 bearish article is in the future at 12:00 -> excluded
    assert f["sent_1d"].loc[asofs[0], "NVDA"] > 0
    # sector sweep not naming AMD, tagged to 4 names -> relevance 0.3
    w_amd = scored[scored.ticker == "AMD"]["weight"].iloc[0]
    assert w_amd == pytest.approx(0.3)


def test_cross_sectional_z_neutral_for_missing_and_excludes_benchmark():
    raw = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [np.nan], "SPY": [9.0]})
    z = sf.cross_sectional_z(raw)
    assert "SPY" not in z.columns and z.loc[0, "C"] == 0.0
    assert z.loc[0, "A"] == -z.loc[0, "B"] != 0


def test_composite_requires_every_factor():
    scores = pd.DataFrame(0.0, index=["A"], columns=list(sf.FACTOR_NAMES))
    w = {f: 0.0 for f in sf.FACTOR_NAMES}
    w["mom_1m"] = 1.0
    assert sf.composite(scores, w).loc["A"] == 0.0
    w.pop("rev_5d")
    with pytest.raises(ValueError, match="missing"):
        sf.composite(scores, w)


def test_hash_is_line_ending_independent(tmp_path, monkeypatch):
    lf = tmp_path / "x.py"
    lf.write_bytes(b"a = 1\nb = 2\n")
    crlf = tmp_path / "y.py"
    crlf.write_bytes(b"a = 1\r\nb = 2\r\n")
    assert version._file_bytes(lf) == version._file_bytes(crlf)


def test_hash_shape_and_stability():
    h = sf.LIBRARY_VERSION_HASH
    assert h.startswith("sha256:") and len(h) == 7 + 64
    assert version.compute_hash() == h


def test_sentiment_mixed_timestamp_resolutions():
    """pandas >= 3 keeps per-input resolution; s-resolution news vs ns as-ofs must still match."""
    from spark_factors.families.sentiment import sentiment_features
    news = pd.DataFrame({"created_at": pd.to_datetime(["2023-08-14T14:00:00Z"]).as_unit("s"),
                         "symbols": [["NVDA"]], "headline": ["Nvidia shares surge"], "summary": [""]})
    scored = sf.score_articles(news, ["NVDA"])
    asofs = pd.DatetimeIndex([pd.Timestamp("2023-08-14 16:00", tz=ET)]).as_unit("ns")
    f = sentiment_features(scored, ["NVDA"], asofs)
    assert f["sent_1d"].iloc[0, 0] > 0 and f["news_count_7d"].iloc[0, 0] > 0


def test_every_factor_is_live_on_the_golden_fixture():
    from spark_factors.selftest import compute_fixture_outputs
    out = compute_fixture_outputs()
    dead = [k for k, v in out.items() if k.startswith("raw__") and np.isnan(v[-1]).all()]
    assert not dead, f"factors with no value on the fixture: {dead}"
