"""Train/serve parity: the engine computes on a MAX_LOOKBACK window, SPARK on years.

For every bar factor:
  * value(last `lookback` rows) == value(full history)       -> finite memory
  * value(last `lookback - 1` rows) is NaN                    -> lookback is exact, not padded
  * changing data OLDER than the window changes nothing       -> no hidden long memory
  * changing FUTURE rows doesn't change the past              -> no look-ahead
"""
import numpy as np
import pytest

from spark_factors.panel import FIELDS, BarPanel
from spark_factors.registry import MAX_LOOKBACK, REGISTRY

BAR_FACTORS = [f for f in REGISTRY if not f.uses_news]
RTOL, ATOL = 1e-9, 1e-12


@pytest.mark.parametrize("spec", BAR_FACTORS, ids=lambda s: s.name)
@pytest.mark.parametrize("end", [300, 360, 419])
def test_window_equals_full_history(panel, spec, end):
    sub = BarPanel(**{f: getattr(panel, f).iloc[: end + 1] for f in FIELDS})
    full = spec.fn(sub).iloc[-1]
    win = spec.fn(sub.tail(spec.lookback)).iloc[-1]
    assert full.notna().all(), f"{spec.name} NaN with full history"
    np.testing.assert_allclose(win.to_numpy(), full.to_numpy(), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("spec", BAR_FACTORS, ids=lambda s: s.name)
def test_lookback_is_exact(panel, spec):
    if spec.lookback == 1:
        return
    short = spec.fn(panel.tail(spec.lookback - 1)).iloc[-1]
    assert short.isna().all(), f"{spec.name}: lookback {spec.lookback} is larger than needed"


@pytest.mark.parametrize("spec", BAR_FACTORS, ids=lambda s: s.name)
def test_older_data_is_irrelevant(panel, spec):
    base = spec.fn(panel).iloc[-1]
    cut = len(panel.index) - spec.lookback
    bumped = {}
    for f in FIELDS:
        df = getattr(panel, f).copy()
        df.iloc[:cut] = df.iloc[:cut] * 1.37
        bumped[f] = df
    np.testing.assert_allclose(spec.fn(BarPanel(**bumped)).iloc[-1].to_numpy(), base.to_numpy(),
                               rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("spec", BAR_FACTORS, ids=lambda s: s.name)
def test_no_lookahead(panel, spec):
    t = 330
    base = spec.fn(panel).iloc[t]
    bumped = {}
    for f in FIELDS:
        df = getattr(panel, f).copy()
        df.iloc[t + 1:] = df.iloc[t + 1:] * 0.5
        bumped[f] = df
    np.testing.assert_allclose(spec.fn(BarPanel(**bumped)).iloc[t].to_numpy(), base.to_numpy(),
                               rtol=RTOL, atol=ATOL)


def test_max_lookback_covers_all():
    assert MAX_LOOKBACK == max(f.lookback for f in REGISTRY) == 253
