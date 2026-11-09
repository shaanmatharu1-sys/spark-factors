"""Golden self-test: catches numeric drift the version hash cannot see.

The hash proves the FORMULAS match. It can't prove a different numpy/pandas build
(or platform) computes them the same way. selftest() runs every factor on a frozen
fixture (fixtures/bars.npz + news.json) and compares against outputs recorded
when the library was released (fixtures/golden.npz).

The engine calls this at startup and must refuse to start on failure; SPARK runs
it before a research job. Regenerate golden ONLY when formulas change (which
changes the hash anyway): `python tools/make_golden.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FIX = Path(__file__).with_name("fixtures")
RTOL, ATOL = 1e-9, 1e-12
GOLDEN_ROWS = 5


class SelfTestError(RuntimeError):
    pass


def load_fixture():
    from spark_factors.families.sentiment import score_articles
    from spark_factors.panel import FIELDS, BarPanel

    z = np.load(FIX / "bars.npz", allow_pickle=False)
    idx = pd.DatetimeIndex(z["dates"].astype("datetime64[ns]"))
    cols = [str(c) for c in z["tickers"]]
    panel = BarPanel(**{f: pd.DataFrame(z[f], index=idx, columns=cols) for f in FIELDS})
    news = pd.DataFrame(json.loads((FIX / "news.json").read_text(encoding="utf-8")))
    news["created_at"] = pd.to_datetime(news["created_at"], utc=True)
    return panel, score_articles(news, cols)


def compute_fixture_outputs() -> dict:
    from spark_factors.api import compute_raw, snapshot_scores, compute_snapshot
    from spark_factors.panel import session_close_asof
    from spark_factors.registry import FACTOR_NAMES

    panel, scored = load_fixture()
    raw = compute_raw(panel, scored)
    out = {f"raw__{n}": raw[n].iloc[-GOLDEN_ROWS:].to_numpy(dtype=float) for n in FACTOR_NAMES}
    snap = compute_snapshot(panel, scored, session_close_asof(panel.index[-1]))
    out["scores"] = snapshot_scores(snap).to_numpy(dtype=float)
    return out


def selftest() -> dict:
    """Raises SelfTestError on any mismatch; returns a small report on success."""
    golden = np.load(FIX / "golden.npz", allow_pickle=False)
    got = compute_fixture_outputs()
    bad = []
    for k in golden.files:
        a, b = got.get(k), golden[k]
        if a is None or a.shape != b.shape:
            bad.append(f"{k}: shape {None if a is None else a.shape} != {b.shape}")
            continue
        if not np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True):
            diff = np.nanmax(np.abs(a - b))
            bad.append(f"{k}: max abs diff {diff:.3e}")
    extra = set(got) - set(golden.files)
    if extra:
        bad.append(f"outputs without golden values: {sorted(extra)}")
    if bad:
        raise SelfTestError("spark-factors selftest FAILED:\n  " + "\n  ".join(bad))
    return {"ok": True, "checked": len(golden.files), "numpy": np.__version__, "pandas": pd.__version__}
