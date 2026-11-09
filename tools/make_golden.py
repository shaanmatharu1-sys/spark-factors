"""Regenerate the self-test fixture and golden outputs. Run from the repo root:

    python tools/make_golden.py            # golden only (fixture kept)
    python tools/make_golden.py --fixture  # also rebuild the input fixture

Only after an intentional formula change. Commit the result with that change.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from spark_factors.panel import FIELDS  # noqa: E402
from spark_factors.selftest import FIX, compute_fixture_outputs  # noqa: E402


def build_fixture():
    from conftest import make_panel

    p = make_panel(n_rows=300, seed=11)
    np.savez_compressed(FIX / "bars.npz",
                        dates=p.index.to_numpy().astype("datetime64[ns]").astype("int64"),
                        tickers=np.array(list(p.tickers)),
                        **{f: getattr(p, f).to_numpy(dtype=float) for f in FIELDS})
    last = p.index[-1]
    days = [str(d.date()) for d in p.index[-35:]]
    heads = ["{n} shares surge after earnings beat", "{n} stock plunges on weak guidance",
             "{n} wins major contract", "Analysts cut {n} rating", "{n} rally continues",
             "Sector stocks tumble amid selloff"]
    news, k = [], 0
    for i, d in enumerate(days):
        for j, tk in enumerate(["AAA", "BBB", "CCC", "DDD"]):
            if (i + j) % 3:
                continue
            hh = 9 + (i * 7 + j * 3) % 9
            syms = [tk] if k % 4 else ["AAA", "BBB", "CCC", "DDD"]
            news.append({"created_at": f"{d}T{hh:02d}:15:00-04:00", "symbols": syms,
                         "headline": heads[k % len(heads)].format(n=tk), "summary": ""})
            k += 1
    (FIX / "news.json").write_text(json.dumps(news, indent=1), encoding="utf-8", newline="\n")
    print(f"fixture: {len(p.index)} rows x {len(p.tickers)} tickers, {len(news)} articles, last {last.date()}")


if __name__ == "__main__":
    FIX.mkdir(exist_ok=True)
    if "--fixture" in sys.argv or not (FIX / "bars.npz").exists():
        build_fixture()
    out = compute_fixture_outputs()
    dead = [k for k, v in out.items() if k.startswith("raw__") and np.isnan(v[-1]).all()]
    if dead:
        sys.exit(f"refusing to write golden: factors with no value on the fixture: {dead}")
    np.savez_compressed(FIX / "golden.npz", **out)
    print(f"golden: {len(out)} arrays written")
