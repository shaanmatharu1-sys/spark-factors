# spark-factors

Every SPARK factor, defined exactly once. SPARK imports it for historical
research; Core Engine imports it for live ticks. Each deployment carries
`LIBRARY_VERSION_HASH`; the engine rejects weights computed under another hash.

```python
import spark_factors as sf
panel  = sf.assemble_asof(daily, intraday, asof, lookback=sf.MAX_LOOKBACK)
snap   = sf.compute_snapshot(panel, sf.score_articles(news, panel.tickers), asof)
scores = sf.snapshot_scores(snap, universe)
alpha  = sf.composite(scores, weights)          # weights must name every factor
```

- 35 factors in 5 families (`sf.families()`), `MAX_LOOKBACK` = 253 bar rows.
- Inputs are part of the definition: fetch bars exactly per `sf.BAR_INPUT_SPEC`
  (Alpaca `Adjustment.ALL`, SIP, daily + 1-hour bars for the current session).
- Every factor has finite memory (no infinite EMA/Wilder/OBV); `tests/test_parity.py`
  proves the declared lookback reproduces the full-history value.
- `spark_factors.selftest.selftest()` checks numerics against golden outputs;
  run it at engine startup and before research jobs.

Install pinned: `pip install "spark-factors @ git+https://github.com/<owner>/spark-factors@v0.1.0"`
Tests: `python -m pytest tests -q`. Regenerate golden only after an intentional
formula change: `python tools/make_golden.py`.
