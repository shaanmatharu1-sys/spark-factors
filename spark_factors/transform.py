"""Cross-sectional standardisation -- part of the factor DEFINITION.

Weights multiply these scores, not the raw values, so research and live must use
the same transform. Per as-of row, over the universe the caller passes (the
benchmark is always excluded):
    rank (average ties) among names with a value -> z-score of the ranks (ddof=0)
    names with no value -> 0 (neutral), so missing data never pushes a name to an extreme.
Ranks make the scores robust to outliers without a separate winsorisation step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spark_factors.spec import BAR_INPUT_SPEC


def cross_sectional_z(raw: pd.DataFrame, universe=None) -> pd.DataFrame:
    cols = [c for c in (universe if universe is not None else raw.columns) if c != BAR_INPUT_SPEC.benchmark]
    x = raw.reindex(columns=cols)
    r = x.rank(axis=1, method="average")
    mu = r.mean(axis=1)
    sd = r.std(axis=1, ddof=0)
    z = r.sub(mu, axis=0).div(sd.where(sd > 0), axis=0)
    return z.fillna(0.0)


def composite(scores: pd.DataFrame, weights: dict) -> pd.Series:
    """scores: [tickers x factors] for one as-of. weights must name EVERY registry factor."""
    from spark_factors.registry import FACTOR_NAMES

    missing = [f for f in FACTOR_NAMES if f not in weights]
    unknown = [f for f in weights if f not in FACTOR_NAMES]
    if missing or unknown:
        raise ValueError(f"weights mismatch: missing={missing} unknown={unknown}")
    w = np.array([float(weights[f]) for f in FACTOR_NAMES])
    if not np.all(np.isfinite(w)):
        raise ValueError("non-finite weight")
    return pd.Series(scores.reindex(columns=list(FACTOR_NAMES)).to_numpy() @ w, index=scores.index)
