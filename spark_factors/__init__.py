"""spark-factors: every SPARK factor defined exactly once.

SPARK imports it for historical research; the engine imports it for live ticks.
Every deployment carries LIBRARY_VERSION_HASH and the engine rejects weights
computed under a different one.
"""
from spark_factors.api import compute_raw, compute_snapshot, snapshot_scores
from spark_factors.families.sentiment import score_articles
from spark_factors.panel import BarPanel, assemble_asof, partial_session_bar, session_close_asof
from spark_factors.registry import BY_NAME, FACTOR_NAMES, MAX_LOOKBACK, REGISTRY, families
from spark_factors.spec import BAR_INPUT_SPEC, NEWS_INPUT_SPEC
from spark_factors.transform import composite, cross_sectional_z
from spark_factors.version import library_version_hash

LIBRARY_VERSION_HASH: str = library_version_hash()

__all__ = [
    "BAR_INPUT_SPEC", "BY_NAME", "BarPanel", "FACTOR_NAMES", "LIBRARY_VERSION_HASH",
    "MAX_LOOKBACK", "NEWS_INPUT_SPEC", "REGISTRY", "assemble_asof", "composite",
    "compute_raw", "compute_snapshot", "cross_sectional_z", "families",
    "library_version_hash", "partial_session_bar", "score_articles", "session_close_asof",
    "snapshot_scores",
]
