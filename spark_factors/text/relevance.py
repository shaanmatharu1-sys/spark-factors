"""How likely a story is ABOUT a ticker (moved from Core Engine core/data/company_names.py).

Difference from the engine original, on purpose: the ticker->name map is a FROZEN
package file (company_names.json, part of LIBRARY_VERSION_HASH). The original
refreshed it from SEC at runtime, which would let research and live score the
same headline differently. Refreshing the map is a library release, not a runtime event.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

NAMES_PATH = Path(__file__).with_name("company_names.json")

SUFFIXES = {
    "inc", "inc.", "incorporated", "corp", "corp.", "corporation", "co",
    "co.", "company", "ltd", "ltd.", "limited", "plc", "llc", "lp", "holdings",
    "holding", "group", "the", "class", "common", "stock", "shares", "trust",
    "international", "industries", "technologies", "technology", "systems",
    "solutions", "enterprises", "partners", "&",
}
STOPWORDS = {
    "us", "usa", "american", "america", "national", "general", "first",
    "global", "united", "new", "north", "south", "east", "west", "capital",
    "financial", "energy", "health", "digital", "advanced", "applied", "public",
    "premier", "select", "core", "prime", "pacific", "atlantic", "central",
}
MIN_TOKEN_LENGTH = 4


@lru_cache(maxsize=1)
def names() -> dict:
    return json.loads(NAMES_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def distinctive_token(ticker: str) -> str:
    name = names().get(str(ticker).upper(), "")
    for w in (w.strip(".,&").lower() for w in re.split(r"[\s/]+", name)):
        if w and w not in SUFFIXES and w not in STOPWORDS and len(w) >= MIN_TOKEN_LENGTH:
            return w
    return ""


def mentions(ticker: str, text: str) -> bool | None:
    token = distinctive_token(ticker)
    if not token:
        return None
    blob = (text or "").lower()
    if token in blob:
        return True
    return bool(re.search(rf"\b{re.escape(str(ticker).lower())}\b", blob))


def relevance(ticker: str, text: str, symbol_count: int = 1) -> float:
    """In [0, 1]; same coarse scale as the engine original."""
    named = mentions(ticker, text)
    if named is True:
        return 1.0
    if named is None:
        return 1.0 if symbol_count <= 2 else 0.5
    return 0.7 if symbol_count <= 2 else 0.3
