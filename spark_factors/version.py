"""LIBRARY_VERSION_HASH: identifies the exact factor definitions.

sha256 over
    * every file in the package (sorted by relative path), text files with line
      endings normalised to LF -- a Windows checkout (CRLF) and the Linux container
      (LF) must hash identically;
    * the registry identity (names, families, lookbacks, news flag) in order;
    * BAR_INPUT_SPEC and NEWS_INPUT_SPEC (inputs are part of the definition).
NOT the package version string, which can change without a formula changing or
stay the same when one does. Numeric drift from a different numpy/pandas build
is caught by selftest(), not by this hash.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

PKG = Path(__file__).resolve().parent
TEXT_SUFFIXES = {".py", ".json", ".txt", ".md", ".toml", ".cfg"}


def _file_bytes(p: Path) -> bytes:
    b = p.read_bytes()
    if p.suffix in TEXT_SUFFIXES:
        b = b.replace(b"\r\n", b"\n")
    return b


def package_files() -> list[Path]:
    return sorted((p for p in PKG.rglob("*") if p.is_file()
                   and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}),
                  key=lambda p: p.relative_to(PKG).as_posix())


def compute_hash() -> str:
    from spark_factors.registry import REGISTRY
    from spark_factors.spec import BAR_INPUT_SPEC, NEWS_INPUT_SPEC

    h = hashlib.sha256()
    for p in package_files():
        rel = p.relative_to(PKG).as_posix().encode()
        data = _file_bytes(p)
        h.update(len(rel).to_bytes(4, "big") + rel + len(data).to_bytes(8, "big") + data)
    meta = {"registry": [f.identity() for f in REGISTRY],
            "bars": BAR_INPUT_SPEC.as_dict(), "news": NEWS_INPUT_SPEC.as_dict()}
    h.update(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode())
    return "sha256:" + h.hexdigest()


@lru_cache(maxsize=1)
def library_version_hash() -> str:
    return compute_hash()
