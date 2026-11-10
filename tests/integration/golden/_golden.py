"""Shared helpers for the 042 golden-fixture tests.

The golden tests freeze the behavior of HoloDeck's retrieval tools, text
chunker, file processor and NLP metrics before the Semantic Kernel, ``evaluate``
and ``markitdown[all]`` dependencies are replaced. They only use HoloDeck's own
public surfaces, so the same tests must pass before and after the swap.

Regenerating expected values
----------------------------
Expected values live in JSON files under ``tests/fixtures/golden/``. To rewrite
them from the current code, run the tests with ``HOLODECK_UPDATE_GOLDEN=1``::

    HOLODECK_UPDATE_GOLDEN=1 uv run pytest tests/integration/golden -n auto -q

In update mode a test writes its file and passes. Review the diff of the
fixture files before committing: a changed golden is a behavior change.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

GOLDEN_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "golden"
RETRIEVAL_ROOT = GOLDEN_ROOT / "retrieval"

QDRANT_URL = os.environ.get("HOLODECK_GOLDEN_QDRANT_URL", "http://127.0.0.1:6333")
CHROMA_URL = os.environ.get("HOLODECK_GOLDEN_CHROMA_URL", "http://127.0.0.1:8765")


def update_mode() -> bool:
    """Return True when the run should rewrite golden files."""
    return os.environ.get("HOLODECK_UPDATE_GOLDEN", "") == "1"


# --------------------------------------------------------------------------
# Golden comparison
# --------------------------------------------------------------------------


def _diff(
    actual: Any, expected: Any, path: str, abs_tol: float, out: list[str]
) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            sub = f"{path}.{key}" if path else str(key)
            if key not in actual:
                out.append(f"{sub}: missing in actual")
            elif key not in expected:
                out.append(f"{sub}: unexpected in actual")
            else:
                _diff(actual[key], expected[key], sub, abs_tol, out)
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            out.append(f"{path}: length {len(actual)} != expected {len(expected)}")
        for idx, (a, e) in enumerate(zip(actual, expected, strict=False)):
            _diff(a, e, f"{path}[{idx}]", abs_tol, out)
    elif (
        isinstance(expected, float)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        if not math.isfinite(float(actual)) or abs(float(actual) - expected) > abs_tol:
            out.append(f"{path}: {actual!r} != expected {expected!r} (tol {abs_tol})")
    elif actual != expected or type(actual) is not type(expected):
        out.append(f"{path}: {actual!r} != expected {expected!r}")


def assert_golden(
    path: Path, actual: Any, *, abs_tol: float = 0.0, label: str = ""
) -> None:
    """Compare *actual* with the JSON file at *path*, or rewrite it in update mode.

    Floats are compared with ``abs_tol``; everything else must be equal.
    """
    text = json.dumps(actual, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    if update_mode():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return
    if not path.exists():
        pytest.fail(
            f"Golden file missing: {path}. "
            "Run with HOLODECK_UPDATE_GOLDEN=1 to create it."
        )
    expected = json.loads(path.read_text(encoding="utf-8"))
    # Round-trip the actual value so tuples/ints serialize like the stored file.
    normalized = json.loads(text)
    problems: list[str] = []
    _diff(normalized, expected, "", abs_tol, problems)
    if problems:
        shown = "\n  ".join(problems[:40])
        more = f"\n  ... and {len(problems) - 40} more" if len(problems) > 40 else ""
        pytest.fail(f"Golden mismatch {label or path.name}:\n  {shown}{more}")


def sha12(text: str) -> str:
    """Short, stable content fingerprint used instead of storing long text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------
# Deterministic offline embedding
# --------------------------------------------------------------------------

_STOPWORDS = frozenset(
    {
        *("a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has"),
        *("have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this"),
        *("to", "was", "were", "what", "when", "which", "who", "with", "does"),
        *("do", "how"),
    }
)
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class HashEmbeddingService:
    """Offline embedding: hashed bag of tokens, L2-normalised.

    Implements the ``generate_embeddings`` surface HoloDeck's tools call on an
    injected embedding service. Each non-stopword token adds weight 1 to one of
    ``BUCKETS`` buckets chosen by SHA-256, so cosine similarity measures lexical
    overlap. The vector is zero-padded to ``dimensions``, which lets tests use
    different dimensions (and so different collection names) while every
    provider sees the same cosine geometry. Text with no tokens (for example
    ``"?!?"``) hashes the raw text into several buckets, so the vector is never
    zero.
    """

    BUCKETS = 256
    EMPTY_TEXT_BUCKETS = 16
    FLOOR = 0.02

    def __init__(self, dimensions: int) -> None:
        if dimensions < self.BUCKETS:
            raise ValueError(f"dimensions must be at least {self.BUCKETS}")
        self.dimensions = dimensions

    def _vector(self, text: str) -> list[float]:
        tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]
        if not tokens:
            # Token-less text (for example "?!?") spreads over several buckets so
            # that many chunks score above zero and top-k has no zero-score ties.
            tokens = [f"{text}#{n}" for n in range(self.EMPTY_TEXT_BUCKETS)]
        # A small floor on every bucket keeps scores non-zero and distinct, so
        # no result depends on how a store orders exact ties.
        vec = [self.FLOOR] * self.BUCKETS + [0.0] * (self.dimensions - self.BUCKETS)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vec[int.from_bytes(digest[:4], "big") % self.BUCKETS] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec]

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


# --------------------------------------------------------------------------
# Docker service reachability
# --------------------------------------------------------------------------


def qdrant_reachable() -> bool:
    try:
        return httpx.get(f"{QDRANT_URL}/readyz", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


def chroma_reachable() -> bool:
    try:
        return (
            httpx.get(f"{CHROMA_URL}/api/v2/heartbeat", timeout=2.0).status_code == 200
        )
    except httpx.HTTPError:
        return False


def delete_remote_collections(provider: str, names: list[str]) -> None:
    """Best-effort removal of collections created by a test."""
    if provider == "qdrant":
        for name in names:
            with contextlib.suppress(httpx.HTTPError):
                httpx.delete(f"{QDRANT_URL}/collections/{name}", timeout=10.0)
    elif provider == "chromadb":
        import chromadb

        host, _, port = CHROMA_URL.split("://", 1)[1].partition(":")
        with contextlib.suppress(Exception):  # cleanup must not mask a result
            client = chromadb.HttpClient(host=host, port=int(port or 8000))
            for name in names:
                with contextlib.suppress(Exception):
                    client.delete_collection(name)


def forget_manifest_entries(provider: str, names: list[str]) -> None:
    """Best-effort removal of the ``holodeck_manifest`` entries of *names*.

    The retrieval tools record each collection in a companion manifest
    collection (042 T15). Deleting a test collection leaves its entries; this
    removes them so runs do not accumulate state. The async manifest API runs
    on its own thread, so this works inside or outside an event loop.
    """
    import asyncio
    import threading

    from holodeck.lib.retrieval.manifest import ManifestStore

    url = {"qdrant": QDRANT_URL, "chromadb": CHROMA_URL}.get(provider)
    if url is None:
        return

    async def forget() -> None:
        store = ManifestStore.open(provider, {"connection_string": url})
        if store is None:
            return
        try:
            for name in names:
                await store.delete_collection_entry(name)
                await store.delete_sources(name)
        finally:
            await store.close()

    def run() -> None:
        with contextlib.suppress(Exception):  # cleanup must not mask a result
            asyncio.run(forget())

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(timeout=60)
