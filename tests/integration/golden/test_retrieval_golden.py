"""Golden retrieval results for VectorStoreTool and HierarchicalDocumentTool.

Both tools ingest a copy of the small corpus in
``tests/fixtures/golden/retrieval/corpus`` (staged at a fixed ``/tmp`` path, see
``STAGE_ROOT``) with the offline ``HashEmbeddingService`` and answer a fixed set
of queries. The ordered results (source, section, content fingerprint, score)
are frozen in
``tests/fixtures/golden/retrieval/expected/<tool>.<provider>.json``.

Providers: ``in-memory`` always runs. ``qdrant`` and ``chromadb`` run against
the Docker containers the live verify suite uses (Qdrant on 6333, Chroma on
8765) and skip when the port does not answer. Override the URLs with
``HOLODECK_GOLDEN_QDRANT_URL`` and ``HOLODECK_GOLDEN_CHROMA_URL``.

Record IDs are not frozen: they embed the source path (and are UUIDv5 over it on
Qdrant). Results are identified by the relative source path, the section ID and
a SHA-256 prefix of the chunk text.

Tie handling: results whose rounded scores are equal are stored sorted by chunk
fingerprint, because stores do not promise an order among ties. Orders between
different scores are frozen.

Regenerate with ``HOLODECK_UPDATE_GOLDEN=1`` (see ``_golden.py``).
"""

from __future__ import annotations

import fcntl
import os
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

import pytest
from pydantic import SecretStr

from holodeck.models.tool import (
    DatabaseConfig,
    DocumentDomain,
    HierarchicalDocumentToolConfig,
    SearchMode,
)
from holodeck.models.tool import VectorstoreTool as VectorstoreToolConfig
from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool
from holodeck.tools.vectorstore_tool import VectorStoreTool

from ._golden import (
    CHROMA_URL,
    QDRANT_URL,
    RETRIEVAL_ROOT,
    HashEmbeddingService,
    assert_golden,
    chroma_reachable,
    delete_remote_collections,
    qdrant_reachable,
    sha12,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.name != "posix", reason="goldens assume a POSIX /tmp path"),
]

CORPUS_SRC = RETRIEVAL_ROOT / "corpus"
# Record IDs embed the absolute source path (UUIDv5 over it on Qdrant), and
# Qdrant's full-text hybrid leg returns matches in point-ID order. The corpus
# is therefore copied to a fixed absolute path per test so results do not
# depend on where the repository is checked out.
STAGE_ROOT = Path("/tmp/holodeck-golden-retrieval")  # noqa: S108
EXPECTED_DIR = RETRIEVAL_ROOT / "expected"

# One entry per query class. Text is chosen so each class has a plausible hit
# in the corpus; "none" and "punct" are the no-match and punctuation-only cases.
QUERIES: dict[str, str] = {
    # Single rare tokens: each matches exactly one chunk's text, so Qdrant's
    # full-text leg has one candidate and its hybrid result is deterministic.
    "definition": "Affiliate",
    "heading": "Timelines",
    "section_id": "503",
    "contextual_text": "accuracy signed corporate",
    "plain_text": "kestrel dawn",
    # Multi-token phrases: several chunks match, so the Qdrant hybrid leg is
    # unordered (see UNSTABLE_QDRANT_HYBRID); the other modes are stable.
    "definition_phrase": (
        '"Force Majeure" means any event beyond the reasonable control'
    ),
    "heading_phrase": "Network Security",
    "section_id_phrase": "Section 202",
    "contextual_phrase": (
        "submit an annual emissions report within ninety days following the end "
        "of each Reporting Period"
    ),
    # No match and punctuation-only (no tokens at all).
    "none": "zzqx plugh xyzzy",
    "punctuation_only": "?!?",
}
UNSTABLE_QDRANT_HYBRID = frozenset(
    {"definition_phrase", "heading_phrase", "section_id_phrase", "contextual_phrase"}
)

HIER_DIMENSIONS = 768  # provider_type="ollama" resolves to 768 dimensions
HIER_TOP_K = 5
HIER_MIN_SCORE = 0.3
VEC_TOP_K = 4
VEC_MIN_SCORE = 0.3
VEC_CHUNK_SIZE = 128
# Distinct dimensions give distinct "documents_dim<N>" collection names, so
# parallel workers and re-runs never share a collection.
VEC_DIMENSIONS = {"in-memory": 301, "qdrant": 303, "chromadb": 305}

HIER_SCORE_TOL = 5e-4
VEC_SCORE_TOL = 0.011  # VectorStoreTool prints scores with two decimals

PROVIDERS = [
    "in-memory",
    pytest.param(
        "qdrant",
        marks=pytest.mark.skipif(
            not qdrant_reachable(), reason=f"Qdrant not reachable at {QDRANT_URL}"
        ),
    ),
    pytest.param(
        "chromadb",
        marks=pytest.mark.skipif(
            not chroma_reachable(), reason=f"Chroma not reachable at {CHROMA_URL}"
        ),
    ),
]


# Behaviors the goldens freeze that look like defects. They are recorded in the
# expected files so a later change that fixes them shows up as a reviewed diff.
_CHROMA_NOTE = (
    "Chroma returns cosine distance. Until 042 T4 HoloDeck used that distance "
    "as the score and sorted it descending, so semantic results came back "
    "worst-first and min_score kept results whose distance was at least the "
    "threshold. The adapter now returns similarity (1 - distance), so the best "
    "match ranks first and min_score filters by similarity."
)
_QDRANT_KEYWORD_NOTE = (
    "search_mode=keyword returns no results on Qdrant: the in-process BM25 "
    "index is skipped for native-hybrid providers."
)
_QDRANT_HYBRID_NOTE = (
    "Qdrant hybrid scores are raw RRF values from the server (1/(k+rank) sums), "
    "not normalised to the best result, unlike the in-process fallback."
)
_QDRANT_UNORDERED_NOTE = (
    "Qdrant's full-text hybrid leg is an unscored filter, so it returns matches "
    "in internal segment order, which changes between runs. When the query "
    "matches more than one chunk (or more than max(4*top_k, 20)), the fused "
    "result is not reproducible. Those queries are listed in excluded_queries "
    "and are not frozen; the single-match queries are."
)
KNOWN_BEHAVIOR: dict[tuple[str, str], list[str]] = {
    ("hierarchical_document", "chromadb"): [_CHROMA_NOTE],
    ("vectorstore", "chromadb"): [_CHROMA_NOTE],
    ("hierarchical_document", "qdrant"): [
        _QDRANT_KEYWORD_NOTE,
        _QDRANT_HYBRID_NOTE,
        _QDRANT_UNORDERED_NOTE,
    ],
}


def _database(provider: str) -> DatabaseConfig | None:
    if provider == "qdrant":
        return DatabaseConfig(
            provider="qdrant", connection_string=SecretStr(QDRANT_URL)
        )
    if provider == "chromadb":
        return DatabaseConfig(
            provider="chromadb", connection_string=SecretStr(CHROMA_URL)
        )
    return None


def _rel(source_path: str, root: Path) -> str:
    """Corpus-relative source path, so goldens do not embed the staging path."""
    path = Path(source_path)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _canonical_ties(
    rows: list[dict[str, Any]], *, only_zero: bool = False
) -> list[dict[str, Any]]:
    """Sort rows with equal scores by fingerprint; keep other order as returned."""
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(rows):
        j = i
        while j < len(rows) and rows[j]["score"] == rows[i]["score"]:
            j += 1
        group = rows[i:j]
        if len(group) > 1 and (not only_zero or rows[i]["score"] == 0.0):
            group = sorted(group, key=lambda r: r["chunk_sha"])
        out.extend(group)
        i = j
    for rank, row in enumerate(out, start=1):
        row["rank"] = rank
    return out


class Cleanup:
    """Remote collections and staged directories to remove after a test."""

    def __init__(self) -> None:
        self.collections: list[tuple[str, str]] = []
        self.dirs: list[Path] = []
        self._locks: list[IO[str]] = []

    def _lock(self, name: str) -> None:
        """Hold a lock on *name* until teardown.

        The staged path and the collection names are fixed, because record
        IDs depend on them. Two runs on one machine (other worktrees, xdist
        workers) must not share them at the same time.
        """
        STAGE_ROOT.mkdir(parents=True, exist_ok=True)
        handle = (STAGE_ROOT / f"{name}.lock").open("w")
        fcntl.flock(handle, fcntl.LOCK_EX)
        self._locks.append(handle)

    def stage_corpus(self, key: str) -> Path:
        """Copy the corpus to a fixed path for *key*; removed after the test."""
        self._lock(key)
        target = STAGE_ROOT / key / "corpus"
        shutil.rmtree(target.parent, ignore_errors=True)
        shutil.copytree(CORPUS_SRC, target)
        self.dirs.append(target.parent)
        return target

    def collection(self, provider: str, name: str) -> None:
        self._lock(f"{provider}-{name}")
        self.collections.append((provider, name))
        delete_remote_collections(provider, [name])  # leftovers of a killed run

    def run(self) -> None:
        for provider, name in self.collections:
            delete_remote_collections(provider, [name])
        for directory in self.dirs:
            shutil.rmtree(directory, ignore_errors=True)
        for handle in self._locks:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()


@pytest.fixture
def cleanup() -> Iterator[Cleanup]:
    """Per-test registry of things to delete in teardown."""
    registry = Cleanup()
    yield registry
    registry.run()


# --------------------------------------------------------------------------
# HierarchicalDocumentTool
# --------------------------------------------------------------------------


async def _run_hierarchical(
    provider: str,
    mode: SearchMode,
    min_score: float | None,
    tool_name: str,
    corpus: Path,
) -> dict[str, list[dict[str, Any]]]:
    config = HierarchicalDocumentToolConfig(
        name=tool_name,
        description="Golden retrieval corpus",
        source=str(corpus),
        contextual_embeddings=False,
        document_domain=DocumentDomain.NONE,
        search_mode=mode,
        top_k=HIER_TOP_K,
        min_score=min_score,
        database=_database(provider),
    )
    tool = HierarchicalDocumentTool(config)
    tool.set_embedding_service(HashEmbeddingService(HIER_DIMENSIONS))
    await tool.initialize(force_ingest=True, provider_type="ollama")
    assert tool._provider == provider, "tool fell back to another provider"

    results: dict[str, list[dict[str, Any]]] = {}
    skipped = (
        UNSTABLE_QDRANT_HYBRID
        if (provider, mode) == ("qdrant", SearchMode.HYBRID)
        else frozenset()
    )
    for query_id, text in QUERIES.items():
        if query_id in skipped:
            continue
        hits = await tool.search(text)
        rows = [
            {
                "rank": rank,
                "source": _rel(hit.source_path, corpus),
                "section_id": hit.section_id,
                "parent_chain": list(hit.parent_chain),
                "chunk_sha": sha12(hit.content),
                "score": round(hit.fused_score, 4),
            }
            for rank, hit in enumerate(hits, start=1)
        ]
        results[query_id] = _canonical_ties(rows)
    return results


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_hierarchical_document_golden(provider: str, cleanup: Cleanup) -> None:
    """Semantic, keyword and hybrid results match the frozen goldens."""
    key = f"hier-{provider}"
    corpus = cleanup.stage_corpus(key)
    tool_name = "golden_hier_" + provider.replace("-", "_")  # UUIDv5 namespace
    cleanup.collection(provider, f"hierarchical_docs_{tool_name}_dim{HIER_DIMENSIONS}")

    actual: dict[str, Any] = {
        "tool": "hierarchical_document",
        "provider": provider,
        "known_behavior": KNOWN_BEHAVIOR.get(("hierarchical_document", provider), []),
        "top_k": HIER_TOP_K,
        "queries": QUERIES,
        "excluded_queries": (
            {"hybrid": sorted(UNSTABLE_QDRANT_HYBRID)} if provider == "qdrant" else {}
        ),
        "modes": {},
    }
    for label, mode, min_score in (
        ("semantic", SearchMode.SEMANTIC, None),
        (f"semantic_min_score_{HIER_MIN_SCORE}", SearchMode.SEMANTIC, HIER_MIN_SCORE),
        ("keyword", SearchMode.KEYWORD, None),
        ("hybrid", SearchMode.HYBRID, None),
    ):
        actual["modes"][label] = await _run_hierarchical(
            provider, mode, min_score, tool_name, corpus
        )

    assert_golden(
        EXPECTED_DIR / f"hierarchical_document.{provider}.json",
        actual,
        abs_tol=HIER_SCORE_TOL,
        label=f"hierarchical_document/{provider}",
    )


# --------------------------------------------------------------------------
# VectorStoreTool
# --------------------------------------------------------------------------

_BLOCK_SPLIT = re.compile(r"\n\n(?=\[\d+\] Score: )")
_BLOCK_HEAD = re.compile(r"^\[(\d+)\] Score: ([0-9.]+) \| Source: ([^\n]*)\n")


def _parse_vectorstore_output(output: str, corpus: Path) -> list[dict[str, Any]]:
    """Parse ``VectorStoreTool.search`` text into rows (public output only)."""
    if output.startswith("No relevant results found"):
        return []
    header, _, body = output.partition("\n\n")
    assert re.fullmatch(r"Found \d+ result\(s\):", header), header
    rows: list[dict[str, Any]] = []
    for block in _BLOCK_SPLIT.split(body):
        match = _BLOCK_HEAD.match(block + "\n")
        assert match, f"unparsable result block: {block[:80]!r}"
        content = block[match.end() - 1 :].lstrip("\n")
        rows.append(
            {
                "rank": int(match.group(1)),
                "source": _rel(match.group(3), corpus),
                "chunk_sha": sha12(content.rstrip("\n")),
                "chunk_head": content.strip().replace("\n", " ")[:60],
                "score": float(match.group(2)),
            }
        )
    return rows


async def _run_vectorstore(
    provider: str, min_score: float | None, dimensions: int, corpus: Path
) -> dict[str, list[dict[str, Any]]]:
    config = VectorstoreToolConfig(
        name="golden_vectorstore",
        description="Golden retrieval corpus",
        source=str(corpus),
        top_k=VEC_TOP_K,
        chunk_size=VEC_CHUNK_SIZE,
        embedding_dimensions=dimensions,
        min_similarity_score=min_score,
        database=_database(provider),
    )
    tool = VectorStoreTool(config)
    tool.set_embedding_service(HashEmbeddingService(dimensions))
    await tool.initialize(force_ingest=True)
    assert tool._provider == provider, "tool fell back to another provider"

    results: dict[str, list[dict[str, Any]]] = {}
    for query_id, text in QUERIES.items():
        rows = _parse_vectorstore_output(await tool.search(text), corpus)
        results[query_id] = _canonical_ties(rows, only_zero=True)
    return results


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_vectorstore_golden(provider: str, cleanup: Cleanup) -> None:
    """Semantic results (and the min-score filter) match the frozen goldens."""
    dimensions = VEC_DIMENSIONS[provider]
    corpus = cleanup.stage_corpus(f"vec-{provider}")
    cleanup.collection(provider, f"documents_dim{dimensions}")

    actual: dict[str, Any] = {
        "tool": "vectorstore",
        "provider": provider,
        "known_behavior": KNOWN_BEHAVIOR.get(("vectorstore", provider), []),
        "top_k": VEC_TOP_K,
        "chunk_size": VEC_CHUNK_SIZE,
        "queries": QUERIES,
        "modes": {
            "semantic": await _run_vectorstore(provider, None, dimensions, corpus),
            f"semantic_min_score_{VEC_MIN_SCORE}": await _run_vectorstore(
                provider, VEC_MIN_SCORE, dimensions, corpus
            ),
        },
    }
    assert_golden(
        EXPECTED_DIR / f"vectorstore.{provider}.json",
        actual,
        abs_tol=VEC_SCORE_TOL,
        label=f"vectorstore/{provider}",
    )
