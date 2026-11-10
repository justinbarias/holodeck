"""VectorStoreTool and HierarchicalDocumentTool on Pinecone Local (042 T8).

Both tools ingest the golden retrieval corpus through the registry into a real
(emulated) Pinecone index and answer queries. The top result of two queries
must equal the top result frozen in the in-memory golden files, which shows the
adapter returns the same ranking as the reference store.

The control plane is Pinecone Local's pre-10 API through
``LocalPineconeAdapter``; the data plane is the production adapter code on the
10.x client. See ``tests/contract/retrieval/pinecone_local.py``. This is not
evidence from the Pinecone service. Skipped when Local is not running.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from pydantic import SecretStr

from holodeck.lib.retrieval.adapters import pinecone as pinecone_module
from holodeck.lib.retrieval.schemas import document_schema, hierarchical_schema
from holodeck.models.tool import DatabaseConfig, SearchMode
from tests.contract.retrieval import pinecone_local
from tests.contract.retrieval.pinecone_local import LocalPineconeAdapter
from tests.integration.golden import test_retrieval_golden as golden
from tests.integration.golden._golden import RETRIEVAL_ROOT

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        pinecone_local.skip_reason() is not None,
        reason=pinecone_local.skip_reason() or "",
    ),
]

QUERY_IDS = ("definition_phrase", "contextual_phrase")
VEC_DIMENSIONS = 307
TOOL_NAME = "pc_hier"


def _frozen_top(tool: str, query_id: str) -> dict[str, Any]:
    path = RETRIEVAL_ROOT / "expected" / f"{tool}.in-memory.json"
    rows: list[dict[str, Any]] = json.loads(path.read_text())["modes"]["semantic"][
        query_id
    ]
    return rows[0]


@pytest_asyncio.fixture
async def pinecone_local_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[None]:
    """Route the ``pinecone`` provider to the Local-compatible adapter."""
    monkeypatch.setattr(pinecone_module, "PineconeAdapter", LocalPineconeAdapter)
    monkeypatch.setattr(
        golden,
        "_database",
        lambda provider: DatabaseConfig(
            provider="pinecone", connection_string=SecretStr("pclocal")
        ),
    )
    yield
    for schema in (
        document_schema(VEC_DIMENSIONS),
        hierarchical_schema(golden.HIER_DIMENSIONS, TOOL_NAME),
    ):
        adapter = LocalPineconeAdapter.build(schema)
        try:
            await adapter.delete_collection()
        finally:
            await adapter.close()


@pytest.mark.asyncio
async def test_vectorstore_tool_ranks_like_the_in_memory_golden(
    pinecone_local_registry: None,
) -> None:
    cleanup = golden.Cleanup()
    try:
        corpus = cleanup.stage_corpus("vec-pinecone-local")
        results = await golden._run_vectorstore(
            "pinecone", None, VEC_DIMENSIONS, corpus
        )
    finally:
        cleanup.run()

    for query_id in QUERY_IDS:
        frozen = _frozen_top("vectorstore", query_id)
        top = results[query_id][0]
        assert (top["source"], top["chunk_sha"]) == (
            frozen["source"],
            frozen["chunk_sha"],
        ), query_id


@pytest.mark.asyncio
async def test_hierarchical_tool_ranks_like_the_in_memory_golden(
    pinecone_local_registry: None,
) -> None:
    cleanup = golden.Cleanup()
    try:
        corpus = cleanup.stage_corpus("hier-pinecone-local")
        results = await golden._run_hierarchical(
            "pinecone", SearchMode.SEMANTIC, None, TOOL_NAME, corpus
        )
    finally:
        cleanup.run()

    for query_id in QUERY_IDS:
        frozen = _frozen_top("hierarchical_document", query_id)
        top = results[query_id][0]
        assert (top["source"], top["section_id"], top["chunk_sha"]) == (
            frozen["source"],
            frozen["section_id"],
            frozen["chunk_sha"],
        ), query_id


@pytest.mark.asyncio
async def test_deleting_a_named_namespace_keeps_the_index_and_other_namespaces() -> (
    None
):
    """A Pinecone collection is (index, namespace) (042 phase 4 review)."""
    import uuid

    from holodeck.lib.retrieval import CollectionSchema, Record
    from holodeck.lib.retrieval.adapters.pinecone import PineconeSettings

    schema = CollectionSchema(f"ns-scope-{uuid.uuid4().hex[:8]}", 4, ())
    first = LocalPineconeAdapter(
        schema, PineconeSettings(api_key=pinecone_local.LOCAL_API_KEY, namespace="a")
    )
    second = LocalPineconeAdapter(
        schema, PineconeSettings(api_key=pinecone_local.LOCAL_API_KEY, namespace="b")
    )
    whole = LocalPineconeAdapter.build(schema)
    try:
        await first.ensure_collection()
        assert await first.collection_exists() is False  # empty namespace
        await first.upsert([Record(id="x", fields={}, vector=[1.0, 0, 0, 0])])
        await second.upsert([Record(id="y", fields={}, vector=[0, 1.0, 0, 0])])
        assert await first.collection_exists() and await first.count() == 1

        await first.delete_collection()

        assert await first.collection_exists() is False
        assert await first.count() == 0
        assert await whole.collection_exists()  # the index stays
        assert await second.count() == 1
        assert [r.id for r in await second.get(["y"])] == ["y"]
    finally:
        for adapter in (first, second):
            await adapter.close()
        try:
            await whole.delete_collection()
        finally:
            await whole.close()
