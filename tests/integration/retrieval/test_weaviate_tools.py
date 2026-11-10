"""Both retrieval tools run end to end on Weaviate (042 T10).

Each tool ingests the golden corpus with the offline hash embedding on
``weaviate`` and on ``in-memory``. For queries that have a clear best chunk the
two providers must agree on the top result, a second tool instance must find the
stored chunks again without ingesting, and hybrid mode must use Weaviate's own
hybrid search (no in-process BM25 index is built).

Needs the Weaviate container on 127.0.0.1:8089 (gRPC 50059).
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio

from holodeck.lib.keyword_search import KeywordSearchStrategy
from holodeck.lib.retrieval.adapters.weaviate import WeaviateAdapter
from holodeck.lib.retrieval.schemas import document_schema, hierarchical_schema
from holodeck.models.tool import (
    DatabaseConfig,
    DocumentDomain,
    HierarchicalDocumentToolConfig,
    SearchMode,
)
from holodeck.models.tool import VectorstoreTool as VectorstoreToolConfig
from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool
from holodeck.tools.vectorstore_tool import VectorStoreTool
from tests.contract.retrieval.targets import WEAVIATE_CONNECTION, port_open
from tests.integration.golden._golden import RETRIEVAL_ROOT, HashEmbeddingService

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

_reason = port_open("127.0.0.1", 8089)()
if _reason is not None:
    pytestmark.append(pytest.mark.skip(reason=_reason))

CORPUS = RETRIEVAL_ROOT / "corpus"
HIER_DIMENSIONS = 768  # provider_type="ollama"
QUERIES = ["Affiliate", "kestrel dawn"]
_SCORE = re.compile(r"Score: [0-9.]+")


def _database(provider: str) -> DatabaseConfig | None:
    if provider == "weaviate":
        return DatabaseConfig(provider="weaviate", **WEAVIATE_CONNECTION)
    return None


@pytest_asyncio.fixture
async def collections() -> AsyncIterator[list[WeaviateAdapter]]:
    """Adapters for the collections a test created; deleted afterwards."""
    created: list[WeaviateAdapter] = []
    yield created
    for adapter in created:
        try:
            await adapter.delete_collection()
        finally:
            await adapter.close()


def _hier_config(
    provider: str, name: str, mode: SearchMode
) -> HierarchicalDocumentToolConfig:
    return HierarchicalDocumentToolConfig(
        name=name,
        description="Golden retrieval corpus",
        source=str(CORPUS),
        contextual_embeddings=False,
        document_domain=DocumentDomain.NONE,
        search_mode=mode,
        top_k=5,
        database=_database(provider),
    )


async def _hierarchical(
    provider: str, name: str, mode: SearchMode
) -> tuple[HierarchicalDocumentTool, list[tuple[str, str, str]]]:
    tool = HierarchicalDocumentTool(_hier_config(provider, name, mode))
    tool.set_embedding_service(HashEmbeddingService(HIER_DIMENSIONS))
    await tool.initialize(force_ingest=True, provider_type="ollama")
    assert tool._provider == provider, "tool fell back to another provider"
    tops: list[tuple[str, str, str]] = []
    for query in QUERIES:
        hits = await tool.search(query)
        assert hits, f"no result for {query!r} on {provider}"
        tops.append((hits[0].source_path, hits[0].section_id, hits[0].content))
    return tool, tops


@pytest.mark.parametrize("mode", [SearchMode.SEMANTIC, SearchMode.HYBRID])
async def test_hierarchical_document_tool_matches_the_in_memory_top_result(
    mode: SearchMode, collections: list[WeaviateAdapter]
) -> None:
    name = f"wvtools_{uuid4().hex[:8]}"
    collections.append(
        WeaviateAdapter.create(
            hierarchical_schema(HIER_DIMENSIONS, name), WEAVIATE_CONNECTION
        )
    )

    weaviate_tool, weaviate_tops = await _hierarchical("weaviate", name, mode)
    memory_tool, memory_tops = await _hierarchical("in-memory", name, mode)
    try:
        assert weaviate_tops == memory_tops
        if mode is SearchMode.HYBRID:
            executor = weaviate_tool._hybrid_executor
            assert executor is not None
            assert executor.strategy is KeywordSearchStrategy.NATIVE_HYBRID
            assert executor._keyword_index is None, "the BM25 fallback was built"
    finally:
        await memory_tool.close()

    # A fresh tool on the same collection reads what the first one stored.
    reopened = HierarchicalDocumentTool(_hier_config("weaviate", name, mode))
    reopened.set_embedding_service(HashEmbeddingService(HIER_DIMENSIONS))
    await reopened.initialize(force_ingest=False, provider_type="ollama")
    try:
        hits = await reopened.search(QUERIES[0])
        assert (hits[0].source_path, hits[0].section_id, hits[0].content) == (
            weaviate_tops[0]
        )
    finally:
        await reopened.close()
        await weaviate_tool.close()


async def _vectorstore(
    provider: str, dimensions: int
) -> tuple[VectorStoreTool, list[str]]:
    config = VectorstoreToolConfig(
        name="wv_vectorstore",
        description="Golden retrieval corpus",
        source=str(CORPUS),
        top_k=4,
        chunk_size=128,
        embedding_dimensions=dimensions,
        database=_database(provider),
    )
    tool = VectorStoreTool(config)
    tool.set_embedding_service(HashEmbeddingService(dimensions))
    await tool.initialize(force_ingest=True)
    assert tool._provider == provider, "tool fell back to another provider"
    tops: list[str] = []
    for query in QUERIES:
        output = await tool.search(query)
        first = output.split("\n\n")[1]
        tops.append(_SCORE.sub("Score: x", first))
    return tool, tops


async def test_vectorstore_tool_matches_the_in_memory_top_result(
    collections: list[WeaviateAdapter],
) -> None:
    dimensions = 300 + int(uuid4().int % 600)  # a collection no other run uses
    collections.append(
        WeaviateAdapter.create(document_schema(dimensions), WEAVIATE_CONNECTION)
    )

    weaviate_tool, weaviate_tops = await _vectorstore("weaviate", dimensions)
    memory_tool, memory_tops = await _vectorstore("in-memory", dimensions)
    try:
        assert weaviate_tops == memory_tops
        assert all(top.startswith("[1] Score: x") for top in weaviate_tops)
    finally:
        await memory_tool.close()
        await weaviate_tool.close()
