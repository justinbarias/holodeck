"""Both retrieval tools run end to end on the Azure AI Search adapter (042 T11).

There is no Azure AI Search emulator and no credentials, so the adapter runs on
the in-memory fake of its two clients (``tests/contract/retrieval/
azure_search_fake.py``). This shows that the tools and the adapter agree on
record shapes, IDs with paths, filters, enumeration and native hybrid search. It
is not evidence from the Azure service.

Each tool ingests the golden corpus with the offline hash embedding on
``azure-ai-search`` and on ``in-memory``. For queries that have a clear best
chunk the two providers must agree on the top result, and a second tool
instance must find the stored chunks again without ingesting.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import cast

import pytest
from pydantic import SecretStr

from holodeck.lib.retrieval import CollectionSchema, Operation
from holodeck.lib.retrieval.adapters.azure_ai_search import (
    AzureAISearchAdapter,
    settings_from_connection,
)
from holodeck.models.tool import (
    DatabaseConfig,
    DocumentDomain,
    HierarchicalDocumentToolConfig,
    SearchMode,
)
from holodeck.models.tool import VectorstoreTool as VectorstoreToolConfig
from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool
from holodeck.tools.vectorstore_tool import VectorStoreTool
from tests.contract.retrieval.azure_search_fake import FakeSearchService
from tests.integration.golden._golden import RETRIEVAL_ROOT, HashEmbeddingService

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

CORPUS = RETRIEVAL_ROOT / "corpus"
HIER_DIMENSIONS = 768  # provider_type="ollama"
QUERIES = ["Affiliate", "kestrel dawn"]
ENDPOINT = "https://fake.search.windows.net"
_SCORE = re.compile(r"Score: [0-9.]+")


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> FakeSearchService:
    """One fake service behind every ``azure-ai-search`` adapter the tools build."""
    fake = FakeSearchService()

    def create(
        cls: type[AzureAISearchAdapter],
        schema: CollectionSchema,
        connection: Mapping[str, object],
    ) -> AzureAISearchAdapter:
        return cls(
            schema, settings_from_connection(connection), client_factory=fake.pair
        )

    monkeypatch.setattr(AzureAISearchAdapter, "create", classmethod(create))
    return fake


def _database(provider: str) -> DatabaseConfig | None:
    if provider == "azure-ai-search":
        return DatabaseConfig(
            provider="azure-ai-search",
            connection_string=SecretStr(ENDPOINT),
            api_key="fake-key",  # type: ignore[call-arg]
        )
    return None


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
    mode: SearchMode, service: FakeSearchService
) -> None:
    name = f"aztools_{mode.value}"

    azure_tool, azure_tops = await _hierarchical("azure-ai-search", name, mode)
    memory_tool, memory_tops = await _hierarchical("in-memory", name, mode)
    try:
        assert azure_tops == memory_tops
    finally:
        await memory_tool.close()

    stored = service.indexes[f"hierarchical_docs_{name}_dim{HIER_DIMENSIONS}"]
    assert len(stored.docs) > 0
    # IDs carry source paths; every key the tool wrote is valid for the service.
    assert all(
        re.fullmatch(r"[A-Za-z0-9=-][A-Za-z0-9_=-]*", key) for key in stored.docs
    )

    searches = service.names("search")
    hybrid = [s for s in searches if s["vector_queries"] and s["search_text"]]
    if mode is SearchMode.HYBRID:
        # The keyword leg ran inside the store: one request with text and vector.
        assert hybrid, "hybrid mode did not use the native hybrid request"
        assert all(s["search_fields"] == ["searchable_text"] for s in hybrid)
        assert all(s["query_type"] == "simple" for s in hybrid)
    else:
        assert not hybrid

    # A fresh tool on the same index reads what the first one stored.
    ingested = len(service.names("merge_or_upload"))
    reopened = HierarchicalDocumentTool(_hier_config("azure-ai-search", name, mode))
    reopened.set_embedding_service(HashEmbeddingService(HIER_DIMENSIONS))
    await reopened.initialize(force_ingest=False, provider_type="ollama")
    try:
        hits = await reopened.search(QUERIES[0])
        assert (hits[0].source_path, hits[0].section_id, hits[0].content) == (
            azure_tops[0]
        )
        assert len(service.names("merge_or_upload")) == ingested, "re-ingested"
    finally:
        await reopened.close()
        await azure_tool.close()


async def _vectorstore(
    provider: str, dimensions: int
) -> tuple[VectorStoreTool, list[str]]:
    config = VectorstoreToolConfig(
        name="az_vectorstore",
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
    service: FakeSearchService,
) -> None:
    dimensions = 307

    azure_tool, azure_tops = await _vectorstore("azure-ai-search", dimensions)
    memory_tool, memory_tops = await _vectorstore("in-memory", dimensions)
    try:
        assert azure_tops == memory_tops
        assert all(top.startswith("[1] Score: x") for top in azure_tops)
    finally:
        await memory_tool.close()
        await azure_tool.close()

    index = service.indexes[f"documents_dim{dimensions}"]
    assert len(index.docs) > 0
    adapter_calls = {name for name, _ in service.calls}
    assert {"create_index", "merge_or_upload", "search"} <= adapter_calls
    # The tool's incremental ingestion reads stored source paths with a filter.
    filters = [cast(str, s["filter"]) for s in service.names("search") if s["filter"]]
    assert any(f.startswith("source_path eq ") for f in filters)
    capabilities = AzureAISearchAdapter.create(
        CollectionSchema("docs", 4, ()), {"endpoint": ENDPOINT}
    ).capabilities
    assert capabilities.supports(Operation.HYBRID_SEARCH)
