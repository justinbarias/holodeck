"""Both retrieval tools run end to end on Azure Cosmos DB NoSQL (042 T12).

Each tool ingests the golden corpus with the offline hash embedding on
``azure-cosmos-nosql`` and on ``in-memory``. For queries that have a clear best
chunk the two providers must agree on the top result, and a second tool
instance must find the stored chunks again without ingesting.

Two environments run the same tests:

* ``emulator``: the Linux Cosmos DB emulator (``vnext-preview``, see
  ``tests/contract/retrieval/cosmos_emulator.py``). It has no full-text search,
  so the keyword leg of hybrid search is HoloDeck's BM25 there. Skipped when
  the emulator is not running.
* ``fake``: ``tests/contract/retrieval/cosmos_fake.py`` with
  ``full_text_search: true``, which is the only place the tools' native hybrid
  path (``ORDER BY RANK RRF``) runs. Its ranking is not the service's.

Neither ran against the Cosmos DB service.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterator
from contextlib import suppress
from uuid import uuid4

import pytest
import pytest_asyncio
from azure.cosmos.aio import CosmosClient
from azure.cosmos.exceptions import CosmosResourceNotFoundError

from holodeck.lib.retrieval import registry
from holodeck.lib.retrieval.registry import ProviderEntry
from holodeck.models.tool import (
    DatabaseConfig,
    DocumentDomain,
    HierarchicalDocumentToolConfig,
    SearchMode,
)
from holodeck.models.tool import VectorstoreTool as VectorstoreToolConfig
from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool
from holodeck.tools.vectorstore_tool import VectorStoreTool
from tests.contract.retrieval import cosmos_emulator
from tests.contract.retrieval.cosmos_fake import FakeAccount, FakeCosmosAdapter
from tests.integration.golden._golden import RETRIEVAL_ROOT, HashEmbeddingService

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

CORPUS = RETRIEVAL_ROOT / "corpus"
HIER_DIMENSIONS = 768  # provider_type="ollama"
QUERIES = ["Affiliate", "kestrel dawn"]
_SCORE = re.compile(r"Score: [0-9.]+")


@pytest.fixture(params=["emulator", "fake"])
def environment(request: pytest.FixtureRequest) -> Iterator[str]:
    """The Cosmos environment under test; the fake replaces the registry entry."""
    name: str = request.param
    if name == "emulator":
        reason = cosmos_emulator.skip_reason()
        if reason is not None:
            pytest.skip(reason)
        yield name
        return
    saved = registry.PROVIDERS["azure-cosmos-nosql"]
    registry.PROVIDERS["azure-cosmos-nosql"] = ProviderEntry(
        "tests.contract.retrieval.cosmos_fake",
        "FakeCosmosAdapter",
        extra="azure-cosmos",
    )
    FakeCosmosAdapter.shared = FakeAccount()
    try:
        yield name
    finally:
        registry.PROVIDERS["azure-cosmos-nosql"] = saved
        FakeCosmosAdapter.shared = None


def _database(provider: str, environment: str) -> DatabaseConfig | None:
    if provider != "azure-cosmos-nosql":
        return None
    settings: dict[str, object] = dict(cosmos_emulator.connection())
    if environment == "fake":
        settings.update(url="https://fake.documents.azure.com:443/", key="fake-key")
        settings.update(full_text_search=True)
    return DatabaseConfig(provider="azure-cosmos-nosql", **settings)  # type: ignore[arg-type]


@pytest_asyncio.fixture
async def containers(environment: str) -> AsyncIterator[list[str]]:
    """Names of the containers a test created; dropped afterwards."""
    created: list[str] = []
    yield created
    if environment == "fake":
        return
    async with CosmosClient(
        cosmos_emulator.EMULATOR_URL, credential=cosmos_emulator.EMULATOR_KEY
    ) as client:
        database = client.get_database_client(cosmos_emulator.DATABASE)
        for name in created:
            with suppress(CosmosResourceNotFoundError):
                await database.delete_container(name)


def _hier_config(
    provider: str, environment: str, name: str, mode: SearchMode
) -> HierarchicalDocumentToolConfig:
    return HierarchicalDocumentToolConfig(
        name=name,
        description="Golden retrieval corpus",
        source=str(CORPUS),
        contextual_embeddings=False,
        document_domain=DocumentDomain.NONE,
        search_mode=mode,
        top_k=5,
        database=_database(provider, environment),
    )


async def _hierarchical(
    provider: str, environment: str, name: str, mode: SearchMode
) -> tuple[HierarchicalDocumentTool, list[tuple[str, str, str]]]:
    tool = HierarchicalDocumentTool(_hier_config(provider, environment, name, mode))
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
    environment: str, mode: SearchMode, containers: list[str]
) -> None:
    name = f"cosmostools_{uuid4().hex[:8]}"
    containers.append(f"hierarchical_docs_{name}_dim{HIER_DIMENSIONS}")

    cosmos_tool, cosmos_tops = await _hierarchical(
        "azure-cosmos-nosql", environment, name, mode
    )
    memory_tool, memory_tops = await _hierarchical("in-memory", environment, name, mode)
    try:
        assert cosmos_tops == memory_tops
        native = cosmos_tool._uses_native_hybrid()
        assert native is (environment == "fake")
    finally:
        await memory_tool.close()

    # A fresh tool on the same container reads what the first one stored.
    reopened = HierarchicalDocumentTool(
        _hier_config("azure-cosmos-nosql", environment, name, mode)
    )
    reopened.set_embedding_service(HashEmbeddingService(HIER_DIMENSIONS))
    await reopened.initialize(force_ingest=False, provider_type="ollama")
    try:
        hits = await reopened.search(QUERIES[0])
        assert (hits[0].source_path, hits[0].section_id, hits[0].content) == (
            cosmos_tops[0]
        )
    finally:
        await reopened.close()
        await cosmos_tool.close()


async def _vectorstore(
    provider: str, environment: str, dimensions: int
) -> tuple[VectorStoreTool, list[str]]:
    config = VectorstoreToolConfig(
        name="cosmos_vectorstore",
        description="Golden retrieval corpus",
        source=str(CORPUS),
        top_k=4,
        chunk_size=128,
        embedding_dimensions=dimensions,
        database=_database(provider, environment),
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
    environment: str, containers: list[str]
) -> None:
    dimensions = 300 + int(uuid4().int % 600)  # a container name no other run uses
    containers.append(f"documents_dim{dimensions}")

    cosmos_tool, cosmos_tops = await _vectorstore(
        "azure-cosmos-nosql", environment, dimensions
    )
    memory_tool, memory_tops = await _vectorstore("in-memory", environment, dimensions)
    try:
        assert cosmos_tops == memory_tops
        assert all(top.startswith("[1] Score: x") for top in cosmos_tops)
    finally:
        await memory_tool.close()
        await cosmos_tool.close()


async def test_chunk_ids_that_are_file_paths_survive_the_round_trip(
    environment: str, containers: list[str]
) -> None:
    """The corpus lives in sub-directories, so chunk IDs contain ``/``."""
    dimensions = 300 + int(uuid4().int % 600)
    containers.append(f"documents_dim{dimensions}")
    tool, _ = await _vectorstore("azure-cosmos-nosql", environment, dimensions)
    try:
        adapter = tool._require_adapter()
        page = await adapter.list_page(limit=500)
        assert page.records
        ids = [record.id for record in page.records]
        assert any("/" in record_id for record_id in ids), ids[:3]
        fetched = await adapter.get(ids[:5])
        assert [r.id for r in fetched] == ids[:5]
    finally:
        await tool.close()
