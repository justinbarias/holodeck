"""Both retrieval tools run end to end on PostgreSQL (042 T7).

Each tool ingests the golden corpus with the offline hash embedding on
``postgres`` and on ``in-memory``. For queries that have a clear best chunk the
two providers must agree on the top result, and a second tool instance must
find the stored chunks again without ingesting.

Needs the pgvector container on 127.0.0.1:54329.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from uuid import uuid4

import psycopg
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
from tests.contract.retrieval.targets import POSTGRES_URL, port_open
from tests.integration.golden._golden import RETRIEVAL_ROOT, HashEmbeddingService

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

_reason = port_open("127.0.0.1", 54329)()
if _reason is not None:
    pytestmark.append(pytest.mark.skip(reason=_reason))

CORPUS = RETRIEVAL_ROOT / "corpus"
HIER_DIMENSIONS = 768  # provider_type="ollama"
QUERIES = ["Affiliate", "kestrel dawn"]
_SCORE = re.compile(r"Score: [0-9.]+")


def _database(provider: str) -> DatabaseConfig | None:
    if provider == "postgres":
        return DatabaseConfig(
            provider="postgres", connection_string=SecretStr(POSTGRES_URL)
        )
    return None


@pytest.fixture
def tables() -> Iterator[list[str]]:
    """Names of the tables a test created; dropped afterwards."""
    created: list[str] = []
    yield created
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        for name in created:
            connection.execute(f'DROP TABLE IF EXISTS public."{name}" CASCADE')


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
    mode: SearchMode, tables: list[str]
) -> None:
    suffix = uuid4().hex[:8]
    name = f"pgtools_{suffix}"
    tables.append(f"hierarchical_docs_{name}_dim{HIER_DIMENSIONS}")

    postgres_tool, postgres_tops = await _hierarchical("postgres", name, mode)
    memory_tool, memory_tops = await _hierarchical("in-memory", name, mode)
    try:
        assert postgres_tops == memory_tops
    finally:
        await memory_tool.close()

    # A fresh tool on the same table reads what the first one stored.
    reopened = HierarchicalDocumentTool(_hier_config("postgres", name, mode))
    reopened.set_embedding_service(HashEmbeddingService(HIER_DIMENSIONS))
    await reopened.initialize(force_ingest=False, provider_type="ollama")
    try:
        hits = await reopened.search(QUERIES[0])
        assert (hits[0].source_path, hits[0].section_id, hits[0].content) == (
            postgres_tops[0]
        )
    finally:
        await reopened.close()
        await postgres_tool.close()


async def _vectorstore(
    provider: str, dimensions: int
) -> tuple[VectorStoreTool, list[str]]:
    config = VectorstoreToolConfig(
        name="pg_vectorstore",
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
    tables: list[str],
) -> None:
    dimensions = 300 + int(uuid4().int % 600)  # a table name no other run uses
    tables.append(f"documents_dim{dimensions}")

    postgres_tool, postgres_tops = await _vectorstore("postgres", dimensions)
    memory_tool, memory_tops = await _vectorstore("in-memory", dimensions)
    try:
        assert postgres_tops == memory_tops
        assert all(top.startswith("[1] Score: x") for top in postgres_tops)
    finally:
        await memory_tool.close()
        await postgres_tool.close()
