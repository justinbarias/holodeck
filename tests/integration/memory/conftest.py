"""Fixtures for the Graphiti memory integration tests (spec 043, T10).

The tests run ``GraphitiBackend`` against a real Neo4j with a stub LLM client
and a stub embedder. The connection comes from the environment:

* ``HOLODECK_TEST_NEO4J_URI`` (default ``bolt://127.0.0.1:7687``)
* ``HOLODECK_TEST_NEO4J_USER`` (default ``neo4j``)
* ``HOLODECK_TEST_NEO4J_PASSWORD`` (default: the password in ``compose.yaml``)

When Neo4j cannot be reached, the tests skip.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
import pytest_asyncio
from pydantic import SecretStr

from holodeck.lib.memory.backends.graphiti import GraphitiBackend

if TYPE_CHECKING:
    from tests.integration.memory.stub_clients import StubLLMClient

DEFAULT_URI = "bolt://127.0.0.1:7687"
DEFAULT_USER = "neo4j"
DEFAULT_PASSWORD = "holodeck-it-pass"  # noqa: S105 - test value, see compose.yaml
CONNECT_TIMEOUT_SECONDS = 3.0


@dataclass(frozen=True, slots=True)
class Neo4jSettings:
    uri: str
    user: str
    password: str


@dataclass(slots=True)
class NamespaceFactory:
    """Makes unique namespaces that start with ``it-`` and remembers them."""

    made: list[str] = field(default_factory=list)

    def __call__(self) -> str:
        namespace = f"it-{uuid4().hex[:12]}"
        self.made.append(namespace)
        return namespace


@pytest.fixture(scope="session")
def neo4j_settings() -> Neo4jSettings:
    """Connection settings; skips the test when Neo4j is not reachable."""
    pytest.importorskip(
        "graphiti_core",
        reason="the graphiti extra is not installed",
        exc_type=ImportError,
    )
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable

    settings = Neo4jSettings(
        uri=os.environ.get("HOLODECK_TEST_NEO4J_URI", DEFAULT_URI),
        user=os.environ.get("HOLODECK_TEST_NEO4J_USER", DEFAULT_USER),
        password=os.environ.get("HOLODECK_TEST_NEO4J_PASSWORD", DEFAULT_PASSWORD),
    )
    driver = GraphDatabase.driver(
        settings.uri,
        auth=(settings.user, settings.password),
        connection_timeout=CONNECT_TIMEOUT_SECONDS,
    )
    try:
        driver.verify_connectivity()
    except (ServiceUnavailable, OSError) as error:
        message = (
            f"Neo4j is not reachable at {settings.uri} ({type(error).__name__}). "
            "Start it with: docker compose -f tests/integration/memory/compose.yaml "
            "up -d --wait"
        )
        # The nightly job sets this, so a database that did not start fails
        # the run and does not pass with every test skipped.
        if os.environ.get("HOLODECK_TEST_NEO4J_REQUIRED") == "1":
            pytest.fail(message)
        pytest.skip(message)
    finally:
        driver.close()
    return settings


@pytest.fixture
def stub_llm(monkeypatch: pytest.MonkeyPatch) -> StubLLMClient:
    """The stub LLM client every backend built in this test uses.

    ``graphiti_clients.build_llm_client`` is the one place the backend creates
    its LLM client, so replacing it keeps the real ``CountingLLMClient``,
    driver, embedder and ``Graphiti`` object in the path.
    """
    from holodeck.lib.memory.backends import graphiti_clients
    from tests.integration.memory.stub_clients import StubLLMClient

    stub = StubLLMClient()
    monkeypatch.setattr(
        graphiti_clients, "build_llm_client", lambda model, **kwargs: stub
    )
    return stub


@pytest.fixture
def new_namespace() -> NamespaceFactory:
    """A factory of unique namespaces. ``backend`` purges them all afterwards."""
    return NamespaceFactory()


@pytest_asyncio.fixture
async def backend(
    neo4j_settings: Neo4jSettings,
    stub_llm: StubLLMClient,
    new_namespace: NamespaceFactory,
) -> AsyncIterator[GraphitiBackend]:
    """A ``GraphitiBackend`` on the real Neo4j, with stub model clients."""
    from holodeck.models.llm import LLMProvider, ProviderEnum
    from holodeck.models.tool import GraphBackendConfig
    from tests.integration.memory.stub_clients import StubEmbeddingService

    config = GraphBackendConfig(
        driver="neo4j",
        uri=neo4j_settings.uri,
        user=neo4j_settings.user,
        password=SecretStr(neo4j_settings.password),
    )
    model = LLMProvider(
        provider=ProviderEnum.OPENAI, name="stub-model", api_key=SecretStr("unused")
    )
    graph_backend = GraphitiBackend(config, model, StubEmbeddingService())
    try:
        yield graph_backend
    finally:
        # Every namespace gets its purge, even when an earlier one fails.
        failures: list[str] = []
        try:
            for namespace in new_namespace.made:
                try:
                    await graph_backend.purge(namespace)
                except Exception as error:  # noqa: BLE001 - reported below
                    failures.append(f"{namespace}: {type(error).__name__}")
        finally:
            await graph_backend.close()
        assert not failures, f"test namespaces were not purged: {failures}"
