"""required_extras picks the holodeck-ai extras an image needs (spec 042, FR-029)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

from holodeck.deploy.extras import required_extras
from holodeck.lib.retrieval.registry import PROVIDERS
from holodeck.models.agent import Agent

PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"


def _agent(
    provider: str = "anthropic",
    tools: list[dict[str, Any]] | None = None,
    observability: dict[str, Any] | None = None,
) -> Agent:
    for index, tool in enumerate(tools or []):
        tool["name"] = f"{tool['name']}_{index}"
    data: dict[str, Any] = {
        "name": "t",
        "model": {"provider": provider, "name": "m"},
        "instructions": {"inline": "x"},
    }
    if provider == "azure_openai":
        data["model"]["endpoint"] = "https://example.openai.azure.com"
    if tools is not None:
        data["tools"] = tools
    if observability is not None:
        data["observability"] = observability
    return Agent.model_validate(data)


def _vectorstore(database: str | None = None, source: str = "./data") -> dict[str, Any]:
    tool: dict[str, Any] = {
        "name": "docs",
        "type": "vectorstore",
        "description": "d",
        "source": source,
    }
    if database is not None:
        tool["database"] = {"provider": database}
    return tool


def _hierarchical(
    database: str | None = None,
    keyword_index: str | None = None,
    source: str = "./data",
) -> dict[str, Any]:
    tool: dict[str, Any] = {
        "name": "laws",
        "type": "hierarchical_document",
        "description": "d",
        "source": source,
    }
    if database is not None:
        tool["database"] = {"provider": database}
    if keyword_index == "opensearch":
        tool["keyword_index"] = {
            "provider": "opensearch",
            "endpoint": "http://os:9200",
            "index_name": "laws",
        }
    elif keyword_index is not None:
        tool["keyword_index"] = {"provider": keyword_index}
    return tool


_TRACING = {"enabled": True}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("openai", ["openai-agents"]),
        ("azure_openai", ["openai-agents"]),
        ("anthropic", []),
        ("ollama", []),
    ],
)
def test_model_provider(provider: str, expected: list[str]) -> None:
    assert required_extras(_agent(provider)) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("database", "expected"),
    [
        ("qdrant", ["qdrant"]),
        ("chromadb", ["chromadb"]),
        ("postgres", ["postgres"]),
        ("pinecone", ["pinecone"]),
        ("weaviate", ["weaviate"]),
        ("azure-ai-search", ["azure-ai-search"]),
        ("azure-cosmos-nosql", ["azure-cosmos"]),
        ("in-memory", []),
        (None, []),
    ],
)
@pytest.mark.parametrize("tool_type", ["vectorstore", "hierarchical_document"])
def test_vector_database_provider(
    tool_type: str, database: str | None, expected: list[str]
) -> None:
    tool = (
        _vectorstore(database)
        if tool_type == "vectorstore"
        else _hierarchical(database)
    )
    assert required_extras(_agent(tools=[tool])) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("az://container/docs", ["azure-blob"]),
        ("s3://bucket/docs", ["s3"]),
        ("./docs", []),
    ],
)
def test_remote_source_scheme(source: str, expected: list[str]) -> None:
    assert required_extras(_agent(tools=[_vectorstore(source=source)])) == expected


@pytest.mark.unit
def test_opensearch_keyword_index_needs_opensearch_extra() -> None:
    tool = _hierarchical(keyword_index="opensearch")
    assert required_extras(_agent(tools=[tool])) == ["opensearch"]


@pytest.mark.unit
def test_in_memory_keyword_index_adds_nothing() -> None:
    tool = _hierarchical(keyword_index="in-memory")
    assert required_extras(_agent(tools=[tool])) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider", "observability", "expected"),
    [
        ("anthropic", {"enabled": True}, ["claude-otel"]),
        ("ollama", {"enabled": True}, ["claude-otel"]),
        ("anthropic", {"enabled": True, "traces": {"enabled": False}}, []),
        ("anthropic", {"enabled": False}, []),
        ("anthropic", None, []),
        # The OpenAI Agents backend has its own tracing bridge.
        ("openai", {"enabled": True}, ["openai-agents"]),
        (
            "openai",
            {"enabled": True, "exporters": {"prometheus": {"enabled": True}}},
            ["openai-agents", "otel-prometheus"],
        ),
        (
            "openai",
            {
                "enabled": True,
                "exporters": {
                    "azure_monitor": {"enabled": True, "connection_string": "x"}
                },
            },
            ["openai-agents", "otel-azmon"],
        ),
        # An exporter that is configured but disabled needs nothing.
        (
            "openai",
            {"enabled": True, "exporters": {"prometheus": {"enabled": False}}},
            ["openai-agents"],
        ),
        # Console and OTLP exporters ship with the base install.
        (
            "openai",
            {"enabled": True, "exporters": {"otlp": {"enabled": True}}},
            ["openai-agents"],
        ),
        # Observability off: its exporters are never loaded.
        (
            "openai",
            {"enabled": False, "exporters": {"prometheus": {"enabled": True}}},
            ["openai-agents"],
        ),
    ],
)
def test_observability(
    provider: str, observability: dict[str, Any] | None, expected: list[str]
) -> None:
    assert required_extras(_agent(provider, observability=observability)) == expected


@pytest.mark.unit
def test_temporal_worker_flag_adds_temporal() -> None:
    assert required_extras(_agent(), temporal_worker=True) == ["temporal"]
    assert required_extras(_agent()) == []


@pytest.mark.unit
def test_combination_is_sorted_and_deduplicated() -> None:
    agent = _agent(
        "openai",
        tools=[
            _vectorstore("qdrant", source="s3://b/a"),
            _vectorstore("qdrant", source="s3://b/b"),
            _hierarchical("weaviate", keyword_index="opensearch"),
        ],
        observability={"enabled": True, "exporters": {"prometheus": {"enabled": True}}},
    )
    extras = required_extras(agent, temporal_worker=True)
    assert extras == [
        "openai-agents",
        "opensearch",
        "otel-prometheus",
        "qdrant",
        "s3",
        "temporal",
        "weaviate",
    ]
    assert extras == sorted(set(extras))


@pytest.mark.unit
def test_nothing_extra_agent() -> None:
    assert required_extras(_agent("anthropic", tools=[])) == []


@pytest.mark.unit
@pytest.mark.parametrize("name", ["eval", "nlp", "optimizer", "dev", "dashboard"])
def test_serving_images_never_get_evaluation_or_dev_extras(name: str) -> None:
    agent = _agent(
        "openai",
        tools=[_vectorstore("qdrant"), _hierarchical("chromadb", "opensearch")],
        observability={"enabled": True},
    )
    assert name not in required_extras(agent, temporal_worker=True)


@pytest.mark.unit
def test_every_registry_provider_extra_is_reachable() -> None:
    """Each provider with an extra maps to it, so the table cannot drift."""
    for provider, entry in PROVIDERS.items():
        if provider == "in-memory":
            continue
        extras = required_extras(_agent(tools=[_vectorstore(provider)]))
        assert extras == ([entry.extra] if entry.extra else [])


@pytest.mark.unit
def test_every_extra_exists_in_pyproject() -> None:
    declared = set(
        tomllib.loads(PYPROJECT.read_text())["project"]["optional-dependencies"]
    )
    agent = _agent(
        "openai",
        tools=[
            _vectorstore(database, source=source)
            for database in PROVIDERS
            for source in ("az://c/d", "s3://b/k")
        ]
        + [_hierarchical("qdrant", keyword_index="opensearch")],
        observability={
            "enabled": True,
            "exporters": {
                "prometheus": {"enabled": True},
                "azure_monitor": {"enabled": True, "connection_string": "x"},
            },
        },
    )
    possible = set(required_extras(agent, temporal_worker=True))
    claude = required_extras(
        _agent("anthropic", observability={"enabled": True}), temporal_worker=True
    )
    possible.update(claude)
    assert possible <= declared, sorted(possible - declared)
