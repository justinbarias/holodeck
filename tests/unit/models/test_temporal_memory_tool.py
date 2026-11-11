"""Tests for ``TemporalMemoryToolConfig`` and its agent-level rules (spec 043)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from holodeck.models.agent import Agent
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import GraphBackendConfig, TemporalMemoryToolConfig

pytestmark = pytest.mark.unit

BACKEND = {
    "provider": "graphiti",
    "driver": "neo4j",
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "not-a-real-secret",
}


def _doc_tool(name: str, **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "type": "hierarchical_document",
        "description": "docs",
        "source": "docs/*.md",
        **extra,
    }


def _memory(**overrides: Any) -> dict[str, Any]:
    return {
        "name": "law_memory",
        "type": "temporal_memory",
        "description": "Facts and when they applied.",
        "source_tools": ["family_law"],
        "backend": dict(BACKEND),
        **overrides,
    }


def _agent(tools: list[dict[str, Any]], model: LLMProvider | None = None) -> Agent:
    return Agent(
        name="a",
        model=model or LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
        instructions={"inline": "hi"},
        tools=tools,
    )


def test_cfg_01_agent_accepts_a_temporal_memory_tool_with_defaults() -> None:
    agent = _agent([_doc_tool("family_law"), _memory()])

    assert agent.tools is not None
    tool = agent.tools[1]
    assert isinstance(tool, TemporalMemoryToolConfig)
    assert tool.resolved_namespace == "law_memory"
    assert tool.extraction_model is None
    assert tool.extraction_instructions is None
    assert tool.effective_date_key == "effective_date"
    assert tool.ingest_on_start is False
    assert (tool.hops, tool.top_k) == (1, 10)
    assert (tool.fusion_weights.chunks, tool.fusion_weights.graph) == (1.0, 1.0)
    assert tool.schema_hints is None


def test_cfg_01_every_spec_field_is_accepted() -> None:
    tool = TemporalMemoryToolConfig.model_validate(
        _memory(
            source_tools=["family_law", "family_guides"],
            namespace="family-assistance",
            extraction_model={"provider": "openai", "name": "gpt-5.4-mini"},
            extraction_instructions="Extract rates as facts.",
            effective_date_key="commenced",
            ingest_on_start=True,
            hops=2,
            top_k=5,
            fusion_weights={"chunks": 0.5, "graph": 2.0},
            schema_hints={"entity_types": ["Act", "Section", "Rate"]},
        )
    )

    assert tool.resolved_namespace == "family-assistance"
    assert tool.schema_hints is not None
    assert tool.schema_hints.entity_types == ["Act", "Section", "Rate"]


def test_cfg_01_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TemporalMemoryToolConfig.model_validate(_memory(source_tool="family_law"))
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TemporalMemoryToolConfig.model_validate(
            _memory(backend={**BACKEND, "pool_size": 5})
        )


def test_backend_password_is_a_secret() -> None:
    backend = GraphBackendConfig.model_validate(BACKEND)

    assert isinstance(backend.password, SecretStr)
    assert "not-a-real-secret" not in repr(backend)


@pytest.mark.parametrize("driver", ["neo4j", "falkordb", "neptune"])
def test_gph_03_a_network_driver_needs_a_uri(driver: str) -> None:
    with pytest.raises(ValidationError, match=f"backend.uri is required for {driver}"):
        GraphBackendConfig.model_validate({"provider": "graphiti", "driver": driver})


def test_gph_03_neptune_needs_a_search_host_and_kuzu_needs_nothing() -> None:
    with pytest.raises(ValidationError, match="backend.aoss_host is required"):
        GraphBackendConfig.model_validate(
            {"driver": "neptune", "uri": "neptune-db://example"}
        )
    assert GraphBackendConfig.model_validate({"driver": "kuzu"}).uri is None


def test_gph_03_an_unknown_driver_or_provider_is_rejected() -> None:
    with pytest.raises(ValidationError):
        GraphBackendConfig.model_validate({"driver": "memgraph", "uri": "bolt://x"})
    with pytest.raises(ValidationError):
        GraphBackendConfig.model_validate(
            {"provider": "native", "driver": "neo4j", "uri": "bolt://x"}
        )


def test_cfg_02_empty_source_tools_fails_with_the_tool_name() -> None:
    with pytest.raises(
        ValidationError,
        match="temporal_memory tool 'law_memory': source_tools must name at least one",
    ):
        TemporalMemoryToolConfig.model_validate(_memory(source_tools=[]))


def test_cfg_02_a_repeated_source_tool_fails_with_both_names() -> None:
    with pytest.raises(
        ValidationError,
        match="temporal_memory tool 'law_memory': source_tools repeats family_law",
    ):
        TemporalMemoryToolConfig.model_validate(
            _memory(source_tools=["family_law", "family_law"])
        )


def test_cfg_02_a_source_tool_that_is_not_in_the_agent_fails() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "temporal_memory tool 'law_memory': source_tools entry 'missing' "
            "names no tool in this agent"
        ),
    ):
        _agent([_doc_tool("family_law"), _memory(source_tools=["missing"])])


def test_cfg_02_a_source_tool_of_another_type_fails() -> None:
    vectorstore = {
        "name": "kb",
        "type": "vectorstore",
        "description": "kb",
        "source": "kb/",
    }
    with pytest.raises(
        ValidationError,
        match=(
            "temporal_memory tool 'law_memory': source_tools entry 'kb' is a "
            "vectorstore tool; only hierarchical_document tools are allowed"
        ),
    ):
        _agent([vectorstore, _memory(source_tools=["kb"])])


def test_cfg_03_a_second_memory_tool_fails_with_both_names() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "Only one temporal_memory tool is allowed per agent; "
            "found: law_memory, other_memory"
        ),
    ):
        _agent([_doc_tool("family_law"), _memory(), _memory(name="other_memory")])


def test_cfg_04_single_valued_predicates_fail_on_graphiti() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "temporal_memory tool 'law_memory': "
            "schema_hints.single_valued_predicates is not supported by the "
            "graphiti backend"
        ),
    ):
        TemporalMemoryToolConfig.model_validate(
            _memory(schema_hints={"single_valued_predicates": ["base_rate"]})
        )


def test_cfg_06_source_tools_with_different_embedding_models_fail() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "temporal_memory tool 'law_memory': source tools use different "
            r"embedding models: family_guides \(model-b\), family_law \(model-a\)"
        ),
    ):
        _agent(
            [
                _doc_tool("family_law", embedding_model="model-a"),
                _doc_tool("family_guides", embedding_model="model-b"),
                _memory(source_tools=["family_law", "family_guides"]),
            ]
        )


def test_cfg_06_an_unset_embedding_model_matches_an_explicit_one() -> None:
    agent = _agent(
        [
            _doc_tool("family_law", embedding_model="model-a"),
            _doc_tool("family_guides"),
            _memory(source_tools=["family_law", "family_guides"]),
        ]
    )

    assert agent.tools is not None and len(agent.tools) == 3


@pytest.mark.parametrize("auth", ["bedrock", "vertex", "foundry", "custom"])
def test_gph_04_an_unsupported_anthropic_auth_provider_fails(auth: str) -> None:
    with pytest.raises(
        ValidationError,
        match=(
            f"temporal_memory tool 'law_memory': extraction_model auth_provider "
            f"'{auth}' is not supported; use api_key or oauth_token"
        ),
    ):
        TemporalMemoryToolConfig.model_validate(
            _memory(
                extraction_model={
                    "provider": "anthropic",
                    "name": "claude-haiku-5-5",
                    "auth_provider": auth,
                }
            )
        )


@pytest.mark.parametrize("auth", [None, "api_key", "oauth_token"])
def test_gph_04_supported_anthropic_auth_providers_load(auth: str | None) -> None:
    model: dict[str, Any] = {"provider": "anthropic", "name": "claude-haiku-5-5"}
    if auth is not None:
        model["auth_provider"] = auth

    tool = TemporalMemoryToolConfig.model_validate(_memory(extraction_model=model))

    assert tool.extraction_model is not None


def test_gph_04_the_agent_model_is_checked_when_it_is_the_extraction_model() -> None:
    bedrock = LLMProvider(
        provider=ProviderEnum.ANTHROPIC,
        name="claude-haiku-5-5",
        auth_provider="bedrock",
    )
    embedding = {
        "embedding_provider": {"provider": "openai", "name": "text-embedding-3-small"}
    }
    with pytest.raises(
        ValidationError,
        match="extraction_model auth_provider 'bedrock' is not supported",
    ):
        Agent(
            name="a",
            model=bedrock,
            instructions={"inline": "hi"},
            tools=[_doc_tool("family_law"), _memory()],
            **embedding,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hops", 0),
        ("hops", 3),
        ("top_k", 0),
        ("namespace", "has space"),
        ("fusion_weights", {"chunks": 0.0, "graph": 0.0}),
        ("fusion_weights", {"chunks": -1.0, "graph": 1.0}),
        ("fusion_weights", {"chunks": float("inf"), "graph": 1.0}),
        ("fusion_weights", {"chunks": 1.0, "graph": float("nan")}),
        ("effective_date_key", " "),
        ("schema_hints", {"entity_types": ["Not Valid"]}),
    ],
)
def test_out_of_range_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        TemporalMemoryToolConfig.model_validate(_memory(**{field: value}))


@pytest.mark.parametrize(
    ("tool", "function"),
    [
        (
            {
                "name": "memory_search",
                "type": "function",
                "description": "f",
                "file": "tools.py",
                "function": "f",
            },
            "memory_search",
        ),
        (
            {
                "name": "memory_timeline",
                "type": "function",
                "description": "f",
                "file": "tools.py",
                "function": "f",
            },
            "memory_timeline",
        ),
        (_doc_tool("memory"), "memory_search"),
    ],
)
def test_hst_01_a_tool_that_takes_a_memory_function_name_fails(
    tool: dict[str, Any], function: str
) -> None:
    with pytest.raises(
        ValidationError,
        match=(
            f"temporal_memory tool 'law_memory': tool '{tool['name']}' would "
            f"expose the function name '{function}'"
        ),
    ):
        _agent([_doc_tool("family_law"), tool, _memory()])
