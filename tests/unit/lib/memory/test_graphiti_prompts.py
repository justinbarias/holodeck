"""Tests for HoloDeck's override of Graphiti's contradiction prompt (spec 043)."""

from __future__ import annotations

import hashlib
import inspect

import pytest
from graphiti_core.prompts import dedupe_edges
from graphiti_core.prompts.lib import (
    PROMPT_GROUPS,
    get_prompt_builder,
    validate_prompt_library,
)
from graphiti_core.prompts.models import ChatPrompt
from pydantic import SecretStr

from holodeck.lib.memory.backends.graphiti_clients import build_graphiti
from holodeck.lib.memory.backends.graphiti_prompts import (
    RESOLVE_EDGE_SYSTEM,
    holodeck_prompt_library,
    resolve_edge,
)
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import GraphBackendConfig

pytestmark = pytest.mark.unit

GRAPHITI_RESOLVE_EDGE_SHA256 = (
    "a3548e1180d876f08698d977668c06e43f25ad8c404d2789d30c974ecaa2c124"
)
"""Hash of Graphiti's own prompt at the version the override was written for."""

CONTEXT: dict[str, object] = {
    "existing_edges": [{"idx": 0, "fact": "existing-fact-text"}],
    "edge_invalidation_candidates": [{"idx": 1, "fact": "candidate-fact-text"}],
    "new_edge": "new-fact-text",
}


class NoEmbeddings:
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("no embedding call is expected")


def test_prompt_carries_the_three_inputs_graphiti_gives() -> None:
    prompt = resolve_edge(CONTEXT)

    assert isinstance(prompt, ChatPrompt)
    assert prompt.system.content == RESOLVE_EDGE_SYSTEM
    for text in ("existing-fact-text", "candidate-fact-text", "new-fact-text"):
        assert text in prompt.user.content


def test_prompt_treats_a_new_value_as_a_replacement() -> None:
    user = resolve_edge(CONTEXT).user.content

    assert "DIFFERENT values of the SAME such" in user
    assert 'does NOT have to say "replaces"' in user
    assert "Do NOT infer contradiction from recency" not in user


def test_prompt_keeps_a_power_that_two_bodies_can_hold() -> None:
    user = resolve_edge(CONTEXT).user.content

    assert "can be held by several bodies or persons at once" in user
    assert "the Commissioner may appoint inspectors" in user
    assert "A subject has ONE value of an attribute at a time" not in user


def test_prompt_keeps_graphiti_output_fields_and_index_rules() -> None:
    user = resolve_edge(CONTEXT).user.content

    assert set(dedupe_edges.EdgeDuplicate.model_fields) == {
        "duplicate_facts",
        "contradicted_facts",
    }
    assert "duplicate_facts: ONLY idx values from EXISTING FACTS" in user
    assert "contradicted_facts: idx values from EITHER list" in user


def test_library_is_complete_and_routes_the_override() -> None:
    library = holodeck_prompt_library()

    validate_prompt_library(library)
    ours = get_prompt_builder(library, "dedupe_edges.resolve_edge")(CONTEXT)
    assert ours.system.content == RESOLVE_EDGE_SYSTEM
    assert "resolve_edge" in PROMPT_GROUPS["dedupe_edges"]


@pytest.mark.asyncio
async def test_graphiti_is_built_with_the_override() -> None:
    runtime = build_graphiti(
        GraphBackendConfig(
            driver="neo4j",
            uri="bolt://127.0.0.1:7687",
            user="neo4j",
            password=SecretStr("db-not-real"),
        ),
        LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-4o",
            api_key=SecretStr("sk-not-real"),
        ),
        NoEmbeddings(),
    )
    try:
        library = runtime.graphiti.clients.prompt_library
        assert library is not None
        prompt = get_prompt_builder(library, "dedupe_edges.resolve_edge")(CONTEXT)
        assert prompt.system.content == RESOLVE_EDGE_SYSTEM
    finally:
        await runtime.graphiti.close()


def test_graphiti_own_prompt_has_not_changed_since_the_override_was_written() -> None:
    source = inspect.getsource(dedupe_edges.resolve_edge)

    assert hashlib.sha256(source.encode()).hexdigest() == (
        GRAPHITI_RESOLVE_EDGE_SHA256
    ), (
        "graphiti_core/prompts/dedupe_edges.py resolve_edge changed. Compare it "
        "with holodeck/lib/memory/backends/graphiti_prompts.py (inputs, output "
        "fields, index rules), update the override, then update this hash."
    )
