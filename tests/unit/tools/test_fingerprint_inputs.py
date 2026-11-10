"""Settings that change stored records are fingerprint inputs (042 review).

Spec 042 FR-023: a setting that changes record identity, stored text or
embeddings must change the pipeline fingerprint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from holodeck.lib.claude_context_generator import (
    ClaudeContextConfig,
    ClaudeSDKContextGenerator,
)
from holodeck.lib.litellm_support import LiteLLMModelSpec
from holodeck.lib.llm_context_generator import LLMContextGenerator
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import HierarchicalDocumentToolConfig
from holodeck.models.tool import VectorstoreTool as VectorstoreToolConfig
from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool
from holodeck.tools.vectorstore_tool import VectorStoreTool


def _structured(tmp_path: Path, **overrides: Any) -> str:
    source = tmp_path / "rows.csv"
    source.write_text("id,title,body\n1,a,b\n")
    settings: dict[str, Any] = {
        "name": "rows",
        "description": "Rows",
        "source": str(source),
        "vector_field": "body",
        "id_field": "id",
        "meta_fields": ["title"],
        "embedding_dimensions": 8,
    }
    settings.update(overrides)
    tool = VectorStoreTool(VectorstoreToolConfig(**settings))
    tool.set_embedding_identity("ollama", "nomic-embed-text")
    tool._setup_collection("ollama")
    return tool.pipeline_inputs().fingerprint


@pytest.mark.parametrize(
    "change",
    [
        {"id_field": "title"},
        {"delimiter": ";"},
        {"meta_fields": None},
        {"field_separator": " | "},
    ],
)
def test_structured_identity_and_content_settings_change_the_fingerprint(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    assert _structured(tmp_path) != _structured(tmp_path, **change)


def test_structured_settings_that_do_not_change_records_keep_the_fingerprint(
    tmp_path: Path,
) -> None:
    assert _structured(tmp_path) == _structured(tmp_path, top_k=9)


def _hier(generator: object | None, **overrides: Any) -> str:
    settings: dict[str, Any] = {
        "name": "docs",
        "description": "Docs",
        "source": "docs",
        "contextual_embeddings": True,
    }
    settings.update(overrides)
    tool = HierarchicalDocumentTool(HierarchicalDocumentToolConfig(**settings))
    if generator is not None:
        tool.set_context_generator(generator)
    tool.set_embedding_identity("ollama", "nomic-embed-text")
    tool._setup_collection("ollama")
    return tool.pipeline_inputs().fingerprint


def _litellm(model: str, max_tokens: int = 100) -> LLMContextGenerator:
    return LLMContextGenerator(
        model_spec=LiteLLMModelSpec(model=model), max_context_tokens=max_tokens
    )


def _context_model(name: str) -> LLMProvider:
    return LLMProvider(provider=ProviderEnum.OPENAI, name=name, api_key="sk-test")


def test_context_model_changes_the_fingerprint() -> None:
    first = _hier(_litellm("gpt-4o-mini"), context_model=_context_model("gpt-4o-mini"))
    second = _hier(_litellm("gpt-4o"), context_model=_context_model("gpt-4o"))
    assert first != second


def test_context_max_tokens_changes_the_fingerprint() -> None:
    model = _context_model("gpt-4o-mini")
    first = _hier(_litellm("gpt-4o-mini", 100), context_model=model)
    second = _hier(
        _litellm("gpt-4o-mini", 200), context_model=model, context_max_tokens=200
    )
    assert first != second


def test_context_generator_kind_changes_the_fingerprint() -> None:
    claude = ClaudeSDKContextGenerator(config=ClaudeContextConfig())
    other_model = ClaudeSDKContextGenerator(
        config=ClaudeContextConfig(model="claude-sonnet-4-5")
    )
    assert _hier(claude) != _hier(_litellm("gpt-4o-mini"))
    assert _hier(claude) != _hier(other_model)


def test_context_off_ignores_the_generator_settings() -> None:
    off = {"contextual_embeddings": False}
    assert _hier(None, **off) == _hier(_litellm("gpt-4o"), **off)
    assert _hier(None, **off) != _hier(_litellm("gpt-4o"))


def test_context_policy_is_recorded_in_the_inputs() -> None:
    tool = HierarchicalDocumentTool(
        HierarchicalDocumentToolConfig(
            name="docs", description="Docs", source="docs", contextual_embeddings=True
        )
    )
    tool.set_context_generator(_litellm("gpt-4o-mini"))
    tool.set_embedding_identity("ollama", "nomic-embed-text")
    tool._setup_collection("ollama")
    inputs = tool.pipeline_inputs().as_dict()
    assert inputs["retrieval.contextual_embeddings"] is True
    assert "gpt-4o-mini" in str(inputs["retrieval.context_generator"])
    assert inputs["retrieval.context_max_tokens"] == 100
