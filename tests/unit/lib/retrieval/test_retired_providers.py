"""Retired vector store providers fail at config load, before any backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from holodeck.config.loader import ConfigLoader
from holodeck.dashboard.data_loader import load_all
from holodeck.dashboard.seed_data import build_seed_runs
from holodeck.lib.errors import ConfigError
from holodeck.lib.retired_providers import RETIRED_PROVIDERS
from holodeck.lib.retrieval import (
    CollectionSchema,
    FieldSpec,
    FieldType,
    RetiredProviderError,
    registry,
)
from holodeck.models.config import VectorstoreConfig
from holodeck.models.eval_run import EvalRun
from holodeck.models.tool import DatabaseConfig, VectorstoreTool

SUPPORTED = {
    "in-memory",
    "qdrant",
    "chromadb",
    "postgres",
    "pinecone",
    "weaviate",
    "azure-ai-search",
    "azure-cosmos-nosql",
}
EXPECTED_TEXT = {
    "faiss": (
        "Use 'chromadb' (embedded, local files) or 'in-memory' for local development."
    ),
    "sql-server": "Use 'postgres' (pgvector) for a SQL database.",
    "azure-cosmos-mongo": "Use 'azure-cosmos-nosql'.",
}
RETIRED = sorted(EXPECTED_TEXT)


def _message(provider: str) -> str:
    return (
        f"Vector store provider '{provider}' was removed in HoloDeck 0.8.0. "
        f"{EXPECTED_TEXT[provider]}"
    )


def test_retired_table_matches_the_decision() -> None:
    assert RETIRED_PROVIDERS == EXPECTED_TEXT


def test_registry_lists_only_supported_providers() -> None:
    assert set(registry.supported_providers()) == SUPPORTED
    assert not set(RETIRED) & set(registry.PROVIDERS)


@pytest.mark.parametrize("provider", RETIRED)
def test_create_adapter_raises_retired_provider_error(provider: str) -> None:
    schema = CollectionSchema(
        name="c", dimensions=4, fields=(FieldSpec("text", FieldType.STRING),)
    )
    with pytest.raises(RetiredProviderError) as caught:
        registry.create_adapter(provider, schema, {})
    assert EXPECTED_TEXT[provider] in str(caught.value)


def test_published_schema_lists_only_supported_providers() -> None:
    path = Path(__file__).resolve().parents[4] / "schemas" / "agent.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    providers = schema["$defs"]["DatabaseConfig"]["properties"]["provider"]["enum"]
    assert set(providers) == SUPPORTED


@pytest.mark.parametrize("provider", RETIRED)
def test_database_config_rejects_retired_provider(provider: str) -> None:
    with pytest.raises(ConfigError) as caught:
        DatabaseConfig(provider=provider)  # type: ignore[arg-type]
    assert _message(provider) in str(caught.value)


@pytest.mark.parametrize("provider", RETIRED)
def test_global_vectorstore_config_rejects_retired_provider(provider: str) -> None:
    with pytest.raises(ConfigError) as caught:
        VectorstoreConfig(provider=provider, connection_string="x://y")
    assert _message(provider) in str(caught.value)


@pytest.mark.parametrize("provider", RETIRED)
def test_agent_yaml_fails_at_load_before_any_adapter(
    provider: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("an adapter was created")

    monkeypatch.setattr(registry, "create_adapter", forbidden)
    (tmp_path / "docs").mkdir()
    (tmp_path / "agent.yaml").write_text(
        f"""
name: demo
model:
  provider: openai
  name: gpt-4o
  api_key: test-key
instructions:
  inline: hello
tools:
  - name: kb
    type: vectorstore
    description: Knowledge base
    source: docs/
    database:
      provider: {provider}
      connection_string: local://x
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as caught:
        ConfigLoader().load_agent_yaml(str(tmp_path / "agent.yaml"))
    assert _message(provider) in str(caught.value)


def _run_with_faiss_tool() -> dict[str, Any]:
    run = build_seed_runs()[0].model_dump(mode="json")
    run["metadata"]["agent_config"]["tools"] = [
        {
            "type": "vectorstore",
            "name": "kb",
            "description": "Knowledge base",
            "source": "docs/",
            "database": {"provider": "faiss"},
        }
    ]
    return run


def test_load_all_keeps_runs_whose_agent_used_a_retired_provider(
    tmp_path: Path,
) -> None:
    (tmp_path / "run.json").write_text(json.dumps(_run_with_faiss_tool()))

    runs = load_all(tmp_path)

    assert len(runs) == 1
    tools = runs[0].metadata.agent_config.tools
    assert tools is not None
    tool = tools[0]
    assert isinstance(tool, VectorstoreTool)
    assert isinstance(tool.database, DatabaseConfig)
    assert tool.database.provider == "faiss"


def test_agent_config_still_rejects_a_retired_provider_in_a_run() -> None:
    with pytest.raises(ConfigError, match="was removed in HoloDeck 0.8.0"):
        EvalRun.model_validate(_run_with_faiss_tool())
