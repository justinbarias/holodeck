"""``holodeck vectorstore`` commands through Click's runner (042 T15, AC-08).

The store is a shared registry of fake adapters, so every command in a test
sees what an earlier one wrote. The embedding service is the offline hash
embedding.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from holodeck.cli.main import main
from holodeck.config.loader import load_agent_with_config
from holodeck.lib.tool_initializer import close_tools, initialize_tools
from tests.integration.golden._golden import HashEmbeddingService
from tests.unit.tools.fakes import StoreRegistry

AGENT_YAML = """\
name: cli_agent
model:
  provider: ollama
  name: llama3
  endpoint: http://127.0.0.1:11434
instructions:
  inline: Test agent.
tools:
  - name: kb
    type: vectorstore
    description: Knowledge base
    source: ./docs
    chunk_size: {chunk_size}
    chunk_overlap: 0
    embedding_dimensions: 256
    database:
      provider: in-memory
      collection_name: cli_kb
"""

INSPECT_KEYS = {
    "tool",
    "tool_type",
    "provider",
    "logical_collection",
    "physical_collection",
    "exists",
    "record_count",
    "manifest_supported",
    "manifest_state",
    "stored_fingerprint",
    "current_fingerprint",
    "differences",
    "compatible",
    "needs_reembedding",
    "source_files",
    "chunks",
    "estimated_tokens",
    "previous_collection",
    "migration_target",
    "estimate_note",
}


class Env:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.registry = StoreRegistry()
        self.embeddings = HashEmbeddingService(256)
        self.embed_calls = 0
        docs = root / "docs"
        docs.mkdir()
        for name in ("a", "b"):
            (docs / f"{name}.md").write_text(
                "\n\n".join(
                    f"Paragraph {n} of {name} explains topic {n} at some length."
                    for n in range(6)
                )
            )

    def write_agent(self, chunk_size: int) -> str:
        path = self.root / "agent.yaml"
        path.write_text(AGENT_YAML.format(chunk_size=chunk_size))
        return str(path)

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        return await HashEmbeddingService.generate_embeddings(self.embeddings, texts)

    def patches(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(
            patch(
                "holodeck.lib.retrieval.create_adapter",
                side_effect=self.registry.create,
            )
        )
        stack.enter_context(
            patch(
                "holodeck.lib.retrieval.registry.create_adapter",
                side_effect=self.registry.create,
            )
        )
        stack.enter_context(
            patch.object(
                self.embeddings, "generate_embeddings", side_effect=self._embed
            )
        )
        stack.enter_context(
            patch(
                "holodeck.lib.tool_initializer.create_embedding_service",
                side_effect=lambda *a, **k: self.embeddings,
            )
        )
        return stack

    async def ingest(self, chunk_size: int) -> None:
        agent, _, _ = load_agent_with_config(self.write_agent(chunk_size))
        tools = await initialize_tools(agent)
        await close_tools(tools.values())


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Env]:
    environment = Env(tmp_path)
    with environment.patches():
        yield environment


def _invoke(*args: str, input: str | None = None):  # type: ignore[no-untyped-def]
    return CliRunner().invoke(main, ["vectorstore", *args], input=input)


def test_inspect_json_shape_on_a_new_collection(env: Env) -> None:
    agent = env.write_agent(60)
    result = _invoke("inspect", agent, "--json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    (report,) = data["tools"]
    assert set(report) == INSPECT_KEYS
    assert report["tool"] == "kb"
    assert report["tool_type"] == "vectorstore"
    assert report["provider"] == "in-memory"
    assert report["logical_collection"] == report["physical_collection"] == "cli_kb"
    assert report["exists"] is False
    assert report["compatible"] == "new"
    assert report["manifest_state"] == "none"
    assert report["source_files"] == 2
    assert report["chunks"] > 0
    assert env.registry.writes == []
    assert env.embed_calls == 0


def test_inspect_reports_differences_and_is_read_only(env: Env) -> None:
    asyncio.run(env.ingest(40))
    env.registry.writes.clear()
    env.embed_calls = 0
    agent = env.write_agent(60)

    result = _invoke("inspect", agent, "--tool", "kb")
    assert result.exit_code == 0, result.output
    assert "Compatible:          no" in result.output
    assert "chunker.chunk_size: 40 -> 60" in result.output
    assert "Needs re-embedding:  yes" in result.output
    assert "Estimate:" in result.output

    data = json.loads(_invoke("inspect", agent, "--json").output)
    assert data["tools"][0]["differences"] == [
        {"input": "chunker.chunk_size", "stored": 40, "current": 60}
    ]
    assert env.registry.writes == []
    assert env.embed_calls == 0


def test_inspect_unknown_tool_fails(env: Env) -> None:
    result = _invoke("inspect", env.write_agent(60), "--tool", "nope")
    assert result.exit_code == 1
    assert "no vectorstore or hierarchical_document tool named 'nope'" in (
        result.output
    )


def test_migrate_asks_before_it_writes(env: Env) -> None:
    asyncio.run(env.ingest(40))
    env.registry.writes.clear()
    result = _invoke("migrate", env.write_agent(60), "--tool", "kb", input="n\n")
    assert result.exit_code == 1
    assert "Aborted." in result.output
    assert env.registry.writes == []


def test_migrate_rollback_and_prune(env: Env) -> None:
    asyncio.run(env.ingest(40))
    old_records = len(env.registry.records("cli_kb"))
    agent = env.write_agent(60)

    migrated = _invoke("migrate", agent, "--tool", "kb", "--yes")
    assert migrated.exit_code == 0, migrated.output
    assert "Migration complete" in migrated.output
    assert "Kept for rollback: cli_kb" in migrated.output
    target = next(name for name in env.registry.existing if name.startswith("cli_kb__"))
    assert env.registry.records(target)

    again = _invoke("migrate", agent, "--tool", "kb", "--yes")
    assert again.exit_code == 0
    assert "already matches" in again.output

    rolled = _invoke("rollback", agent, "--tool", "kb", "--yes")
    assert rolled.exit_code == 0, rolled.output
    assert "Active collection: cli_kb" in rolled.output

    pruned = _invoke("prune", agent, "--tool", "kb", "--yes")
    assert pruned.exit_code == 0, pruned.output
    assert f"Deleted collection: {target}" in pruned.output
    assert target not in env.registry.existing
    assert len(env.registry.records("cli_kb")) == old_records

    nothing = _invoke("prune", agent, "--tool", "kb", "--yes")
    assert nothing.exit_code == 1
    assert "no kept previous collection" in nothing.output


def test_migrate_requires_a_tool_name(env: Env) -> None:
    result = _invoke("migrate", env.write_agent(60))
    assert result.exit_code == 2
    assert "--tool" in result.output
