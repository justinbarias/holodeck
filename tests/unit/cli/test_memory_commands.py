"""Tests for ``holodeck memory`` (spec 043: CLI-01, CLI-02, CLI-03, ING-08).

``open_memory`` is replaced by a session over a real ``TemporalMemory`` and
``FakeBackend``, with small chunk sources over real files. No network,
database or model call.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result
from pydantic import SecretStr

from holodeck.cli.commands import memory as memory_commands
from holodeck.cli.commands.memory import memory
from holodeck.lib.memory.backends.fake import FakeBackend
from holodeck.lib.memory.chunk_source import ChunkMatch, ChunkRef, SourceDocument
from holodeck.lib.memory.service import TemporalMemory, episode_id_for
from holodeck.lib.memory.types import MemoryOperation
from holodeck.models.agent import Agent
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import TemporalMemoryToolConfig
from holodeck.tools.temporal_memory import MemorySession

pytestmark = pytest.mark.unit

NAMESPACE = "law_memory"
TEXTS = ("Rate is 10 dollars.", "Rate is 12 dollars.", "Threshold is 500 dollars.")


class FileSource:
    """A ``ChunkSource`` over one real file with one chunk per text."""

    def __init__(self, name: str, path: Path, texts: Sequence[str]) -> None:
        self._name = name
        self._path = path
        self._texts = texts

    @property
    def name(self) -> str:
        return self._name

    async def documents(self) -> Sequence[SourceDocument]:
        return [SourceDocument(self._name, str(self._path), self._path, True)]

    async def document(self, source_uri: str) -> SourceDocument | None:
        return SourceDocument(self._name, source_uri, self._path, True)

    async def chunks(self, document: SourceDocument) -> Sequence[ChunkRef]:
        return [
            ChunkRef(self._name, f"c{i}", document.source_uri, f"s{i}", i, text)
            for i, text in enumerate(self._texts)
        ]

    async def find(self, query: str, top_k: int) -> list[ChunkMatch]:
        return []


def _agent() -> Agent:
    return Agent(
        name="a",
        model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
        instructions={"inline": "hi"},
        tools=[
            {
                "name": "acts",
                "type": "hierarchical_document",
                "description": "acts",
                "source": "acts/",
            },
            {
                "name": NAMESPACE,
                "type": "temporal_memory",
                "description": "memory",
                "source_tools": ["acts"],
                "backend": {"driver": "kuzu"},
            },
        ],
    )


class Harness:
    def __init__(self, tmp_path: Path) -> None:
        document = tmp_path / "acts.md"
        document.write_text("# Acts\n")
        self.backend = FakeBackend()
        self.seconds_per_episode: float | None = None
        self.agent = _agent()
        self.sources = [FileSource("acts", document, TEXTS)]
        self.agent_file = tmp_path / "agent.yaml"
        self.agent_file.write_text("name: a\n")
        self.opened: list[dict[str, Any]] = []

    def episode_ids(self) -> list[Any]:
        return [
            episode_id_for(NAMESPACE, f"s{i}", text) for i, text in enumerate(TEXTS)
        ]

    def run(self, *args: str, input: str | None = None) -> Result:
        return CliRunner().invoke(
            memory, [args[0], str(self.agent_file), *args[1:]], input=input
        )


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    harness = Harness(tmp_path)
    config = harness.agent.tools[1]  # type: ignore[index]
    assert isinstance(config, TemporalMemoryToolConfig)

    @asynccontextmanager
    async def fake_open(agent: Agent, **kwargs: Any) -> AsyncIterator[MemorySession]:
        harness.opened.append(kwargs)
        service = TemporalMemory(
            namespace=config.resolved_namespace,
            backend=harness.backend,
            sources=(
                [] if kwargs.get("initialize_sources") is False else harness.sources
            ),
            seconds_per_episode=harness.seconds_per_episode,
        )
        # The fake store lives in memory, so it is not closed between runs.
        yield MemorySession(memory=service, backend=harness.backend, config=config)

    def fake_load(agent_config: str) -> tuple[Agent, None, str]:
        return harness.agent, None, str(Path(agent_config).parent)

    monkeypatch.setattr("holodeck.tools.temporal_memory.open_memory", fake_open)
    monkeypatch.setattr(memory_commands, "_load", fake_load)
    return harness


def _value(result: Result, label: str) -> str:
    for line in result.output.splitlines():
        if line.startswith(label):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"no '{label}' line in:\n{result.output}")


def test_cli_01_ingest_writes_every_new_episode_then_skips_them(
    harness: Harness,
) -> None:
    first = harness.run("ingest")

    assert first.exit_code == 0, first.output
    assert _value(first, "Documents") == "1"
    assert _value(first, "Episodes new") == "3"
    assert _value(first, "Episodes skipped") == "0"
    assert _value(first, "Episodes written") == "3"
    assert _value(first, "LLM calls") == "3"
    assert _value(first, "Tokens in") == "not reported"
    assert _value(first, "Tokens out") == "not reported"
    assert _value(first, "Duration").endswith(" s")
    assert "Estimated duration" not in first.output
    assert harness.backend.calls["ingest"] == 3

    second = harness.run("ingest")

    assert second.exit_code == 0, second.output
    assert _value(second, "Episodes new") == "0"
    assert _value(second, "Episodes skipped") == "3"
    assert _value(second, "Episodes written") == "0"
    assert harness.backend.calls["ingest"] == 3


def test_cli_01_ingest_passes_each_tool_option_to_the_service(
    harness: Harness,
) -> None:
    ok = harness.run("ingest", "--tool", "acts")

    assert ok.exit_code == 0, ok.output
    assert _value(ok, "Episodes written") == "3"


def test_cli_01_unknown_tool_exits_non_zero_with_the_service_message(
    harness: Harness,
) -> None:
    result = harness.run("ingest", "--tool", "nope")

    assert result.exit_code != 0
    assert "unknown source tool 'nope'; known: acts" in result.output
    assert "Traceback" not in result.output
    assert harness.backend.calls["ingest"] == 0


def test_cli_01_failed_episode_exits_non_zero_and_names_it(harness: Harness) -> None:
    failing = harness.episode_ids()[1]
    harness.backend.fail_on.add(failing)

    result = harness.run("ingest")

    assert result.exit_code != 0
    assert f"Failed episode: {failing}" in result.output
    assert "extraction failed (set by the test)" in result.output
    assert _value(result, "Episodes written") == "1"


def test_ing_08_dry_run_counts_new_episodes_and_writes_nothing(
    harness: Harness,
) -> None:
    result = harness.run("ingest", "--dry-run")

    assert result.exit_code == 0, result.output
    assert _value(result, "Episodes new") == "3"
    assert _value(result, "Episodes written") == "0"
    assert _value(result, "LLM calls") == "0"
    assert harness.backend.calls["ingest"] == 0
    assert _value(result, "Estimated duration") == "no estimate for this model"


def test_ing_08_dry_run_previews_the_sources_and_a_full_run_does_not(
    harness: Harness,
) -> None:
    harness.run("ingest", "--dry-run")
    harness.run("ingest")

    assert [opened["preview_sources"] for opened in harness.opened] == [True, False]


def test_ing_08_dry_run_estimate_is_minutes_and_seconds(harness: Harness) -> None:
    harness.seconds_per_episode = 11.0

    result = harness.run("ingest", "--dry-run")

    assert result.exit_code == 0, result.output
    assert _value(result, "Estimated duration") == "0 min 33 s"
    assert harness.backend.calls["ingest"] == 0


def test_ing_08_estimate_over_a_minute(harness: Harness) -> None:
    harness.seconds_per_episode = 25.0

    result = harness.run("ingest", "--dry-run")

    assert _value(result, "Estimated duration") == "1 min 15 s"


def test_cli_02_status_prints_backend_namespace_count_and_capabilities(
    harness: Harness,
) -> None:
    harness.run("ingest")

    result = harness.run("status")

    assert result.exit_code == 0, result.output
    assert _value(result, "Backend") == "fake"
    assert _value(result, "Namespace") == NAMESPACE
    assert _value(result, "Episodes") == "3"
    lines = [line.strip() for line in result.output.splitlines()]
    printed = lines[lines.index("Capabilities:") + 1 :]
    assert printed == sorted(operation.value for operation in MemoryOperation)
    assert len(printed) == len(MemoryOperation) > 10


def test_cli_02_status_does_not_initialize_the_source_tools(harness: Harness) -> None:
    result = harness.run("status")

    assert result.exit_code == 0, result.output
    assert harness.opened == [
        {
            "base_dir": str(harness.agent_file.parent),
            "execution_config": None,
            "initialize_sources": False,
        }
    ]
    assert _value(result, "Episodes") == "0"


def test_cli_03_purge_without_yes_asks_and_no_purges_nothing(
    harness: Harness,
) -> None:
    harness.run("ingest")

    result = harness.run("purge", input="n\n")

    assert result.exit_code == 0, result.output
    assert harness.backend.calls["purge"] == 0
    assert f"namespace '{NAMESPACE}'" in result.output
    assert "backend 'graphiti'" in result.output
    assert "Nothing was purged" in result.output
    assert harness.backend.calls["episode_count"] == 0
    assert harness.backend.calls["ingest"] == 3


def test_cli_03_purge_with_yes_clears_the_namespace(harness: Harness) -> None:
    harness.run("ingest")

    result = harness.run("purge", "--yes")

    assert result.exit_code == 0, result.output
    assert harness.backend.calls["purge"] == 1
    assert f"Cleared namespace '{NAMESPACE}'" in result.output
    assert "(3 episodes)" in result.output
    assert "Remove every episode" not in result.output


def test_cli_03_purge_confirmed_by_answering_yes(harness: Harness) -> None:
    harness.run("ingest")

    result = harness.run("purge", input="y\n")

    assert result.exit_code == 0, result.output
    assert harness.backend.calls["purge"] == 1


def test_cli_unavailable_backend_prints_one_line_and_exits_non_zero(
    harness: Harness,
) -> None:
    harness.backend.available = False

    result = harness.run("ingest")

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert len(lines) == 1
    assert "unavailable" in lines[0]


def test_cli_agent_without_memory_tool_prints_one_line(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    from holodeck.lib.memory.errors import MemoryConfigError

    @asynccontextmanager
    async def refuse(agent: Agent, **kwargs: Any) -> AsyncIterator[MemorySession]:
        raise MemoryConfigError("agent has no temporal_memory tool")
        yield  # pragma: no cover

    monkeypatch.setattr("holodeck.tools.temporal_memory.open_memory", refuse)

    result = harness.run("ingest")

    assert result.exit_code != 0
    assert result.output.strip() == "Error: agent has no temporal_memory tool"


def test_cli_a_credential_in_a_backend_error_is_not_printed(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    from holodeck.lib.memory.backends import graphiti_clients
    from holodeck.lib.memory.backends.graphiti import GraphitiBackend
    from holodeck.models.llm import LLMProvider, ProviderEnum
    from holodeck.models.tool import GraphBackendConfig

    secret = "hunter2-secret"  # noqa: S105 - a canary, not a credential

    def build_graphiti(*args: object) -> None:
        raise RuntimeError(f"authentication failed for password {secret}")

    class NoEmbeddings:
        async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
            return []

    monkeypatch.setattr(graphiti_clients, "build_graphiti", build_graphiti)
    harness.backend = GraphitiBackend(  # type: ignore[assignment]
        GraphBackendConfig(
            driver="neo4j", uri="bolt://127.0.0.1:1", password=SecretStr(secret)
        ),
        LLMProvider(provider=ProviderEnum.OPENAI, name="m", api_key=SecretStr("k")),
        NoEmbeddings(),
    )

    result = harness.run("status")

    assert result.exit_code != 0
    assert "authentication failed for password ***" in result.output
    assert secret not in result.output


def test_cli_a_yaml_that_does_not_load_prints_one_line(tmp_path: Path) -> None:
    broken = tmp_path / "agent.yaml"
    broken.write_text("name: [\n")

    result = CliRunner().invoke(memory, ["status", str(broken)])

    assert result.exit_code != 0
    assert result.output.startswith("Error: ")
    assert "Traceback" not in result.output


def test_cli_memory_group_lists_its_commands() -> None:
    result = CliRunner().invoke(memory, ["--help"])

    assert result.exit_code == 0
    for command in ("ingest", "status", "purge"):
        assert command in result.output


@pytest.mark.parametrize("argv", [["--help"], ["init", "--help"]])
def test_cli_memory_stack_is_not_imported_by_help_or_init(argv: list[str]) -> None:
    # No memory module at all: the group is lazy, and loading an agent model
    # does not import the memory package.
    watched = (
        "holodeck.lib.memory",
        "holodeck.tools.temporal_memory",
        "holodeck.lib.memory.service",
        "holodeck.lib.memory.registry",
        "holodeck.lib.memory.backends.graphiti",
        "graphiti_core",
    )
    code = (
        "import json, sys\n"
        "from holodeck.cli.main import main\n"
        "try:\n"
        f"    main({argv!r}, standalone_mode=False)\n"
        "except SystemExit:\n"
        "    pass\n"
        f"print(json.dumps([m for m in {watched!r} if m in sys.modules]))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )

    assert json.loads(out.stdout.strip().splitlines()[-1]) == []
