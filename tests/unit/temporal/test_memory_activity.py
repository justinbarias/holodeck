"""Temporal memory ingestion activity and worker refusal (spec 043, T15).

HST-03: under Temporal, ingestion and every memory tool call run in
activities. ING-07: a worker refuses ``ingest_on_start: true`` and names the
ingestion activity.

The source tools are real ``HierarchicalDocumentTool`` instances on the
in-memory store with keyword search; the backend is ``FakeBackend``, injected
at the composition module's backend factory. The activity runs in Temporal's
``ActivityEnvironment``: no server, network, database or model.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
import pytest_asyncio
from temporalio import activity, workflow
from temporalio.api.failure.v1 import Failure
from temporalio.converter import DefaultFailureConverter, DefaultPayloadConverter
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment
from temporalio.worker.workflow_sandbox import (
    SandboxedWorkflowRunner,
    SandboxRestrictions,
)

from holodeck.lib import tool_initializer
from holodeck.lib.backends.base import BackendInitError
from holodeck.lib.errors import ConfigError
from holodeck.lib.memory.backends.fake import FakeBackend
from holodeck.lib.memory.errors import (
    MemoryBackendUnavailableError,
    MemoryNotSupportedError,
)
from holodeck.lib.memory.types import EpisodeIn, IngestReport
from holodeck.models.tool import HierarchicalDocumentToolConfig, SearchMode
from holodeck.models.workflow import EdgeNode
from holodeck.temporal.activity import agent_activity
from holodeck.temporal.memory_activity import (
    EPISODE_FAILED_ERROR_TYPE,
    memory_ingest_activity,
)
from holodeck.temporal.models import (
    MEMORY_INGEST_ACTIVITY,
    MemoryIngestInput,
    MemoryIngestResult,
)
from holodeck.tools import temporal_memory as composition
from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool
from tests.unit.temporal import sandbox_memory_workflows
from tests.unit.temporal.test_worker_command import (  # noqa: F401 - fixtures
    RecordingWorker,
    connect_calls,
    recording_worker,
)

pytestmark = [pytest.mark.unit]

QUOTED = "pension rate is 100 dollars"
# Several sections per document: keyword search needs more than two chunks.
DOCS = {
    "alpha": (
        "---\neffective_date: 2020-01-01\n---\n# Rates\n\n"
        f"## Pension\n\nThe {QUOTED} per fortnight.\n\n"
        "## Carer\n\nThe carer payment is paid weekly to a carer.\n\n"
        "## Youth\n\nYouth support is paid to students under age limits.\n"
    ),
    "beta": (
        "---\neffective_date: 2022-07-01\n---\n# Allowance\n\n"
        "## Rent\n\nThe rent allowance is 50 dollars per fortnight.\n\n"
        "## Travel\n\nTravel concessions apply to public transport.\n\n"
        "## Phone\n\nA phone allowance is paid each quarter.\n"
    ),
}
# The corpus above, as the service sees it.
DOCUMENTS = 2
EPISODES = 6
# Episodes written before the "carer payment" episode, oldest first.
FAILS_AFTER = 1

AGENT_YAML = """\
name: law-agent
description: Memory agent under test
model:
  provider: openai
  name: gpt-4o
instructions:
  inline: "Answer from memory."
response_format:
  type: object
  properties:
    answer:
      type: string
tools:
  - name: alpha
    type: hierarchical_document
    description: alpha
    source: alpha/
    search_mode: keyword
  - name: beta
    type: hierarchical_document
    description: beta
    source: beta/
    search_mode: keyword
  - name: law_memory
    type: temporal_memory
    description: Law memory.
    source_tools: [alpha, beta]
    backend: {{driver: kuzu}}
    ingest_on_start: {ingest_on_start}
"""

GATE = '{"type": "object", "properties": {"answer": {"type": "string"}}}'


class _KeptBackend(FakeBackend):
    """A ``FakeBackend`` whose data survives ``close``, as a real store's does.

    Attributes:
        closed: How many times ``close`` was called.
        fail_text: An episode whose text contains it fails extraction, with
            an error that quotes the text.
        gate: When set, ``ingest`` waits on it (for the cancellation test).
        entered: Set when ``ingest`` is first entered.
    """

    def __init__(self) -> None:
        super().__init__(extract=self._extract_or_fail)
        self.closed = 0
        self.fail_text: str | None = None
        self.gate: asyncio.Event | None = None
        self.entered = asyncio.Event()

    def _extract_or_fail(self, episode: EpisodeIn) -> Sequence[tuple[str, str]]:
        if self.fail_text is not None and self.fail_text in episode.text:
            raise RuntimeError(f"extraction failed on: {episode.text}")
        return [("subject", episode.text)]

    async def ingest(
        self, namespace: str, episodes: Sequence[EpisodeIn], instructions: str
    ) -> IngestReport:
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        return await super().ingest(namespace, episodes, instructions)

    async def close(self) -> None:
        self.calls["close"] += 1
        self.closed += 1


class _Rig:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.backend = _KeptBackend()
        self.tool_closes = 0
        self.initialize_kwargs: list[dict[str, Any]] = []
        self.node = EdgeNode(
            id="law",
            edge={"agent": "agents/law.yaml"},  # type: ignore[arg-type]
            gate={"schema": "gates/law.schema.json"},  # type: ignore[arg-type]
        )

    def write_agent(self, *, ingest_on_start: bool) -> None:
        flag = "true" if ingest_on_start else "false"
        (self.root / "agents" / "law.yaml").write_text(
            AGENT_YAML.format(ingest_on_start=flag)
        )

    def activity(self) -> Any:
        fn = memory_ingest_activity([self.node], self.root)
        assert fn is not None
        return fn

    def in_use(self) -> int:
        """Resources opened and not closed: the backend plus the source tools."""
        opened = 1 + len(DOCS)
        return opened - self.backend.closed - self.tool_closes


@pytest_asyncio.fixture
async def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Rig:
    made = _Rig(tmp_path)
    agents = tmp_path / "agents"
    agents.mkdir()
    for name, text in DOCS.items():
        (agents / name).mkdir()
        (agents / name / "doc.md").write_text(text)
    (tmp_path / "gates").mkdir()
    (tmp_path / "gates" / "law.schema.json").write_text(GATE)
    made.write_agent(ingest_on_start=False)

    async def fake_initialize(agent: Any, **kwargs: Any) -> dict[str, Any]:
        made.initialize_kwargs.append(kwargs)
        tools: dict[str, Any] = {}
        for name in DOCS:
            tool = HierarchicalDocumentTool(
                HierarchicalDocumentToolConfig(
                    name=name,
                    description=name,
                    source=str(agents / name),
                    search_mode=SearchMode.KEYWORD,
                )
            )
            await tool.initialize()
            real_close = tool.close

            async def counted_close(real: Any = real_close) -> None:
                made.tool_closes += 1
                await real()

            tool.close = counted_close  # type: ignore[method-assign]
            tools[name] = tool
        return tools

    monkeypatch.setattr(tool_initializer, "initialize_tools", fake_initialize)
    monkeypatch.setattr(composition, "create_backend", lambda *args: made.backend)
    monkeypatch.setattr(composition, "create_embedding_service", lambda agent: None)
    return made


async def _run(
    fn: Any, beats: list[tuple[Any, ...]] | None = None
) -> MemoryIngestResult:
    env = ActivityEnvironment()
    if beats is not None:
        env.on_heartbeat = lambda *details: beats.append(details)
    result: MemoryIngestResult = await env.run(fn, MemoryIngestInput(node_id="law"))
    return result


# --- registration ----------------------------------------------------------


def test_memory_ingest_activity_is_registered_under_its_name(rig: _Rig) -> None:
    definition = activity._Definition.must_from_callable(rig.activity())
    assert definition.name == MEMORY_INGEST_ACTIVITY == "holodeck.memory_ingest"


def test_memory_ingest_activity_is_none_without_a_memory_agent(
    tmp_path: Path,
) -> None:
    (tmp_path / "plain.yaml").write_text(
        AGENT_YAML.split("tools:")[0], encoding="utf-8"
    )
    node = EdgeNode(
        id="plain",
        edge={"agent": "plain.yaml"},  # type: ignore[arg-type]
        gate={"schema": "gate.json"},  # type: ignore[arg-type]
    )
    assert memory_ingest_activity([node], tmp_path) is None


# --- the activity ----------------------------------------------------------


@pytest.mark.asyncio
async def test_hst_03_memory_ingest_returns_exact_counts(rig: _Rig) -> None:
    result = await _run(rig.activity())

    assert result.node_id == "law"
    assert result.documents == DOCUMENTS
    assert result.episodes_new == EPISODES
    assert result.episodes_skipped == 0
    assert result.episodes_written == EPISODES
    assert len(result.written) == EPISODES
    assert result.failed is None
    assert result.llm_calls == EPISODES
    assert rig.backend.calls["ingest"] == EPISODES
    # The source tools are built for the sources only: no memory tool.
    assert rig.initialize_kwargs[0]["initialize_memory"] is False
    # Round-trips as the wire payload.
    assert MemoryIngestResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.asyncio
async def test_memory_ingest_second_run_skips_every_episode(rig: _Rig) -> None:
    await _run(rig.activity())
    again = await _run(rig.activity())

    assert again.episodes_new == 0
    assert again.episodes_skipped == EPISODES
    assert again.episodes_written == 0
    assert rig.backend.calls["ingest"] == EPISODES


@pytest.mark.asyncio
async def test_memory_ingest_result_carries_no_document_text(rig: _Rig) -> None:
    result = await _run(rig.activity())

    payload = result.model_dump_json()
    for text in DOCS.values():
        for line in text.splitlines():
            if len(line) > 12 and not line.startswith(("#", "---", "effective")):
                assert line not in payload
    assert QUOTED not in payload


@pytest.mark.asyncio
async def test_memory_ingest_heartbeats_per_episode(rig: _Rig) -> None:
    beats: list[tuple[Any, ...]] = []
    await _run(rig.activity(), beats)

    # One beat after each completion check and one after each write.
    assert len(beats) == 2 * EPISODES
    assert beats[-1] == (EPISODES, EPISODES)
    submitted = [beat[1] for beat in beats]
    assert submitted == sorted(submitted)
    assert all(isinstance(value, int) for beat in beats for value in beat)


@pytest.mark.asyncio
async def test_memory_ingest_skips_only_heartbeat_per_check(rig: _Rig) -> None:
    await _run(rig.activity())
    beats: list[tuple[Any, ...]] = []
    await _run(rig.activity(), beats)

    assert beats == [(checked, 0) for checked in range(1, EPISODES + 1)]


@pytest.mark.asyncio
async def test_memory_ingest_failed_episode_is_retryable_and_quotes_no_text(
    rig: _Rig,
) -> None:
    rig.backend.fail_text = QUOTED

    with pytest.raises(ApplicationError) as excinfo:
        await _run(rig.activity())

    error = excinfo.value
    assert error.type == EPISODE_FAILED_ERROR_TYPE
    assert not error.non_retryable
    report = MemoryIngestResult.model_validate(error.details[0])
    assert report.failed is not None
    assert report.failed in str(error)
    assert "holodeck memory ingest" in str(error)
    # The backend's error quotes the episode; neither channel repeats it.
    assert QUOTED not in str(error)
    assert QUOTED not in report.model_dump_json()


@pytest.mark.asyncio
async def test_memory_ingest_retry_submits_only_the_remaining_episodes(
    rig: _Rig,
) -> None:
    rig.backend.fail_text = "carer payment"
    with pytest.raises(ApplicationError) as excinfo:
        await _run(rig.activity())
    first = MemoryIngestResult.model_validate(excinfo.value.details[0])
    submitted_first = rig.backend.calls["ingest"]
    assert first.episodes_written == FAILS_AFTER
    assert submitted_first == FAILS_AFTER + 1

    rig.backend.fail_text = None
    retry = await _run(rig.activity())

    assert retry.episodes_skipped == FAILS_AFTER
    assert retry.episodes_new == EPISODES - FAILS_AFTER
    assert retry.episodes_written == EPISODES - FAILS_AFTER
    assert set(retry.written).isdisjoint(first.written)
    assert rig.backend.calls["ingest"] - submitted_first == EPISODES - FAILS_AFTER
    assert await rig.backend.episode_count("law_memory") == EPISODES


@pytest.mark.asyncio
async def test_memory_ingest_closes_resources_on_success(rig: _Rig) -> None:
    await _run(rig.activity())

    assert rig.backend.closed == 1
    assert rig.tool_closes == len(DOCS)
    assert rig.in_use() == 0


@pytest.mark.asyncio
async def test_memory_ingest_closes_resources_on_failure(rig: _Rig) -> None:
    rig.backend.fail_text = "rent allowance"

    with pytest.raises(ApplicationError):
        await _run(rig.activity())

    assert rig.backend.closed == 1
    assert rig.tool_closes == len(DOCS)


@pytest.mark.asyncio
async def test_memory_ingest_closes_resources_on_cancellation(rig: _Rig) -> None:
    rig.backend.gate = asyncio.Event()
    env = ActivityEnvironment()
    task = asyncio.create_task(
        env.run(rig.activity(), MemoryIngestInput(node_id="law"))
    )
    await asyncio.wait_for(rig.backend.entered.wait(), timeout=10)

    env.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert rig.backend.closed == 1
    assert rig.tool_closes == len(DOCS)


@pytest.mark.asyncio
async def test_memory_ingest_unknown_node_is_non_retryable(rig: _Rig) -> None:
    env = ActivityEnvironment()

    with pytest.raises(ApplicationError) as excinfo:
        await env.run(rig.activity(), MemoryIngestInput(node_id="nope"))

    assert excinfo.value.non_retryable
    assert rig.backend.calls["ingest"] == 0


# --- failures carry no document text, credential or host ------------------

SECRETS = ("confidential", "hunter2", "db.internal.example", "secret")
DOC_BODY = "# Rates\n\n## Pension\n\nThe pension is paid.\n"


def _serialized(error: BaseException) -> Failure:
    """The failure Temporal stores in workflow history for ``error``."""
    failure = Failure()
    DefaultFailureConverter().to_failure(error, DefaultPayloadConverter(), failure)
    return failure


def _assert_clean(error: ApplicationError, type_name: str) -> Failure:
    """Assert the serialized failure names the type and carries no secret."""
    failure = _serialized(error)
    wire = str(failure)
    for secret in SECRETS:
        assert secret not in wire
        assert secret not in str(error)
    assert error.type == type_name
    assert failure.application_failure_info.type == type_name
    assert type_name in failure.message
    assert "'law'" in failure.message
    assert not error.details
    assert not failure.application_failure_info.HasField("details")
    assert error.__cause__ is None
    assert not failure.HasField("cause")
    return failure


async def _failing_run(rig: _Rig) -> ApplicationError:
    with pytest.raises(ApplicationError) as excinfo:
        await _run(rig.activity())
    return excinfo.value


@pytest.mark.asyncio
async def test_memory_ingest_bad_effective_date_quotes_no_document_text(
    rig: _Rig, caplog: pytest.LogCaptureFixture
) -> None:
    (rig.root / "agents" / "alpha" / "doc.md").write_text(
        "---\neffective_date: confidential benefit assessment text\n---\n" + DOC_BODY
    )
    caplog.set_level(logging.DEBUG)

    error = await _failing_run(rig)

    failure = _assert_clean(error, "MemoryValueError")
    assert failure.application_failure_info.non_retryable
    assert error.non_retryable
    _assert_no_secret_logged(caplog)


@pytest.mark.asyncio
async def test_memory_ingest_malformed_front_matter_quotes_no_document_text(
    rig: _Rig, caplog: pytest.LogCaptureFixture
) -> None:
    (rig.root / "agents" / "alpha" / "doc.md").write_text(
        "---\neffective_date: [confidential: {unclosed\n---\n" + DOC_BODY
    )
    caplog.set_level(logging.DEBUG)

    error = await _failing_run(rig)

    failure = _assert_clean(error, "MemoryValueError")
    assert failure.application_failure_info.non_retryable
    _assert_no_secret_logged(caplog)


def _assert_no_secret_logged(caplog: pytest.LogCaptureFixture) -> None:
    for record in caplog.records:
        if record.levelno < logging.INFO:
            continue
        text = record.getMessage()
        for secret in SECRETS:
            assert secret not in text, (record.name, text)
        assert record.exc_info is None


_DSN = "bolt://neo4j:hunter2@db.internal.example:7687"


@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        (MemoryBackendUnavailableError("graphiti", f"cannot reach {_DSN}"), True),
        (BackendInitError(f"login failed for {_DSN}"), False),
        (ConfigError("tools.law_memory", f"bad uri {_DSN}"), False),
        (MemoryNotSupportedError("graphiti", "purge", f"see {_DSN}"), False),
        (RuntimeError("document says: secret"), False),
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, Exception) else "",
)
@pytest.mark.asyncio
async def test_memory_ingest_failure_is_classified_and_sanitized(
    rig: _Rig,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    retryable: bool,
) -> None:
    async def broken(namespace: str, episode_id: UUID) -> bool:
        raise error

    monkeypatch.setattr(rig.backend, "has_episode", broken)
    caplog.set_level(logging.DEBUG)

    raised = await _failing_run(rig)

    failure = _assert_clean(raised, type(error).__name__)
    assert raised.non_retryable is not retryable
    assert failure.application_failure_info.non_retryable is not retryable
    _assert_no_secret_logged(caplog)
    # Resources still close on the failure path.
    assert rig.backend.closed == 1
    assert rig.tool_closes == len(DOCS)


@pytest.mark.asyncio
async def test_memory_ingest_failure_in_tool_start_is_sanitized(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken_start(agent: Any, **kwargs: Any) -> dict[str, Any]:
        raise ConfigError("tools.alpha", f"cannot download {_DSN}")

    monkeypatch.setattr(tool_initializer, "initialize_tools", broken_start)

    error = await _failing_run(rig)

    failure = _assert_clean(error, "ConfigError")
    assert failure.application_failure_info.non_retryable


@pytest.mark.asyncio
async def test_memory_ingest_failed_episode_serializes_without_text(
    rig: _Rig,
) -> None:
    rig.backend.fail_text = QUOTED

    error = await _failing_run(rig)

    failure = _serialized(error)
    assert QUOTED not in str(failure)
    assert not failure.application_failure_info.non_retryable
    assert failure.application_failure_info.type == EPISODE_FAILED_ERROR_TYPE
    assert not failure.HasField("cause")


# --- the worker refusal (ING-07) --------------------------------------------


def test_ing_07_worker_refuses_memory_ingest_on_start(rig: _Rig) -> None:
    rig.write_agent(ingest_on_start=True)

    with pytest.raises(ConfigError) as excinfo:
        agent_activity(rig.node, rig.root)

    message = str(excinfo.value)
    assert MEMORY_INGEST_ACTIVITY in message
    assert "holodeck memory ingest" in message
    assert "ingest_on_start" in message


def test_ing_07_plugin_refuses_memory_ingest_on_start(rig: _Rig) -> None:
    from holodeck.temporal.plugin import HoloDeckPlugin

    rig.write_agent(ingest_on_start=True)

    with pytest.raises(ConfigError, match="holodeck memory ingest"):
        HoloDeckPlugin(nodes=[rig.node], base_dir=rig.root)


def test_ing_07_worker_accepts_memory_agent_without_ingest_on_start(
    rig: _Rig,
) -> None:
    from holodeck.temporal.plugin import HoloDeckPlugin

    fn = agent_activity(rig.node, rig.root)
    plugin = HoloDeckPlugin(nodes=[rig.node], base_dir=rig.root)

    assert activity._Definition.must_from_callable(fn).name == "law"
    assert plugin.memory_ingest is not None
    names = [
        activity._Definition.must_from_callable(f).name for f in plugin.agent_activities
    ]
    names.append(activity._Definition.must_from_callable(plugin.memory_ingest).name)
    assert names == ["law", MEMORY_INGEST_ACTIVITY]


def _write_worker_yaml(rig: _Rig) -> Path:
    path = rig.root / "worker.yaml"
    path.write_text(
        "temporal:\n  task_queue: law\n"
        "nodes:\n  - id: law\n    edge: {agent: agents/law.yaml}\n"
        "    gate: {schema: gates/law.schema.json}\n"
    )
    return path


def test_ing_07_worker_command_refuses_memory_ingest_on_start(
    rig: _Rig,
    connect_calls: list[dict[str, Any]],  # noqa: F811
    recording_worker: type[RecordingWorker],  # noqa: F811
) -> None:
    from click.testing import CliRunner

    from holodeck.cli.commands.worker import worker

    rig.write_agent(ingest_on_start=True)

    result = CliRunner().invoke(worker, ["--config", str(_write_worker_yaml(rig))])

    assert result.exit_code == 1
    assert MEMORY_INGEST_ACTIVITY in result.output
    assert "holodeck memory ingest" in result.output
    assert connect_calls == []
    assert recording_worker.last is None


def test_ing_07_worker_command_registers_memory_ingest_activity(
    rig: _Rig,
    connect_calls: list[dict[str, Any]],  # noqa: F811
    recording_worker: type[RecordingWorker],  # noqa: F811
) -> None:
    from click.testing import CliRunner

    from holodeck.cli.commands.worker import worker

    result = CliRunner().invoke(worker, ["--config", str(_write_worker_yaml(rig))])

    assert result.exit_code == 0, result.output
    assert recording_worker.last is not None
    names = [
        activity._Definition.must_from_callable(fn).name
        for fn in recording_worker.last.kwargs["activities"]
    ]
    assert names == ["law", MEMORY_INGEST_ACTIVITY]
    assert f"activity:   {MEMORY_INGEST_ACTIVITY}" in result.output


# --- the workflow sandbox ---------------------------------------------------


def test_hst_03_sandbox_validates_memory_workflow_with_memory_agent_loaded(
    rig: _Rig,
) -> None:
    # A memory agent is loaded and both its activities are built in this
    # process, so the memory modules are imported when the sandbox runs.
    agent_activity(rig.node, rig.root)
    rig.activity()
    assert "holodeck.lib.memory.service" in sys.modules

    # No exception is the assertion.
    _prepare(sandbox_memory_workflows.MemoryIngestWorkflow)


# The same pass-through list as test_sandbox_safety.py, for the same reasons.
_PASSTHROUGH = ("regex", "holodeck.models.chat", "holodeck.config")


def _prepare(cls: type) -> None:
    """Run Temporal's sandbox validation for a workflow class."""
    definition = workflow._Definition.must_from_class(cls)
    restrictions = SandboxRestrictions.default.with_passthrough_modules(*_PASSTHROUGH)

    async def _run() -> None:
        SandboxedWorkflowRunner(restrictions=restrictions).prepare_workflow(definition)

    asyncio.run(_run())


_PROBE = """
import sys
import {module}  # noqa: F401
leaked = [name for name in {forbidden!r} if name in sys.modules]
print("LEAKED:", ", ".join(leaked) if leaked else "none")
sys.exit(1 if leaked else 0)
"""

_MEMORY_MODULES = (
    "holodeck.lib.memory",
    "graphiti_core",
    "holodeck.tools.temporal_memory",
    "holodeck.tools.temporal_memory_tool",
)


@pytest.mark.parametrize(
    ("module", "forbidden"),
    [
        (
            "tests.unit.temporal.sandbox_memory_workflows",
            (
                *_MEMORY_MODULES,
                "holodeck.tools",
                "holodeck.lib.backends",
                "temporalio.worker",
            ),
        ),
        ("holodeck.temporal.memory_activity", _MEMORY_MODULES),
    ],
)
def test_memory_workflow_imports_no_memory_code(
    module: str, forbidden: tuple[str, ...]
) -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _PROBE.format(module=module, forbidden=forbidden)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=Path(__file__).resolve().parents[3],
    )
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_memory_episode_ids_are_strings_in_the_result() -> None:
    episode = UUID(int=1)
    result = MemoryIngestResult(
        node_id="n",
        documents=1,
        episodes_new=1,
        episodes_skipped=0,
        episodes_written=1,
        written=[str(episode)],
        llm_calls=1,
    )
    assert MemoryIngestResult.model_validate_json(result.model_dump_json()) == result
