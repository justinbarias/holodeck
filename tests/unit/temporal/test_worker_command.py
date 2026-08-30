"""``holodeck worker`` wiring (spec 040, T8).

The command is checked without a Temporal server and without a model: the
client connect and the ``Worker`` class are replaced with recorders, and the
assertions are about what the command *asks for* — the data converter, the
tracing interceptor, the task queue, and one activity per configured node.

The lazy-import contract gets its own subprocess probe: importing the command
module must not pull ``temporalio`` in, or ``holodeck --help`` would break on
an installation without the ``temporal`` extra.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
import temporalio.client
import temporalio.worker
from click.testing import CliRunner
from temporalio import activity
from temporalio.contrib.opentelemetry import TracingInterceptor
from temporalio.contrib.pydantic import pydantic_data_converter

import holodeck.cli.commands.worker as worker_module
from holodeck.cli.commands.worker import GRACEFUL_SHUTDOWN_SECONDS, worker
from holodeck.lib.errors import ConfigError

pytestmark = pytest.mark.unit

AGENT_YAML = """\
name: {name}
description: Edge agent under test
model:
  provider: anthropic
  name: claude-sonnet-4-20250514
instructions:
  inline: "Extract the applicant's income evidence."
response_format:
  type: object
  properties:
    net_income:
      type: number
"""

GATE_JSON = """\
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {"net_income": {"type": "number"}},
  "required": ["net_income"]
}
"""

WORKER_YAML = """\
temporal:
  address: temporal.internal:7233
  namespace: hardship-ns
  task_queue: hardship
nodes:
  - id: evidence
    edge:
      agent: agents/evidence.yaml
    gate:
      schema: gates/evidence.schema.json
  - id: letter
    edge:
      agent: agents/letter.yaml
    gate:
      schema: gates/letter.schema.json
"""


class RecordingWorker:
    """Stands in for ``temporalio.worker.Worker``.

    Records the constructor arguments and completes its context manager
    immediately, so the command's shutdown path runs without a real poller.
    """

    last: RecordingWorker | None = None

    def __init__(self, client: Any, **kwargs: Any) -> None:
        self.client = client
        self.kwargs = kwargs
        RecordingWorker.last = self

    async def __aenter__(self) -> RecordingWorker:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        return None


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Write a two-node worker project (config, agents, gates) under tmp_path.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The project directory containing ``worker.yaml``.
    """
    (tmp_path / "agents").mkdir()
    (tmp_path / "gates").mkdir()
    for node in ("evidence", "letter"):
        (tmp_path / "agents" / f"{node}.yaml").write_text(
            AGENT_YAML.format(name=f"hardship-{node}"), encoding="utf-8"
        )
        (tmp_path / "gates" / f"{node}.schema.json").write_text(
            GATE_JSON, encoding="utf-8"
        )
    (tmp_path / "worker.yaml").write_text(
        textwrap.dedent(WORKER_YAML), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def connect_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace ``Client.connect`` with a recorder returning a sentinel client.

    Args:
        monkeypatch: pytest's patcher.

    Returns:
        The list the recorder appends each call's arguments to.
    """
    calls: list[dict[str, Any]] = []
    sentinel = object()

    async def fake_connect(target_host: str, **kwargs: Any) -> Any:
        calls.append({"target_host": target_host, **kwargs})
        return sentinel

    # Patch the module object, not a dotted string: the command imports the
    # class lazily at call time, so the attribute is what it resolves.
    monkeypatch.setattr(temporalio.client.Client, "connect", fake_connect)
    return calls


@pytest.fixture
def recording_worker(monkeypatch: pytest.MonkeyPatch) -> type[RecordingWorker]:
    """Replace ``temporalio.worker.Worker`` with :class:`RecordingWorker`.

    Also stands down the command's one unbounded wait, so an invocation
    returns as soon as the worker would have started polling. The signal
    wiring that normally ends that wait is tested directly in
    :class:`TestShutdownWiring`.

    Args:
        monkeypatch: pytest's patcher.

    Returns:
        The recorder class.
    """
    RecordingWorker.last = None
    monkeypatch.setattr(temporalio.worker, "Worker", RecordingWorker)

    async def immediate(shutdown: asyncio.Event) -> None:
        shutdown.set()

    monkeypatch.setattr(worker_module, "_wait_for_shutdown", immediate)
    return RecordingWorker


@pytest.fixture(autouse=True)
def forbid_real_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blow up if a concrete backend is constructed (proves zero LLM calls)."""
    from holodeck.lib.backends.claude_backend import ClaudeBackend

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a real backend was constructed")

    monkeypatch.setattr(ClaudeBackend, "__init__", _boom)


class TestHelp:
    """The command surface is available without touching Temporal."""

    def test_help_exits_zero(self) -> None:
        # Act
        result = CliRunner().invoke(worker, ["--help"])

        # Assert
        assert result.exit_code == 0
        assert "--task-queue" in result.output
        assert "--config" in result.output

    def test_registered_on_the_main_group(self) -> None:
        # Arrange
        from holodeck.cli.main import main

        # Act
        result = CliRunner().invoke(main, ["worker", "--help"])

        # Assert
        assert result.exit_code == 0


_PROBE = """
import sys
import holodeck.cli.commands.worker
leaked = [n for n in sys.modules if n == "temporalio" or n.startswith("temporalio.")]
print("LEAKED:", ", ".join(sorted(leaked)) if leaked else "none")
sys.exit(1 if leaked else 0)
"""


class TestLazyImports:
    """Importing the command module must not require the ``temporal`` extra."""

    def test_module_import_pulls_no_temporalio(self) -> None:
        # Act — a subprocess, so a temporalio already imported by the test
        # session cannot mask a regression.
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", _PROBE],
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Assert
        assert result.returncode == 0, (
            "importing holodeck.cli.commands.worker pulled in temporalio "
            f"(stdout: {result.stdout.strip()!r}, "
            f"stderr: {result.stderr.strip()!r})"
        )


class TestStartup:
    """A successful start registers exactly the configured activities."""

    def test_registers_one_activity_per_node(
        self,
        project: Path,
        connect_calls: list[dict[str, Any]],
        recording_worker: type[RecordingWorker],
    ) -> None:
        # Act
        result = CliRunner().invoke(worker, ["--config", str(project / "worker.yaml")])

        # Assert
        assert result.exit_code == 0, result.output
        assert recording_worker.last is not None
        activities = recording_worker.last.kwargs["activities"]
        names = [activity._Definition.must_from_callable(fn).name for fn in activities]
        assert names == ["evidence", "letter"]
        # Activities only: no workflows are registered at all.
        assert "workflows" not in recording_worker.last.kwargs

    def test_worker_uses_the_configured_task_queue_and_grace_period(
        self,
        project: Path,
        connect_calls: list[dict[str, Any]],
        recording_worker: type[RecordingWorker],
    ) -> None:
        # Act
        result = CliRunner().invoke(worker, ["--config", str(project / "worker.yaml")])

        # Assert
        assert result.exit_code == 0, result.output
        assert recording_worker.last is not None
        assert recording_worker.last.kwargs["task_queue"] == "hardship"
        grace = recording_worker.last.kwargs["graceful_shutdown_timeout"]
        assert grace.total_seconds() == GRACEFUL_SHUTDOWN_SECONDS

    def test_connect_uses_converter_interceptor_and_connection(
        self,
        project: Path,
        connect_calls: list[dict[str, Any]],
        recording_worker: type[RecordingWorker],
    ) -> None:
        # Act
        result = CliRunner().invoke(worker, ["--config", str(project / "worker.yaml")])

        # Assert
        assert result.exit_code == 0, result.output
        assert len(connect_calls) == 1
        call = connect_calls[0]
        assert call["target_host"] == "temporal.internal:7233"
        assert call["namespace"] == "hardship-ns"
        assert call["data_converter"] is pydantic_data_converter
        assert call["tls"] is False
        assert any(
            isinstance(item, TracingInterceptor) for item in call["interceptors"]
        )

    def test_tls_flag_reaches_connect(
        self,
        project: Path,
        connect_calls: list[dict[str, Any]],
        recording_worker: type[RecordingWorker],
    ) -> None:
        # Arrange
        config = project / "worker.yaml"
        config.write_text(
            config.read_text(encoding="utf-8").replace(
                "  task_queue: hardship", "  task_queue: hardship\n  tls: true"
            ),
            encoding="utf-8",
        )

        # Act
        result = CliRunner().invoke(worker, ["--config", str(config)])

        # Assert
        assert result.exit_code == 0, result.output
        assert connect_calls[0]["tls"] is True

    def test_startup_lines_name_the_connection_and_activities(
        self,
        project: Path,
        connect_calls: list[dict[str, Any]],
        recording_worker: type[RecordingWorker],
    ) -> None:
        # Act
        result = CliRunner().invoke(worker, ["--config", str(project / "worker.yaml")])

        # Assert
        assert "temporal.internal:7233" in result.output
        assert "hardship-ns" in result.output
        assert "hardship" in result.output
        assert "evidence" in result.output
        assert "letter" in result.output


class TestTaskQueueOverride:
    """``--task-queue`` beats the configured value."""

    def test_override_reaches_the_worker(
        self,
        project: Path,
        connect_calls: list[dict[str, Any]],
        recording_worker: type[RecordingWorker],
    ) -> None:
        # Act
        result = CliRunner().invoke(
            worker,
            ["--config", str(project / "worker.yaml"), "--task-queue", "urgent"],
        )

        # Assert
        assert result.exit_code == 0, result.output
        assert recording_worker.last is not None
        assert recording_worker.last.kwargs["task_queue"] == "urgent"

    def test_override_leaves_the_rest_of_the_connection_intact(
        self,
        project: Path,
        connect_calls: list[dict[str, Any]],
        recording_worker: type[RecordingWorker],
    ) -> None:
        # Act
        CliRunner().invoke(
            worker,
            ["--config", str(project / "worker.yaml"), "--task-queue", "urgent"],
        )

        # Assert
        assert connect_calls[0]["namespace"] == "hardship-ns"
        assert connect_calls[0]["target_host"] == "temporal.internal:7233"

    def test_blank_override_refused(
        self,
        project: Path,
        connect_calls: list[dict[str, Any]],
        recording_worker: type[RecordingWorker],
    ) -> None:
        # Act
        result = CliRunner().invoke(
            worker, ["--config", str(project / "worker.yaml"), "--task-queue", "  "]
        )

        # Assert
        assert result.exit_code == 1
        assert "task_queue" in result.output
        assert connect_calls == []


class TestFailurePaths:
    """Every HoloDeck error reaches the operator as a message, not a traceback."""

    def test_missing_config_exits_one(self, tmp_path: Path) -> None:
        # Act
        result = CliRunner().invoke(worker, ["--config", str(tmp_path / "absent.yaml")])

        # Assert
        assert result.exit_code == 1
        assert "Failed to start the Temporal worker" in result.output
        assert "Traceback" not in result.output

    def test_missing_extra_shows_the_guard_message(
        self, monkeypatch: pytest.MonkeyPatch, project: Path
    ) -> None:
        # Arrange — stand in for the T1 guard firing on the lazy import.
        guard = ConfigError(
            "dependencies",
            "The Temporal integration requires the 'temporal' extra. Install "
            "it with:\n  uv add 'holodeck-ai[temporal]'",
        )

        def _raise(*args: Any, **kwargs: Any) -> None:
            raise guard

        import holodeck.temporal.worker_config as worker_config_module

        monkeypatch.setattr(worker_config_module, "load_worker_config", _raise)

        # Act
        result = CliRunner().invoke(worker, ["--config", str(project / "worker.yaml")])

        # Assert
        assert result.exit_code == 1
        assert "temporal" in result.output
        assert "Traceback" not in result.output

    def test_node_agent_escape_refused_before_connect(
        self,
        project: Path,
        connect_calls: list[dict[str, Any]],
        recording_worker: type[RecordingWorker],
    ) -> None:
        # Arrange
        config = project / "worker.yaml"
        config.write_text(
            config.read_text(encoding="utf-8").replace(
                "agents/evidence.yaml", "../evidence.yaml"
            ),
            encoding="utf-8",
        )

        # Act
        result = CliRunner().invoke(worker, ["--config", str(config)])

        # Assert
        assert result.exit_code == 1
        assert connect_calls == []


class TestShutdownWiring:
    """A termination signal sets the event the worker waits on."""

    def test_sigterm_sets_the_shutdown_event(self) -> None:
        # Arrange
        import signal

        async def scenario() -> bool:
            event = asyncio.Event()
            loop = asyncio.get_running_loop()
            try:
                loop.add_signal_handler(signal.SIGTERM, lambda: None)
            except (NotImplementedError, RuntimeError, ValueError):
                pytest.skip("this platform cannot install loop signal handlers")
            loop.remove_signal_handler(signal.SIGTERM)

            worker_module._install_signal_handlers(event)
            try:
                # Safe only because the handler above is now installed: the
                # loop absorbs the signal instead of the default terminate.
                signal.raise_signal(signal.SIGTERM)
                await asyncio.wait_for(event.wait(), timeout=5)
            finally:
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.remove_signal_handler(sig)
            return event.is_set()

        # Act
        was_set = asyncio.run(scenario())

        # Assert
        assert was_set is True
