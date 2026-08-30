"""Worker-process, replay, and sandbox acceptance for spec 040 Task 11.

AC-4 runs the production ``holodeck worker`` Click path in a separate OS
process. Only its deferred backend selector is replaced by a deterministic
driver; configuration loading, activity construction, client connection,
polling, payload conversion, gates, and graceful signal handling are real.

AC-5 records a completed hardship workflow against the T10 scripted seam and
replays its full :class:`~temporalio.client.WorkflowHistory`. Call counters and
a fail-loudly activity sentinel prove replay executes the deterministic table
logic without executing activities or contacting a backend.

The final test constructs the public :class:`~temporalio.worker.Worker` with
the fixture workflow and activities. Worker construction performs the same
sandbox preparation a consumer registration does, complementing the direct
``prepare_workflow`` unit tests.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import IO, Any

import pytest
from temporalio.client import WorkflowHistory
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

from holodeck.temporal import activity as activity_module
from holodeck.temporal.activity import agent_activity
from holodeck.temporal.models import AgentActivityResult
from holodeck.temporal.worker_config import WorkerConfig, load_worker_config
from tests.integration.temporal.fixtures.hardship.workflow import HardshipWorkflow
from tests.integration.temporal.scripted_worker_driver import LETTER_OUTPUT
from tests.integration.temporal.test_activity_acceptance import (
    PASSING_SCRIPT,
    STATEMENT,
    _install,
    _run,
    _ScriptedSelector,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).parents[3]
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "hardship"
WORKER_YAML = FIXTURE_DIR / "worker.yaml"

STARTUP_TIMEOUT_SECONDS = 15.0
SHUTDOWN_TIMEOUT_SECONDS = 45

_TEMPORAL_ENV = (
    "TEMPORAL_ADDRESS",
    "TEMPORAL_NAMESPACE",
    "TEMPORAL_TASK_QUEUE",
)
_MODEL_CREDENTIAL_ENV = (
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "OPENAI_API_KEY",
    "AZURE_OPENAI_KEY",
)


@pytest.fixture(autouse=True)
def _isolate_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove credentials and inherited Temporal routing from every test."""
    for name in (*_TEMPORAL_ENV, *_MODEL_CREDENTIAL_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def worker_config() -> WorkerConfig:
    """Load the committed hardship worker configuration."""
    return load_worker_config(WORKER_YAML)


def _read(path: Path) -> str:
    """Read a captured subprocess stream.

    Args:
        path: Captured stdout or stderr file.

    Returns:
        Its text, or a placeholder if the file is not present.
    """
    return path.read_text(encoding="utf-8") if path.exists() else "<no output>"


def _diagnose(proc: subprocess.Popen[bytes], stdout: Path, stderr: Path) -> str:
    """Describe a worker subprocess failure with both captured streams.

    Args:
        proc: Worker subprocess.
        stdout: Captured standard-output path.
        stderr: Captured standard-error path.

    Returns:
        Diagnostic text containing the return code and both streams.
    """
    return (
        f"worker subprocess returncode={proc.returncode}\n"
        f"--- stdout ---\n{_read(stdout)}\n"
        f"--- stderr ---\n{_read(stderr)}"
    )


def _kill_process_group(proc: subprocess.Popen[bytes]) -> None:
    """Stop the subprocess session, escalating after a bounded wait.

    Args:
        proc: Process-group leader created with ``start_new_session=True``.
    """
    for sig, timeout in ((signal.SIGTERM, 10), (signal.SIGKILL, 10)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue


def _subprocess_env(address: str, task_queue: str) -> dict[str, str]:
    """Build an isolated child environment with explicit Temporal overrides.

    Args:
        address: Local Temporal dev-server address.
        task_queue: Unique queue for this test.

    Returns:
        Environment for the scripted worker subprocess.
    """
    child_env = dict(os.environ)
    child_env.pop("CLAUDECODE", None)
    for name in (*_TEMPORAL_ENV, *_MODEL_CREDENTIAL_ENV):
        child_env.pop(name, None)

    # Explicit empty values stop CLI dotenv loading from refilling credentials
    # from ~/.holodeck/.env. The fake selector means these values are never read.
    for name in _MODEL_CREDENTIAL_ENV:
        child_env[name] = ""

    # These win over the localhost/hardship values committed in worker.yaml.
    child_env["TEMPORAL_ADDRESS"] = address
    child_env["TEMPORAL_NAMESPACE"] = "default"
    child_env["TEMPORAL_TASK_QUEUE"] = task_queue

    python_path = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT), python_path) if part
    )
    return child_env


async def _completed_history(
    worker_config: WorkerConfig, monkeypatch: pytest.MonkeyPatch
) -> tuple[WorkflowHistory, _ScriptedSelector]:
    """Run the hardship workflow in-process and return its history and counts.

    Args:
        worker_config: Loaded hardship registration.
        monkeypatch: Patcher used by T10's selector installer.

    Returns:
        The completed history and its call-counting scripted selector.
    """
    selector = _install(monkeypatch, PASSING_SCRIPT)
    env = await WorkflowEnvironment.start_local(data_converter=pydantic_data_converter)
    try:
        task_queue = f"replay-{uuid.uuid4()}"
        workflow_id = f"replay-{uuid.uuid4()}"
        activities = [
            agent_activity(node, worker_config.base_dir) for node in worker_config.nodes
        ]
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[HardshipWorkflow],
            activities=activities,
        ):
            output = await _run(env.client, task_queue, workflow_id)
            handle = env.client.get_workflow_handle(workflow_id)
            history = await handle.fetch_history()

        assert output == LETTER_OUTPUT
        return history, selector
    finally:
        await env.shutdown()


class TestWorkerReplayAcceptance:
    """Task 11's AC-4, AC-5, and Worker-init sandbox backstop."""

    @pytest.mark.asyncio
    async def test_ac4_cli_worker_subprocess_runs_hardship_workflow(
        self, tmp_path: Path
    ) -> None:
        """AC-4: the real CLI subprocess returns the exact gated letter."""
        # Arrange
        env = await WorkflowEnvironment.start_local(
            data_converter=pydantic_data_converter
        )
        proc: subprocess.Popen[bytes] | None = None
        stdout_path = tmp_path / "worker.stdout"
        stderr_path = tmp_path / "worker.stderr"
        stdout_file: IO[bytes] | None = None
        stderr_file: IO[bytes] | None = None
        try:
            address = env.client.service_client.config.target_host
            task_queue = f"worker-e2e-{uuid.uuid4()}"
            workflow_id = f"worker-e2e-{uuid.uuid4()}"
            child_env = _subprocess_env(address, task_queue)
            assert not child_env["CLAUDE_CODE_OAUTH_TOKEN"]

            stdout_file = stdout_path.open("wb")
            stderr_file = stderr_path.open("wb")
            proc = subprocess.Popen(  # noqa: S603
                [
                    sys.executable,
                    "-m",
                    "tests.integration.temporal.scripted_worker_driver",
                    "worker",
                    "--config",
                    str(WORKER_YAML),
                ],
                cwd=tmp_path,
                stdout=stdout_file,
                stderr=stderr_file,
                env=child_env,
                start_new_session=True,
            )

            deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                assert proc.poll() is None, _diagnose(proc, stdout_path, stderr_path)
                output = _read(stdout_path)
                if "activity:   evidence" in output and "activity:   letter" in output:
                    break
                time.sleep(0.25)
            else:
                pytest.fail(
                    "worker subprocess did not report both activities before "
                    f"the startup deadline\n{_diagnose(proc, stdout_path, stderr_path)}"
                )

            # Act — this process owns only the workflow. Both activities are
            # registered and executed exclusively by the CLI subprocess.
            async with Worker(
                env.client,
                task_queue=task_queue,
                workflows=[HardshipWorkflow],
            ):
                result: dict[str, Any] = await env.client.execute_workflow(
                    HardshipWorkflow.run,
                    STATEMENT,
                    id=workflow_id,
                    task_queue=task_queue,
                    execution_timeout=timedelta(minutes=2),
                )

            # Assert
            assert result == LETTER_OUTPUT
            assert proc.poll() is None, _diagnose(proc, stdout_path, stderr_path)
            startup_output = _read(stdout_path)
            assert address in startup_output
            assert task_queue in startup_output

            proc.send_signal(signal.SIGINT)
            returncode = proc.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
            assert returncode == 0, _diagnose(proc, stdout_path, stderr_path)
            assert "Worker stopped." in _read(stdout_path)
        finally:
            if proc is not None and proc.poll() is None:
                _kill_process_group(proc)
            for handle in (stdout_file, stderr_file):
                if handle is not None:
                    handle.close()
            await env.shutdown()

    @pytest.mark.asyncio
    async def test_ac5_replay_makes_zero_backend_and_activity_calls(
        self,
        worker_config: WorkerConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """AC-5: replay executes table logic with no activity re-execution."""
        # Arrange — first create a real completed history with two model-seam
        # invocations, one for each activity.
        history, scripted_selector = await _completed_history(
            worker_config, monkeypatch
        )
        calls_before_replay = dict(scripted_selector.calls)
        activity_calls = 0

        async def fail_if_activity_executes(
            *_args: Any, **_kwargs: Any
        ) -> AgentActivityResult:
            """Fail loudly if replay crosses the activity implementation seam."""
            nonlocal activity_calls
            activity_calls += 1
            raise AssertionError("workflow replay executed agent activity code")

        monkeypatch.setattr(
            activity_module, "_run_gated_turn", fail_if_activity_executes
        )

        # Act — temporalio 1.32.0 accepts one WorkflowHistory here. The same
        # Pydantic converter used to record typed payloads is required to replay.
        with ThreadPoolExecutor() as workflow_task_executor:
            replay_result = await Replayer(
                workflows=[HardshipWorkflow],
                workflow_task_executor=workflow_task_executor,
                data_converter=pydantic_data_converter,
            ).replay_workflow(history)

        # Assert
        assert replay_result.replay_failure is None
        assert calls_before_replay == {
            "hardship-evidence-extractor": 1,
            "hardship-letter-writer": 1,
        }
        # T10's installed selector owns this mutable counter; replay must not
        # increment it because Temporal resolves activity results from history.
        assert dict(scripted_selector.calls) == calls_before_replay
        assert activity_calls == 0

    @pytest.mark.asyncio
    async def test_worker_construction_validates_fixture_workflow_sandbox(
        self, worker_config: WorkerConfig
    ) -> None:
        """Public Worker construction accepts the fixture's sandbox imports."""
        # Arrange
        env = await WorkflowEnvironment.start_local(
            data_converter=pydantic_data_converter
        )
        try:
            activities = [
                agent_activity(node, worker_config.base_dir)
                for node in worker_config.nodes
            ]

            # Act — construction calls the default sandbox runner's prepare
            # path. No custom passthrough list is rederived here: the consumer
            # idiom under test is the fixture's imports_passed_through block.
            worker = Worker(
                env.client,
                task_queue=f"sandbox-backstop-{uuid.uuid4()}",
                workflows=[HardshipWorkflow],
                activities=activities,
            )

            # Assert
            assert isinstance(worker, Worker)
        finally:
            await env.shutdown()
