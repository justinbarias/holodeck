"""Live end-to-end for the ``holodeck worker`` CLI: a real subprocess, no mocks.

What this proves beyond ``test_live_agent_workflow.py`` (T15): there, the test
process itself built the activity and the worker in memory. Here the unit under
test is **the command**, running as its own OS process with nothing shared but
a socket:

* it loads ``worker.yaml`` from disk and resolves the node's agent and gate
  paths relative to it (T7),
* it registers the activity under the node id — asserted from the process's own
  startup output, so registration provably came from the file,
* it sets ``pydantic_data_converter`` on its client independently of this test
  process, which is the only reason the typed payload survives the wire, and
* it shuts down gracefully on SIGINT, exiting 0 inside the grace window.

This test process hosts only the *workflow*; the activity exists solely inside
the subprocess. If the CLI failed to register it, the workflow would hang until
its start-to-close timeout rather than pass.

Requires ``CLAUDE_CODE_OAUTH_TOKEN`` in ``tests/integration/.env`` and
``SKIP_LLM_INTEGRATION_TESTS=false`` in the shell environment:

    SKIP_LLM_INTEGRATION_TESTS=false pytest \
        tests/integration/temporal/test_live_worker_command.py -v
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import timedelta
from pathlib import Path
from typing import IO, Any

import pytest
from dotenv import load_dotenv
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from tests.integration.temporal.test_live_agent_workflow import (
    AGENT_YAML,
    GATE_SCHEMA,
    STATEMENT,
)
from tests.integration.temporal.workflows import EvidenceWorkflow

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

SKIP_LLM_TESTS = os.getenv("SKIP_LLM_INTEGRATION_TESTS", "false").lower() == "true"
CLAUDE_CODE_OAUTH_TOKEN = os.getenv("CLAUDE_CODE_OAUTH_TOKEN")

skip_if_no_claude_oauth = pytest.mark.skipif(
    SKIP_LLM_TESTS or not CLAUDE_CODE_OAUTH_TOKEN,
    reason="CLAUDE_CODE_OAUTH_TOKEN not configured or LLM tests disabled",
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# The command's own grace period is 30s (GRACEFUL_SHUTDOWN_SECONDS); allow
# margin for interpreter teardown before calling the shutdown a failure.
SHUTDOWN_TIMEOUT_SECONDS = 45

# A worker that has not yet polled is not an error — Temporal queues the
# activity task. This wait only makes a *crashed* subprocess fail fast.
STARTUP_GRACE_SECONDS = 5.0

WORKER_YAML = """\
temporal:
  address: {address}
  namespace: default
  task_queue: {task_queue}
nodes:
  - id: evidence
    edge:
      agent: evidence.yaml
    gate:
      schema: evidence.schema.json
"""


@pytest.fixture(autouse=True)
def _unset_claudecode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset CLAUDECODE so the Agent SDK subprocess doesn't reject nesting."""
    monkeypatch.delenv("CLAUDECODE", raising=False)


def _read(path: Path) -> str:
    """Return a captured stream's contents, or a placeholder when absent.

    Args:
        path: File the subprocess's stdout or stderr was captured to.

    Returns:
        The captured text.
    """
    return path.read_text(encoding="utf-8") if path.exists() else "<no output>"


def _diagnose(proc: subprocess.Popen[bytes], out: Path, err: Path) -> str:
    """Build a failure message carrying the subprocess's own output.

    Args:
        proc: The worker subprocess.
        out: File its stdout was captured to.
        err: File its stderr was captured to.

    Returns:
        A message naming the exit state and both streams.
    """
    return (
        f"worker subprocess returncode={proc.returncode}\n"
        f"--- stdout ---\n{_read(out)}\n"
        f"--- stderr ---\n{_read(err)}"
    )


class TestLiveWorkerCommand:
    """The CLI process serves a real workflow execution and stops cleanly."""

    @skip_if_no_claude_oauth
    @pytest.mark.asyncio
    async def test_cli_worker_serves_a_live_workflow_and_exits_on_sigint(
        self, tmp_path: Path
    ) -> None:
        """Dev server, CLI subprocess, live Claude, gated dict, clean SIGINT."""
        # Arrange — dev server; this process's client needs the converter too,
        # since it hosts the workflow and reads the typed result.
        from temporalio.contrib.pydantic import pydantic_data_converter

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
            task_queue = f"live-cli-{uuid.uuid4()}"

            (tmp_path / "evidence.yaml").write_text(AGENT_YAML, encoding="utf-8")
            (tmp_path / "evidence.schema.json").write_text(
                json.dumps(GATE_SCHEMA), encoding="utf-8"
            )
            config_path = tmp_path / "worker.yaml"
            config_path.write_text(
                WORKER_YAML.format(address=address, task_queue=task_queue),
                encoding="utf-8",
            )

            child_env: dict[str, Any] = dict(os.environ)
            child_env.pop("CLAUDECODE", None)
            assert child_env.get("CLAUDE_CODE_OAUTH_TOKEN"), (
                "the worker subprocess needs CLAUDE_CODE_OAUTH_TOKEN in its "
                "environment to reach Claude"
            )

            stdout_file = stdout_path.open("wb")
            stderr_file = stderr_path.open("wb")
            # Same interpreter, therefore the same virtualenv: the subprocess
            # inherits no shell activation of its own.
            proc = subprocess.Popen(  # noqa: S603
                [
                    sys.executable,
                    "-m",
                    "holodeck.cli.main",
                    "worker",
                    "--config",
                    str(config_path),
                ],
                stdout=stdout_file,
                stderr=stderr_file,
                env=child_env,
            )

            deadline = time.monotonic() + STARTUP_GRACE_SECONDS
            while time.monotonic() < deadline:
                assert proc.poll() is None, _diagnose(proc, stdout_path, stderr_path)
                if "activity:   evidence" in _read(stdout_path):
                    break
                time.sleep(0.25)

            # Act — this process hosts the workflow only; the activity lives in
            # the subprocess and reaches Claude for real.
            async with Worker(
                env.client,
                task_queue=task_queue,
                workflows=[EvidenceWorkflow],
            ):
                output = await env.client.execute_workflow(
                    EvidenceWorkflow.run,
                    STATEMENT,
                    id=f"cli-evidence-{uuid.uuid4()}",
                    task_queue=task_queue,
                    execution_timeout=timedelta(minutes=5),
                )

            # Assert — the gate-validated dict crossed the wire typed
            assert isinstance(output, dict)
            assert output == {"net_income": 4200, "residency_status": "verified"}

            assert proc.poll() is None, _diagnose(proc, stdout_path, stderr_path)

            # Registration provably came from worker.yaml, not from this test.
            assert "activity:   evidence" in _read(stdout_path)
            assert address in _read(stdout_path)

            # Assert — SIGINT shuts the worker down gracefully, exit code 0
            proc.send_signal(signal.SIGINT)
            returncode = proc.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
            assert returncode == 0, _diagnose(proc, stdout_path, stderr_path)
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=30)
            for handle in (stdout_file, stderr_file):
                if handle is not None:
                    handle.close()
            await env.shutdown()
