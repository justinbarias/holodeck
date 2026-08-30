"""Live end-to-end for ``HoloDeckPlugin``: the one-liner wiring, no mocks.

The T6 unit suite drives ``configure_client``/``configure_worker`` directly;
this test proves the real SDK plumbing instead. The plugin is handed to the
dev server's client and nothing else is wired by hand:

* the client gets ``pydantic_data_converter`` from the plugin — the
  environment is started **without** one, so typed payloads crossing the wire
  proves the plugin set it (decision 15), and
* the worker registers **no activities itself** — the agent activity arrives
  through client-plugin propagation alone (decision 14's sugar contract).

Same conventions as ``test_live_agent_workflow.py``: requires
``CLAUDE_CODE_OAUTH_TOKEN`` in ``tests/integration/.env`` and
``SKIP_LLM_INTEGRATION_TESTS=false`` in the shell environment.
"""

from __future__ import annotations

import os
import uuid
from datetime import timedelta
from pathlib import Path

import pytest
from dotenv import load_dotenv
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from holodeck.temporal.plugin import HoloDeckPlugin
from tests.integration.temporal.test_live_agent_workflow import (
    AGENT_YAML,
    GATE_SCHEMA,
    STATEMENT,
    base_dir,
    node,
)
from tests.integration.temporal.workflows import EvidenceWorkflow

__all__ = ["AGENT_YAML", "GATE_SCHEMA", "base_dir", "node"]

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


@pytest.fixture(autouse=True)
def _unset_claudecode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset CLAUDECODE so the Agent SDK subprocess doesn't reject nesting."""
    monkeypatch.delenv("CLAUDECODE", raising=False)


class TestLivePlugin:
    """The plugin alone wires converter and activities for a live run."""

    @skip_if_no_claude_oauth
    @pytest.mark.asyncio
    async def test_plugin_wires_converter_and_activities_end_to_end(
        self, base_dir: Path, node: object
    ) -> None:
        """Client plugin propagation: no manual converter, no manual activities."""
        # Arrange — the plugin is the ONLY wiring: no data_converter here.
        plugin = HoloDeckPlugin(nodes=[node], base_dir=base_dir)  # type: ignore[list-item]
        env = await WorkflowEnvironment.start_local(plugins=[plugin])
        try:
            task_queue = f"live-plugin-{uuid.uuid4()}"

            # No activities passed: they must arrive via plugin propagation.
            async with Worker(
                env.client,
                task_queue=task_queue,
                workflows=[EvidenceWorkflow],
            ):
                # Act
                output = await env.client.execute_workflow(
                    EvidenceWorkflow.run,
                    STATEMENT,
                    id=f"plugin-evidence-{uuid.uuid4()}",
                    task_queue=task_queue,
                    execution_timeout=timedelta(minutes=5),
                )

            # Assert — gated dict crossed the pydantic converter the plugin set
            assert isinstance(output, dict)
            assert output["net_income"] == 4200
            assert output["residency_status"] == "verified"
            assert set(output) <= {"net_income", "residency_status"}
        finally:
            await env.shutdown()
