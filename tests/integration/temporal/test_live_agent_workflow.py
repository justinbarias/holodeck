"""Live end-to-end: user-authored Temporal workflow → agent activity → Claude.

The real thing, no mocks: a local Temporal dev server
(``WorkflowEnvironment.start_local``, auto-downloaded binary), a worker
registering the T3 activity factory's output, a user-authored workflow that
schedules it with :class:`ActivityParameters`, and a live Claude call through
``ClaudeBackend``. The gate-validated dict — never raw model text — is what the
workflow receives (FR-008).

Requires ``CLAUDE_CODE_OAUTH_TOKEN`` in ``tests/integration/.env`` and
``SKIP_LLM_INTEGRATION_TESTS=false`` in the shell environment (the committed
``.env`` sets it to ``true``; shell env wins because ``load_dotenv`` does not
override existing variables):

    SKIP_LLM_INTEGRATION_TESTS=false pytest tests/integration/temporal/ -m slow
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from holodeck.models.workflow import EdgeNode
from holodeck.temporal.activity import agent_activity
from tests.integration.temporal.workflows import EvidenceWorkflow

# ---------------------------------------------------------------------------
# Environment & skip logic (same conventions as the other live suites)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Worker-side fixtures: an edge agent and its gate, written to disk
# ---------------------------------------------------------------------------

AGENT_YAML = """\
name: evidence-extractor
description: Extracts structured income evidence from an applicant statement.
model:
  provider: anthropic
  name: claude-sonnet-4-6
  auth_provider: oauth_token
  temperature: 0.0
  max_tokens: 1024
instructions:
  inline: |
    Extract the applicant's income evidence from the user's message.
    Respond only with the structured output.
response_format:
  type: object
  properties:
    net_income:
      type: number
      description: Monthly net income in dollars.
    residency_status:
      type: string
      enum: [verified, unverified]
  required: [net_income, residency_status]
"""

GATE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "net_income": {"type": "number"},
        "residency_status": {"type": "string", "enum": ["verified", "unverified"]},
    },
    "required": ["net_income", "residency_status"],
    "additionalProperties": False,
}

STATEMENT = (
    "The applicant reports a net income of $4,200 per month. "
    "Their residency status has been verified by the case officer."
)


@pytest.fixture(autouse=True)
def _unset_claudecode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset CLAUDECODE so the Agent SDK subprocess doesn't reject nesting."""
    monkeypatch.delenv("CLAUDECODE", raising=False)


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    """A worker base directory holding the edge agent and its gate schema."""
    (tmp_path / "evidence.yaml").write_text(AGENT_YAML, encoding="utf-8")
    (tmp_path / "evidence.schema.json").write_text(
        json.dumps(GATE_SCHEMA), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def node() -> EdgeNode:
    """The edge node exposed to the worker."""
    return EdgeNode(
        id="evidence",
        edge={"agent": "evidence.yaml"},  # type: ignore[arg-type]
        gate={"schema": "evidence.schema.json"},  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestLiveAgentWorkflow:
    """A real workflow drives a real agent activity against live Claude."""

    @skip_if_no_claude_oauth
    @pytest.mark.asyncio
    async def test_workflow_runs_live_agent_and_receives_gated_output(
        self, base_dir: Path, node: EdgeNode
    ) -> None:
        """End to end: dev server, worker, workflow, live Claude, gated dict."""
        # Arrange — local dev server with the pydantic converter (decision 15)
        env = await WorkflowEnvironment.start_local(
            data_converter=pydantic_data_converter
        )
        try:
            client: Client = env.client
            task_queue = f"live-evidence-{uuid.uuid4()}"
            activity_fn = agent_activity(node, base_dir)

            async with Worker(
                client,
                task_queue=task_queue,
                workflows=[EvidenceWorkflow],
                activities=[activity_fn],
            ):
                # Act
                output = await client.execute_workflow(
                    EvidenceWorkflow.run,
                    STATEMENT,
                    id=f"evidence-{uuid.uuid4()}",
                    task_queue=task_queue,
                    execution_timeout=timedelta(minutes=5),
                )

            # Assert — the workflow saw the gate-validated dict, nothing else
            assert isinstance(output, dict)
            assert output["net_income"] == 4200
            assert output["residency_status"] == "verified"
            assert set(output) <= {"net_income", "residency_status"}
        finally:
            await env.shutdown()
