"""Live smoke for the complete hardship workflow against Claude.

The Temporal dev server, worker, activities, gates, decision table, workflow,
and Claude calls are all real. Run this test manually with credentials:

    SKIP_LLM_INTEGRATION_TESTS=false pytest \
        tests/integration/temporal/test_smoke_live.py -v
"""

from __future__ import annotations

import os
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from holodeck.lib.workflow.edge import check_gate, load_gate_schema
from holodeck.temporal.activity import agent_activity
from holodeck.temporal.worker_config import load_worker_config
from tests.integration.temporal.fixtures.hardship.workflow import HardshipWorkflow

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

SKIP_LLM_TESTS = os.getenv("SKIP_LLM_INTEGRATION_TESTS", "false").lower() == "true"
CLAUDE_CODE_OAUTH_TOKEN = os.getenv("CLAUDE_CODE_OAUTH_TOKEN")

skip_if_no_claude_oauth = pytest.mark.skipif(
    SKIP_LLM_TESTS or not CLAUDE_CODE_OAUTH_TOKEN,
    reason="CLAUDE_CODE_OAUTH_TOKEN not configured or LLM tests disabled",
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "hardship"
WORKER_YAML = FIXTURE_DIR / "worker.yaml"
STATEMENT = (
    "I take home $5,000 a month and my outgoings are $3,500. My residency was "
    "verified by the case officer in March."
)


@pytest.fixture(autouse=True)
def _unset_claudecode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset CLAUDECODE so the Agent SDK subprocess doesn't reject nesting."""
    monkeypatch.delenv("CLAUDECODE", raising=False)


@skip_if_no_claude_oauth
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_hardship_workflow_returns_gated_policy_letter() -> None:
    """Run both live agents and prove the deterministic verdict reached the letter."""
    worker_config = load_worker_config(WORKER_YAML)
    activities = [
        agent_activity(node, worker_config.base_dir) for node in worker_config.nodes
    ]
    letter_node = next(node for node in worker_config.nodes if node.id == "letter")
    letter_gate = load_gate_schema(letter_node, worker_config.base_dir)

    env = await WorkflowEnvironment.start_local(data_converter=pydantic_data_converter)
    try:
        task_queue = f"live-hardship-{uuid.uuid4()}"
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[HardshipWorkflow],
            activities=activities,
        ):
            output: dict[str, Any] = await env.client.execute_workflow(
                HardshipWorkflow.run,
                STATEMENT,
                id=f"live-hardship-{uuid.uuid4()}",
                task_queue=task_queue,
                execution_timeout=timedelta(minutes=5),
            )

        assert check_gate(output, letter_gate, node_id="letter") == output
        letter = output["letter"]
        assert isinstance(letter, str)
        assert "affordable" in letter.casefold()
        assert "2026-06-01.1" in letter
    finally:
        await env.shutdown()
