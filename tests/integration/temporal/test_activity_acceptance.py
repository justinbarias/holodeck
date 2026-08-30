"""AC-1/AC-2/AC-3 acceptance for the hardship sample (spec 040, T10).

Everything here is real except the model. A local Temporal dev server
(``WorkflowEnvironment.start_local``, auto-downloaded binary), the pydantic
data converter, the T3 activity factory over the committed hardship fixture,
the gate, the decision table, and the user-authored ``HardshipWorkflow`` all
run unmodified. Only the backend is replaced — a scripted fake injected at the
``BackendSelector`` seam ``holodeck.temporal.activity`` imports — so the suite
needs no credentials, spends no tokens, and is deterministic in CI. The
live-LLM counterpart is the T13 smoke.

The three acceptance criteria:

* **AC-1** — the workflow receives a gate-validated envelope, and the object
  the letter activity is handed is that same gated dict, not model text.
* **AC-2** — a gate rejection is retryable evidence about the model: the
  backend emits a gate-failing object first and a passing one second, the run
  still completes, and the history shows the activity took two attempts.
* **AC-3** — the verdict the workflow computes over the gated object is the
  same :class:`Verdict` the 036 conformance suite produces for the same
  inputs.

Plus the history assertion the plan pairs with them: every completed activity
result in the history carries an object that crosses its node's gate, and the
failed attempt left no completed result behind at all.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from temporalio.api.enums.v1 import EventType
from temporalio.api.history.v1 import HistoryEvent
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from holodeck.lib.backends import selector as selector_module
from holodeck.lib.backends.base import ExecutionResult
from holodeck.lib.workflow.edge import check_gate, load_gate_schema
from holodeck.lib.workflow.table_eval import Verdict, evaluate
from holodeck.temporal.activity import agent_activity
from holodeck.temporal.worker_config import WorkerConfig, load_worker_config
from tests.integration.temporal.fixtures.hardship.policy import TABLE
from tests.integration.temporal.fixtures.hardship.workflow import HardshipWorkflow

# The 036 conformance suite is the source of truth AC-3 compares against.
from tests.unit.workflow.test_table_eval import _table

pytestmark = pytest.mark.integration

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "hardship"
WORKER_YAML = FIXTURE_DIR / "worker.yaml"

# Agent names, from the fixture's agents/*.yaml — how the scripted backend
# tells the two activities apart at the selector seam.
EVIDENCE_AGENT = "hardship-evidence-extractor"
LETTER_AGENT = "hardship-letter-writer"

STATEMENT = (
    "I take home $5,000 a month and my outgoings are $3,500. My residency was "
    "verified by the case officer in March."
)

# surplus_ratio = (5000 - 3500) / 5000 = 0.3 -> >= 0.25 and verified -> affordable
EVIDENCE_OUTPUT: dict[str, Any] = {
    "income": {"net": 5000, "expenses": 3500},
    "residency": {"status": "verified"},
}

# Missing the required `residency` object: the evidence gate rejects it, which
# is retryable evidence about the model (SC-003), not an authoring fault.
BAD_EVIDENCE_OUTPUT: dict[str, Any] = {"income": {"net": 5000, "expenses": 3500}}

LETTER_OUTPUT: dict[str, Any] = {
    "letter": "Dear applicant, we have completed our review of your case.",
    "tone": "neutral",
}

# The header `_compose_message` renders a caller-supplied context object under.
CONTEXT_HEADER = "Context (JSON):"


# ---------------------------------------------------------------------------
# Scripted backend — the only thing that is not real
# ---------------------------------------------------------------------------


class _ScriptedBackend:
    """Stands in for an ``AgentBackend``; replays a canned script, never calls out."""

    def __init__(self, selector: _ScriptedSelector, agent_name: str) -> None:
        self._selector = selector
        self._agent_name = agent_name

    async def invoke_once(
        self, message: str, context: list[dict[str, Any]] | None = None
    ) -> ExecutionResult:
        """Record the prompt and return the next canned result for this agent.

        Args:
            message: The composed prompt the activity built.
            context: Unused; present to match the backend protocol.

        Returns:
            The next scripted :class:`ExecutionResult` for this agent. The last
            entry of a script repeats once exhausted.
        """
        self._selector.messages.append((self._agent_name, message))
        script = self._selector.script[self._agent_name]
        index = min(self._selector.calls[self._agent_name], len(script) - 1)
        self._selector.calls[self._agent_name] += 1
        return ExecutionResult(response="", structured_output=script[index])

    async def teardown(self) -> None:
        """Record that the activity released the backend on every exit path."""
        self._selector.teardowns.append(self._agent_name)


class _ScriptedSelector:
    """Stands in for ``BackendSelector``, routing by the agent's name."""

    def __init__(self, script: dict[str, list[Any]]) -> None:
        self.script = script
        self.calls: dict[str, int] = dict.fromkeys(script, 0)
        self.messages: list[tuple[str, str]] = []
        self.teardowns: list[str] = []

    async def select(
        self,
        agent: Any,
        tool_instances: dict[str, Any] | None = None,
        mode: str = "test",
    ) -> _ScriptedBackend:
        """Return the scripted backend for the agent being invoked.

        Args:
            agent: The loaded agent configuration.
            tool_instances: Unused; present to match ``BackendSelector.select``.
            mode: Unused; present to match ``BackendSelector.select``.

        Returns:
            A backend replaying this agent's script.
        """
        return _ScriptedBackend(self, agent.name)

    def context_sent_to(self, agent_name: str) -> dict[str, Any]:
        """Decode the context object rendered into that agent's first prompt.

        Args:
            agent_name: Name of the agent whose prompt to read.

        Returns:
            The JSON block the activity appended under the context header.
        """
        for name, message in self.messages:
            if name == agent_name and CONTEXT_HEADER in message:
                return json.loads(message.split(CONTEXT_HEADER, 1)[1])
        raise AssertionError(f"no context-carrying prompt reached {agent_name!r}")


def _install(
    monkeypatch: pytest.MonkeyPatch, script: dict[str, list[Any]]
) -> _ScriptedSelector:
    """Replace the selector the activity's deferred import resolves.

    Patched as an attribute of the module object rather than by dotted string
    so the binding is unambiguous under xdist.

    Args:
        monkeypatch: The pytest patcher.
        script: Per-agent-name list of ``structured_output`` values to emit.

    Returns:
        The installed selector, for its recorded prompts.
    """
    selector = _ScriptedSelector(script)
    monkeypatch.setattr(selector_module, "BackendSelector", selector)
    return selector


PASSING_SCRIPT: dict[str, list[Any]] = {
    EVIDENCE_AGENT: [EVIDENCE_OUTPUT],
    LETTER_AGENT: [LETTER_OUTPUT],
}
# Gate-failing first, passing second: AC-2's one failed attempt then a retry.
RETRY_SCRIPT: dict[str, list[Any]] = {
    EVIDENCE_AGENT: [BAD_EVIDENCE_OUTPUT, EVIDENCE_OUTPUT],
    LETTER_AGENT: [LETTER_OUTPUT],
}


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def worker_config() -> WorkerConfig:
    """The committed hardship ``worker.yaml``, loaded and confined."""
    return load_worker_config(WORKER_YAML)


@asynccontextmanager
async def _running_worker(config: WorkerConfig) -> AsyncIterator[tuple[Client, str]]:
    """Start a dev server and a worker registering the fixture's activities.

    Args:
        config: The loaded worker configuration.

    Yields:
        The connected client and the unique task queue the worker polls.
    """
    env = await WorkflowEnvironment.start_local(data_converter=pydantic_data_converter)
    try:
        task_queue = f"hardship-{uuid.uuid4()}"
        activities = [agent_activity(node, config.base_dir) for node in config.nodes]
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[HardshipWorkflow],
            activities=activities,
        ):
            yield env.client, task_queue
    finally:
        await env.shutdown()


async def _run(client: Client, task_queue: str, workflow_id: str) -> dict[str, Any]:
    """Execute one hardship workflow to completion.

    Args:
        client: Connected Temporal client.
        task_queue: Queue the worker polls.
        workflow_id: Id to run under, so the history can be fetched after.

    Returns:
        The workflow's return value — the gate-validated letter object.
    """
    result: dict[str, Any] = await client.execute_workflow(
        HardshipWorkflow.run,
        STATEMENT,
        id=workflow_id,
        task_queue=task_queue,
        execution_timeout=timedelta(minutes=2),
    )
    return result


async def _history(client: Client, workflow_id: str) -> Sequence[HistoryEvent]:
    """Fetch the full event history for a finished workflow.

    Args:
        client: Connected Temporal client.
        workflow_id: The workflow whose history to read.

    Returns:
        Every history event, in order.
    """
    handle = client.get_workflow_handle(workflow_id)
    history = await handle.fetch_history()
    return history.events


def _activity_names_by_scheduled_id(events: Sequence[HistoryEvent]) -> dict[int, str]:
    """Map each ActivityTaskScheduled event id to the activity name it scheduled.

    Args:
        events: The workflow's history events.

    Returns:
        Scheduled event id → activity type name.
    """
    scheduled = EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
    return {
        event.event_id: (
            event.activity_task_scheduled_event_attributes.activity_type.name
        )
        for event in events
        if event.event_type == scheduled
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestActivityAcceptance:
    """The hardship sample end to end, against a scripted backend."""

    @pytest.mark.asyncio
    async def test_ac1_workflow_receives_gate_validated_envelope(
        self, worker_config: WorkerConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-1: the gated object — not model text — crosses every boundary."""
        # Arrange
        selector = _install(monkeypatch, PASSING_SCRIPT)

        # Act
        async with _running_worker(worker_config) as (client, task_queue):
            output = await _run(client, task_queue, f"ac1-{uuid.uuid4()}")

        # Assert — the workflow returned exactly the gated letter object
        assert output == LETTER_OUTPUT

        # Assert — the evidence the letter activity was handed is the gated dict
        context = selector.context_sent_to(LETTER_AGENT)
        assert context["evidence"] == EVIDENCE_OUTPUT
        assert selector.teardowns == [EVIDENCE_AGENT, LETTER_AGENT]

    @pytest.mark.asyncio
    async def test_ac2_gate_failure_retries_and_history_shows_both_attempts(
        self, worker_config: WorkerConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-2: a gate rejection is retried, and the history records attempt 2."""
        # Arrange
        selector = _install(monkeypatch, RETRY_SCRIPT)
        workflow_id = f"ac2-{uuid.uuid4()}"

        # Act
        async with _running_worker(worker_config) as (client, task_queue):
            output = await _run(client, task_queue, workflow_id)
            events = await _history(client, workflow_id)

        # Assert — the run still completed, on the second evidence attempt
        assert output == LETTER_OUTPUT
        assert selector.calls[EVIDENCE_AGENT] == 2
        assert selector.calls[LETTER_AGENT] == 1

        # Assert — the history shows the evidence activity started twice
        names = _activity_names_by_scheduled_id(events)
        attempts = {
            names[event.activity_task_started_event_attributes.scheduled_event_id]: (
                event.activity_task_started_event_attributes.attempt
            )
            for event in events
            if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_STARTED
        }
        assert attempts["evidence"] == 2
        assert attempts["letter"] == 1

        # Assert — the failed attempt is recorded as the gate rejection it was
        started = next(
            event
            for event in events
            if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_STARTED
            and names[event.activity_task_started_event_attributes.scheduled_event_id]
            == "evidence"
        )
        last_failure = started.activity_task_started_event_attributes.last_failure
        assert last_failure.application_failure_info.type == "GateValidationError"
        assert last_failure.application_failure_info.non_retryable is False

    @pytest.mark.asyncio
    async def test_ac3_workflow_verdict_matches_036_suite(
        self, worker_config: WorkerConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-3: the workflow's verdict is the 036 suite's verdict."""
        # Arrange — the 036 conformance table, evaluated over the gated object
        selector = _install(monkeypatch, PASSING_SCRIPT)
        expected: Verdict = evaluate(_table(), EVIDENCE_OUTPUT)

        # Act
        async with _running_worker(worker_config) as (client, task_queue):
            await _run(client, task_queue, f"ac3-{uuid.uuid4()}")

        # Assert — the fixture table and the 036 suite's table agree entirely
        actual: Verdict = evaluate(TABLE, EVIDENCE_OUTPUT)
        assert actual.outputs == expected.outputs
        assert actual.rule_identity == expected.rule_identity
        assert actual.table_id == expected.table_id
        assert actual.table_version == expected.table_version
        assert actual.is_default == expected.is_default

        # Assert — and that is the verdict the workflow actually carried forward
        context = selector.context_sent_to(LETTER_AGENT)
        assert context["affordability"] == expected.outputs["affordability"]
        assert context["policy"] == expected.table_id
        assert context["policy_version"] == expected.table_version

    @pytest.mark.asyncio
    async def test_history_contains_no_unvalidated_payload(
        self, worker_config: WorkerConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every completed activity result in the history crosses its node's gate."""
        # Arrange — the AC-2 script, so a gate-failing attempt is in the history
        _install(monkeypatch, RETRY_SCRIPT)
        workflow_id = f"history-{uuid.uuid4()}"
        schemas: dict[str, dict[str, Any]] = {
            node.id: load_gate_schema(node, worker_config.base_dir)
            for node in worker_config.nodes
        }
        assert set(schemas) == {"evidence", "letter"}

        # Act
        async with _running_worker(worker_config) as (client, task_queue):
            await _run(client, task_queue, workflow_id)
            events = await _history(client, workflow_id)

        # Assert — decode every completed activity result and re-gate it
        names = _activity_names_by_scheduled_id(events)
        completed: list[tuple[str, Any]] = []
        for event in events:
            if event.event_type != EventType.EVENT_TYPE_ACTIVITY_TASK_COMPLETED:
                continue
            attributes = event.activity_task_completed_event_attributes
            node_id = names[attributes.scheduled_event_id]
            decoded = await client.data_converter.decode(
                list(attributes.result.payloads)
            )
            completed.append((node_id, decoded[0]["output"]))

        assert [node_id for node_id, _ in completed] == ["evidence", "letter"]
        for node_id, output in completed:
            # check_gate raises GateValidationError if the object does not
            # cross — the same gate the activity applied.
            assert check_gate(output, schemas[node_id], node_id=node_id) == output

        # Assert — the rejected object never reached a completed result
        assert all(output != BAD_EVIDENCE_OUTPUT for _, output in completed)
