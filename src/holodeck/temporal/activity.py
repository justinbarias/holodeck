"""Activity factory: an edge node becomes a Temporal activity (spec 040, T3).

:func:`agent_activity` binds one :class:`~holodeck.models.workflow.EdgeNode` to
a worker and returns the async activity a user-authored workflow calls by name.
The agent's configuration, its resolved path, and its gate schema are read once
at factory time — worker-side state, so credentials never enter an activity
payload or the workflow history (decision 14).

Two seams from the 036 spine are reused, never rewritten:
``resolve_agent_path``/``load_gate_schema`` (the only path and schema seams,
decision 1) and ``edge.check_gate`` (the same gate the D3 surface exposes to
workflow code). The invocation half of
``execute_edge_node`` is deliberately *not* reused: as that function's own
docstring anticipates, the activity invokes the backend against Temporal's
retry and cancellation seams and needs the token usage and turn count the
``GatedOutput`` does not carry.

Timeouts and retries are absent by design — in Temporal they ride the caller's
``execute_activity`` command, which is what
:class:`~holodeck.temporal.models.ActivityParameters` builds (decision 10).

Error classification (T4) happens at the activity boundary, translating
SC-003's model-fault vs authoring-fault split into Temporal retry semantics:

* **Retryable — evidence about the model or the transport.** A gate rejection
  (``GateValidationError``) and a broken invocation (``ExecutionError``)
  propagate as plain exceptions. Temporal converts a plain exception into a
  retryable ``ApplicationError`` whose ``type`` is the exception class name,
  so a workflow can still opt out per class with
  ``RetryPolicy(non_retryable_error_types=["GateValidationError"])`` — the
  match is by that class-name string, which is therefore part of the public
  contract of this module.
* **Non-retryable — authoring faults.** ``ConfigError``, ``GateSchemaError``,
  :class:`holodeck.lib.errors.FileNotFoundError` and ``BackendInitError``
  (initialization normalizes configuration failures — missing credentials,
  absent Node.js — to it) are re-raised as
  ``ApplicationError(non_retryable=True)`` typed by the original class name:
  no number of retries fixes a broken worker configuration, and retrying
  would bill a model call per attempt. Most authoring faults already fail at
  factory time, before the worker starts; this classification covers the ones
  only reachable per call.

The two channels never mix: a gate rejection is never non-retryable, and an
authoring fault is never allowed to surface as a plain (retryable) exception.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

from temporalio import activity
from temporalio.exceptions import ApplicationError

from holodeck.config.context import agent_base_dir
from holodeck.config.loader import ConfigLoader
from holodeck.lib.backends.base import BackendInitError
from holodeck.lib.errors import ConfigError, ExecutionError, GateSchemaError
from holodeck.lib.errors import FileNotFoundError as HoloDeckFileNotFoundError
from holodeck.lib.workflow.edge import (
    _teardown,
    check_gate,
    load_gate_schema,
    resolve_agent_path,
)
from holodeck.models.agent import Agent
from holodeck.models.workflow import EdgeNode
from holodeck.temporal.models import AgentActivityInput, AgentActivityResult

logger = logging.getLogger(__name__)

_CONTEXT_HEADER = "Context (JSON):"

# Authoring faults: defects in the worker's configuration, not evidence about
# the model (SC-003). Retrying cannot fix them, so they cross the activity
# boundary as ApplicationError(non_retryable=True) typed by class name.
_AUTHORING_FAULTS = (
    ConfigError,
    GateSchemaError,
    HoloDeckFileNotFoundError,
    BackendInitError,
)


def _compose_message(activity_input: AgentActivityInput) -> str:
    """Render the activity input as the single prompt handed to the agent.

    A caller-supplied ``context`` object is appended as a JSON block under a
    fixed header. The rendering is deterministic (sorted keys) so the same
    input always produces the same prompt.

    Args:
        activity_input: The activity's typed input payload.

    Returns:
        The message for ``invoke_once``. Identical to ``input.message`` when no
        context was supplied.
    """
    if activity_input.context is None:
        return activity_input.message
    rendered = json.dumps(activity_input.context, sort_keys=True, default=str)
    return f"{activity_input.message}\n\n{_CONTEXT_HEADER}\n{rendered}"


async def _run_gated_turn(
    node: EdgeNode,
    agent: Agent,
    agent_path: Path,
    gate_schema: dict[str, Any],
    message: str,
) -> AgentActivityResult:
    """Run one stateless agent turn and gate its structured output.

    One activity call is one ``invoke_once`` (decision 5): the backend is built,
    used, and torn down inside this call. Nothing is held between calls, so
    concurrent activity executions on the same worker never share an SDK
    subprocess.

    This is the single seam T4 wraps for retry classification — every failure
    channel of the turn passes through here.

    Args:
        node: The edge node being executed.
        agent: The node's agent configuration, loaded at factory time.
        agent_path: Path the agent was loaded from. Its parent is the base
            directory a backend resolves relative ``file:`` tool references
            against.
        gate_schema: The node's gate schema, loaded at factory time.
        message: The composed prompt for this turn.

    Returns:
        The result envelope carrying the gate-validated object. The raw model
        response text is deliberately not part of it (FR-008).

    Raises:
        ExecutionError: If the invocation raised, or failed with nothing to
            judge. A broken invocation is not evidence about the model.
        GateValidationError: If the agent returned free text or output the gate
            rejected.
        GateSchemaError: If the gate could not be applied (an authoring fault
            that load-time validation could not settle).
        BackendInitError: If building or initialising the backend fails.
    """
    # Deferred import for the same reason edge.py defers it: keep the backend
    # SDKs off the import path of anything that only needs the factory.
    from holodeck.lib.backends.selector import BackendSelector

    # Backends resolve a tool's relative `file:` against this contextvar. A
    # worker runs many activities in one process, so it is set per call and
    # reset by token — never left pointing at the last node that ran.
    token = agent_base_dir.set(str(agent_path.parent))
    try:
        backend = await BackendSelector.select(agent, mode="test")
        # Teardown rides a finally so cancellation cannot leak the backend:
        # Temporal cancellation surfaces as asyncio.CancelledError (a
        # BaseException), which an `except Exception` never sees, and the SDK
        # subprocess, MCP servers, and tool resources must be released on
        # every exit path.
        try:
            result = await backend.invoke_once(message)
        except Exception as exc:
            raise ExecutionError(
                f"activity '{node.id}': agent invocation raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        finally:
            await _teardown(backend, node.id)

        # is_error with structured output still went to the gate: the model
        # produced something, and what it produced is evidence about the model
        # (SC-003). Only a failure with nothing to judge is an ExecutionError.
        if result.is_error and result.structured_output is None:
            raise ExecutionError(
                f"activity '{node.id}': agent invocation failed: "
                f"{result.error_reason}"
            )

        output = check_gate(result.structured_output, gate_schema, node_id=node.id)
    finally:
        agent_base_dir.reset(token)

    return AgentActivityResult(
        output=output,
        token_usage=result.token_usage,
        num_turns=result.num_turns,
        agent_id=node.id,
    )


def agent_activity(
    node: EdgeNode, base_dir: Path
) -> Callable[[AgentActivityInput], Awaitable[AgentActivityResult]]:
    """Build the Temporal activity that runs an edge node's agent.

    The activity's name is the node id (decision 11): it is replay-load-bearing
    and survives moving the agent's files. Everything the activity needs is
    resolved and validated here, once, before the worker starts — an unusable
    gate, an agent outside ``base_dir``, or an agent that could never produce
    structured output fails at registration rather than costing a model call
    per execution.

    Args:
        node: The edge node to expose as an activity. Its ``gate`` makes the
            schema check mandatory by construction — there is no ungated
            variant (decision 2).
        base_dir: Directory the node's ``edge.agent`` and ``gate.schema`` paths
            resolve against, and which they may not escape.

    Returns:
        An async callable decorated with ``@activity.defn(name=node.id)``,
        taking an :class:`~holodeck.temporal.models.AgentActivityInput` and
        returning an :class:`~holodeck.temporal.models.AgentActivityResult`.

    Raises:
        ConfigError: If the agent path escapes ``base_dir``, or the agent
            declares no ``response_format`` and so could never produce
            structured output for the gate.
        GateSchemaError: If the gate schema is missing, unparseable, or
            unusable as a gate.
        FileNotFoundError: If the node's ``agent.yaml`` does not exist.
    """
    agent_path = resolve_agent_path(node, base_dir)
    gate_schema = load_gate_schema(node, base_dir)
    agent = ConfigLoader().load_agent_yaml(str(agent_path))

    # Same guard execute_edge_node applies, hoisted to registration: an agent
    # with no response_format can never produce structured_output, so every
    # execution would spend a model call to reach the gate's "free text"
    # rejection — a rejection charged to a model that was never asked for
    # structure.
    if agent.response_format is None:
        raise ConfigError(
            f"nodes.{node.id}.edge.agent",
            "edge agent declares no response_format, so it can never produce "
            "structured output for the gate",
        )

    @activity.defn(name=node.id)
    async def run_agent(activity_input: AgentActivityInput) -> AgentActivityResult:
        """Run the edge node's agent for one message and gate its output.

        Args:
            activity_input: The message and optional context for this turn.

        Returns:
            The envelope carrying the gate-validated object.
        """
        logger.debug("activity '%s': starting agent turn", node.id)
        try:
            return await _run_gated_turn(
                node, agent, agent_path, gate_schema, _compose_message(activity_input)
            )
        except _AUTHORING_FAULTS as exc:
            raise ApplicationError(
                str(exc), type=type(exc).__name__, non_retryable=True
            ) from exc

    # activity.defn's overloads erase the callable's type for mypy under the
    # pre-commit configuration; the decorator wraps without changing it.
    return cast(
        "Callable[[AgentActivityInput], Awaitable[AgentActivityResult]]", run_agent
    )


__all__ = ["agent_activity"]
