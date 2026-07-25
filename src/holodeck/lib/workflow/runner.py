"""Workflow runner for the deterministic spine (036, T6).

Executes a ``workflow.yaml`` DAG: edge nodes run an agent behind a schema gate,
policy nodes evaluate a DMN table over named inputs, and each node's result
becomes a named input to the nodes above it.

The module is split into two halves on purpose:

* :func:`prepare_workflow` does **all** validation — parse the graph (cycles and
  unresolved references), validate every declared fact of record, resolve every
  gate schema, load every decision table, and apply the FR-030 review gate. It
  never constructs a backend, so any of those failures costs zero LLM calls
  (FR-003).
* :func:`execute_workflow` walks the prepared graph in topological order and is
  the only half that can reach a model.

The review gate lives in ``prepare_workflow`` rather than at the point of
evaluation: a table drafted by a model and not signed off by a human must not
influence a run at all, and refusing it before the first agent invocation is the
only placement where that is true of a multi-node workflow (FR-030, SC-009).

Human nodes are not implemented here — the CLI prompt, ``decided_by``, and the
override path land in T8. A workflow containing one is refused at preparation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from graphlib import TopologicalSorter
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import yaml
from pydantic import BaseModel, ConfigDict, Field

from holodeck.lib.errors import ConfigError, PolicyReviewError, WorkflowError
from holodeck.lib.logging_config import get_logger
from holodeck.lib.observability import get_tracer

# load_gate_schema is edge.py's own definition of "this gate is usable". The
# runner must apply it to *every* edge node before the first agent runs (FR-003)
# — execute_edge_node only checks its own node, by which time an earlier node
# may already have spent a model call. Re-implementing the check here would let
# the two drift, so the owning module's helper is reused as-is.
from holodeck.lib.workflow.edge import GatedOutput, execute_edge_node, load_gate_schema
from holodeck.lib.workflow.input_data import validate_input_data
from holodeck.lib.workflow.table_eval import Verdict, evaluate
from holodeck.models.decision_table import DecisionTable, load_decision_table
from holodeck.models.observability import ObservabilityConfig
from holodeck.models.workflow import EdgeNode, HumanNode, PolicyNode, Workflow

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

logger = get_logger(__name__)

_HUMAN_NODE_UNSUPPORTED = (
    "requires a human determination, which this runner does not execute; "
    "human determination (the CLI prompt, decided_by, and the override path) "
    "lands in T8"
)


class PreparedWorkflow(BaseModel):
    """A fully validated workflow, ready to execute without further checks.

    Holding the loaded tables and validated facts here is what makes "no LLM
    call until everything resolves" structural: building this object is the
    entire validation surface, and it touches no backend.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow: Workflow = Field(description="The parsed, DAG-validated workflow.")
    workflow_dir: Path = Field(
        description="Directory holding workflow.yaml; every relative path in the "
        "workflow resolves against it.",
    )
    order: tuple[str, ...] = Field(
        description="Node ids in topological order. input_data names are absent: "
        "they are facts already in hand, not work to schedule.",
    )
    tables: dict[str, DecisionTable] = Field(
        description="Loaded, review-gated decision table per policy node id.",
    )
    facts: dict[str, Any] = Field(
        description="The validated facts of record, keyed by declared name.",
    )


class WorkflowRunResult(BaseModel):
    """Everything one workflow run produced, keyed by node id."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Workflow name.")
    version: str = Field(description="Workflow version label.")
    order: tuple[str, ...] = Field(description="Node ids in execution order.")
    gated_outputs: dict[str, GatedOutput] = Field(
        description="Gate-validated object per edge node id.",
    )
    verdicts: dict[str, Verdict] = Field(
        description="Verdict per policy node id.",
    )


def _reject_human_node(node_id: str) -> NoReturn:
    """Refuse a human node, which this runner does not implement.

    Args:
        node_id: The offending node.

    Raises:
        WorkflowError: Always.
    """
    raise WorkflowError(f"node '{node_id}' {_HUMAN_NODE_UNSUPPORTED}")


def _load_workflow(path: Path) -> Workflow:
    """Read and validate ``workflow.yaml``.

    Args:
        path: Path to the workflow file.

    Returns:
        The validated :class:`~holodeck.models.workflow.Workflow`.

    Raises:
        ConfigError: If the file cannot be read or is not a YAML mapping.
        pydantic.ValidationError: If the graph has a duplicate id, an unresolved
            input reference, or a cycle (FR-003).
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            "workflow", f"could not read workflow '{path}': {exc}"
        ) from exc
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ConfigError(
            "workflow",
            f"workflow '{path}' must be a YAML mapping, got {type(data).__name__}",
        )
    return Workflow.model_validate(data)


def _execution_order(workflow: Workflow) -> tuple[str, ...]:
    """Return the node ids in topological order.

    ``input_data`` names are excluded — they are supplied facts, not nodes, and
    the workflow model already keeps them out of the dependency graph.

    Args:
        workflow: The validated workflow.

    Returns:
        Node ids, dependencies first.
    """
    ids = {node.id for node in workflow.nodes}
    graph: dict[str, set[str]] = {
        node.id: {ref for ref in (getattr(node, "inputs", None) or []) if ref in ids}
        for node in workflow.nodes
    }
    # Cycles are already rejected by Workflow's model validator, so static_order
    # cannot raise here.
    return tuple(TopologicalSorter(graph).static_order())


def _enforce_review_gate(table: DecisionTable) -> None:
    """Refuse a generated decision table that no human has signed off (FR-030).

    Args:
        table: The loaded decision table.

    Raises:
        PolicyReviewError: If ``provenance.generated_by`` is set and
            ``provenance.reviewed_by`` is absent. Hand-authored tables (no
            ``provenance`` block) are unaffected.
    """
    provenance = table.provenance
    if provenance is not None and provenance.awaiting_review:
        raise PolicyReviewError(
            table.id,
            f"it was generated by '{provenance.generated_by}' and records no "
            f"reviewed_by; AI-drafted policy must be reviewed by a human before "
            f"it can reach a determination",
        )


def prepare_workflow(
    workflow_path: str | Path, payload: Mapping[str, Any] | None = None
) -> PreparedWorkflow:
    """Load and validate everything a run needs, before any agent is invoked.

    Parses the graph, validates the supplied facts of record against their
    declared schemas, resolves every edge node's gate schema, loads every
    referenced decision table, and applies the FR-030 review gate. No backend is
    constructed, so every failure here costs zero LLM calls (FR-003).

    Args:
        workflow_path: Path to ``workflow.yaml``.
        payload: Caller-supplied facts of record, keyed by declared
            ``input_data`` name. Undeclared keys are dropped.

    Returns:
        The :class:`PreparedWorkflow`.

    Raises:
        ConfigError: If the workflow file cannot be read or is not a mapping.
        WorkflowError: If the workflow contains a human node (T8).
        InputDataError: If a declared fact is missing or schema-invalid (FR-025).
        GateSchemaError: If an edge node's gate schema is unusable.
        DecisionTableError: If a referenced decision table is missing or invalid.
        FeelValidationError: If a table carries an out-of-subset expression.
        PolicyReviewError: If a table is generated but unreviewed (FR-030).
        pydantic.ValidationError: If the graph has a cycle or an unresolved
            reference.
    """
    path = Path(workflow_path)
    workflow = _load_workflow(path)
    workflow_dir = path.resolve().parent
    facts = validate_input_data(workflow, dict(payload or {}), workflow_dir)

    tables: dict[str, DecisionTable] = {}
    for node in workflow.nodes:
        if isinstance(node, EdgeNode):
            load_gate_schema(node, workflow_dir)
            continue
        if isinstance(node, HumanNode):
            _reject_human_node(node.id)
        table = load_decision_table(workflow_dir / node.decision)
        _enforce_review_gate(table)
        tables[node.id] = table

    return PreparedWorkflow(
        workflow=workflow,
        workflow_dir=workflow_dir,
        order=_execution_order(workflow),
        tables=tables,
        facts=facts,
    )


def _edge_prompt(facts: Mapping[str, Any]) -> str:
    """Compose the message handed to an edge agent.

    Phase 1 prompts every edge agent with the validated facts of record and
    nothing else. That is not a placeholder: an ``EdgeNode`` has no ``inputs``
    field, so an edge node is structurally a leaf and the facts of record are
    the only workflow state that exists when it runs. What the agent should
    *do* with them is its own ``agent.yaml`` instructions.

    Args:
        facts: The validated facts of record.

    Returns:
        The prompt string.
    """
    return (
        "Facts of record for this determination run (JSON):\n"
        f"{json.dumps(dict(facts), indent=2, sort_keys=True, default=str)}"
    )


def _node_kind(node: EdgeNode | PolicyNode | HumanNode) -> str:
    """Return the span-attribute kind for a node."""
    if isinstance(node, EdgeNode):
        return "edge"
    if isinstance(node, HumanNode):
        return "human"
    return "policy"


def _node_span(
    tracer: Tracer | None, node: EdgeNode | PolicyNode | HumanNode
) -> AbstractContextManager[Any]:
    """Open a span for a node, or nothing at all when observability is off.

    Args:
        tracer: The tracer, or ``None`` when observability is disabled.
        node: The node about to execute.

    Returns:
        The span context manager, or a no-op context manager.
    """
    if tracer is None:
        return nullcontext()
    # start_as_current_span is untyped in the installed opentelemetry stubs, so
    # it returns Any; bind it before returning to keep the declared type honest.
    span: AbstractContextManager[Any] = tracer.start_as_current_span(
        f"holodeck.workflow.node.{node.id}",
        attributes={
            "holodeck.workflow.node_id": node.id,
            "holodeck.workflow.node_kind": _node_kind(node),
        },
    )
    return span


def _named_inputs(
    node: PolicyNode,
    prepared: PreparedWorkflow,
    gated_outputs: Mapping[str, GatedOutput],
    verdicts: Mapping[str, Verdict],
) -> dict[str, Any]:
    """Build the FEEL variable context for a policy node.

    Each name in the node's ``inputs`` resolves to either a supplied fact of
    record, an upstream edge node's gated object, or an upstream policy node's
    verdict outputs. The context is built in full: the FEEL evaluator now fails
    loudly on an unbound name, so a name the runner cannot bind must be an
    error, never a silent ``None``.

    Args:
        node: The policy node about to be evaluated.
        prepared: The prepared workflow, for its validated facts.
        gated_outputs: Gated objects of the edge nodes executed so far.
        verdicts: Verdicts of the policy nodes evaluated so far.

    Returns:
        The named-input context, keyed by the node's declared input names.

    Raises:
        WorkflowError: If a declared input cannot be bound.
    """
    context: dict[str, Any] = {}
    for name in node.inputs:
        if name in prepared.facts:
            context[name] = prepared.facts[name]
        elif name in gated_outputs:
            context[name] = gated_outputs[name].value
        elif name in verdicts:
            context[name] = verdicts[name].outputs
        else:
            raise WorkflowError(
                f"node '{node.id}': input '{name}' resolved to nothing — no such "
                f"fact of record and no result from an upstream node"
            )
    return context


async def execute_workflow(
    prepared: PreparedWorkflow,
    observability: ObservabilityConfig | None = None,
) -> WorkflowRunResult:
    """Execute a prepared workflow in topological order.

    Args:
        prepared: The validated workflow from :func:`prepare_workflow`.
        observability: Observability configuration. A span per node is emitted
            only when this is supplied and enabled; otherwise no tracer is even
            acquired (FR-018).

    Returns:
        The :class:`WorkflowRunResult` holding every node's result.

    Raises:
        WorkflowError: If a human node is reached (T8), or an input cannot be
            bound.
        GateValidationError: If an edge agent's output is rejected at its gate.
        GateSchemaError: If a gate schema turns out to be an invalid JSON Schema.
        TableEvalError: If a table cannot produce exactly one verdict.
        FeelEvaluationError: If a table expression or rule cell fails to evaluate.
        ExecutionError: If an edge agent invocation itself failed.
    """
    tracer: Tracer | None = None
    if observability is not None and observability.enabled:
        tracer = get_tracer(__name__)

    nodes_by_id = {node.id: node for node in prepared.workflow.nodes}
    gated_outputs: dict[str, GatedOutput] = {}
    verdicts: dict[str, Verdict] = {}
    prompt = _edge_prompt(prepared.facts)

    for node_id in prepared.order:
        node = nodes_by_id[node_id]
        with _node_span(tracer, node):
            if isinstance(node, EdgeNode):
                logger.info("Executing edge node '%s'", node_id)
                gated_outputs[node_id] = await execute_edge_node(
                    node, prepared.workflow_dir, prompt
                )
            elif isinstance(node, PolicyNode):
                logger.info("Evaluating policy node '%s'", node_id)
                verdicts[node_id] = evaluate(
                    prepared.tables[node_id],
                    _named_inputs(node, prepared, gated_outputs, verdicts),
                )
            else:
                _reject_human_node(node_id)

    return WorkflowRunResult(
        name=prepared.workflow.name,
        version=prepared.workflow.version,
        order=prepared.order,
        gated_outputs=gated_outputs,
        verdicts=verdicts,
    )


async def run_workflow(
    workflow_path: str | Path,
    payload: Mapping[str, Any] | None = None,
    observability: ObservabilityConfig | None = None,
) -> WorkflowRunResult:
    """Validate and then execute a workflow.

    Args:
        workflow_path: Path to ``workflow.yaml``.
        payload: Caller-supplied facts of record, keyed by declared
            ``input_data`` name.
        observability: Observability configuration; spans are emitted only when
            supplied and enabled.

    Returns:
        The :class:`WorkflowRunResult`.

    Raises:
        WorkflowError: On any spine failure; see :func:`prepare_workflow` and
            :func:`execute_workflow` for the specific subclasses.
        ConfigError: If the workflow or a referenced agent cannot be loaded.
    """
    prepared = prepare_workflow(workflow_path, payload)
    return await execute_workflow(prepared, observability)
