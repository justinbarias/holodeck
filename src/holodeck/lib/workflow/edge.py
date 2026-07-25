"""Edge-node executor and schema gate for the deterministic spine (036, T5).

An edge node is the only place an LLM touches a workflow run, and this module
is the gate it must cross. The agent named by ``edge.agent`` is invoked through
``BackendSelector``/``invoke_once`` (FR-006), and its ``structured_output`` is
validated against the node's ``gate.schema`` (FR-007). Free text or
schema-invalid output is rejected with :class:`GateValidationError` — nothing
ungated ever reaches the spine (SC-003).

Three failure channels, deliberately distinct because SC-003 counts gate
rejections as evidence about model output: :class:`GateValidationError` (the
model's output was rejected), :class:`GateSchemaError` (the gate itself is
unusable — a workflow-authoring defect), and ``ExecutionError`` (the invocation
never produced output to judge). The discriminator between the last two is
*whether there is something to judge*, not the backend's ``is_error`` flag: a
backend that flags an error while still returning an object — what
``ClaudeBackend`` does when the model violates its own ``response_format`` —
has produced evidence about the model, and that evidence goes to the gate.

Gate validation never touches the network. Remote ``$ref`` retrieval is
refused outright, so the schema snapshotted in :class:`GatedOutput` is exactly
the schema that was enforced.

What crosses is the :class:`GatedOutput`: the *validated object*, never the raw
model text (FR-008). It carries the gate schema by **content**, not by path, so
a run record can snapshot what the output was actually judged against (T10).

Per refinements §1 the POC validates the Claude backend only; dispatch still
goes through ``BackendSelector`` so the protocol contract is unchanged.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from pathlib import Path
from typing import Any

import jsonschema
import jsonschema.validators
from pydantic import BaseModel, ConfigDict, Field, field_validator
from referencing import Registry, Resource
from referencing.exceptions import NoSuchResource, Unresolvable

from holodeck.config.context import agent_base_dir
from holodeck.config.loader import ConfigLoader
from holodeck.lib.backends.base import AgentBackend, ExecutionResult
from holodeck.lib.backends.selector import BackendSelector
from holodeck.lib.errors import ExecutionError, GateSchemaError, GateValidationError
from holodeck.models.workflow import EdgeNode

logger = logging.getLogger(__name__)


def _refuse_retrieval(uri: str) -> Resource[Any]:
    """Refuse to fetch a reference the gate schema did not carry itself.

    ``jsonschema`` resolves ``$ref`` lazily at validate time and, left at its
    defaults, retrieves remote references over the network with a blocking
    ``urlopen``. That would make a workflow file an SSRF vector, put sync I/O
    on the event loop, and — worst — mean the gate actually enforced is not the
    ``gate_schema`` snapshotted for replay (T10).

    Args:
        uri: The reference the validator asked to retrieve.

    Returns:
        Never returns.

    Raises:
        NoSuchResource: Always. Retrieval is not a capability the gate has.
    """
    raise NoSuchResource(ref=uri)  # type: ignore[call-arg]


#: Registry used for every gate validation: it can resolve references the
#: schema document carries itself (``#/$defs/..``) and nothing else.
_NO_RETRIEVAL_REGISTRY: Registry[Any] = Registry(
    retrieve=_refuse_retrieval  # type: ignore[call-arg]
)


class GatedOutput(BaseModel):
    """The one thing an edge node is allowed to hand the spine.

    ``value`` is the object that satisfied the gate — the canonical value for
    every downstream node (FR-008). The raw model text is deliberately absent:
    if it is not in here, it cannot be consumed by a decision table.

    Both mapping fields are deep-copied on construction. Pydantic's ``frozen``
    only rebinds-guards the top-level attributes; without the copy a caller
    holding the source object could still mutate a value that has crossed the
    gate and been persisted for replay.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(description="Id of the edge node that produced this value.")
    value: dict[str, Any] = Field(
        description="The gate-validated object; the canonical value that crosses "
        "into the spine (never the raw LLM text).",
    )
    gate_schema: dict[str, Any] = Field(
        description="Content of the JSON Schema the value was validated against, "
        "carried by value so a run record snapshots the schema itself rather "
        "than a path into the working tree (T10).",
    )

    @field_validator("value", "gate_schema", mode="before")
    @classmethod
    def _detach(cls, value: Any) -> Any:
        """Deep-copy the mapping so ``frozen`` holds all the way down.

        Args:
            value: The mapping as supplied by the caller.

        Returns:
            A copy that shares no nested container with the caller's object.
        """
        return copy.deepcopy(value)


def load_gate_schema(node: EdgeNode, workflow_dir: Path) -> dict[str, Any]:
    """Read and parse an edge node's gate schema.

    Args:
        node: The edge node whose ``gate.schema`` is being resolved.
        workflow_dir: Directory containing ``workflow.yaml``; the gate schema
            path is relative to it.

    Returns:
        The parsed JSON Schema.

    Raises:
        GateSchemaError: If the schema file cannot be read, is not a JSON
            object, or declares a type the spine cannot address. These are
            authoring defects, deliberately not ``GateValidationError`` — see
            that class's docstring.
    """
    path = (workflow_dir / node.gate.schema_path).resolve()
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GateSchemaError(
            node.id, f"gate schema '{path}' could not be read: {exc}"
        ) from exc
    try:
        schema = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GateSchemaError(
            node.id, f"gate schema '{path}' is not valid JSON: {exc}"
        ) from exc
    if not isinstance(schema, dict):
        raise GateSchemaError(
            node.id,
            f"gate schema '{path}' must be a JSON object, got "
            f"{type(schema).__name__}",
        )

    # The spine addresses an edge value by node id (`inputs: [<node id>]`) and
    # dot-paths its fields in FEEL, so a gate that asks the model for anything
    # but an object is unaddressable. Caught here, at load, because it is the
    # author's defect: enforcing it on the output instead would spend an agent
    # call and charge an SC-003 gate rejection to a model that did as asked.
    declared = schema.get("type")
    if isinstance(declared, str):
        permitted: list[Any] | None = [declared]
    elif isinstance(declared, list):
        permitted = declared
    else:
        permitted = None
    if permitted is not None and "object" not in permitted:
        raise GateSchemaError(
            node.id,
            f"gate schema '{path}' must describe an object the spine can name "
            f"fields on, but declares type {declared!r}",
        )
    return schema


def _apply_gate(
    node_id: str, result: ExecutionResult, gate_schema: dict[str, Any]
) -> GatedOutput:
    """Validate an agent result against the gate schema.

    Args:
        node_id: Id of the edge node, used to locate any failure.
        result: The agent's execution result.
        gate_schema: The parsed gate JSON Schema.

    Returns:
        The :class:`GatedOutput` carrying the validated object.

    Raises:
        GateSchemaError: If the gate schema is not a valid JSON Schema, or
            refers to a resource it does not carry — authoring defects, not
            rejections of the model's output.
        GateValidationError: If the agent produced free text, or the structured
            output does not satisfy the gate schema.
    """
    value = result.structured_output
    if value is None:
        raise GateValidationError(
            node_id,
            "agent produced free text, not structured output; nothing was "
            "presented to the gate",
        )

    validator_cls = jsonschema.validators.validator_for(gate_schema)
    try:
        validator_cls.check_schema(gate_schema)
    except jsonschema.SchemaError as exc:
        raise GateSchemaError(
            node_id, f"gate schema is not a valid JSON Schema: {exc.message}"
        ) from exc

    validator = validator_cls(
        gate_schema,
        registry=_NO_RETRIEVAL_REGISTRY,
        format_checker=validator_cls.FORMAT_CHECKER,
    )
    try:
        validator.validate(value)
    except Unresolvable as exc:
        # Includes jsonschema's _WrappedReferencingError, which is neither a
        # ValidationError nor a SchemaError and would otherwise escape every
        # declared channel.
        raise GateSchemaError(
            node_id,
            f"gate schema reference could not be resolved (the gate is not "
            f"self-contained and retrieval is refused): {exc}",
        ) from exc
    except jsonschema.ValidationError as exc:
        raise GateValidationError(
            node_id,
            f"structured output violates the gate schema at "
            f"{exc.json_path}: {exc.message}",
        ) from exc

    if not isinstance(value, dict):
        # load_gate_schema rejects a gate that *declares* a non-object type,
        # but a gate with no declared type (e.g. `{}`) cannot be judged
        # statically, so the spine's addressability requirement is enforced
        # here too.
        raise GateValidationError(
            node_id,
            f"gate output must be a JSON object the spine can name fields on, "
            f"got {type(value).__name__}",
        )
    return GatedOutput(node_id=node_id, value=value, gate_schema=gate_schema)


async def _teardown(backend: AgentBackend, node_id: str) -> None:
    """Tear a backend down without letting its failure change the outcome.

    A raising ``teardown`` (``ClaudeBackend`` tears down an SDK subprocess)
    must never mask the real error nor discard a result that already crossed
    the gate.

    Args:
        backend: The backend to tear down.
        node_id: Id of the edge node, for the log record.
    """
    try:
        await backend.teardown()
    except Exception:
        logger.warning(
            "edge node '%s': backend teardown failed; continuing",
            node_id,
            exc_info=True,
        )


async def execute_edge_node(
    node: EdgeNode,
    workflow_dir: Path,
    message: str,
) -> GatedOutput:
    """Run an edge node's agent and gate its structured output.

    The gate schema is loaded *before* the agent is invoked: a node that cannot
    gate must not spend an agent call, and the failure is the workflow author's,
    not the model's.

    Args:
        node: The edge node to execute.
        workflow_dir: Directory containing ``workflow.yaml``; both
            ``edge.agent`` and ``gate.schema`` are resolved relative to it.
        message: The prompt handed to the edge agent. Composed by the caller
            (the runner, T6) — this function orchestrates a single node only.

    Returns:
        The :class:`GatedOutput` whose ``value`` is the canonical object
        crossing into the spine.

    Raises:
        GateSchemaError: If the gate schema cannot be loaded, is not a valid
            JSON Schema, describes something the spine cannot address, or is
            not self-contained — all workflow-authoring defects.
        GateValidationError: If the agent returned free text or schema-invalid
            output. An ``is_error`` result that nonetheless carries structured
            output lands here, not in ``ExecutionError``: the model produced
            something, and what it produced is evidence about the model.
        ExecutionError: If the invocation produced nothing to judge — it raised,
            or it failed with no structured output. Distinct from a gate
            rejection: a broken invocation is not evidence about the model.
        holodeck.lib.errors.FileNotFoundError: If the referenced ``agent.yaml``
            does not exist.
        ConfigError: If the referenced ``agent.yaml`` is invalid.
        BackendInitError: If no backend supports the agent's provider.
    """
    gate_schema = await asyncio.to_thread(load_gate_schema, node, workflow_dir)

    agent_path = (workflow_dir / node.edge.agent).resolve()
    agent = await asyncio.to_thread(ConfigLoader().load_agent_yaml, str(agent_path))

    # Backends resolve a tool's relative `file:` against the agent_base_dir
    # contextvar. Only load_agent_and_config (the `test`/`chat` path) sets it;
    # loading the YAML directly does not, so without this an edge agent's
    # function tool resolves against the process CWD and is found only when the
    # run happens to start in the agent's directory. Same value that path sets:
    # the agent YAML's own parent.
    agent_base_dir.set(str(agent_path.parent))

    backend: AgentBackend = await BackendSelector.select(agent)
    try:
        result = await backend.invoke_once(message)
    except Exception as exc:
        await _teardown(backend, node.id)
        raise ExecutionError(
            f"edge node '{node.id}': agent invocation raised "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    await _teardown(backend, node.id)

    if result.is_error and result.structured_output is None:
        raise ExecutionError(
            f"edge node '{node.id}': agent invocation failed: {result.error_reason}"
        )
    return _apply_gate(node.id, result, gate_schema)
