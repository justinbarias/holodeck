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

Gate validation never touches the network: retrieval is refused outright, so
the only references that resolve are the ones the document carries itself and
the metaschemas ``jsonschema`` bundles. Whichever way a reference is settled,
the schema snapshotted in :class:`GatedOutput` is exactly the schema enforced.

The load-time ``$ref`` walk below exists for correctness first, economy
second: ``validate()`` resolves references lazily, so an unresolvable ``$ref``
in a branch no instance happens to reach is silently never checked — a gate
that looks enforced but has a dead limb. The walk makes that an authoring
error at load, deterministically; the saved agent call is a side benefit.

*When* an unresolvable reference is discovered depends on the gate's dialect,
and only two dialects are checked at load:

* **2020-12 and 2019-09** — including a gate with no ``$schema`` at all, since
  that is the dialect ``jsonschema`` itself falls back to. For these,
  :func:`load_gate_schema` walks the document and resolves every ``$ref``
  :func:`_collect_refs` reaches, so an unresolvable one — remote,
  sibling-file, or a mistyped pointer into the gate's own ``$defs`` — is an
  authoring defect found before an agent call. Even here the walk is partial:
  what it still does not model is listed in :func:`_collect_refs`.
* **Every other dialect** — draft-07, draft-06, draft-04, draft-03, or the
  ``Specification.OPAQUE`` fallback, whose subresource table is empty and
  would make the walk a no-op anyway. (OPAQUE is narrower than it reads:
  ``validator_for`` falls back to 2020-12 for a ``$schema`` it does not
  recognise at all, so OPAQUE is reachable only for a dialect ``jsonschema``
  knows but ``referencing`` does not.) For these **no load-time reference
  check runs at all**; the gate loads unexamined and any resolution failure surfaces
  at validate time, after one agent call, through the backstop in
  :func:`_apply_gate`. Why the line is drawn there is in
  :data:`_WALKED_SPECIFICATIONS`.

What crosses is the :class:`GatedOutput`: the *validated object*, never the raw
model text. It carries the gate schema by **content**, not by path, so any
record of the run can snapshot what the output was actually judged against.

Per refinements §1 the POC validates the Claude backend only; dispatch still
goes through ``BackendSelector`` so the protocol contract is unchanged.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jsonschema
import jsonschema.validators
from jsonschema.validators import SPECIFICATIONS
from pydantic import BaseModel, ConfigDict, Field, field_validator
from referencing import Registry, Resource, Specification
from referencing.exceptions import NoSuchResource, Unresolvable
from referencing.jsonschema import DRAFT201909, DRAFT202012, specification_with

from holodeck.config.context import agent_base_dir

if TYPE_CHECKING:
    # Annotation-only: importing anything under holodeck.lib.backends at
    # runtime executes that package's __init__, which eagerly imports the
    # concrete backends and with them the Claude Agent SDK. Agent is here for
    # consistency with the module's import-purity stance — it is used only as
    # an annotation on execute_edge_node.
    from holodeck.lib.backends.base import AgentBackend, ExecutionResult
    from holodeck.models.agent import Agent
# BackendSelector is imported lazily inside execute_edge_node, not here:
# selector.py imports the concrete backends (and with them the Claude Agent
# SDK) at module scope, so a module-level import would drag the whole backend
# stack into any importer of the pure gate half — load_gate_schema and
# _apply_gate must stay importable from Temporal workflow code, which
# forbids I/O imports (spec 040 section 7, specs/040-holodeck-temporal/spec.md).
# test_import_purity.py pins this.
from holodeck.lib.errors import (
    ConfigError,
    ExecutionError,
    GateSchemaError,
    GateValidationError,
)
from holodeck.models.workflow import EdgeNode

logger = logging.getLogger(__name__)


def _refuse_retrieval(uri: str) -> Resource[Any]:
    """Refuse to fetch a reference the gate schema did not carry itself.

    Enforced rather than inherited: since 4.18 ``jsonschema``'s own default
    registry already refuses unknown references (the network-fetching
    ``RefResolver`` is the deprecated path), so today this makes the
    invariant explicit rather than closing a live hole. It stays because the
    stakes are the module's, not the library's: a retrieving registry passed
    by a future caller would make a workflow file an SSRF vector, put sync
    I/O on the event loop, and — worst — mean the gate actually enforced is
    not the ``gate_schema`` snapshotted for replay (T10).

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

#: Registry the load-time ``$ref`` walk resolves against. It must be exactly
#: what the validator itself would use, or the walk would refuse a reference
#: the gate can in fact resolve: ``jsonschema`` combines any caller-supplied
#: registry with ``SPECIFICATIONS`` (the metaschemas it ships) before building
#: a resolver, so a schema referring to a metaschema resolves at validate time
#: and must resolve here too. Nothing in ``SPECIFICATIONS`` is retrieved — it
#: is bundled — so this adds no network reach.
_GATE_REF_REGISTRY: Registry[Any] = SPECIFICATIONS.combine(_NO_RETRIEVAL_REGISTRY)

#: The dialects whose ``subresources_of`` table the load-time walk trusts.
#:
#: The invariant: the walk only runs where ``referencing``'s subresource table
#: is known-complete and known to yield only schemas (2020-12, 2019-09). The
#: pre-2019 tables both *hide* references (draft-07/06/04 ``dependencies``,
#: draft-03 object-form ``extends``), and hand-rolling descent for them risks
#: the worse error of refusing a gate ``validate()`` would accept. An unwalked
#: dialect degrades to the runtime backstop in :func:`_apply_gate` — a quieter
#: check, never a wrong one. The per-dialect behaviour is pinned by
#: ``test_dialect_matrix_pins_where_an_unresolvable_ref_is_caught``.
_WALKED_SPECIFICATIONS: frozenset[Specification[Any]] = frozenset(
    {DRAFT202012, DRAFT201909}
)

#: Load-time bounds on the gate document. The file is attacker-influenceable
#: (same posture as the retrieval refusal), and both ``check_schema`` and the
#: ``$ref`` walk recurse — an unbounded document would escape the module's
#: error taxonomy as a bare ``RecursionError``. 100 levels / 5 MB are far past
#: any real gate and far short of the interpreter's limits.
_MAX_GATE_DEPTH = 100
_MAX_GATE_BYTES = 5 * 1024 * 1024


def _gate_specification(validator_cls: type[Any]) -> Specification[Any]:
    """Return the ``referencing`` specification ``validator_cls`` resolves under.

    Derived exactly the way ``jsonschema.validators.create`` derives it — from
    the dialect id of the validator's own metaschema, falling back to
    ``Specification.OPAQUE`` for a dialect ``referencing`` does not know. Using
    the same object means the walk's notion of "this is a subschema" and its
    base-URI arithmetic are the validator's, not an approximation of them.

    Args:
        validator_cls: The validator class ``validator_for`` chose for the gate.

    Returns:
        The specification whose subresource table and ``$id`` rules apply.
    """
    return specification_with(
        validator_cls.ID_OF(validator_cls.META_SCHEMA) or "urn:unknown-dialect",
        default=Specification.OPAQUE,
    )


def _exceeds_depth(document: Any, limit: int) -> bool:
    """Report whether ``document`` nests deeper than ``limit`` levels.

    Iterative on purpose: this runs *before* ``check_schema`` and the ``$ref``
    walk, both of which recurse and would surface an overdeep document as a
    bare ``RecursionError`` outside the module's declared error taxonomy. It
    counts raw container depth, which bounds every later recursion: each step
    of the ``$ref`` walk descends at least one container level, so a document
    that passes here cannot push the walk past the same limit.
    """
    stack: list[tuple[Any, int]] = [(document, 0)]
    while stack:
        node, depth = stack.pop()
        if depth > limit:
            return True
        if isinstance(node, dict):
            stack.extend((child, depth + 1) for child in node.values())
        elif isinstance(node, list):
            stack.extend((child, depth + 1) for child in node)
    return False


def _collect_refs(
    spec: Specification[Any], schema: Any, resolver: Any
) -> list[tuple[str, Any]]:
    r"""Collect the ``$ref``\ s in schema position, each with its own scope.

    Only ever called for a dialect in :data:`_WALKED_SPECIFICATIONS`. Descent
    is delegated to ``spec.subresources_of`` — ``referencing``'s own table of
    which keywords hold subschemas — so the walk enters a value exactly where
    a subschema begins: a *property named* ``$ref`` is data, a ``$ref``-shaped
    literal under ``const``/``enum`` is data, and ``$ref`` is read only in
    keyword position. Each reference is paired with the resolver in force
    where it appears, so a nested ``$id`` rebases it by the validator's own
    rule (``Resolver.in_subresource``).

    Not modelled, and therefore left to the runtime backstop in
    :func:`_apply_gate`: ``$dynamicRef``/``$recursiveRef`` (target chosen from
    the dynamic scope, not the document); a subschema that re-dialects itself
    with its own ``$schema`` (re-dialecting would mean walking by the very
    tables :data:`_WALKED_SPECIFICATIONS` refuses to trust); the deprecated,
    never-evaluated ``dependencies``; anything the table does not call a
    subschema; and a non-string ``$ref``. The per-case behaviour is pinned by
    the walk tests in ``test_edge_gate.py``.

    Recursion here is bounded by the ``_exceeds_depth`` pre-check in
    :func:`load_gate_schema`: every step descends at least one container
    level, so the walk can never out-recurse a document that check accepted.

    Args:
        spec: The specification returned by :func:`_gate_specification`.
        schema: The parsed gate schema, or any fragment of it.
        resolver: The ``referencing`` resolver in force at ``schema``.

    Returns:
        ``(reference, resolver)`` pairs in document order, duplicates kept.
    """
    found: list[tuple[str, Any]] = []

    def walk(node: Any, scope: Any) -> None:
        if not isinstance(node, dict):
            return
        ref = node.get("$ref")
        if isinstance(ref, str):
            found.append((ref, scope))
        for child in spec.subresources_of(node):
            # Both walked dialects allow a boolean wherever they allow a
            # schema. One carries neither a reference nor an `$id`, so there is
            # nothing to collect and nothing to rebase against.
            if not isinstance(child, dict):
                continue
            walk(child, scope.in_subresource(spec.create_resource(child)))

    walk(schema, resolver)
    return found


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


def _escapes(candidate: Path, root: Path) -> bool:
    """Report whether a resolved path lands outside ``root``.

    The module treats the workflow file as attacker-influenceable (that is why
    remote ``$ref`` retrieval is refused); letting a path field name
    ``/etc/passwd`` or ``../../x`` would be the same hole through the front
    door. An absolute path replaces ``root`` entirely under ``/``, so
    resolve-then-check is the only shape that catches both forms.

    Args:
        candidate: The already-``resolve()``-d path to judge.
        root: The directory the path must stay inside.

    Returns:
        ``True`` when ``candidate`` is not inside ``root``.
    """
    return not candidate.is_relative_to(root.resolve())


def resolve_agent_path(node: EdgeNode, workflow_dir: Path) -> Path:
    """Resolve an edge node's agent path, confined to the workflow directory.

    The confinement seam for ``EdgeRef.agent`` — the same control
    :func:`load_gate_schema` applies to ``gate.schema``. The 036 loader that
    resolved this path was removed with the overlay engine; whatever consumes
    an :class:`EdgeNode` next (the Temporal activity wrapper, spec 040 D1)
    must resolve the reference through here rather than joining the path
    itself.

    Args:
        node: The edge node whose ``edge.agent`` is being resolved.
        workflow_dir: Directory containing the workflow file; the agent path
            is relative to it.

    Returns:
        The resolved path to the agent's ``agent.yaml``.

    Raises:
        ConfigError: If the path escapes the workflow directory.
    """
    path = (workflow_dir / node.edge.agent).resolve()
    if _escapes(path, workflow_dir):
        raise ConfigError(
            f"nodes.{node.id}.edge.agent",
            f"agent path '{node.edge.agent}' escapes the workflow directory "
            f"'{workflow_dir}'",
        )
    return path


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
            object, is not a valid JSON Schema, or declares a type the spine
            cannot address — and, *for a gate whose dialect is one of*
            :data:`_WALKED_SPECIFICATIONS` only, if it carries a ``$ref`` in a
            schema position :func:`_collect_refs` reaches that cannot be
            resolved without retrieval. A gate in any other dialect gets no
            reference check here at all; its references are settled at validate
            time by :func:`_apply_gate`. These are authoring defects,
            deliberately not ``GateValidationError`` — see that class's
            docstring.
    """
    path = (workflow_dir / node.gate.schema_path).resolve()
    if _escapes(path, workflow_dir):
        raise GateSchemaError(
            node.id,
            f"gate schema '{node.gate.schema_path}' escapes the workflow "
            f"directory '{workflow_dir}'",
        )
    try:
        # Size-capped for the same reason the depth is bounded below: the file
        # is attacker-influenceable (see _MAX_GATE_BYTES).
        if path.stat().st_size > _MAX_GATE_BYTES:
            raise GateSchemaError(
                node.id,
                f"gate schema '{path}' exceeds {_MAX_GATE_BYTES} bytes; "
                f"refusing to load it",
            )
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
    except RecursionError as exc:
        # json.loads is the first recursive consumer of the document — a
        # '['-bomb well under the size cap out-recurses the C scanner before
        # _exceeds_depth below ever sees the parse. Same authoring defect,
        # same channel.
        raise GateSchemaError(
            node.id, f"gate schema '{path}' nests too deeply to parse"
        ) from exc
    if not isinstance(schema, dict):
        raise GateSchemaError(
            node.id,
            f"gate schema '{path}' must be a JSON object, got "
            f"{type(schema).__name__}",
        )
    # Checked before check_schema and the $ref walk, both of which recurse:
    # without this, an overdeep document escapes as a bare RecursionError
    # instead of the authoring error it is. This is the only depth guard —
    # container depth bounds the walk's recursion (see _exceeds_depth).
    if _exceeds_depth(schema, _MAX_GATE_DEPTH):
        raise GateSchemaError(
            node.id,
            f"gate schema '{path}' nests deeper than {_MAX_GATE_DEPTH} levels; "
            f"refusing to walk it",
        )

    # Whether the document is a *schema* is judged here, not at validate time:
    # `{"required": "name"}` parses, reads, and looks like a gate, so nothing
    # short of the metaschema catches it. Deferring it to the first output would
    # spend an agent call to discover a typo in a file the author already wrote.
    validator_cls = jsonschema.validators.validator_for(schema)
    try:
        validator_cls.check_schema(schema)
    except jsonschema.SchemaError as exc:
        raise GateSchemaError(
            node.id, f"gate schema '{path}' is not a valid JSON Schema: {exc.message}"
        ) from exc

    # References are settled here too — but only for the dialects
    # `_WALKED_SPECIFICATIONS` names; every other dialect keeps jsonschema's
    # lazy validate-time resolution. Deliberately stricter than validate time:
    # a reference is refused even in a branch no instance reaches (an unused
    # `$defs` entry, the losing arm of an `if`/`then`), because an unresolvable
    # ref is an authoring defect whether or not this run's output touches it —
    # accepted, it ships a gate that stops constraining anything the first time
    # an instance does reach that branch. The reverse error — refusing a ref
    # `validate()` would resolve — is what `_collect_refs` is built to avoid.
    spec = _gate_specification(validator_cls)
    if spec in _WALKED_SPECIFICATIONS:
        resolver = _GATE_REF_REGISTRY.resolver_with_root(spec.create_resource(schema))
        for ref, scope in _collect_refs(spec, schema, resolver):
            try:
                scope.lookup(ref)
            except Unresolvable as exc:
                raise GateSchemaError(
                    node.id,
                    f"gate schema '{path}' references '{ref}', which it does "
                    f"not carry and retrieval is refused: {exc}",
                ) from exc
            except AttributeError as exc:
                # Resolving a reference makes `referencing` crawl the whole
                # document, and the crawl re-dialects an embedded resource from
                # its own `$schema` — including into the pre-2019 tables
                # `_WALKED_SPECIFICATIONS` refuses to walk by. Those hand a
                # `str`/`list` to `_legacy_anchor_in_id`, which calls `.get` on
                # it. Narrowing the walk does not put this out of reach: the
                # root dialect can be 2020-12 and the embedded one draft-03.
                # It is still an authoring defect, so it leaves through the
                # channel this function declares rather than as a traceback —
                # but the catch is wider than that one cause, so the message
                # hedges and the real traceback goes to the debug log.
                logger.debug(
                    "gate schema '%s': AttributeError while resolving '%s'",
                    path,
                    ref,
                    exc_info=True,
                )
                raise GateSchemaError(
                    node.id,
                    f"gate schema '{path}' references '{ref}', and the "
                    f"document could not be crawled to resolve it "
                    f"({type(exc).__name__}: {exc}); most commonly an embedded "
                    f"subschema puts a non-schema where its declared dialect "
                    f"expects one",
                ) from exc

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
    # `null` is a promise the executor cannot keep: _apply_gate reads
    # `structured_output is None` as "the agent produced free text, nothing to
    # validate", so a null value would be rejected with a misleading message
    # and charged to the model. An authoring defect, settled at load like the
    # non-object case above.
    if permitted is not None and "null" in permitted:
        raise GateSchemaError(
            node.id,
            f"gate schema '{path}' permits type 'null', which the gate can "
            f"never accept: an absent structured output is indistinguishable "
            f"from a null one",
        )
    return schema


def check_gate(
    obj: Any, schema: dict[str, Any], *, node_id: str = "gate"
) -> dict[str, Any]:
    """Validate a plain object against a gate schema.

    The gate-check half of this module, on plain values: no
    ``ExecutionResult``, no backend import, nothing but ``jsonschema``. That is
    what lets Temporal workflow code call it (spec 040 D3) as well as the
    activity that produced the object in the first place.

    Args:
        obj: The candidate object. ``None`` means the agent produced free text
            and there is nothing to present to the gate — a rejection, not a
            crash.
        schema: The gate JSON Schema, already checked by
            :func:`load_gate_schema`.
        node_id: Id of the edge node the object came from, used to locate any
            failure in the message.

    Returns:
        The validated object, unchanged.

    Raises:
        GateSchemaError: If a reference cannot be resolved here. For a gate in
            a dialect :data:`_WALKED_SPECIFICATIONS` does not name — draft-07,
            draft-06, draft-04, draft-03 — this is not a backstop but the
            *only* check there is: :func:`load_gate_schema` performs no
            reference walk for those at all. For the two dialects it does walk,
            this is the backstop, and what still arrives here is what that walk
            does not model: ``$dynamicRef``/``$recursiveRef``, whose target
            comes from the dynamic scope of this validation rather than from
            the document, and a reference inside a subschema that declares its
            own ``$schema``. The full list is in :func:`_collect_refs`.
        GateValidationError: If the agent produced free text, or the structured
            output does not satisfy the gate schema.
    """
    value = obj
    if value is None:
        raise GateValidationError(
            node_id,
            "agent produced free text, not structured output; nothing was "
            "presented to the gate",
        )

    validator_cls = jsonschema.validators.validator_for(schema)
    validator = validator_cls(
        schema,
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
    except AttributeError as exc:
        # Same upstream defect `load_gate_schema` converts: resolving a
        # reference crawls the document, and the pre-2019 subresource tables
        # yield a `str` (draft-03 `extends` written as an object) or a `list`
        # (a `dependencies` whose first value is a schema and a later one an
        # array) where a schema is expected, so the crawl calls `.get` on it.
        # Reachable from any dialect, because an embedded subschema may declare
        # a legacy `$schema` of its own. The gate cannot be applied; that is an
        # authoring defect, and it must not escape as a bare AttributeError.
        # The catch is wider than that one cause (it wraps the whole
        # validate()), so the message hedges and the traceback goes to debug.
        logger.debug(
            "edge node '%s': AttributeError while applying the gate",
            node_id,
            exc_info=True,
        )
        raise GateSchemaError(
            node_id,
            f"gate schema could not be applied under its declared dialect "
            f"({type(exc).__name__}: {exc}); most commonly a subschema puts a "
            f"non-schema where that dialect expects one",
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
    return value


def _apply_gate(
    node_id: str, result: ExecutionResult, gate_schema: dict[str, Any]
) -> GatedOutput:
    """Validate an agent result against the gate schema.

    The ``ExecutionResult``-shaped wrapper over :func:`check_gate`, for the
    in-process executor. Callers holding a plain object — the Temporal
    activity, workflow code — call :func:`check_gate` directly.

    Args:
        node_id: Id of the edge node, used to locate any failure.
        result: The agent's execution result.
        gate_schema: The gate JSON Schema, already checked by
            :func:`load_gate_schema`.

    Returns:
        The :class:`GatedOutput` carrying the validated object.

    Raises:
        GateSchemaError: As described in :func:`check_gate`.
        GateValidationError: As described in :func:`check_gate`.
    """
    return GatedOutput(
        node_id=node_id,
        value=check_gate(result.structured_output, gate_schema, node_id=node_id),
        gate_schema=gate_schema,
    )


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
    agent: Agent,
    agent_path: Path,
    message: str,
    gate_schema: dict[str, Any],
) -> GatedOutput:
    """Run an edge node's agent and gate its structured output.

    Provisional pending the Temporal activity design (spec 040 D1): the
    activity wrapper may invoke the backend against Temporal's own
    retry/timeout/cancellation seams rather than through this function, in
    which case this invocation half is replaced and only the pure gate half
    of the module survives. Kept meanwhile as the one tested reference for
    the invocation-plus-gate sequencing.

    Neither the gate nor the agent is read here: :func:`load_gate_schema` and
    ``ConfigLoader.load_agent_yaml`` both run at preparation, and handing their
    results down is what makes the schema enforced here — and snapshotted into
    the :class:`GatedOutput` — and the agent invoked here the very objects that
    were approved. Re-reading either file would reopen a window the width of a
    model call between validation and use.

    Args:
        node: The edge node to execute.
        agent: The node's agent configuration, loaded at preparation.
        agent_path: Path the ``agent`` was loaded from. Its parent becomes the
            ``agent_base_dir`` a backend resolves relative ``file:`` tool
            references against; the file itself is not re-read.
        message: The prompt handed to the edge agent. Composed by the caller —
            this function orchestrates a single node only.
        gate_schema: The node's gate schema as returned by
            :func:`load_gate_schema`.

    Returns:
        The :class:`GatedOutput` whose ``value`` is the canonical object
        crossing into the spine.

    Raises:
        ConfigError: If the agent declares no ``response_format`` — it could
            never produce structured output, so every run would spend a model
            call to discover an authoring defect.
        GateValidationError: If the agent returned free text or schema-invalid
            output. An ``is_error`` result that nonetheless carries structured
            output lands here, not in ``ExecutionError``: the model produced
            something, and what it produced is evidence about the model.
        GateSchemaError: Only as the backstop described in :func:`_apply_gate` —
            the authoring defects preparation is able to judge are settled
            there.
        ExecutionError: If the invocation produced nothing to judge — it raised,
            or it failed with no structured output. Distinct from a gate
            rejection: a broken invocation is not evidence about the model.
        BackendInitError: If constructing or initialising the backend fails.
            Not surfaced by any load-time validation: knowing it requires
            *building* a backend, which costs a connection attempt before the
            first agent call. Its unsupported-provider case is unreachable from an
            ``agent.yaml`` — ``model.provider`` is an enum every branch of
            ``BackendSelector`` covers, and an unknown value is a ``ConfigError``
            at load.
    """
    # An agent with no response_format can never produce structured_output
    # (ClaudeBackend builds its output format from it), so every run would
    # burn a model call and land at _apply_gate's "free text" rejection —
    # an SC-003 gate rejection charged to a model that was never asked for
    # structure. The author's defect, caught before any backend is built.
    if agent.response_format is None:
        raise ConfigError(
            f"nodes.{node.id}.edge.agent",
            "edge agent declares no response_format, so it can never produce "
            "structured output for the gate",
        )

    # Backends resolve a tool's relative `file:` against the agent_base_dir
    # contextvar. Only load_agent_and_config (the `test`/`chat` path) sets it;
    # loading the YAML directly does not, so without this an edge agent's
    # function tool resolves against the process CWD and is found only when the
    # run happens to start in the agent's directory. Same value that path sets:
    # the agent YAML's own parent.
    #
    # Reset via token rather than left set: this module may be called
    # repeatedly by a long-lived caller (a future server/embedded run), and a
    # ContextVar mutated with no restore would leak this node's directory into
    # whatever runs after it.
    # Deferred import — see the note at the top of the module's import block.
    from holodeck.lib.backends.selector import BackendSelector

    token = agent_base_dir.set(str(agent_path.parent))
    try:
        # mode="test" spelled out (it is the default): its permission mapping
        # is the `holodeck test` one, which is what a non-interactive edge run
        # wants today. Whether a headless Temporal activity needs its own mode
        # is a D1 question.
        backend: AgentBackend = await BackendSelector.select(agent, mode="test")
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
                f"edge node '{node.id}': agent invocation failed: "
                f"{result.error_reason}"
            )
        return _apply_gate(node.id, result, gate_schema)
    finally:
        agent_base_dir.reset(token)
