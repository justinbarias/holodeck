"""Pre-execution validation of a workflow's ``input_data`` facts (036, T2a).

A run computes one state transition from facts of record supplied by the caller
(prior state, case facts). Those facts are declared in ``workflow.yaml`` under
``input_data:`` with a JSON Schema each, and every one of them is validated
**before any node executes** — a missing or schema-invalid fact stops the run
rather than flowing silently into a determination (FR-025).

Facts of record never come from an agent (FR-027); this module only checks what
the caller supplied against what the workflow declared.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, NoReturn

import jsonschema
import jsonschema.exceptions
import jsonschema.validators
import referencing
from referencing.exceptions import Unresolvable

from holodeck.lib.errors import InputDataError
from holodeck.models.workflow import Workflow

logger = logging.getLogger(__name__)


def _refuse_remote_ref(uri: str) -> NoReturn:
    """Retrieve hook that refuses to fetch any ``$ref`` over the network.

    ``jsonschema``'s own default registry retrieves an unresolved ``$ref`` via
    a blocking ``urlopen`` call (with a ``DeprecationWarning``). That is an
    SSRF reachable straight from a workflow-declared schema, and it makes
    validation non-deterministic: the run record must reproduce a transition
    from the same facts (FR-028), which is impossible if the schema those
    facts were checked against depends on whatever a remote URL served that
    day. A ``$ref`` that is not already crawled from the schema document
    itself must fail loudly, never fetch.

    Args:
        uri: The unresolved reference's URI. Unused; retrieval is refused
            unconditionally.

    Raises:
        Unresolvable: Always.
    """
    raise Unresolvable(uri)


_NO_REMOTE_REGISTRY: referencing.Registry[Any] = referencing.Registry(
    retrieve=_refuse_remote_ref  # type: ignore[call-arg]
)


def _load_schema(name: str, schema_file: Path) -> dict[str, Any]:
    """Read and parse a declared fact's JSON Schema.

    Args:
        name: The declared ``input_data`` fact the schema belongs to.
        schema_file: Resolved path to the schema document.

    Returns:
        The parsed JSON Schema.

    Raises:
        InputDataError: If the file cannot be read or is not a JSON object.
    """
    try:
        raw = schema_file.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        # ValueError covers UnicodeDecodeError, which does not subclass OSError:
        # a schema file must decode the same way regardless of platform locale,
        # so a genuinely undecodable file is a fact-of-record failure too.
        raise InputDataError(
            name, f"cannot read schema '{schema_file}': {exc}"
        ) from exc

    try:
        schema = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InputDataError(
            name, f"schema '{schema_file}' is not valid JSON: {exc}"
        ) from exc

    if not isinstance(schema, dict):
        raise InputDataError(
            name, f"schema '{schema_file}' must be a JSON object, got {type(schema)}"
        )
    return schema


def validate_input_data(
    workflow: Workflow, payload: dict[str, Any], base_dir: Path
) -> dict[str, Any]:
    """Validate every declared fact of record before any node executes.

    Each fact declared under ``workflow.input_data`` must be present in the
    payload and valid against its JSON Schema, resolved relative to the
    directory holding ``workflow.yaml``. Keys the workflow did not declare are
    dropped: only declared facts are referenceable from a node's ``inputs:``,
    so undeclared data has no way into the spine.

    Args:
        workflow: The loaded workflow whose ``input_data`` block is authoritative.
        payload: The caller-supplied facts, keyed by declared name.
        base_dir: Directory containing ``workflow.yaml``; schema paths resolve
            relative to it.

    Returns:
        The validated facts, keyed by declared name. Empty if the workflow
        declares no ``input_data``.

    Raises:
        InputDataError: If a declared fact is missing, its schema cannot be read
            or is not a valid JSON Schema, its schema has an unresolvable
            ``$ref``, or its value does not satisfy that schema. The error
            names the offending fact.
    """
    declared = workflow.input_data or {}
    validated: dict[str, Any] = {}

    for name, ref in declared.items():
        if name not in payload:
            raise InputDataError(name, "missing from the supplied input payload")

        # base_dir / ref.schema_path resolves `..`-escapes and absolute paths
        # unconstrained. Deliberate: workflow.yaml is operator-authored and
        # in-repo, and sibling node kinds (edge.agent, gate.schema, decision
        # tables) already name arbitrary paths with no such constraint, so
        # fencing only this one path would be theatre. Revisit if workflows
        # stop being hand-authored (spec 039 generates policy artifacts).
        schema = _load_schema(name, base_dir / ref.schema_path)
        value = payload[name]
        try:
            validator_cls = jsonschema.validators.validator_for(schema)
            validator_cls.check_schema(schema)
            # format_checker matches the edge gate (edge.py). Both are typed
            # boundaries into the spine, so a `format:` a fact schema declares
            # must be enforced, not treated as an annotation — otherwise a
            # declared constraint silently does nothing. Also makes
            # research.md caveat 6's `format: date` -> datetime.date
            # conversion safe: an unparseable date is rejected here rather
            # than exploding at conversion.
            validator = validator_cls(
                schema,
                registry=_NO_REMOTE_REGISTRY,
                format_checker=validator_cls.FORMAT_CHECKER,
            )
            error = jsonschema.exceptions.best_match(validator.iter_errors(value))
            if error is not None:
                raise error
        except jsonschema.SchemaError as exc:
            raise InputDataError(
                name, f"invalid JSON Schema '{ref.schema_path}': {exc.message}"
            ) from exc
        except jsonschema.ValidationError as exc:
            raise InputDataError(
                name, f"value is schema-invalid: {exc.message}"
            ) from exc
        except Unresolvable as exc:
            # jsonschema wraps referencing's resolution failures (unknown
            # internal $ref, relative/remote $ref with no retrieve hook) in a
            # private subclass of this public base; jsonschema's own
            # deprecation notice for RefResolutionError directs callers to
            # catch referencing.exceptions.Unresolvable directly.
            raise InputDataError(
                name, f"schema '{ref.schema_path}' has an unresolvable $ref: {exc}"
            ) from exc

        validated[name] = value

    undeclared = payload.keys() - declared.keys()
    if undeclared:
        logger.warning(
            "input_data: payload keys not declared by workflow are dropped: %s",
            sorted(undeclared),
        )

    return validated
