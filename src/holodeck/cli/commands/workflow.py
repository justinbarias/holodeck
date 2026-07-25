"""CLI commands for running deterministic-spine workflows (036, T6).

Implements the ``holodeck workflow`` command group. ``run`` loads a
``workflow.yaml``, binds the facts of record from ``--input``, validates the
whole graph, and then executes it — emitting each node's result to stdout.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn

import click
from pydantic import ValidationError as PydanticValidationError

from holodeck.lib.backends.base import BackendInitError
from holodeck.lib.errors import (
    ConfigError,
    ExecutionError,
    FeelEvaluationError,
    GateValidationError,
    TableEvalError,
    WorkflowError,
)
from holodeck.lib.logging_config import get_logger
from holodeck.lib.workflow.runner import WorkflowRunResult, run_workflow

logger = get_logger(__name__)


def _fail(banner: str, error: Exception, code: int) -> NoReturn:
    """Log an error, print it, and exit with ``code``."""
    logger.error(f"{banner}: {error}")
    click.secho(f"Error: {banner}", fg="red", err=True)
    click.echo(f"  {error}", err=True)
    sys.exit(code)


@contextmanager
def handle_workflow_errors() -> Generator[None, None, None]:
    """Context manager mapping spine failures to exit codes.

    The codes separate the three things an operator does differently: fix a
    file, read the model's rejected output, or retry a flaky invocation. A gate
    rejection in particular is evidence about the *model* (SC-003), so it is
    deliberately not collapsed into a generic "run failed".

    Exit codes:
        1: Anything unexpected.
        2: The workflow, a decision table, a gate schema, or the input payload
           is invalid — an authoring or input defect, including a
           generated-but-unreviewed table (FR-030). No determination was
           attempted.
        3: Valid policy could not produce a determination (no rule matched and
           no default, a ``UNIQUE`` multi-match, or a FEEL evaluation failure).
        4: The gate rejected an edge agent's output (FR-007, SC-003).
        5: An edge agent invocation failed before producing output, or its
           backend could not be constructed (e.g. a missing API key).
    """
    try:
        yield
    except PydanticValidationError as e:
        _fail("Invalid workflow", e, 2)
    except GateValidationError as e:
        _fail("Agent output rejected at the gate", e, 4)
    except BackendInitError as e:
        # A missing/invalid credential is the commonest first-run condition,
        # not an internal fault: without this it fell to the generic handler
        # and printed a traceback before the message. Grouped with 5 because
        # the operator's next action is the same — fix the environment and
        # retry — and no determination was attempted either way.
        _fail("Edge agent backend could not be initialized", e, 5)
    except ExecutionError as e:
        _fail("Edge agent invocation failed", e, 5)
    except (TableEvalError, FeelEvaluationError) as e:
        _fail("Policy could not produce a determination", e, 3)
    except (ConfigError, WorkflowError) as e:
        # Everything else in the WorkflowError family is an authoring or input
        # defect: DecisionTableError, FeelValidationError, GateSchemaError,
        # InputDataError, PolicyReviewError, and the base itself.
        _fail("Invalid workflow", e, 2)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


def _load_payload(input_path: Path | None) -> dict[str, Any]:
    """Read the ``--input`` payload binding the workflow's facts of record.

    Args:
        input_path: Path to the JSON payload, or ``None`` when not supplied.

    Returns:
        The payload, or an empty mapping when no ``--input`` was given.

    Raises:
        ConfigError: If the file cannot be read, is not valid JSON, or is not a
            JSON object.
    """
    if input_path is None:
        return {}
    try:
        raw = input_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError("--input", f"could not read '{input_path}': {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            "--input", f"'{input_path}' is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ConfigError(
            "--input",
            f"'{input_path}' must be a JSON object keyed by input_data name, "
            f"got {type(payload).__name__}",
        )
    return payload


def _echo_result(result: WorkflowRunResult) -> None:
    """Print each node's result in execution order.

    Args:
        result: The completed run.
    """
    click.echo(f"Workflow '{result.name}' v{result.version}")
    for node_id in result.order:
        if node_id in result.gated_outputs:
            value = json.dumps(
                result.gated_outputs[node_id].value, sort_keys=True, default=str
            )
            click.echo(f"  edge   {node_id}: {value}")
        else:
            verdict = result.verdicts[node_id]
            outputs = json.dumps(verdict.outputs, sort_keys=True, default=str)
            click.echo(
                f"  policy {node_id}: {outputs} "
                f"[{verdict.table_id} {verdict.table_version}, "
                f"{verdict.rule_identity}]"
            )


@click.group(name="workflow", invoke_without_command=True)
@click.pass_context
def workflow(ctx: click.Context) -> None:
    """Run deterministic-spine workflows.

    Subcommands:

        run   Run a workflow to a verdict

    Example:

        holodeck workflow run workflow.yaml --input case.json
    """
    ctx.ensure_object(dict)
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@workflow.command(name="run")
@click.argument(
    "workflow_config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="JSON payload binding the workflow's input_data facts of record.",
)
def run(workflow_config: Path, input_path: Path | None) -> None:
    """Run WORKFLOW_CONFIG and print every node's result.

    The whole graph — decision tables, gate schemas, and the input payload — is
    validated before any agent is invoked, so an invalid workflow costs no
    model calls.
    """
    with handle_workflow_errors():
        payload = _load_payload(input_path)
        result = asyncio.run(run_workflow(workflow_config, payload))
        _echo_result(result)
