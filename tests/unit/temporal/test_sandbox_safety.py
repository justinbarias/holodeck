"""The D3 surface passes Temporal's workflow sandbox (spec 040, T5).

Decision 16 layer 1: a fast, serverless check that
``SandboxedWorkflowRunner.prepare_workflow`` accepts a workflow built on
:mod:`holodeck.temporal.deterministic`. The integration suite keeps the
public-API backstop (``Worker`` init against a dev server).

The workflows under test live in their own modules
(``sandbox_workflows.py``, ``sandbox_workflows_nondeterministic.py``) because
the sandbox re-imports the module a workflow is defined in — see those files.
A positive control proves the harness can fail: a workflow module that reads
the wall clock at import time must be rejected by the same call.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys

import pytest
from temporalio import workflow
from temporalio.worker.workflow_sandbox import (
    SandboxedWorkflowRunner,
    SandboxRestrictions,
)
from temporalio.worker.workflow_sandbox._restrictions import (
    RestrictedWorkflowAccessError,
)

from tests.unit.temporal import sandbox_workflows, sandbox_workflows_nondeterministic

pytestmark = pytest.mark.unit

# The three modules the harness passes through, each for a reason that is not
# about the D3 surface itself:
#
# * ``regex`` — a third-party transitive dependency (bkflow-feel → lark →
#   regex) that reads the process locale while being imported. Passing through
#   third-party libraries is Temporal's documented remedy, and the SDK's own
#   default restrictions already do it for dozens of them.
# * ``holodeck.models.chat`` — a chat model whose field default is
#   ``datetime.utcnow``; the sandbox refuses that attribute at class-definition
#   time. Importing any model submodule executes the package ``__init__``,
#   which builds this one. It is a pure data definition and no part of the D3
#   surface.
# * ``holodeck.config`` — the I/O config package, which forms an import cycle
#   with ``holodeck.models.agent`` that only resolves under the import order a
#   normal process happens to use. It has no business inside a workflow.
#
# Everything else stays under sandbox validation and is really re-imported:
# ``holodeck.temporal.deterministic`` and ``holodeck.temporal.models``, the
# 036 primitives ``edge`` / ``table_eval`` / ``feel``, and the
# ``decision_table`` and ``workflow`` models they rest on.
_PASSTHROUGH = ("regex", "holodeck.models.chat", "holodeck.config")

# The D3 logic modules the acceptance criterion names, plus the surface itself.
_D3_MODULES = (
    "holodeck.temporal.deterministic",
    "holodeck.temporal.models",
    "holodeck.lib.workflow.edge",
    "holodeck.lib.workflow.table_eval",
    "holodeck.lib.workflow.feel",
    "holodeck.models.decision_table",
)

# Nothing in the D3 import graph may pull the backend stack in. Run in a
# subprocess so an SDK already imported by the test session cannot mask it.
_PROBE = """
import sys
import holodeck.temporal.deterministic  # noqa: F401
leaked = [name for name in {forbidden!r} if name in sys.modules]
print("LEAKED:", ", ".join(leaked) if leaked else "none")
sys.exit(1 if leaked else 0)
"""

_FORBIDDEN = (
    "claude_agent_sdk",
    "holodeck.lib.backends",
    "holodeck.lib.backends.selector",
    "temporalio.client",
    "temporalio.worker",
)


def _prepare(cls: type) -> None:
    """Run sandbox validation for a workflow class.

    ``prepare_workflow`` builds a workflow instance to validate it, which
    needs a running event loop, so it is driven through ``asyncio.run``.

    Args:
        cls: The ``@workflow.defn`` class to validate.

    Raises:
        Exception: Whatever the sandbox rejects the workflow with.
    """
    definition = workflow._Definition.must_from_class(cls)
    restrictions = SandboxRestrictions.default.with_passthrough_modules(*_PASSTHROUGH)

    async def _run() -> None:
        SandboxedWorkflowRunner(restrictions=restrictions).prepare_workflow(definition)

    asyncio.run(_run())


class TestSandboxValidation:
    """``SandboxedWorkflowRunner.prepare_workflow`` on the D3 surface."""

    def test_workflow_importing_the_d3_surface_validates(self):
        """The helpers are re-imported inside the sandbox and pass."""
        # Act / Assert — no exception is the assertion.
        _prepare(sandbox_workflows.DeterministicWorkflow)

    def test_workflow_passing_the_surface_through_validates(self):
        """The shape a consumer writes validates too."""
        # Act / Assert
        _prepare(sandbox_workflows.PassedThroughWorkflow)

    def test_import_time_nondeterminism_is_rejected(self):
        """Positive control: the harness can fail (decision 16)."""
        # Act / Assert
        with pytest.raises(RestrictedWorkflowAccessError) as excinfo:
            _prepare(sandbox_workflows_nondeterministic.NondeterministicWorkflow)
        assert "utcnow" in str(excinfo.value)

    def test_d3_modules_are_imported_inside_the_sandbox(self):
        """The passing case is not passing because nothing was re-imported."""
        # Arrange
        restrictions = SandboxRestrictions.default.with_passthrough_modules(
            *_PASSTHROUGH
        )

        # Act
        passed_through = restrictions.passthrough_modules

        # Assert
        for module in _D3_MODULES:
            assert module not in passed_through
            assert not any(
                module == prefix or module.startswith(f"{prefix}.")
                for prefix in _PASSTHROUGH
            )


class TestImportPurity:
    """The D3 surface must not drag the backend or the Temporal client in."""

    def test_deterministic_module_imports_without_the_backend_stack(self):
        """A workflow author's import must not reach an SDK."""
        # Act
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", _PROBE.format(forbidden=_FORBIDDEN)],
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Assert
        assert result.returncode == 0, (
            "importing holodeck.temporal.deterministic failed or pulled in a "
            f"forbidden module (stdout: {result.stdout.strip()!r}, "
            f"stderr: {result.stderr.strip()!r})"
        )

    def test_surface_exports_the_documented_names(self):
        """The D3 surface is exactly what the plan names (decision 7)."""
        # Arrange
        from holodeck.temporal import deterministic

        # Assert
        assert set(deterministic.__all__) == {
            "ActivityParameters",
            "DecisionTable",
            "Verdict",
            "check_gate",
            "evaluate",
            "load_decision_table",
        }
