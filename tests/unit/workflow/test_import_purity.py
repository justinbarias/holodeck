"""The workflow-safe modules must not drag the backend stack in.

SPEC.md section 7: helpers destined for Temporal workflow code must not import
I/O modules. ``edge.py`` therefore imports ``BackendSelector`` lazily inside
``execute_edge_node`` — a module-scope import would pull the Claude Agent SDK
into any importer of the pure gate half (``load_gate_schema``/``_apply_gate``).
These tests run in a subprocess so a previously imported SDK in the test
runner cannot mask a regression.
"""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit

_FORBIDDEN = (
    "claude_agent_sdk",
    # The package __init__ is what eagerly imports the concrete backends, so
    # pin it as well as the selector module.
    "holodeck.lib.backends",
    "holodeck.lib.backends.selector",
)

_PROBE = """
import sys
import holodeck.lib.workflow.edge
import holodeck.lib.workflow.table_eval
import holodeck.lib.workflow.feel
import holodeck.models.workflow
import holodeck.models.decision_table
leaked = [name for name in {forbidden!r} if name in sys.modules]
sys.exit(1 if leaked else 0)
"""


def test_gate_and_table_modules_import_without_the_backend_stack() -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _PROBE.format(forbidden=_FORBIDDEN)],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        "importing the workflow-safe modules pulled in the backend stack "
        f"(stderr: {result.stderr.strip()})"
    )
