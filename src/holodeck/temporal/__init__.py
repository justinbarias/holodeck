"""Temporal integration for HoloDeck agents (spec 040).

Turns an agent definition into a Temporal activity so that a user-authored
Temporal workflow can call it durably. HoloDeck supplies the agents; it does
not ship a workflow engine, a workflow YAML format, or a DAG runner.

The ``temporalio`` SDK is an optional dependency, shipped in the ``temporal``
extra. Importing this package without it raises a :class:`ConfigError` naming
the install command, rather than a bare ``ModuleNotFoundError`` from somewhere
deeper in the import graph.
"""

from __future__ import annotations

from importlib.util import find_spec

from holodeck.lib.errors import ConfigError

_EXTRA_HINT = (
    "The Temporal integration requires the 'temporal' extra. Install it with:\n"
    "  uv add 'holodeck-ai[temporal]'   # or: pip install 'holodeck-ai[temporal]'"
)


def require_temporalio() -> None:
    """Verify that the ``temporalio`` SDK is importable.

    Uses :func:`importlib.util.find_spec` rather than a trial import so the
    check stays cheap and does not pull the SDK into modules that only need to
    know whether it is present. A finder that refuses the name counts as
    absent: ``find_spec`` raises rather than returning ``None`` when an import
    hook rejects it, and both outcomes mean the same thing to the caller.

    Raises:
        ConfigError: If ``temporalio`` is not installed.
    """
    try:
        installed = find_spec("temporalio") is not None
    except (ImportError, ValueError):
        installed = False
    if not installed:
        raise ConfigError("dependencies", _EXTRA_HINT)


require_temporalio()

__all__ = ["require_temporalio"]
