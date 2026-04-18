"""EvalRun persistence package.

Public API:
- :func:`write_eval_run` — atomic JSON writer for :class:`EvalRun` artifacts
- :func:`redact` — two-rule redactor for the agent-config snapshot
- :func:`slugify` — filesystem-safe agent-name slug for ``results/<slug>/``
- :func:`build_eval_run_metadata` — assembles provenance from agent + env
- :data:`REDACTED_FIELD_NAMES` — centralised allowlist constant (FR-005)
"""

from holodeck.lib.eval_run.metadata import build_eval_run_metadata
from holodeck.lib.eval_run.redactor import (
    REDACTED_FIELD_NAMES,
    REDACTED_PLACEHOLDER,
    redact,
)
from holodeck.lib.eval_run.slugify import slugify
from holodeck.lib.eval_run.writer import write_eval_run

__all__ = [
    "REDACTED_FIELD_NAMES",
    "REDACTED_PLACEHOLDER",
    "build_eval_run_metadata",
    "redact",
    "slugify",
    "write_eval_run",
]
