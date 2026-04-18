"""Builder for :class:`EvalRunMetadata` — captures provenance for one run.

Snapshot invariants
-------------------

The ``agent_config`` field on :class:`EvalRunMetadata` is a **frozen deep-copy
snapshot** of the validated :class:`Agent` that produced the run. The builder
enforces a strict ordering so that later edits to ``agent.yaml`` cannot leak
into previously-written :class:`EvalRun` files and so that redaction never
mutates the live :class:`Agent` instance held by the running test executor:

1. **Deep copy.** ``agent.model_copy(deep=True)`` — every nested model
   (``LLMProvider``, ``ToolUnion`` members, ``ClaudeConfig`` sub-blocks,
   ``EvaluationConfig`` metrics, ``TestCaseModel`` files, ...) is rebuilt as a
   fresh Python object. Callers may safely continue using the original
   ``Agent`` for agent execution after this function returns.
2. **Redact the copy.** The two-rule redactor in
   :mod:`holodeck.lib.eval_run.redactor` (name allowlist + ``SecretStr``
   type-driven) walks the copy and replaces secret leaves with ``"***"``.
3. **Freeze into metadata.** The redacted copy is attached to the returned
   :class:`EvalRunMetadata`. Once this object is serialised by
   :func:`holodeck.lib.eval_run.writer.write_eval_run`, the on-disk JSON file
   is the authoritative record — no mutation of the in-memory graph can
   retroactively change it.

This module also captures run provenance (holodeck version, ``sys.argv``, and
best-effort git commit) inside side-effecting helpers so that unit tests can
mock them trivially.

See ``specs/031-eval-runs-dashboard/data-model.md`` §"Snapshot semantics" and
``spec.md`` User Story 3 AC1/AC5/AC6 for the authoritative contract.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as importlib_version

from holodeck.lib.eval_run.redactor import redact
from holodeck.models.agent import Agent
from holodeck.models.eval_run import EvalRunMetadata, PromptVersion

logger = logging.getLogger(__name__)

_HOLODECK_VERSION_FALLBACK = "0.0.0.dev0"
_GIT_TIMEOUT_SECONDS = 2


def _resolve_git_commit() -> str | None:
    """Best-effort ``git rev-parse HEAD``; return ``None`` on any failure."""
    try:
        completed = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.debug("git rev-parse failed: %s", exc)
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None


def _resolve_holodeck_version() -> str:
    try:
        return importlib_version("holodeck-ai")
    except PackageNotFoundError:
        return _HOLODECK_VERSION_FALLBACK


def build_eval_run_metadata(
    agent: Agent,
    prompt_version: PromptVersion,
    argv: list[str] | None = None,
) -> EvalRunMetadata:
    """Assemble an :class:`EvalRunMetadata` for the current invocation.

    The returned metadata embeds a **redacted deep copy** of the supplied
    :class:`Agent`. The caller's own :class:`Agent` instance is never
    mutated, so the running test executor can continue using the original
    secret-bearing configuration after this function returns. See the module
    docstring for the full snapshot invariants.

    Args:
        agent: The live, validated :class:`Agent` that produced this run.
            Must be the post-validation model, not a raw YAML dict. This
            function will deep-copy and redact it internally.
        prompt_version: The prompt-identity record resolved from either the
            frontmatter on ``instructions.file`` or the inline body hash.
        argv: Override for ``sys.argv[1:]``; when ``None``, the live
            ``sys.argv`` is consulted.

    Returns:
        A fully-populated :class:`EvalRunMetadata` instance whose
        ``agent_config`` is a frozen, redacted snapshot of the input agent.
    """
    cli_args = list(argv) if argv is not None else list(sys.argv[1:])
    # Step 1 — deep copy first so redaction cannot mutate the live Agent
    # that is still in use by the running test executor.
    snapshot = agent.model_copy(deep=True)
    # Step 2 — redact on the copy. ``redact`` returns a new deep copy with
    # secret leaves replaced; the intermediate ``snapshot`` becomes garbage.
    redacted_snapshot = redact(snapshot)
    # Step 3 — freeze into metadata.
    return EvalRunMetadata(
        agent_config=redacted_snapshot,
        prompt_version=prompt_version,
        holodeck_version=_resolve_holodeck_version(),
        cli_args=cli_args,
        git_commit=_resolve_git_commit(),
    )
