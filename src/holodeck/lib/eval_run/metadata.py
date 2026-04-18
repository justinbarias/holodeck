"""Builder for :class:`EvalRunMetadata` — captures provenance for one run.

Encapsulates the side-effecting bits (subprocess, importlib.metadata, sys.argv)
in a single function so they are trivial to mock in unit tests.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as importlib_version

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

    Args:
        agent: The (already redacted) agent snapshot to embed.
        prompt_version: The prompt-identity record (US2 derives; US1 stub).
        argv: Override for ``sys.argv[1:]``; ``None`` consults ``sys.argv``.

    Returns:
        A fully-populated :class:`EvalRunMetadata` instance.
    """
    cli_args = list(argv) if argv is not None else list(sys.argv[1:])
    return EvalRunMetadata(
        agent_config=agent,
        prompt_version=prompt_version,
        holodeck_version=_resolve_holodeck_version(),
        cli_args=cli_args,
        git_commit=_resolve_git_commit(),
    )
