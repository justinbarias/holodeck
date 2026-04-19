"""Prompt-version resolver for the 031-eval-runs-dashboard feature (US2).

Parses optional YAML frontmatter from an agent's ``instructions.file`` via
``python-frontmatter`` and produces a :class:`PromptVersion` record suitable
for embedding in :class:`EvalRunMetadata`. This module is purely additive —
it does NOT alter :func:`holodeck.lib.instruction_resolver.resolve_instructions`,
whose contract (return the raw instruction text) is kept byte-equivalent so
that the prompt body reaching the LLM is unchanged (FR-015).

See ``specs/031-eval-runs-dashboard/research.md`` R1 for the parser choice and
``specs/031-eval-runs-dashboard/data-model.md`` §PromptVersion for field rules.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import frontmatter
import yaml

from holodeck.config.context import agent_base_dir
from holodeck.lib.errors import ConfigError
from holodeck.models.agent import Instructions
from holodeck.models.eval_run import PromptVersion

_RECOGNISED_KEYS: frozenset[str] = frozenset(
    {"version", "author", "description", "tags"}
)


def resolve_prompt_version(
    instructions: Instructions,
    base_dir: Path | None = None,
) -> PromptVersion:
    """Derive a :class:`PromptVersion` from an :class:`Instructions` config.

    Behaviour:
        * When ``instructions.inline`` is set, frontmatter parsing is skipped.
          The full inline string is hashed; ``version`` becomes
          ``"auto-<first 8 hex of sha256>"``; ``source='inline'`` (FR-014).
        * When ``instructions.file`` is set, the file is parsed with
          ``python-frontmatter``. Recognised keys (``version``, ``author``,
          ``description``, ``tags``) populate the typed fields; all other
          frontmatter keys land in ``extra`` (FR-016). ``body_hash`` is the
          SHA-256 of the frontmatter-stripped body. If no ``version:`` key is
          present, ``version`` defaults to ``"auto-<body_hash[:8]>"`` (FR-013).
          Files with NO frontmatter are fine — ``python-frontmatter`` returns
          an empty metadata dict (FR-011 / AC3).

    Args:
        instructions: The agent's :class:`Instructions` config.
        base_dir: Directory for resolving a relative ``instructions.file``.
            Falls back to the ``agent_base_dir`` context variable, then CWD —
            matching the convention in
            :func:`holodeck.lib.instruction_resolver.resolve_instructions`.

    Returns:
        A fully-validated :class:`PromptVersion`.

    Raises:
        ConfigError: When ``instructions.file`` has malformed YAML
            frontmatter (FR-017), or when neither ``file`` nor ``inline`` is
            provided.
    """
    if instructions.inline is not None:
        return _resolve_inline(instructions.inline)

    if instructions.file is not None:
        resolved_path = _resolve_file_path(instructions.file, base_dir)
        return _resolve_file(resolved_path)

    raise ConfigError("instructions", "No instructions provided (file or inline)")


def _resolve_inline(inline: str) -> PromptVersion:
    """Short-circuit resolver for the ``instructions.inline`` branch (FR-014)."""
    body_hash = hashlib.sha256(inline.encode("utf-8")).hexdigest()
    return PromptVersion(
        version=f"auto-{body_hash[:8]}",
        source="inline",
        file_path=None,
        body_hash=body_hash,
        tags=[],
        extra={},
    )


def _resolve_file_path(file_ref: str, base_dir: Path | None) -> Path:
    """Resolve a (possibly relative) ``instructions.file`` against ``base_dir``.

    Mirrors the resolution order used by
    :func:`holodeck.lib.instruction_resolver.resolve_instructions` so both
    helpers agree on the same on-disk path.
    """
    resolved_base = base_dir or _resolve_base_dir_from_context()
    return resolved_base / file_ref if resolved_base else Path(file_ref)


def _resolve_base_dir_from_context() -> Path | None:
    ctx_dir = agent_base_dir.get()
    return Path(ctx_dir) if ctx_dir else None


def _resolve_file(path: Path) -> PromptVersion:
    """Parse a file's optional frontmatter and build a :class:`PromptVersion`."""
    if not path.exists():
        raise ConfigError("instructions.file", f"Instructions file not found: {path}")
    try:
        post = frontmatter.load(str(path))
    except yaml.YAMLError as exc:
        raise ConfigError(
            "instructions.file",
            f"Malformed YAML frontmatter in {path}: {exc}",
        ) from exc
    except OSError as exc:
        raise ConfigError(
            "instructions.file",
            f"Failed to read instructions file {path}: {exc}",
        ) from exc

    body: str = post.content
    metadata: dict[str, Any] = dict(post.metadata or {})
    recognised, extra = _partition_metadata(metadata)

    body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    version = recognised.get("version") or f"auto-{body_hash[:8]}"

    return PromptVersion(
        version=str(version),
        author=recognised.get("author"),
        description=recognised.get("description"),
        tags=list(recognised.get("tags") or []),
        source="file",
        file_path=str(path),
        body_hash=body_hash,
        extra=extra,
    )


def _partition_metadata(
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split parsed frontmatter into recognised vs. unknown keys.

    Returns:
        (recognised_dict, extra_dict) — keys in _RECOGNISED_KEYS land in the
        first dict; everything else lands in ``extra`` so unknown keys remain
        addressable without polluting the typed fields (FR-016).
    """
    recognised: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in _RECOGNISED_KEYS:
            recognised[key] = value
        else:
            extra[key] = value
    return recognised, extra


__all__ = ["PromptVersion", "resolve_prompt_version"]
