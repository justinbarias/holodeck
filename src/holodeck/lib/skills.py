"""SKILL.md loading for ``type: skill`` tools (spec 023 FR-023, spec 035 FR-070).

A file-based skill points at a directory containing ``SKILL.md`` in the
`Agent Skills <https://agentskills.io/specification>`_ layout: a YAML
frontmatter block (``---`` delimited) carrying at least ``name`` and
``description``, followed by a Markdown body that becomes the skill agent's
instructions. This module parses that file into a :class:`SkillDefinition`
without importing any SDK, so both config-time validation
(``models/tool.py``) and the backend adapters can share one reader.

Only ``name``, ``description``, and the body are consumed. The optional
``allowed-tools`` frontmatter key belongs to native skill runtimes and is
deliberately **not** merged: HoloDeck tool scoping comes exclusively from the
YAML ``allowed_tools`` field (data-model rule for spec 023 §8).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

SKILL_FILE_NAME = "SKILL.md"
_FRONTMATTER_DELIMITER = "---"


class SkillLoadError(ValueError):
    """Raised when a skill directory or its ``SKILL.md`` is invalid."""


@dataclass(frozen=True)
class SkillDefinition:
    """The parsed content of one ``SKILL.md``.

    Attributes:
        name: The ``name`` frontmatter value.
        description: The ``description`` frontmatter value.
        instructions: The Markdown body following the frontmatter, stripped of
            surrounding whitespace.
        path: The skill directory the definition was read from.
    """

    name: str
    description: str
    instructions: str
    path: Path


def resolve_skill_dir(path: str, base_dir: Path | None) -> Path:
    """Resolve a skill ``path`` against *base_dir* (or the cwd when ``None``).

    Args:
        path: The YAML ``path`` value; absolute paths are used as-is.
        base_dir: The agent.yaml directory, when known.

    Returns:
        The resolved directory path (not verified to exist).
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root = Path.cwd() if base_dir is None else base_dir
    return root / candidate


def load_skill_definition(skill_dir: Path) -> SkillDefinition:
    """Parse ``SKILL.md`` under *skill_dir*.

    Args:
        skill_dir: A directory expected to contain ``SKILL.md``.

    Returns:
        The parsed :class:`SkillDefinition`.

    Raises:
        SkillLoadError: If the directory or file is missing, the frontmatter
            is absent or malformed, or ``name`` / ``description`` are missing
            or empty. The message names the offending path and, for
            frontmatter problems, the missing keys.
    """
    if not skill_dir.is_dir():
        raise SkillLoadError(f"skill path is not a directory: {skill_dir}")
    skill_file = skill_dir / SKILL_FILE_NAME
    if not skill_file.is_file():
        raise SkillLoadError(
            f"skill directory has no {SKILL_FILE_NAME}: {skill_dir} "
            "(see https://agentskills.io/specification)"
        )
    text = skill_file.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text, skill_file)

    values: dict[str, str] = {}
    missing: list[str] = []
    for key in ("name", "description"):
        value = frontmatter.get(key)
        if isinstance(value, str) and value.strip():
            values[key] = value.strip()
        else:
            missing.append(key)
    if missing:
        raise SkillLoadError(
            f"{skill_file}: SKILL.md frontmatter is missing required "
            f"field(s): {', '.join(missing)}"
        )
    instructions = body.strip()
    if not instructions:
        raise SkillLoadError(f"{skill_file}: SKILL.md body (instructions) is empty")
    return SkillDefinition(
        name=values["name"],
        description=values["description"],
        instructions=instructions,
        path=skill_dir,
    )


def _split_frontmatter(text: str, source: Path) -> tuple[dict[str, object], str]:
    """Split *text* into its YAML frontmatter mapping and Markdown body.

    Args:
        text: The full ``SKILL.md`` content.
        source: The file path, used only for error messages.

    Returns:
        ``(frontmatter, body)``.

    Raises:
        SkillLoadError: If the frontmatter block is missing, unterminated, not
            valid YAML, or not a mapping.
    """
    # A delimiter is a line that is exactly ``---`` (trailing whitespace
    # tolerated). Indented ``---`` inside a block-scalar value is content.
    lines = text.splitlines()
    if not lines or lines[0].rstrip() != _FRONTMATTER_DELIMITER:
        raise SkillLoadError(
            f"{source}: SKILL.md must start with a '---' YAML frontmatter block"
        )
    for index in range(1, len(lines)):
        if lines[index].rstrip() == _FRONTMATTER_DELIMITER:
            raw = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1 :])
            break
    else:
        raise SkillLoadError(f"{source}: SKILL.md frontmatter is not terminated")

    try:
        loaded = yaml.safe_load(raw) if raw.strip() else {}
    except yaml.YAMLError as exc:
        raise SkillLoadError(f"{source}: invalid SKILL.md frontmatter: {exc}") from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise SkillLoadError(f"{source}: SKILL.md frontmatter must be a mapping")
    return loaded, body
