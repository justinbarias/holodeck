"""Shared instruction resolver for loading agent instructions.

Extracts the instruction resolution logic so both SK and Claude backends
can resolve ``Instructions`` (inline text or file path) without duplication.
"""

from pathlib import Path

from holodeck.config.context import agent_base_dir
from holodeck.lib.errors import ConfigError
from holodeck.models.agent import Instructions


def resolve_instructions(
    instructions: Instructions, base_dir: Path | None = None
) -> str:
    """Resolve agent instructions from an ``Instructions`` config object.

    Args:
        instructions: Instructions config with either ``inline`` text or ``file`` path.
        base_dir: Explicit base directory for resolving relative file paths.
            Falls back to the ``agent_base_dir`` context variable, then CWD.

    Returns:
        The resolved instruction text.

    Raises:
        ConfigError: If the instructions file is missing or cannot be read.
    """
    if instructions.inline:
        return instructions.inline

    if instructions.file:
        resolved_base = base_dir or _resolve_base_dir()
        file_path = (
            resolved_base / instructions.file
            if resolved_base
            else Path(instructions.file)
        )

        if not file_path.exists():
            raise ConfigError(
                "instructions.file", f"Instructions file not found: {file_path}"
            )

        try:
            return file_path.read_text()
        except OSError as exc:
            raise ConfigError(
                "instructions.file", f"Failed to read instructions file: {exc}"
            ) from exc

    raise ConfigError("instructions", "No instructions provided (file or inline)")


def _resolve_base_dir() -> Path | None:
    """Resolve the base directory from the ``agent_base_dir`` context variable."""
    ctx_dir = agent_base_dir.get()
    return Path(ctx_dir) if ctx_dir else None
