"""Dynamic loader for FunctionTool callables.

Imports the user-provided Python file declared by a :class:`FunctionTool`
configuration and returns the named attribute as a callable. Shared by the
Semantic Kernel and Claude Agent SDK backends so function tools dispatch
identically on both runtimes.

Any failure (missing file, import error, missing attribute, non-callable
attribute) is re-raised as :class:`ConfigError` so it surfaces at config
load time with a clear, actionable message (FR-025).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

from holodeck.lib.errors import ConfigError
from holodeck.models.tool import FunctionTool

_MODULE_NAME_PREFIX = "holodeck_function_tool_"


def load_function_tool(
    tool: FunctionTool,
    base_dir: Path | None,
) -> Callable[..., Any]:
    """Import *tool.file* relative to *base_dir* and return *tool.function*.

    Args:
        tool: The ``FunctionTool`` configuration from the agent YAML.
        base_dir: Directory used to resolve ``tool.file`` when it is a
            relative path. Typically the agent project root (``agent.yaml``'s
            parent). When ``None``, only absolute paths in ``tool.file``
            resolve successfully.

    Returns:
        The callable referenced by ``tool.function``.

    Raises:
        ConfigError: If the file does not exist, the import fails, the
            attribute is missing, or the attribute is not callable. The
            error's ``field`` is ``tools.<tool.name>``.
    """
    field = f"tools.{tool.name}"

    raw_path = Path(tool.file)
    if raw_path.is_absolute():
        resolved = raw_path
    elif base_dir is not None:
        resolved = (base_dir / raw_path).resolve()
    else:
        resolved = raw_path.resolve()

    if not resolved.is_file():
        raise ConfigError(
            field,
            f"Function tool file not found: {resolved} (from file='{tool.file}').",
        )

    module_name = f"{_MODULE_NAME_PREFIX}{tool.name}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(resolved))
        if spec is None or spec.loader is None:
            raise ConfigError(
                field,
                f"Could not build import spec for {resolved}.",
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(
            field,
            f"Failed to import {resolved} for function tool '{tool.name}': {exc}",
        ) from exc

    if not hasattr(module, tool.function):
        raise ConfigError(
            field,
            (
                f"Function '{tool.function}' not found in {resolved} "
                f"(for function tool '{tool.name}')."
            ),
        )

    attr = getattr(module, tool.function)
    if not callable(attr):
        raise ConfigError(
            field,
            (
                f"Attribute '{tool.function}' in {resolved} is not callable "
                f"(got {type(attr).__name__})."
            ),
        )

    return attr  # type: ignore[no-any-return]
