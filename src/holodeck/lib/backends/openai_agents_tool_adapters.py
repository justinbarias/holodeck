"""Tool adapters bridging HoloDeck tools to the OpenAI Agents SDK.

Translates HoloDeck ``FunctionTool`` configs into low-level
``agents.FunctionTool`` instances. The user callable is resolved via the shared
:func:`holodeck.lib.function_tool_loader.load_function_tool` (the same loader the
SK and Claude backends use) and wrapped with an ``on_invoke_tool`` coroutine
that parses the model-supplied JSON arguments, invokes the callable, and returns
its stringified result.

MVP scope is **function tools only**. Any other tool type on an
``openai_agents`` agent raises :class:`ConfigError` naming the unsupported type,
so misconfigured agents fail fast rather than silently dropping tools.

All ``import agents`` happen inside functions to keep the optional SDK import
lazy (SC-005).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any

from holodeck.lib.errors import ConfigError
from holodeck.lib.function_tool_loader import load_function_tool
from holodeck.models.tool import FunctionTool, ToolUnion

# Python annotation -> JSON Schema type. Anything unrecognised falls back to a
# permissive ``string`` (the SDK is non-strict here, so the model still sees a
# usable parameter).
_JSON_TYPE_BY_PY: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type_for(annotation: Any) -> str:
    """Best-effort map a Python annotation to a JSON Schema ``type`` string."""
    if isinstance(annotation, type) and annotation in _JSON_TYPE_BY_PY:
        return _JSON_TYPE_BY_PY[annotation]
    # Handle typing constructs (e.g. ``list[str]``) via their origin.
    origin = typing.get_origin(annotation)
    if origin is not None and isinstance(origin, type) and origin in _JSON_TYPE_BY_PY:
        return _JSON_TYPE_BY_PY[origin]
    return "string"


def _derive_params_schema(
    func: Callable[..., Any],
    declared: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """Build a JSON Schema object for the SDK ``params_json_schema``.

    When the YAML config declares ``parameters`` it is used verbatim as the
    ``properties`` map; otherwise the schema is derived from *func*'s signature.
    Parameters without a default are marked required.

    Args:
        func: The resolved tool callable.
        declared: The ``FunctionTool.parameters`` mapping from YAML, or ``None``.

    Returns:
        A JSON Schema ``object`` with ``properties`` and ``required`` keys.
    """
    if declared:
        return {
            "type": "object",
            "properties": dict(declared),
            "required": list(declared.keys()),
            "additionalProperties": False,
        }

    properties: dict[str, Any] = {}
    required: list[str] = []
    sig = inspect.signature(func)
    try:
        hints = typing.get_type_hints(func)
    except (NameError, TypeError):
        # Annotations referencing names not in scope (forward refs) or otherwise
        # unresolvable — fall back to the raw signature annotations below.
        hints = {}
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = hints.get(name, param.annotation)
        properties[name] = {"type": _json_type_for(annotation)}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _make_on_invoke(
    func: Callable[..., Any],
    tool_name: str,
) -> Callable[[Any, str], Any]:
    """Build the SDK ``on_invoke_tool`` coroutine that dispatches to *func*.

    The SDK passes a ``ToolContext`` and the raw JSON argument string. We parse
    the arguments, call the (sync or async) callable, and return its result as a
    string for the model.
    """
    is_coro = asyncio.iscoroutinefunction(func)

    async def _on_invoke(_ctx: Any, input_json: str) -> str:
        try:
            args = json.loads(input_json) if input_json else {}
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"tools.{tool_name}",
                f"Tool '{tool_name}' received malformed JSON arguments: {exc}",
            ) from exc
        if not isinstance(args, dict):
            args = {}
        if is_coro:
            result = await func(**args)
        else:
            result = func(**args)
        return str(result)

    return _on_invoke


def build_sdk_tools(
    tool_configs: list[ToolUnion] | None,
    base_dir: Path | None,
) -> list[Any]:
    """Translate HoloDeck tool configs into SDK ``FunctionTool`` instances.

    Args:
        tool_configs: All tool configurations from the agent YAML (may be None).
        base_dir: Directory used to resolve relative ``FunctionTool.file`` paths
            (typically the agent project root).

    Returns:
        A list of ``agents.FunctionTool`` objects ready to pass to ``Agent``.

    Raises:
        ConfigError: If a non-function tool type is configured (unsupported on
            this backend for the MVP), or a function tool fails to load.
    """
    from agents import FunctionTool as SDKFunctionTool

    tools: list[Any] = []
    for cfg in tool_configs or []:
        if isinstance(cfg, FunctionTool):
            func = load_function_tool(cfg, base_dir=base_dir)
            schema = _derive_params_schema(func, cfg.parameters)
            tools.append(
                SDKFunctionTool(
                    name=cfg.name,
                    description=cfg.description,
                    params_json_schema=schema,
                    on_invoke_tool=_make_on_invoke(func, cfg.name),
                    strict_json_schema=False,
                )
            )
        else:
            raise ConfigError(
                f"tools.{cfg.name}",
                f"'{cfg.type}' tools are not yet supported on the openai_agents "
                "backend (MVP supports function tools only).",
            )

    return tools
