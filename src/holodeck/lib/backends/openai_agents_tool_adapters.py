"""Tool adapters bridging HoloDeck tools to the OpenAI Agents SDK.

Translates HoloDeck ``FunctionTool`` configs into low-level
``agents.FunctionTool`` instances. The user callable is resolved via the shared
:func:`holodeck.lib.function_tool_loader.load_function_tool` (the same loader the
SK and Claude backends use) and wrapped with an ``on_invoke_tool`` coroutine
that parses the model-supplied JSON arguments, invokes the callable, and returns
its stringified result.

Supported types: function, vectorstore, and hierarchical_document tools (the
last two wrap the same initialized ``.search()`` instances the Claude adapter
uses, supplied via ``tool_instances``). ``type: prompt`` tools are skipped with
a warning — no backend has a runtime adapter for them. Any other tool type
raises :class:`ConfigError` naming the unsupported type, so misconfigured agents
fail fast rather than silently dropping tools.

All ``import agents`` happen inside functions to keep the optional SDK import
lazy (SC-005).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import typing
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from holodeck.lib.backends.base import BackendInitError
from holodeck.lib.errors import ConfigError
from holodeck.lib.function_tool_loader import load_function_tool
from holodeck.models.tool import (
    FunctionTool,
    HierarchicalDocumentToolConfig,
    PromptTool,
    ToolUnion,
    VectorstoreTool,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime SDK import
    from agents import Tool as SDKTool

logger = logging.getLogger(__name__)

# Sentinel returned to the model when a search yields no hits — mirrors the
# Claude adapter (``lib/backends/tool_adapters.py``).
_NO_RESULTS = "No results found."

# JSON Schema for the single-``query`` search tools, matching the Claude
# adapter's ``{"query": str}`` shape.
_SEARCH_PARAMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}

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


def _make_vectorstore_on_invoke(
    instance: Any,
    tool_name: str,
) -> Callable[[Any, str], Any]:
    """Build an ``on_invoke_tool`` coroutine for a vectorstore ``.search()``.

    The instance's ``search`` returns a single formatted string; an empty result
    is reported to the model as the ``_NO_RESULTS`` sentinel.
    """

    async def _on_invoke(_ctx: Any, input_json: str) -> str:
        query = _extract_query(input_json)
        result: str = await instance.search(query)
        return result if result else _NO_RESULTS

    del tool_name  # carried for symmetry / future failure-wrapping (E1)
    return _on_invoke


def _make_hierarchical_on_invoke(
    instance: Any,
    tool_name: str,
) -> Callable[[Any, str], Any]:
    """Build an ``on_invoke_tool`` coroutine for a hierarchical-doc ``.search()``.

    The instance's ``search`` returns a list of results; each is ``.format()``-ed
    and joined with ``\\n---\\n`` (matching the Claude adapter), or the
    ``_NO_RESULTS`` sentinel when empty.
    """

    async def _on_invoke(_ctx: Any, input_json: str) -> str:
        query = _extract_query(input_json)
        results = await instance.search(query)
        if not results:
            return _NO_RESULTS
        return "\n---\n".join(r.format() for r in results)

    del tool_name
    return _on_invoke


def _extract_query(input_json: str) -> str:
    """Parse the model-supplied JSON arguments and return the ``query`` string."""
    try:
        args = json.loads(input_json) if input_json else {}
    except json.JSONDecodeError:
        return ""
    if isinstance(args, dict):
        return str(args.get("query", ""))
    return ""


def build_sdk_tools(
    tool_configs: list[ToolUnion] | None,
    base_dir: Path | None,
    tool_instances: dict[str, Any] | None = None,
) -> list[SDKTool]:
    """Translate HoloDeck tool configs into SDK ``FunctionTool`` instances.

    Args:
        tool_configs: All tool configurations from the agent YAML (may be None).
        base_dir: Directory used to resolve relative ``FunctionTool.file`` paths
            (typically the agent project root).
        tool_instances: Initialized vectorstore / hierarchical-document tool
            instances keyed by config name (built by ``initialize_tools``).
            Required for those two tool types.

    Returns:
        A list of ``agents.FunctionTool`` objects ready to pass to ``Agent``.

    Raises:
        BackendInitError: If a vectorstore / hierarchical-document tool has no
            matching initialized instance.
        ConfigError: If a tool type is unsupported on this backend, or a function
            tool fails to load.
    """
    from agents import FunctionTool as SDKFunctionTool

    instances = tool_instances or {}
    tools: list[SDKTool] = []
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
        elif isinstance(cfg, VectorstoreTool):
            instance = _require_instance(instances, cfg.name, cfg.type)
            tools.append(
                SDKFunctionTool(
                    name=f"{cfg.name}_search",
                    description=f"Search {cfg.name}: {cfg.description}",
                    params_json_schema=_SEARCH_PARAMS_SCHEMA,
                    on_invoke_tool=_make_vectorstore_on_invoke(instance, cfg.name),
                    strict_json_schema=False,
                )
            )
        elif isinstance(cfg, HierarchicalDocumentToolConfig):
            instance = _require_instance(instances, cfg.name, cfg.type)
            tools.append(
                SDKFunctionTool(
                    name=f"{cfg.name}_search",
                    description=f"Search {cfg.name}: {cfg.description}",
                    params_json_schema=_SEARCH_PARAMS_SCHEMA,
                    on_invoke_tool=_make_hierarchical_on_invoke(instance, cfg.name),
                    strict_json_schema=False,
                )
            )
        elif isinstance(cfg, PromptTool):
            logger.warning(
                "Tool '%s' (type: prompt) has no runtime adapter on any backend; "
                "skipping it on the openai_agents backend.",
                cfg.name,
            )
            continue
        else:
            raise ConfigError(
                f"tools.{cfg.name}",
                f"'{cfg.type}' tools are not yet supported on the openai_agents "
                "backend.",
            )

    return tools


def _require_instance(
    instances: dict[str, Any],
    name: str,
    tool_type: str,
) -> Any:
    """Return the initialized instance for *name* or raise ``BackendInitError``."""
    instance = instances.get(name)
    if instance is None:
        raise BackendInitError(
            f"No initialized instance found for tool '{name}' (type: {tool_type}). "
            "Ensure tool initialization completed before building SDK tools."
        )
    return instance
