"""Tool adapters bridging HoloDeck tools to the Claude Agent SDK.

Wraps VectorStoreTool and HierarchicalDocumentTool search methods as
``@tool``-decorated functions, bundles them into an in-process MCP server
via ``create_sdk_mcp_server()``, and provides a factory for ClaudeBackend
to call during initialization.
"""

from __future__ import annotations

import asyncio
import inspect
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any

from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server, tool
from claude_agent_sdk.types import McpSdkServerConfig

from holodeck.lib.backends.base import BackendInitError, BackendSessionError
from holodeck.lib.function_tool_loader import load_function_tool
from holodeck.models.tool import (
    FunctionTool,
    HierarchicalDocumentToolConfig,
    ToolUnion,
)
from holodeck.models.tool import (
    VectorstoreTool as VectorstoreToolConfig,
)
from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool
from holodeck.tools.vectorstore_tool import VectorStoreTool

_NO_RESULTS = "No results found."


def _truncate_description(desc: str, max_len: int = 200) -> str:
    """Truncate tool description to *max_len* chars for the SDK manifest."""
    if len(desc) <= max_len:
        return desc
    return desc[: max_len - 3] + "..."


# ---------------------------------------------------------------------------
# Factory functions (module-level to avoid closure-by-reference bug)
# ---------------------------------------------------------------------------


def _make_vectorstore_search_fn(
    instance: VectorStoreTool,
    name: str,
    description: str,
) -> SdkMcpTool[Any]:
    """Create an ``@tool``-decorated async handler for a VectorStoreTool."""

    @tool(name, description, {"query": str})
    async def _search(args: dict[str, Any]) -> dict[str, Any]:
        if not instance.is_initialized:
            raise BackendSessionError(
                f"Tool '{name.removesuffix('_search')}' is not initialized"
            )
        result: str = await instance.search(args["query"])
        text = result if result else _NO_RESULTS
        return {"content": [{"type": "text", "text": text}]}

    return _search


def _derive_input_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Build the Claude SDK ``@tool`` input_schema dict from *func*'s signature.

    Returns a mapping of parameter name -> Python type (matches the existing
    ``{"query": str}`` pattern used for vectorstore tools). Parameters without
    annotations fall back to :class:`typing.Any`. The ``self`` / ``cls``
    implicit first argument on bound methods is skipped.
    """
    schema: dict[str, Any] = {}
    sig = inspect.signature(func)
    try:
        hints = typing.get_type_hints(func)
    except Exception:
        hints = {}
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if name in hints:
            schema[name] = hints[name]
        elif param.annotation is not inspect.Parameter.empty:
            schema[name] = param.annotation
        else:
            schema[name] = Any
    return schema


def _make_function_tool_fn(
    func: Callable[..., Any],
    name: str,
    description: str,
    schema: dict[str, Any],
) -> SdkMcpTool[Any]:
    """Create an ``@tool``-decorated async handler that invokes *func*."""
    is_coro = asyncio.iscoroutinefunction(func)

    @tool(name, description, schema)
    async def _handler(args: dict[str, Any]) -> dict[str, Any]:
        try:
            if is_coro:
                result = await func(**args)
            else:
                result = func(**args)
        except Exception as exc:
            raise BackendSessionError(
                f"Function tool '{name}' raised {type(exc).__name__}: {exc}"
            ) from exc
        return {"content": [{"type": "text", "text": str(result)}]}

    return _handler


def _make_hierarchical_search_fn(
    instance: HierarchicalDocumentTool,
    name: str,
    description: str,
) -> SdkMcpTool[Any]:
    """Create an ``@tool``-decorated async handler for a HierarchicalDocumentTool."""

    @tool(name, description, {"query": str})
    async def _search(args: dict[str, Any]) -> dict[str, Any]:
        if not instance._initialized:
            raise BackendSessionError(
                f"Tool '{name.removesuffix('_search')}' is not initialized"
            )
        results = await instance.search(args["query"])
        if not results:
            text = _NO_RESULTS
        else:
            text = "\n---\n".join(r.format() for r in results)
        return {"content": [{"type": "text", "text": text}]}

    return _search


# ---------------------------------------------------------------------------
# Adapter classes
# ---------------------------------------------------------------------------


class VectorStoreToolAdapter:
    """Wraps a ``VectorStoreTool`` for use with the Claude Agent SDK.

    Args:
        config: The vectorstore tool configuration from the agent YAML.
        instance: An initialized ``VectorStoreTool`` instance.
    """

    def __init__(
        self,
        config: VectorstoreToolConfig,
        instance: VectorStoreTool,
    ) -> None:
        self.config = config
        self.instance = instance

    def to_sdk_tool(self) -> SdkMcpTool[Any]:
        """Return an ``SdkMcpTool`` backed by this adapter's search method."""
        name = f"{self.config.name}_search"
        desc = _truncate_description(
            f"Search {self.config.name}: {self.config.description}"
        )
        return _make_vectorstore_search_fn(self.instance, name, desc)


class FunctionToolAdapter:
    """Wraps a user-provided Python callable for use with the Claude Agent SDK.

    Args:
        config: The function tool configuration from the agent YAML.
        callable: The Python callable resolved by
            :func:`holodeck.lib.function_tool_loader.load_function_tool`.
    """

    def __init__(
        self,
        config: FunctionTool,
        callable: Callable[..., Any],
    ) -> None:
        self.config = config
        self.callable = callable

    def to_sdk_tool(self) -> SdkMcpTool[Any]:
        """Return an ``SdkMcpTool`` backed by this adapter's callable."""
        desc = _truncate_description(self.config.description)
        schema = _derive_input_schema(self.callable)
        return _make_function_tool_fn(
            self.callable,
            self.config.name,
            desc,
            schema,
        )


class HierarchicalDocToolAdapter:
    """Wraps a ``HierarchicalDocumentTool`` for use with the Claude Agent SDK.

    Args:
        config: The hierarchical document tool configuration from the agent YAML.
        instance: An initialized ``HierarchicalDocumentTool`` instance.
    """

    def __init__(
        self,
        config: HierarchicalDocumentToolConfig,
        instance: HierarchicalDocumentTool,
    ) -> None:
        self.config = config
        self.instance = instance

    def to_sdk_tool(self) -> SdkMcpTool[Any]:
        """Return an ``SdkMcpTool`` backed by this adapter's search method."""
        name = f"{self.config.name}_search"
        desc = _truncate_description(
            f"Search {self.config.name}: {self.config.description}"
        )
        return _make_hierarchical_search_fn(self.instance, name, desc)


# ---------------------------------------------------------------------------
# Factory & server builder
# ---------------------------------------------------------------------------

_SERVER_NAME = "holodeck_tools"


def create_tool_adapters(
    tool_configs: list[ToolUnion],
    tool_instances: dict[str, VectorStoreTool | HierarchicalDocumentTool],
    base_dir: Path | None = None,
) -> list[VectorStoreToolAdapter | HierarchicalDocToolAdapter | FunctionToolAdapter]:
    """Build adapters for vectorstore, hierarchical-document, and function tools.

    Filters *tool_configs* for supported types, matches each to its
    initialized instance (vectorstore / hierarchical) or loads the Python
    callable (function tools) and returns adapter objects.

    Args:
        tool_configs: All tool configurations from the agent YAML.
        tool_instances: Initialized tool instances keyed by config name.
        base_dir: Directory used to resolve relative ``FunctionTool.file``
            paths. Typically the agent project root. Unused by vectorstore /
            hierarchical-document adapters.

    Returns:
        List of adapter objects ready for ``build_holodeck_sdk_server()``.

    Raises:
        BackendInitError: If a supported tool config has no matching instance.
        ConfigError: If a function tool fails to load.
    """
    adapters: list[
        VectorStoreToolAdapter | HierarchicalDocToolAdapter | FunctionToolAdapter
    ] = []

    for cfg in tool_configs:
        if isinstance(cfg, VectorstoreToolConfig):
            instance = tool_instances.get(cfg.name)
            if instance is None:
                raise BackendInitError(
                    f"No initialized instance found for tool '{cfg.name}' "
                    f"(type: {cfg.type}). Ensure tool initialization "
                    "completed before creating adapters."
                )
            adapters.append(
                VectorStoreToolAdapter(
                    config=cfg,
                    instance=instance,  # type: ignore[arg-type]
                )
            )
        elif isinstance(cfg, HierarchicalDocumentToolConfig):
            instance = tool_instances.get(cfg.name)
            if instance is None:
                raise BackendInitError(
                    f"No initialized instance found for tool '{cfg.name}' "
                    f"(type: {cfg.type}). Ensure tool initialization "
                    "completed before creating adapters."
                )
            adapters.append(
                HierarchicalDocToolAdapter(
                    config=cfg,
                    instance=instance,  # type: ignore[arg-type]
                )
            )
        elif isinstance(cfg, FunctionTool):
            func = load_function_tool(cfg, base_dir=base_dir)
            adapters.append(FunctionToolAdapter(config=cfg, callable=func))

    return adapters


def build_holodeck_sdk_server(
    adapters: list[
        VectorStoreToolAdapter | HierarchicalDocToolAdapter | FunctionToolAdapter
    ],
) -> tuple[McpSdkServerConfig, list[str]]:
    """Bundle adapters into an in-process MCP server for the Claude subprocess.

    Args:
        adapters: Adapter objects produced by ``create_tool_adapters()``.

    Returns:
        A tuple of ``(server_config, allowed_tool_names)`` where
        *server_config* is a ``McpSdkServerConfig`` TypedDict and
        *allowed_tool_names* are the fully-qualified MCP tool names.
        Search-backed adapters contribute ``<name>_search``; function-tool
        adapters contribute the raw tool name.
    """
    sdk_tools: list[SdkMcpTool[Any]] = [a.to_sdk_tool() for a in adapters]

    server_config: McpSdkServerConfig = create_sdk_mcp_server(
        name=_SERVER_NAME,
        tools=sdk_tools,
    )

    allowed_tools: list[str] = []
    for a in adapters:
        if isinstance(a, FunctionToolAdapter):
            allowed_tools.append(f"mcp__{_SERVER_NAME}__{a.config.name}")
        else:
            allowed_tools.append(f"mcp__{_SERVER_NAME}__{a.config.name}_search")

    return server_config, allowed_tools
