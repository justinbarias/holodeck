"""Provider backend abstraction layer for HoloDeck."""

from holodeck.lib.backends.base import (
    AgentBackend,
    AgentSession,
    BackendError,
    BackendInitError,
    BackendSessionError,
    BackendTimeoutError,
    ContextGenerator,
    ExecutionResult,
)
from holodeck.lib.backends.claude_backend import ClaudeBackend, ClaudeSession
from holodeck.lib.backends.mcp_bridge import build_claude_mcp_configs
from holodeck.lib.backends.otel_bridge import translate_observability
from holodeck.lib.backends.selector import BackendSelector
from holodeck.lib.backends.sk_backend import SKBackend, SKSession
from holodeck.lib.backends.tool_adapters import (
    HierarchicalDocToolAdapter,
    VectorStoreToolAdapter,
    build_holodeck_sdk_server,
    create_tool_adapters,
)

__all__ = [
    "ExecutionResult",
    "AgentSession",
    "AgentBackend",
    "BackendError",
    "BackendInitError",
    "BackendSessionError",
    "BackendTimeoutError",
    "ContextGenerator",
    "BackendSelector",
    "ClaudeBackend",
    "ClaudeSession",
    "SKBackend",
    "SKSession",
    "HierarchicalDocToolAdapter",
    "VectorStoreToolAdapter",
    "build_claude_mcp_configs",
    "build_holodeck_sdk_server",
    "create_tool_adapters",
    "translate_observability",
]
