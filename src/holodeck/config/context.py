"""Context variables for request-scoped configuration.

This module provides context variables that flow through async call stacks
without explicit parameter passing. Similar to .NET's IHttpContextAccessor
or ASP.NET Core's scoped services.

Usage:
    # At CLI entry point (test.py, chat.py):
    from holodeck.config.context import agent_base_dir
    agent_base_dir.set(str(Path(agent_yaml_path).parent))

    # Anywhere downstream (e.g., VectorStoreTool):
    from holodeck.config.context import agent_base_dir
    base_dir = agent_base_dir.get()  # Returns str | None
"""

from contextvars import ContextVar

# Base directory for the current agent configuration file.
# Set at CLI entry point, available throughout the request lifecycle.
# Used to resolve relative file paths in tool configurations.
agent_base_dir: ContextVar[str | None] = ContextVar("agent_base_dir", default=None)
