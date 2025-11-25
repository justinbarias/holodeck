"""Chat session progress tracking and display."""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Any

from holodeck.models.token_usage import TokenUsage
from holodeck.models.tool_execution import ToolExecution, ToolStatus


class ChatProgressIndicator:
    """Track and display chat session progress with spinner and status information.

    Provides animated spinner during agent execution and adaptive status display
    (minimal in default mode, rich in verbose mode). Tracks message count,
    tokens, session time, response timing, and tool executions.
    """

    # Braille spinner characters
    _SPINNER_CHARS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, max_messages: int, quiet: bool, verbose: bool) -> None:
        """Initialize progress indicator.

        Args:
            max_messages: Maximum messages for session before warning.
            quiet: Suppress status display (spinner still shows).
            verbose: Show rich status panel instead of inline status.
        """
        self.max_messages = max_messages
        self.quiet = quiet
        self.verbose = verbose

        # State tracking
        self.current_messages = 0
        self.total_tokens = TokenUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
        self.last_response_time: float | None = None
        self.session_start = datetime.now()
        self._spinner_index = 0

        # Tool execution tracking
        self.last_tool_executions: list[ToolExecution] = []
        self.total_tool_calls: int = 0

    @property
    def is_tty(self) -> bool:
        """Check if stdout is connected to a terminal.

        Returns:
            True if stdout is a TTY, False otherwise.
        """
        return sys.stdout.isatty()

    def get_spinner_line(self) -> str:
        """Get current spinner animation frame.

        Returns:
            Animated spinner text, or empty string if not TTY.
        """
        if not self.is_tty:
            return ""

        spinner_char = self._SPINNER_CHARS[
            self._spinner_index % len(self._SPINNER_CHARS)
        ]
        self._spinner_index += 1
        return f"{spinner_char} Thinking..."

    def update(self, response: Any) -> None:
        """Update progress after agent response.

        Args:
            response: AgentResponse object with execution_time, tokens_used,
                and tool_executions.
        """
        # Update message count
        self.current_messages += 1

        # Update execution time
        if hasattr(response, "execution_time"):
            self.last_response_time = response.execution_time

        # Accumulate token usage
        if hasattr(response, "tokens_used") and response.tokens_used:
            tokens = response.tokens_used
            self.total_tokens = TokenUsage(
                prompt_tokens=self.total_tokens.prompt_tokens + tokens.prompt_tokens,
                completion_tokens=self.total_tokens.completion_tokens
                + tokens.completion_tokens,
                total_tokens=self.total_tokens.total_tokens + tokens.total_tokens,
            )

        # Track tool executions
        if hasattr(response, "tool_executions") and response.tool_executions:
            self.last_tool_executions = response.tool_executions
            self.total_tool_calls += len(response.tool_executions)
        else:
            self.last_tool_executions = []

    def get_status_inline(self) -> str:
        """Get minimal inline status for default mode.

        Format: [messages_current/messages_max | execution_time]

        Returns:
            Inline status string.
        """
        status_parts = []

        # Message count
        status_parts.append(f"{self.current_messages}/{self.max_messages}")

        # Execution time
        if self.last_response_time is not None:
            time_str = f"{self.last_response_time:.1f}s"
            status_parts.append(time_str)

        return f"[{' | '.join(status_parts)}]" if status_parts else ""

    def get_status_panel(self) -> str:
        """Get rich status panel for verbose mode.

        Returns:
            Multi-line status panel string.
        """
        lines = []
        content_width = 39  # Width of content area (excluding "│ " and " │")

        # Top border
        lines.append("╭─── Chat Status ─────────────────────────╮")

        # Session time
        session_duration = (datetime.now() - self.session_start).total_seconds()
        hours = int(session_duration // 3600)
        minutes = int((session_duration % 3600) // 60)
        seconds = int(session_duration % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        content = f"Session Time: {time_str}"
        lines.append(f"│ {content:<{content_width}} │")

        # Message count with percentage
        percentage = int((self.current_messages / self.max_messages) * 100)
        msg_str = f"{self.current_messages} / {self.max_messages} ({percentage}%)"
        content = f"Messages: {msg_str}"
        lines.append(f"│ {content:<{content_width}} │")

        # Token usage
        total_str = f"{self.total_tokens.total_tokens:,}"
        content = f"Total Tokens: {total_str}"
        lines.append(f"│ {content:<{content_width}} │")

        # Token breakdown - prompt
        prompt_str = f"{self.total_tokens.prompt_tokens:,}"
        content = f"  ├─ Prompt: {prompt_str}"
        lines.append(f"│ {content:<{content_width}} │")

        # Token breakdown - completion
        completion_str = f"{self.total_tokens.completion_tokens:,}"
        content = f"  └─ Completion: {completion_str}"
        lines.append(f"│ {content:<{content_width}} │")

        # Last response time
        if self.last_response_time is not None:
            time_str = f"{self.last_response_time:.1f}s"
            content = f"Last Response: {time_str}"
            lines.append(f"│ {content:<{content_width}} │")

        # Tool execution summary
        content = f"Tool Calls (Total): {self.total_tool_calls}"
        lines.append(f"│ {content:<{content_width}} │")

        # Last tool executions (if any)
        if self.last_tool_executions:
            lines.append(f"│ {'Last Tools Called:':<{content_width}} │")
            for tool_exec in self.last_tool_executions:
                # Get status indicator
                status_icon = self._get_status_icon(tool_exec.status)
                # Truncate tool name if too long
                tool_name = tool_exec.tool_name
                max_name_len = content_width - 6  # Account for "  └─ " and icon
                if len(tool_name) > max_name_len:
                    tool_name = tool_name[: max_name_len - 3] + "..."
                content = f"  └─ {status_icon} {tool_name}"
                lines.append(f"│ {content:<{content_width}} │")

        # Bottom border
        lines.append("╰─────────────────────────────────────────╯")

        return "\n".join(lines)

    @staticmethod
    def _get_status_icon(status: ToolStatus) -> str:
        """Get status icon for tool execution status.

        Args:
            status: Tool execution status.

        Returns:
            Unicode icon representing the status.
        """
        status_icons = {
            ToolStatus.PENDING: "○",
            ToolStatus.RUNNING: "◐",
            ToolStatus.SUCCESS: "✓",
            ToolStatus.FAILED: "✗",
        }
        return status_icons.get(status, "?")
