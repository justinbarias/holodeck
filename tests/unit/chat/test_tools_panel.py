"""Unit tests for the chat ToolsPanel state + renderer.

Drives :class:`holodeck.chat.tools_panel.ToolsPanel` with crafted
:class:`holodeck.lib.backends.base.ToolEvent` sequences and asserts both
state transitions (snapshot, line_count) and rendered content.
"""

from __future__ import annotations

from unittest.mock import patch

from holodeck.chat.tools_panel import ToolsPanel
from holodeck.lib.backends.base import ToolEvent


def _force_tty(value: bool = True) -> patch:
    """Patch is_tty in the tools_panel module."""
    return patch("holodeck.chat.tools_panel.is_tty", return_value=value)


class TestEmptyState:
    """Empty panels render nothing regardless of TTY state."""

    def test_no_active_tools_returns_empty_lines_in_tty(self) -> None:
        panel = ToolsPanel()
        with _force_tty(True):
            assert panel.render_lines() == []
            assert panel.line_count == 0

    def test_non_tty_renders_nothing_even_with_active_tools(self) -> None:
        panel = ToolsPanel()
        panel.apply(ToolEvent(kind="start", tool_name="Read", tool_use_id="tu_1"))
        with _force_tty(False):
            assert panel.render_lines() == []
            assert panel.line_count == 0


class TestRegularToolLifecycle:
    """start → end / error transitions for non-Task tools."""

    def test_start_creates_active_entry(self) -> None:
        panel = ToolsPanel()
        panel.apply(ToolEvent(kind="start", tool_name="Read", tool_use_id="tu_1"))
        with _force_tty(True):
            lines = panel.render_lines()
        assert any("Read" in line for line in lines)
        # 2 borders + 1 entry
        with _force_tty(True):
            assert panel.line_count == 3

    def test_end_removes_active_entry(self) -> None:
        panel = ToolsPanel()
        panel.apply(ToolEvent(kind="start", tool_name="Read", tool_use_id="tu_1"))
        panel.apply(ToolEvent(kind="end", tool_name="Read", tool_use_id="tu_1"))
        with _force_tty(True):
            assert panel.render_lines() == []
            assert panel.line_count == 0

    def test_error_removes_active_entry(self) -> None:
        panel = ToolsPanel()
        panel.apply(ToolEvent(kind="start", tool_name="Bash", tool_use_id="tu_1"))
        panel.apply(
            ToolEvent(
                kind="error",
                tool_name="Bash",
                tool_use_id="tu_1",
                error="boom",
            )
        )
        with _force_tty(True):
            assert panel.render_lines() == []

    def test_multiple_tools_render_in_order(self) -> None:
        panel = ToolsPanel()
        for name, tu in [("Read", "tu_1"), ("Bash", "tu_2"), ("Grep", "tu_3")]:
            panel.apply(ToolEvent(kind="start", tool_name=name, tool_use_id=tu))
        with _force_tty(True):
            lines = panel.render_lines()
        joined = "\n".join(lines)
        # All names present, in insertion order
        assert joined.index("Read") < joined.index("Bash") < joined.index("Grep")


class TestSubagentLifecycle:
    """Task tool calls render as subagents and surface status text."""

    def test_task_tool_labels_with_subagent_type(self) -> None:
        panel = ToolsPanel()
        panel.apply(
            ToolEvent(
                kind="start",
                tool_name="Task",
                tool_use_id="task_1",
                tool_input={"subagent_type": "code-reviewer"},
            )
        )
        with _force_tty(True):
            lines = panel.render_lines()
        assert any("Task[code-reviewer]" in line for line in lines)

    def test_subagent_message_attaches_status_line(self) -> None:
        panel = ToolsPanel()
        panel.apply(
            ToolEvent(
                kind="start",
                tool_name="Task",
                tool_use_id="task_1",
                tool_input={"subagent_type": "researcher"},
            )
        )
        panel.apply(
            ToolEvent(
                kind="subagent_message",
                tool_name="Task",
                tool_use_id="task_1",
                parent_tool_use_id="task_1",
                text="Found 3 candidate files\nReading src/foo.py…",
            )
        )
        with _force_tty(True):
            lines = panel.render_lines()
            count = panel.line_count
        # Status uses the LAST non-empty line
        assert any("Reading src/foo.py" in line for line in lines)
        # 2 borders + 1 entry + 1 status
        assert count == 4

    def test_subagent_message_replaces_previous_status(self) -> None:
        panel = ToolsPanel()
        panel.apply(
            ToolEvent(
                kind="start",
                tool_name="Task",
                tool_use_id="task_1",
                tool_input={"subagent_type": "x"},
            )
        )
        panel.apply(
            ToolEvent(
                kind="subagent_message",
                tool_name="Task",
                tool_use_id="task_1",
                parent_tool_use_id="task_1",
                text="step one",
            )
        )
        panel.apply(
            ToolEvent(
                kind="subagent_message",
                tool_name="Task",
                tool_use_id="task_1",
                parent_tool_use_id="task_1",
                text="step two",
            )
        )
        with _force_tty(True):
            joined = "\n".join(panel.render_lines())
        assert "step two" in joined
        assert "step one" not in joined

    def test_subagent_message_for_unknown_id_is_dropped(self) -> None:
        panel = ToolsPanel()
        # No corresponding start event — message must not crash and must
        # not introduce a phantom entry.
        panel.apply(
            ToolEvent(
                kind="subagent_message",
                tool_name="Task",
                tool_use_id="task_unknown",
                parent_tool_use_id="task_unknown",
                text="hello",
            )
        )
        with _force_tty(True):
            assert panel.render_lines() == []

    def test_end_clears_subagent_and_status(self) -> None:
        panel = ToolsPanel()
        panel.apply(
            ToolEvent(
                kind="start",
                tool_name="Task",
                tool_use_id="task_1",
                tool_input={"subagent_type": "rev"},
            )
        )
        panel.apply(
            ToolEvent(
                kind="subagent_message",
                tool_name="Task",
                tool_use_id="task_1",
                parent_tool_use_id="task_1",
                text="working",
            )
        )
        panel.apply(ToolEvent(kind="end", tool_name="Task", tool_use_id="task_1"))
        with _force_tty(True):
            assert panel.render_lines() == []


class TestSnapshot:
    """`snapshot()` returns the current in-flight entries."""

    def test_snapshot_reflects_active_tools(self) -> None:
        panel = ToolsPanel()
        panel.apply(ToolEvent(kind="start", tool_name="Read", tool_use_id="tu_1"))
        panel.apply(
            ToolEvent(
                kind="start",
                tool_name="Task",
                tool_use_id="task_1",
                tool_input={"subagent_type": "rev"},
            )
        )
        snap = panel.snapshot()
        assert {e.tool_name for e in snap} == {"Read", "Task"}
        task_entry = next(e for e in snap if e.is_subagent)
        assert task_entry.subagent_type == "rev"
