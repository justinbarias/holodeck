"""Live panel of in-flight tools and subagents for the chat REPL.

Maintains state from the ``ToolEvent`` queue published by
:class:`holodeck.lib.backends.claude_backend.ClaudeSession` and renders a
multi-line ANSI panel above the spinner.  Two kinds of entries are tracked:

* Regular tools — created on ``"start"``, removed on ``"end"`` / ``"error"``.
  Their input args are recorded from the ``"start"`` event and rendered
  as a compact status line under the entry.
* Subagents — Task tool calls (``tool_name == "Task"``).  Their last
  assistant text snippet is recorded from ``"subagent_message"`` events and
  shown as a status line under the entry, replacing the args view once
  the subagent starts streaming.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from holodeck.lib.backends.base import ToolEvent
from holodeck.lib.ui import SpinnerMixin, is_tty


@dataclass
class _Active:
    """In-flight tool or subagent tracked by :class:`ToolsPanel`."""

    tool_name: str
    tool_use_id: str
    started_at: float
    is_subagent: bool
    subagent_type: str | None = None
    last_status: str | None = None
    tool_input: dict[str, Any] | None = None
    parent_tool_use_id: str | None = None


_PANEL_WIDTH = 60
_STATUS_MAX = _PANEL_WIDTH - 8  # account for "│   └─ " and trailing " │"
_BORDER_TOP = "╭─── Active Tools " + "─" * (_PANEL_WIDTH - 19) + "╮"
_BORDER_BOTTOM = "╰" + "─" * (_PANEL_WIDTH - 2) + "╯"


class ToolsPanel(SpinnerMixin):
    """State + ANSI renderer for in-flight tools and subagents.

    Not coroutine-safe across producers — consumers are expected to
    serialise calls (the chat REPL runs the drainer on a single asyncio
    task).  No locking is required because mutations and reads happen on
    the same event loop.
    """

    def __init__(self) -> None:
        self._active: dict[str, _Active] = {}
        # parent_link can arrive before the child's "start" event (the SDK
        # surfaces parent_tool_use_id via AssistantMessage on a separate path
        # from PreToolUse hooks), so stash early arrivals and apply them when
        # the start finally lands.
        self._pending_parents: dict[str, str] = {}
        self._spinner_index = 0

    def apply(self, evt: ToolEvent) -> None:
        """Update state from a ToolEvent.

        Args:
            evt: Event from the backend's tool event queue.
        """
        if evt.kind == "start":
            is_subagent = evt.tool_name == "Task"
            subagent_type: str | None = None
            if is_subagent and evt.tool_input:
                raw = evt.tool_input.get("subagent_type")
                if isinstance(raw, str):
                    subagent_type = raw
            self._active[evt.tool_use_id] = _Active(
                tool_name=evt.tool_name,
                tool_use_id=evt.tool_use_id,
                started_at=time.monotonic(),
                is_subagent=is_subagent,
                subagent_type=subagent_type,
                tool_input=evt.tool_input,
                parent_tool_use_id=self._pending_parents.pop(evt.tool_use_id, None),
            )
        elif evt.kind in ("end", "error"):
            self._active.pop(evt.tool_use_id, None)
            self._pending_parents.pop(evt.tool_use_id, None)
        elif evt.kind == "subagent_message" and evt.text:
            entry = self._active.get(evt.tool_use_id)
            if entry is not None:
                entry.last_status = _summarize_status(evt.text)
        elif evt.kind == "parent_link" and evt.parent_tool_use_id:
            entry = self._active.get(evt.tool_use_id)
            if entry is None:
                self._pending_parents[evt.tool_use_id] = evt.parent_tool_use_id
            else:
                entry.parent_tool_use_id = evt.parent_tool_use_id

    def render_lines(self) -> list[str]:
        """Render current state as panel lines.

        Returns:
            A list of fully-formed lines (no trailing newline).  Empty list
            when nothing is active or stdout is not a TTY.
        """
        if not is_tty() or not self._active:
            return []

        spinner_char = self.get_spinner_char()
        children_by_parent: dict[str, list[_Active]] = {}
        for entry in self._active.values():
            if entry.parent_tool_use_id:
                children_by_parent.setdefault(entry.parent_tool_use_id, []).append(
                    entry
                )

        lines: list[str] = [_BORDER_TOP]
        for entry in self._active.values():
            # Top-level pass: skip entries whose parent is still active — they
            # are rendered nested under the parent below.
            if entry.parent_tool_use_id and entry.parent_tool_use_id in self._active:
                continue
            self._render_entry(entry, 0, spinner_char, lines, children_by_parent)
        lines.append(_BORDER_BOTTOM)
        return lines

    def _render_entry(
        self,
        entry: _Active,
        indent: int,
        spinner_char: str,
        lines: list[str],
        children_by_parent: dict[str, list[_Active]],
    ) -> None:
        """Append entry header, secondary line, and children to *lines*."""
        inner_width = _PANEL_WIDTH - 4 - indent
        pad = " " * indent
        label = _entry_label(entry)
        elapsed = f"{time.monotonic() - entry.started_at:.1f}s"
        content = _fit_row(f"{spinner_char} {label}", elapsed, inner_width)
        lines.append(f"│ {pad}{content} │")
        secondary = _secondary_text(entry)
        if secondary:
            status_line = _truncate(f"  └─ {secondary}", inner_width)
            lines.append(f"│ {pad}{status_line:<{inner_width}} │")
        for child in children_by_parent.get(entry.tool_use_id, []):
            self._render_entry(
                child, indent + 2, spinner_char, lines, children_by_parent
            )

    @property
    def line_count(self) -> int:
        """Number of lines :meth:`render_lines` would produce right now."""
        return len(self.render_lines())

    def snapshot(self) -> list[_Active]:
        """Return a copy of the in-flight entries (for post-turn display)."""
        return list(self._active.values())


def _entry_label(entry: _Active) -> str:
    """Build the human label for a panel entry."""
    if entry.is_subagent and entry.subagent_type:
        return f"Task[{entry.subagent_type}]"
    return entry.tool_name


def _secondary_text(entry: _Active) -> str | None:
    """Choose the text shown under the entry header.

    Subagent status (from streamed assistant text) wins once it arrives.
    Otherwise we render compact ``key=value`` pairs from the tool input so
    the user can see *what* the tool was invoked with.  ``subagent_type``
    is suppressed because it's already shown in the entry label.
    """
    if entry.is_subagent and entry.last_status:
        return entry.last_status
    if entry.tool_input:
        skip = {"subagent_type"} if entry.is_subagent else set()
        args = _format_args(entry.tool_input, skip_keys=skip)
        return args or None
    return None


def _format_args(tool_input: dict[str, Any], skip_keys: set[str] | None = None) -> str:
    """Format tool input as ``k=v`` pairs joined by spaces."""
    skip = skip_keys or set()
    parts: list[str] = []
    for key, value in tool_input.items():
        if key in skip:
            continue
        parts.append(f"{key}={_format_value(value)}")
    return " ".join(parts)


def _format_value(value: Any) -> str:
    """Render a single arg value as a single-line string."""
    if isinstance(value, str):
        return value.replace("\n", " ").replace("\t", " ").strip()
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _summarize_status(text: str) -> str:
    """Reduce a multi-line subagent text snapshot to a single status line."""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return _truncate(stripped, _STATUS_MAX - len("  └─ "))
    return _truncate(text.strip(), _STATUS_MAX - len("  └─ "))


def _truncate(text: str, max_len: int) -> str:
    """Truncate *text* to *max_len* characters with an ellipsis."""
    if max_len <= 1:
        return text[:max_len]
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _fit_row(label: str, suffix: str, content_width: int) -> str:
    """Pack a row so *label* sits left-aligned and *suffix* sits right-aligned.

    Total content width is *content_width*.  If *label* is too long, it is
    truncated so the suffix always remains visible.
    """
    suffix_room = len(suffix) + 1  # at least one space between
    label_room = content_width - suffix_room
    if label_room < 1:
        return _truncate(label, content_width)
    label_fit = _truncate(label, label_room)
    return f"{label_fit:<{label_room}} {suffix:>{suffix_room - 1}}"
