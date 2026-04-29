"""Live panel of in-flight tools and subagents for the chat REPL.

Maintains state from the ``ToolEvent`` queue published by
:class:`holodeck.lib.backends.claude_backend.ClaudeSession` and renders a
multi-line ANSI panel above the spinner.  Two kinds of entries are tracked:

* Regular tools — created on ``"start"``, removed on ``"end"`` / ``"error"``.
* Subagents — Task tool calls (``tool_name == "Task"``).  Their last
  assistant text snippet is recorded from ``"subagent_message"`` events and
  shown as a status line under the entry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

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
            )
        elif evt.kind in ("end", "error"):
            self._active.pop(evt.tool_use_id, None)
        elif evt.kind == "subagent_message" and evt.text:
            entry = self._active.get(evt.tool_use_id)
            if entry is not None:
                entry.last_status = _summarize_status(evt.text)

    def render_lines(self) -> list[str]:
        """Render current state as panel lines.

        Returns:
            A list of fully-formed lines (no trailing newline).  Empty list
            when nothing is active or stdout is not a TTY.
        """
        if not is_tty() or not self._active:
            return []

        spinner_char = self.get_spinner_char()
        lines: list[str] = [_BORDER_TOP]
        for entry in self._active.values():
            label = _entry_label(entry)
            elapsed = f"{time.monotonic() - entry.started_at:.1f}s"
            content = _fit_row(f"{spinner_char} {label}", elapsed)
            lines.append(f"│ {content} │")
            if entry.is_subagent and entry.last_status:
                status_line = _truncate(f"  └─ {entry.last_status}", _STATUS_MAX)
                lines.append(f"│ {status_line:<{_PANEL_WIDTH - 4}} │")
        lines.append(_BORDER_BOTTOM)
        return lines

    @property
    def line_count(self) -> int:
        """Number of lines :meth:`render_lines` would produce right now."""
        if not is_tty() or not self._active:
            return 0
        # 2 borders + 1 line per entry + 1 status line per subagent w/ status
        n = 2 + len(self._active)
        for entry in self._active.values():
            if entry.is_subagent and entry.last_status:
                n += 1
        return n

    def snapshot(self) -> list[_Active]:
        """Return a copy of the in-flight entries (for post-turn display)."""
        return list(self._active.values())


def _entry_label(entry: _Active) -> str:
    """Build the human label for a panel entry."""
    if entry.is_subagent and entry.subagent_type:
        return f"Task[{entry.subagent_type}]"
    return entry.tool_name


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


def _fit_row(label: str, suffix: str) -> str:
    """Pack a row so *label* sits left-aligned and *suffix* sits right-aligned.

    Total content width is :data:`_PANEL_WIDTH` minus the 4 chars of border
    padding (``"│ "`` + ``" │"``).  If *label* is too long, it is truncated
    so the suffix always remains visible.
    """
    content_width = _PANEL_WIDTH - 4
    suffix_room = len(suffix) + 1  # at least one space between
    label_room = content_width - suffix_room
    if label_room < 1:
        return _truncate(label, content_width)
    label_fit = _truncate(label, label_room)
    return f"{label_fit:<{label_room}} {suffix:>{suffix_room - 1}}"
