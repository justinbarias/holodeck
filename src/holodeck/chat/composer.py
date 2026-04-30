"""Live screen compositor for the chat REPL.

Owns the region below the streaming-text cursor for the duration of one
turn. Streamed text is written at the cursor while a multi-line panel
(tools + spinner) sits below it and is erased + redrawn around every text
write so it stays visible throughout the response.

Cursor invariant (between operations):
* Cursor is positioned at the end of the streamed text.
* Lines immediately below the cursor hold the panel (or nothing if the
  panel is empty).

The compositor uses DECSC / DECRC (``\\033 7`` / ``\\033 8``) to save and
restore the cursor around panel paints. This avoids tracking the
streaming text's column manually — the terminal does it for us.

Limitations: when the cursor is on the bottom row of the terminal, the
``\\n`` used to descend into the panel region scrolls the screen, which
invalidates the saved cursor on most terminals. In practice that means
the panel may flicker if the chat output reaches the very bottom of the
window.
"""

from __future__ import annotations

import asyncio
import sys

from holodeck.lib.ui import is_tty


class LiveComposer:
    """Serialises stdout writes around a panel pinned below the cursor."""

    def __init__(self) -> None:
        self._panel_lines: list[str] = []
        self._lock = asyncio.Lock()
        self._active = False

    async def begin(self) -> None:
        """Mark the start of a turn — the panel may now be drawn."""
        async with self._lock:
            self._panel_lines = []
            self._active = True

    async def end(self) -> None:
        """End the turn — erase any visible panel and reset state."""
        async with self._lock:
            self._erase_panel()
            self._panel_lines = []
            self._active = False
            sys.stdout.flush()

    async def update_panel(self, lines: list[str]) -> None:
        """Replace the visible panel with *lines* (no-op if unchanged)."""
        async with self._lock:
            if not self._active:
                return
            if lines == self._panel_lines:
                return
            self._erase_panel()
            self._panel_lines = list(lines)
            self._draw_panel()
            sys.stdout.flush()

    async def write_text(self, chunk: str) -> None:
        """Append *chunk* at the cursor while keeping the panel visible."""
        async with self._lock:
            self._erase_panel()
            sys.stdout.write(chunk)
            self._draw_panel()
            sys.stdout.flush()

    def _erase_panel(self) -> None:
        if not self._panel_lines or not is_tty():
            return
        # Save cursor at text-end, descend one line into the panel region,
        # clear from there to end of screen, restore cursor.
        sys.stdout.write("\0337\n\033[J\0338")

    def _draw_panel(self) -> None:
        if not self._panel_lines or not is_tty():
            return
        body = "\n".join(self._panel_lines)
        sys.stdout.write(f"\0337\n{body}\0338")
