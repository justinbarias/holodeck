"""Unit tests for :class:`holodeck.chat.composer.LiveComposer`."""

from __future__ import annotations

import io
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from holodeck.chat.composer import LiveComposer


@contextmanager
def _captured_stdout() -> Iterator[io.StringIO]:
    """Patch ``sys.stdout`` with an in-memory buffer for the duration of the test."""
    buf = io.StringIO()
    with patch("holodeck.chat.composer.sys.stdout", buf):
        yield buf


def _force_tty(value: bool) -> Iterator[None]:
    return patch("holodeck.chat.composer.is_tty", return_value=value)


class TestLiveComposer:
    @pytest.mark.asyncio
    async def test_inactive_composer_skips_panel_updates(self) -> None:
        composer = LiveComposer()
        with _captured_stdout() as buf, _force_tty(True):
            await composer.update_panel(["one"])
        # Never began — must not paint.
        assert buf.getvalue() == ""

    @pytest.mark.asyncio
    async def test_update_panel_writes_when_active_and_tty(self) -> None:
        composer = LiveComposer()
        with _captured_stdout() as buf, _force_tty(True):
            await composer.begin()
            await composer.update_panel(["row-a", "row-b"])
            output = buf.getvalue()
        # DECSC + newline + body + DECRC.
        assert "\0337\nrow-a\nrow-b\0338" in output

    @pytest.mark.asyncio
    async def test_update_panel_noops_when_lines_unchanged(self) -> None:
        composer = LiveComposer()
        with _captured_stdout() as buf, _force_tty(True):
            await composer.begin()
            await composer.update_panel(["row-a"])
            buf.truncate(0)
            buf.seek(0)
            await composer.update_panel(["row-a"])
            assert buf.getvalue() == ""

    @pytest.mark.asyncio
    async def test_write_text_erases_then_redraws_panel(self) -> None:
        composer = LiveComposer()
        with _captured_stdout() as buf, _force_tty(True):
            await composer.begin()
            await composer.update_panel(["row-a"])
            buf.truncate(0)
            buf.seek(0)
            await composer.write_text("hello")
            output = buf.getvalue()
        # Erase sequence + chunk + redraw sequence.
        assert output.startswith("\0337\n\033[J\0338")
        assert "hello" in output
        assert output.endswith("\0337\nrow-a\0338")

    @pytest.mark.asyncio
    async def test_end_clears_panel_and_resets_state(self) -> None:
        composer = LiveComposer()
        with _captured_stdout() as buf, _force_tty(True):
            await composer.begin()
            await composer.update_panel(["row-a"])
            buf.truncate(0)
            buf.seek(0)
            await composer.end()
            output = buf.getvalue()
        # Erase escape was emitted.
        assert "\0337\n\033[J\0338" in output

        # After end(), update_panel must noop (composer inactive).
        with _captured_stdout() as buf2, _force_tty(True):
            await composer.update_panel(["new-row"])
            assert buf2.getvalue() == ""

    @pytest.mark.asyncio
    async def test_non_tty_skips_ansi_writes_but_streams_text(self) -> None:
        composer = LiveComposer()
        with _captured_stdout() as buf, _force_tty(False):
            await composer.begin()
            await composer.update_panel(["row-a"])
            await composer.write_text("hello")
            output = buf.getvalue()
        # No ANSI sequences when not a TTY; text still arrives so logs are usable.
        assert "\0337" not in output
        assert "\033[J" not in output
        assert output == "hello"
