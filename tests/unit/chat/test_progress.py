"""Unit tests for chat progress tracking and display.

Tests cover:
- ChatProgressIndicator initialization
- Spinner animation (get_spinner_line)
- Progress updates with token tracking
- Status display formatting (inline and panel modes)
- TTY detection
"""

from unittest.mock import patch

from holodeck.chat.executor import AgentResponse
from holodeck.chat.progress import ChatProgressIndicator
from holodeck.models.token_usage import TokenUsage


class TestChatProgressIndicatorInitialization:
    """Tests for ChatProgressIndicator initialization."""

    def test_initialization_with_defaults(self) -> None:
        """ChatProgressIndicator initializes with correct defaults."""
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)

        assert progress.max_messages == 50
        assert progress.quiet is False
        assert progress.verbose is False
        assert progress.current_messages == 0
        assert progress.last_response_time is None
        assert progress.total_tokens.prompt_tokens == 0
        assert progress.total_tokens.completion_tokens == 0
        assert progress.total_tokens.total_tokens == 0

    def test_initialization_with_quiet_mode(self) -> None:
        """ChatProgressIndicator respects quiet flag."""
        progress = ChatProgressIndicator(max_messages=100, quiet=True, verbose=False)

        assert progress.quiet is True
        assert progress.max_messages == 100

    def test_initialization_with_verbose_mode(self) -> None:
        """ChatProgressIndicator respects verbose flag."""
        progress = ChatProgressIndicator(max_messages=75, quiet=False, verbose=True)

        assert progress.verbose is True
        assert progress.max_messages == 75


class TestSpinnerAnimation:
    """Tests for spinner animation."""

    def test_spinner_line_returns_empty_when_not_tty(self) -> None:
        """Spinner returns empty string when not a TTY."""
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)

        with patch("sys.stdout.isatty", return_value=False):
            line = progress.get_spinner_line()
            assert line == ""

    def test_spinner_line_returns_animated_text_when_tty(self) -> None:
        """Spinner returns animated text when TTY."""
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)

        with patch("sys.stdout.isatty", return_value=True):
            line = progress.get_spinner_line()
            assert "Thinking..." in line
            assert line.startswith("⠋")  # First braille character

    def test_spinner_animation_cycles(self) -> None:
        """Spinner cycles through all animation frames."""
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)

        with patch("sys.stdout.isatty", return_value=True):
            characters = []
            # Get 10 spinner lines to see cycling
            for _ in range(10):
                line = progress.get_spinner_line()
                char = line[0] if line else ""
                characters.append(char)

            # Should have cycled through multiple different characters
            unique_chars = set(characters)
            assert len(unique_chars) > 1

    def test_spinner_index_increments(self) -> None:
        """Spinner index increments with each call."""
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)

        with patch("sys.stdout.isatty", return_value=True):
            initial_index = progress._spinner_index
            progress.get_spinner_line()
            assert progress._spinner_index > initial_index


class TestProgressUpdate:
    """Tests for progress tracking with updates."""

    def test_update_increments_message_count(self) -> None:
        """Update increments current_messages."""
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)
        response = AgentResponse(
            content="Test response",
            tool_executions=[],
            tokens_used=None,
            execution_time=1.0,
        )

        progress.update(response)

        assert progress.current_messages == 1

    def test_update_tracks_execution_time(self) -> None:
        """Update tracks last response execution time."""
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)
        response = AgentResponse(
            content="Test response",
            tool_executions=[],
            tokens_used=None,
            execution_time=2.5,
        )

        progress.update(response)

        assert progress.last_response_time == 2.5

    def test_update_accumulates_token_usage(self) -> None:
        """Update accumulates token usage from multiple responses."""
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)

        # First response
        response1 = AgentResponse(
            content="First response",
            tool_executions=[],
            tokens_used=TokenUsage(
                prompt_tokens=100, completion_tokens=50, total_tokens=150
            ),
            execution_time=1.0,
        )
        progress.update(response1)

        assert progress.total_tokens.prompt_tokens == 100
        assert progress.total_tokens.completion_tokens == 50
        assert progress.total_tokens.total_tokens == 150

        # Second response
        response2 = AgentResponse(
            content="Second response",
            tool_executions=[],
            tokens_used=TokenUsage(
                prompt_tokens=80, completion_tokens=40, total_tokens=120
            ),
            execution_time=1.5,
        )
        progress.update(response2)

        assert progress.total_tokens.prompt_tokens == 180
        assert progress.total_tokens.completion_tokens == 90
        assert progress.total_tokens.total_tokens == 270

    def test_update_handles_none_tokens(self) -> None:
        """Update handles response with None token_usage."""
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)
        response = AgentResponse(
            content="Test response",
            tool_executions=[],
            tokens_used=None,
            execution_time=1.0,
        )

        # Should not raise an error
        progress.update(response)
        assert progress.total_tokens.total_tokens == 0


class TestInlineStatus:
    """Tests for inline status display."""

    def test_inline_status_format(self) -> None:
        """Inline status has correct format."""
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)
        progress.current_messages = 3
        progress.last_response_time = 1.2

        status = progress.get_status_inline()

        assert status == "[3/50 | 1.2s]"

    def test_inline_status_without_execution_time(self) -> None:
        """Inline status without execution time."""
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)
        progress.current_messages = 5
        progress.last_response_time = None

        status = progress.get_status_inline()

        assert status == "[5/50]"

    def test_inline_status_with_message_count_only(self) -> None:
        """Inline status shows message count even without execution time."""
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)
        progress.current_messages = 0
        progress.last_response_time = None

        status = progress.get_status_inline()

        assert status == "[0/50]"


class TestStatusPanel:
    """Tests for rich status panel display."""

    def test_panel_structure_has_borders(self) -> None:
        """Panel has top and bottom borders."""
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=True)
        progress.current_messages = 1
        progress.last_response_time = 1.0

        panel = progress.get_status_panel()
        lines = panel.split("\n")

        assert lines[0].startswith("╭")
        assert lines[-1].startswith("╰")

    def test_panel_all_lines_same_width(self) -> None:
        """All panel lines have consistent width."""
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=True)
        progress.current_messages = 3
        progress.last_response_time = 1.2
        progress.total_tokens = TokenUsage(
            prompt_tokens=890, completion_tokens=344, total_tokens=1234
        )

        panel = progress.get_status_panel()
        lines = panel.split("\n")

        # All lines should be exactly 43 characters
        widths = [len(line) for line in lines]
        assert all(w == 43 for w in widths), f"Inconsistent line widths: {widths}"

    def test_panel_includes_session_time(self) -> None:
        """Panel includes session time."""
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=True)

        panel = progress.get_status_panel()

        assert "Session Time:" in panel
        assert "00:00:00" in panel  # Initial time

    def test_panel_includes_message_count(self) -> None:
        """Panel includes message count and percentage."""
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=True)
        progress.current_messages = 3

        panel = progress.get_status_panel()

        assert "Messages: 3 / 50 (6%)" in panel

    def test_panel_includes_token_totals(self) -> None:
        """Panel includes token totals and breakdown."""
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=True)
        progress.total_tokens = TokenUsage(
            prompt_tokens=890, completion_tokens=344, total_tokens=1234
        )

        panel = progress.get_status_panel()

        assert "Total Tokens: 1,234" in panel
        assert "Prompt: 890" in panel
        assert "Completion: 344" in panel

    def test_panel_includes_last_response_time(self) -> None:
        """Panel includes last response time when available."""
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=True)
        progress.last_response_time = 2.3

        panel = progress.get_status_panel()

        assert "Last Response: 2.3s" in panel

    def test_panel_without_response_time(self) -> None:
        """Panel handles missing last response time."""
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=True)
        progress.last_response_time = None

        panel = progress.get_status_panel()

        # Should not have "Last Response:" line
        assert "Last Response:" not in panel

    def test_panel_formats_large_token_counts(self) -> None:
        """Panel formats large token counts with commas."""
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=True)
        progress.total_tokens = TokenUsage(
            prompt_tokens=10000, completion_tokens=5000, total_tokens=15000
        )

        panel = progress.get_status_panel()

        assert "15,000" in panel
        assert "10,000" in panel
        assert "5,000" in panel


class TestTTYDetection:
    """Tests for TTY detection."""

    def test_is_tty_property_returns_true_for_tty(self) -> None:
        """is_tty returns True when stdout is a TTY."""
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)

        with patch("sys.stdout.isatty", return_value=True):
            assert progress.is_tty is True

    def test_is_tty_property_returns_false_for_non_tty(self) -> None:
        """is_tty returns False when stdout is not a TTY."""
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)

        with patch("sys.stdout.isatty", return_value=False):
            assert progress.is_tty is False


class TestChatProgressIntegration:
    """Integration tests for chat progress tracking."""

    def test_full_conversation_tracking(self) -> None:
        """Track a full multi-turn conversation."""
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)

        # Turn 1
        response1 = AgentResponse(
            content="First response",
            tool_executions=[],
            tokens_used=TokenUsage(
                prompt_tokens=100, completion_tokens=50, total_tokens=150
            ),
            execution_time=1.0,
        )
        progress.update(response1)

        # Turn 2
        response2 = AgentResponse(
            content="Second response",
            tool_executions=[],
            tokens_used=TokenUsage(
                prompt_tokens=80, completion_tokens=40, total_tokens=120
            ),
            execution_time=1.2,
        )
        progress.update(response2)

        # Turn 3
        response3 = AgentResponse(
            content="Third response",
            tool_executions=[],
            tokens_used=TokenUsage(
                prompt_tokens=90, completion_tokens=35, total_tokens=125
            ),
            execution_time=0.9,
        )
        progress.update(response3)

        # Verify accumulated state
        assert progress.current_messages == 3
        assert progress.total_tokens.prompt_tokens == 270
        assert progress.total_tokens.completion_tokens == 125
        assert progress.total_tokens.total_tokens == 395
        assert progress.last_response_time == 0.9

        # Verify inline status
        status = progress.get_status_inline()
        assert "3/50" in status
        assert "0.9s" in status

        # Verify panel status
        panel = progress.get_status_panel()
        assert "Messages: 3 / 50 (6%)" in panel
        assert "395" in panel  # Total tokens
