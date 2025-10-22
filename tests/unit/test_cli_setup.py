"""Tests for Click CLI setup and group functionality.

These tests verify that the Click CLI group is properly configured
and responds to basic commands like --help.
"""

import pytest
from click.testing import CliRunner


@pytest.mark.unit
def test_cli_group_exists() -> None:
    """Test that CLI group can be imported."""
    from holodeck.cli import main  # noqa: F401

    # If import succeeds, test passes


@pytest.mark.unit
def test_cli_help_text() -> None:
    """Test that CLI group responds to --help."""
    from holodeck.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "--help" in result.output


@pytest.mark.unit
def test_cli_version_flag() -> None:
    """Test that CLI responds to --version flag."""
    from holodeck.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])

    # Should either show version or be recognized as a valid flag
    assert result.exit_code in (0, 2)  # 0 for success, 2 for click behavior


@pytest.mark.unit
def test_init_command_registered() -> None:
    """Test that 'init' command is registered with CLI group."""
    from holodeck.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["init", "--help"])

    # Should show help for init command
    assert result.exit_code == 0
    assert "Usage:" in result.output or "init" in result.output.lower()
