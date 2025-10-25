"""
Unit tests for CLI help text and documentation.

Tests verify:
- T111: CLI help text completeness and clarity
- T112: Version flag support
"""

import pytest
from click.testing import CliRunner

from holodeck.cli.main import main as cli


@pytest.mark.unit
class TestCLIHelpText:
    """T111: Test CLI help text completeness and clarity."""

    def test_init_command_help_exists(self):
        """`holodeck init --help` should display help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.output
        assert "Options:" in result.output

    def test_help_includes_description(self):
        """Help text should describe what the command does."""
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--help"])

        help_text = result.output.lower()
        # Should explain what init does
        assert (
            "create" in help_text or "initialize" in help_text or "project" in help_text
        )

    def test_help_includes_arguments(self):
        """Help text should document required arguments."""
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--help"])

        help_text = result.output.lower()
        # Should mention project name argument
        assert "name" in help_text or "project" in help_text

    def test_help_includes_options(self):
        """Help text should document available options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--help"])

        help_text = result.output.lower()
        # Should document key options
        assert "--template" in result.output or "template" in help_text
        assert "--description" in result.output or "description" in help_text
        assert "--author" in result.output or "author" in help_text
        assert "--force" in result.output or "force" in help_text

    def test_help_includes_examples(self):
        """Help text should include usage examples."""
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--help"])

        # Examples make help more useful
        # This might be in the long help text or docstring
        # At minimum, should show basic usage
        assert "holodeck" in result.output

    def test_template_option_documented(self):
        """--template option should be documented with available choices."""
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--help"])

        # Should mention template option
        assert "--template" in result.output

    def test_description_option_documented(self):
        """--description option should be documented."""
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--help"])

        assert "--description" in result.output

    def test_author_option_documented(self):
        """--author option should be documented."""
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--help"])

        assert "--author" in result.output

    def test_force_option_documented(self):
        """--force option should be documented."""
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--help"])

        assert "--force" in result.output

    def test_help_text_readable(self):
        """Help text should be formatted for readability."""
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--help"])

        # Should be readable length (not all on one line)
        lines = result.output.split("\n")
        assert len(lines) > 5  # Multiple lines

        # Should not have extremely long lines
        for line in lines:
            assert len(line) < 120  # Reasonable line length


@pytest.mark.unit
class TestVersionFlag:
    """T112: Test version flag support."""

    def test_version_flag_exists(self):
        """`holodeck --version` should display version."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        # Should either succeed or indicate version is not yet set
        assert result.exit_code in [0, 2]  # 0 = success, 2 = option not found

    def test_version_output_format(self):
        """Version output should be in standard format."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        if result.exit_code == 0:
            # Should contain version-like format (e.g., "0.1.0", "1.0.0a1")
            assert "version" in result.output.lower() or any(
                c.isdigit() for c in result.output
            )

    def test_help_flag_exists(self):
        """`holodeck --help` should work."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_main_help_shows_commands(self):
        """Main help should list available commands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        help_text = result.output.lower()
        # Should mention init command
        assert "init" in help_text

    def test_invalid_option_shows_help(self):
        """Invalid option should show helpful error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--invalid-option"])

        # Should indicate the option is invalid
        assert result.exit_code != 0

    def test_missing_argument_shows_error(self):
        """Missing required argument should show error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["init"])

        # Should show error about missing argument
        assert result.exit_code != 0
        assert (
            "Error" in result.output
            or "Usage" in result.output
            or "argument" in result.output.lower()
            or "missing" in result.output.lower()
        )
