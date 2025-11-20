"""Unit tests for the config command."""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from holodeck.cli.commands.config import init
from holodeck.config.manager import ConfigManager
from holodeck.models.config import GlobalConfig


class TestConfigManager:
    """Test the ConfigManager class."""

    def test_create_default_config(self):
        """Test that create_default_config returns a valid GlobalConfig."""
        config = ConfigManager.create_default_config()
        assert isinstance(config, GlobalConfig)
        assert config.providers["openai"].name == "gpt-4"
        assert config.vectorstores["postgres"].provider == "postgres"

    def test_get_config_path_global(self):
        """Test get_config_path for global config."""
        path, name = ConfigManager.get_config_path(
            global_config=True, project_config=False
        )
        assert path == Path.home() / ".holodeck" / "config.yaml"
        assert name == "global"

    def test_get_config_path_project(self):
        """Test get_config_path for project config."""
        path, name = ConfigManager.get_config_path(
            global_config=False, project_config=True
        )
        assert path == Path.cwd() / "config.yaml"
        assert name == "project"

    def test_generate_config_content(self):
        """Test generate_config_content produces valid YAML."""
        config = ConfigManager.create_default_config()
        content = ConfigManager.generate_config_content(config)
        assert "providers:" in content
        assert "gpt-4" in content
        assert "vectorstores:" in content

    def test_write_config(self, tmp_path):
        """Test write_config writes content to file."""
        config_file = tmp_path / "config.yaml"
        content = "test: content"
        ConfigManager.write_config(config_file, content)
        assert config_file.exists()
        assert config_file.read_text(encoding="utf-8") == content


class TestConfigInitCommand:
    """Test the config init command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_init_help(self, runner):
        """Test that help message is displayed."""
        result = runner.invoke(init, ["--help"])
        assert result.exit_code == 0
        assert "Initialize HoloDeck global or project configuration" in result.output

    @patch("holodeck.config.manager.ConfigManager.write_config")
    def test_init_global(self, mock_write, runner):
        """Test initializing global configuration."""
        result = runner.invoke(init, ["-g"])
        assert result.exit_code == 0
        assert "Global configuration initialized successfully" in result.output
        mock_write.assert_called_once()

    @patch("holodeck.config.manager.ConfigManager.write_config")
    def test_init_project(self, mock_write, runner):
        """Test initializing project configuration."""
        result = runner.invoke(init, ["-p"])
        assert result.exit_code == 0
        assert "Project configuration initialized successfully" in result.output
        mock_write.assert_called_once()

    @patch("holodeck.config.manager.ConfigManager.write_config")
    def test_init_prompt_default(self, mock_write, runner):
        """Test prompting for config type (default global)."""
        # Simulate pressing enter for default
        result = runner.invoke(init, input="\n")
        assert result.exit_code == 0
        assert "Initialize global" in result.output
        assert "Global configuration initialized successfully" in result.output
        mock_write.assert_called_once()

    @patch("holodeck.config.manager.ConfigManager.write_config")
    def test_init_prompt_project(self, mock_write, runner):
        """Test prompting for config type (choose project)."""
        result = runner.invoke(init, input="p\n")
        assert result.exit_code == 0
        assert "Project configuration initialized successfully" in result.output
        mock_write.assert_called_once()

    def test_init_existing_file_no_force(self, runner, tmp_path):
        """Test that existing file prompts for overwrite (decline)."""
        # Create a dummy config file
        config_path = tmp_path / "config.yaml"
        config_path.touch()

        with patch(
            "holodeck.config.manager.ConfigManager.get_config_path"
        ) as mock_get_path:
            mock_get_path.return_value = (config_path, "project")

            # Simulate answering 'n' to overwrite
            result = runner.invoke(init, ["-p"], input="n\n")

            assert result.exit_code == 0
            assert "already exists" in result.output
            assert "Initialization cancelled" in result.output

    def test_init_existing_file_force(self, runner, tmp_path):
        """Test that force overwrites existing file."""
        # Create a dummy config file
        config_path = tmp_path / "config.yaml"
        config_path.touch()

        with patch(
            "holodeck.config.manager.ConfigManager.get_config_path"
        ) as mock_get_path:
            mock_get_path.return_value = (config_path, "project")

            # We need to mock write_config to avoid actually writing to the temp file
            # if we want, but since we're using tmp_path it's fine to write.
            # However, the command uses ConfigManager.write_config, so let's verify
            # it's called.
            with patch(
                "holodeck.config.manager.ConfigManager.write_config"
            ) as mock_write:
                result = runner.invoke(init, ["-p", "--force"])

                assert result.exit_code == 0
                assert "Project configuration initialized successfully" in result.output
                mock_write.assert_called_once()
