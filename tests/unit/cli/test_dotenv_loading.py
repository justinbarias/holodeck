"""Unit tests for CLI dotenv loading functionality.

Tests verify that the HoloDeck CLI correctly loads .env files from:
1. ~/.holodeck/.env (user-level defaults)
2. .env in current working directory (project-level)

Shell environment variables should always take precedence.
"""

import os
from pathlib import Path

import pytest

# Import the _load_dotenv_files function from the cli package
from holodeck.cli import _load_dotenv_files


@pytest.mark.unit
class TestLoadDotenvFiles:
    """Test the _load_dotenv_files function."""

    def test_loads_env_from_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Verify .env in current working directory is loaded."""
        # Create .env file in temp directory
        env_file = tmp_path / ".env"
        env_file.write_text("CWD_TEST_VAR=cwd_value\n")

        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Clear any existing value
        monkeypatch.delenv("CWD_TEST_VAR", raising=False)

        # Call the function directly
        _load_dotenv_files()

        assert os.environ.get("CWD_TEST_VAR") == "cwd_value"

    def test_loads_env_from_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Verify ~/.holodeck/.env is loaded."""
        # Create fake home directory structure
        fake_home = tmp_path / "fake_home"
        holodeck_dir = fake_home / ".holodeck"
        holodeck_dir.mkdir(parents=True)
        env_file = holodeck_dir / ".env"
        env_file.write_text("HOME_TEST_VAR=home_value\n")

        # Mock Path.home() to return our fake home
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        # Create empty CWD (no .env there)
        cwd_path = tmp_path / "project"
        cwd_path.mkdir()
        monkeypatch.chdir(cwd_path)

        # Clear any existing value
        monkeypatch.delenv("HOME_TEST_VAR", raising=False)

        # Call the function directly
        _load_dotenv_files()

        assert os.environ.get("HOME_TEST_VAR") == "home_value"

    def test_shell_vars_take_precedence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Verify existing shell environment variables are not overwritten."""
        # Create .env file with a value
        env_file = tmp_path / ".env"
        env_file.write_text("PRECEDENCE_TEST_VAR=dotenv_value\n")

        monkeypatch.chdir(tmp_path)

        # Set shell env var BEFORE loading dotenv
        monkeypatch.setenv("PRECEDENCE_TEST_VAR", "shell_value")

        _load_dotenv_files()

        # Shell value should take precedence
        assert os.environ.get("PRECEDENCE_TEST_VAR") == "shell_value"

    def test_project_env_overrides_user_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Verify CWD .env values override ~/.holodeck/.env values."""
        # Create fake home directory with .env
        fake_home = tmp_path / "fake_home"
        holodeck_dir = fake_home / ".holodeck"
        holodeck_dir.mkdir(parents=True)
        home_env = holodeck_dir / ".env"
        home_env.write_text("OVERRIDE_TEST_VAR=home_value\n")

        # Create project directory with .env
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        project_env = project_dir / ".env"
        project_env.write_text("OVERRIDE_TEST_VAR=project_value\n")

        # Mock Path.home() and change to project directory
        monkeypatch.setattr(Path, "home", lambda: fake_home)
        monkeypatch.chdir(project_dir)

        # Clear any existing value
        monkeypatch.delenv("OVERRIDE_TEST_VAR", raising=False)

        _load_dotenv_files()

        # Project value should override home value
        assert os.environ.get("OVERRIDE_TEST_VAR") == "project_value"

    def test_missing_files_handled_gracefully(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Verify no errors when .env files don't exist."""
        # Create empty directories (no .env files)
        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        monkeypatch.setattr(Path, "home", lambda: fake_home)
        monkeypatch.chdir(project_dir)

        # Should not raise any exception
        _load_dotenv_files()

    def test_only_home_env_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Verify loading works when only ~/.holodeck/.env exists."""
        # Create home .env only
        fake_home = tmp_path / "fake_home"
        holodeck_dir = fake_home / ".holodeck"
        holodeck_dir.mkdir(parents=True)
        home_env = holodeck_dir / ".env"
        home_env.write_text("HOME_ONLY_VAR=home_only_value\n")

        # Create project directory without .env
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        monkeypatch.setattr(Path, "home", lambda: fake_home)
        monkeypatch.chdir(project_dir)
        monkeypatch.delenv("HOME_ONLY_VAR", raising=False)

        _load_dotenv_files()

        assert os.environ.get("HOME_ONLY_VAR") == "home_only_value"

    def test_only_project_env_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Verify loading works when only project .env exists."""
        # Create empty home directory
        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()

        # Create project .env
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        project_env = project_dir / ".env"
        project_env.write_text("PROJECT_ONLY_VAR=project_only_value\n")

        monkeypatch.setattr(Path, "home", lambda: fake_home)
        monkeypatch.chdir(project_dir)
        monkeypatch.delenv("PROJECT_ONLY_VAR", raising=False)

        _load_dotenv_files()

        assert os.environ.get("PROJECT_ONLY_VAR") == "project_only_value"

    def test_both_envs_merge_without_conflict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Verify non-conflicting vars from both files are loaded."""
        # Create home .env with VAR_A
        fake_home = tmp_path / "fake_home"
        holodeck_dir = fake_home / ".holodeck"
        holodeck_dir.mkdir(parents=True)
        home_env = holodeck_dir / ".env"
        home_env.write_text("MERGE_VAR_A=home_a\n")

        # Create project .env with VAR_B
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        project_env = project_dir / ".env"
        project_env.write_text("MERGE_VAR_B=project_b\n")

        monkeypatch.setattr(Path, "home", lambda: fake_home)
        monkeypatch.chdir(project_dir)
        monkeypatch.delenv("MERGE_VAR_A", raising=False)
        monkeypatch.delenv("MERGE_VAR_B", raising=False)

        _load_dotenv_files()

        # Both vars should be loaded
        assert os.environ.get("MERGE_VAR_A") == "home_a"
        assert os.environ.get("MERGE_VAR_B") == "project_b"
