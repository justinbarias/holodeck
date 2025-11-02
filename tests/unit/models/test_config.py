"""Unit tests for configuration models.

Tests ExecutionConfig validation including timeout constraints,
cache settings, and output mode flags.
"""

import pytest
from pydantic import ValidationError

from holodeck.models.config import ExecutionConfig


class TestExecutionConfig:
    """Tests for ExecutionConfig model."""

    def test_execution_config_all_fields(self) -> None:
        """Test ExecutionConfig with all fields set."""
        config = ExecutionConfig(
            file_timeout=30,
            llm_timeout=60,
            download_timeout=30,
            cache_enabled=True,
            cache_dir=".holodeck/cache",
            verbose=False,
            quiet=False,
        )

        assert config.file_timeout == 30
        assert config.llm_timeout == 60
        assert config.download_timeout == 30
        assert config.cache_enabled is True
        assert config.cache_dir == ".holodeck/cache"
        assert config.verbose is False
        assert config.quiet is False

    def test_execution_config_minimal(self) -> None:
        """Test ExecutionConfig with no fields (all optional)."""
        config = ExecutionConfig()

        assert config.file_timeout is None
        assert config.llm_timeout is None
        assert config.download_timeout is None
        assert config.cache_enabled is None
        assert config.cache_dir is None
        assert config.verbose is None
        assert config.quiet is None

    def test_execution_config_partial(self) -> None:
        """Test ExecutionConfig with some fields set."""
        config = ExecutionConfig(
            file_timeout=30,
            cache_enabled=True,
        )

        assert config.file_timeout == 30
        assert config.cache_enabled is True
        assert config.llm_timeout is None
        assert config.cache_dir is None

    def test_execution_config_file_timeout_valid(self) -> None:
        """Test file_timeout accepts valid values."""
        config = ExecutionConfig(file_timeout=1)
        assert config.file_timeout == 1

        config = ExecutionConfig(file_timeout=300)
        assert config.file_timeout == 300

    def test_execution_config_llm_timeout_valid(self) -> None:
        """Test llm_timeout accepts valid values."""
        config = ExecutionConfig(llm_timeout=1)
        assert config.llm_timeout == 1

        config = ExecutionConfig(llm_timeout=600)
        assert config.llm_timeout == 600

    def test_execution_config_download_timeout_valid(self) -> None:
        """Test download_timeout accepts valid values."""
        config = ExecutionConfig(download_timeout=1)
        assert config.download_timeout == 1

        config = ExecutionConfig(download_timeout=300)
        assert config.download_timeout == 300

    def test_execution_config_cache_dir_string(self) -> None:
        """Test cache_dir accepts string values."""
        config = ExecutionConfig(cache_dir="/custom/cache")
        assert config.cache_dir == "/custom/cache"

        config = ExecutionConfig(cache_dir=".holodeck/cache")
        assert config.cache_dir == ".holodeck/cache"

    def test_execution_config_boolean_flags(self) -> None:
        """Test boolean configuration flags."""
        config = ExecutionConfig(
            cache_enabled=True,
            verbose=True,
            quiet=False,
        )

        assert config.cache_enabled is True
        assert config.verbose is True
        assert config.quiet is False

    def test_execution_config_conflicting_flags(self) -> None:
        """Test that both verbose and quiet can be set (no validation constraint)."""
        # This is allowed - validation would happen at usage time
        config = ExecutionConfig(verbose=True, quiet=True)
        assert config.verbose is True
        assert config.quiet is True

    def test_execution_config_forbids_extra_fields(self) -> None:
        """Test that ExecutionConfig forbids extra fields."""
        with pytest.raises(ValidationError) as exc_info:
            ExecutionConfig(invalid_field=True)  # type: ignore

        assert (
            "extra_forbidden" in str(exc_info.value).lower()
            or "unexpected keyword" in str(exc_info.value).lower()
        )
