"""Tests for Claude-specific configuration models in holodeck.models.claude_config."""

import pytest
from pydantic import ValidationError

from holodeck.models.claude_config import (
    AuthProvider,
    BashConfig,
    ClaudeConfig,
    ExtendedThinkingConfig,
    FileSystemConfig,
    PermissionMode,
    SubagentConfig,
)


class TestAuthProvider:
    """Tests for AuthProvider enum."""

    @pytest.mark.parametrize(
        "value",
        ["api_key", "oauth_token", "bedrock", "vertex", "foundry"],
        ids=["api_key", "oauth_token", "bedrock", "vertex", "foundry"],
    )
    def test_auth_provider_valid_values(self, value: str) -> None:
        """Test that all expected AuthProvider values are accepted."""
        provider = AuthProvider(value)
        assert provider.value == value

    def test_auth_provider_invalid_value(self) -> None:
        """Test that invalid AuthProvider value is rejected."""
        with pytest.raises(ValueError):
            AuthProvider("invalid")

    def test_auth_provider_is_str_enum(self) -> None:
        """Test that AuthProvider values can be used as strings."""
        assert str(AuthProvider.api_key) == "AuthProvider.api_key"
        assert AuthProvider.api_key.value == "api_key"


class TestPermissionMode:
    """Tests for PermissionMode enum."""

    @pytest.mark.parametrize(
        "value",
        ["manual", "acceptEdits", "acceptAll"],
        ids=["manual", "acceptEdits", "acceptAll"],
    )
    def test_permission_mode_valid_values(self, value: str) -> None:
        """Test that all expected PermissionMode values are accepted."""
        mode = PermissionMode(value)
        assert mode.value == value

    def test_permission_mode_invalid_value(self) -> None:
        """Test that invalid PermissionMode value is rejected."""
        with pytest.raises(ValueError):
            PermissionMode("invalid")


class TestExtendedThinkingConfig:
    """Tests for ExtendedThinkingConfig model."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = ExtendedThinkingConfig()
        assert config.enabled is False
        assert config.budget_tokens == 10_000

    def test_enabled_with_custom_budget(self) -> None:
        """Test enabling with custom budget."""
        config = ExtendedThinkingConfig(enabled=True, budget_tokens=50_000)
        assert config.enabled is True
        assert config.budget_tokens == 50_000

    @pytest.mark.parametrize(
        "budget",
        [999, 100_001],
        ids=["below_min", "above_max"],
    )
    def test_budget_tokens_out_of_range(self, budget: int) -> None:
        """Test that budget_tokens outside 1000-100000 is rejected."""
        with pytest.raises(ValidationError):
            ExtendedThinkingConfig(budget_tokens=budget)

    @pytest.mark.parametrize(
        "budget",
        [1_000, 100_000],
        ids=["at_min", "at_max"],
    )
    def test_budget_tokens_at_boundaries(self, budget: int) -> None:
        """Test that budget_tokens at boundaries is accepted."""
        config = ExtendedThinkingConfig(budget_tokens=budget)
        assert config.budget_tokens == budget

    def test_extra_fields_rejected(self) -> None:
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            ExtendedThinkingConfig(enabled=True, unknown_field="value")


class TestBashConfig:
    """Tests for BashConfig model."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = BashConfig()
        assert config.enabled is False
        assert config.excluded_commands == []
        assert config.allow_unsafe is False

    def test_with_excluded_commands(self) -> None:
        """Test with excluded commands list."""
        config = BashConfig(
            enabled=True,
            excluded_commands=["rm -rf", "sudo"],
        )
        assert config.enabled is True
        assert config.excluded_commands == ["rm -rf", "sudo"]

    def test_extra_fields_rejected(self) -> None:
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            BashConfig(extra_field="value")


class TestFileSystemConfig:
    """Tests for FileSystemConfig model."""

    def test_defaults(self) -> None:
        """Test default values (all disabled)."""
        config = FileSystemConfig()
        assert config.read is False
        assert config.write is False
        assert config.edit is False

    def test_all_enabled(self) -> None:
        """Test with all flags enabled."""
        config = FileSystemConfig(read=True, write=True, edit=True)
        assert config.read is True
        assert config.write is True
        assert config.edit is True

    def test_partial_flags(self) -> None:
        """Test with partial flags enabled."""
        config = FileSystemConfig(read=True, write=False, edit=True)
        assert config.read is True
        assert config.write is False
        assert config.edit is True

    def test_extra_fields_rejected(self) -> None:
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            FileSystemConfig(execute=True)


class TestSubagentConfig:
    """Tests for SubagentConfig model."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = SubagentConfig()
        assert config.enabled is False
        assert config.max_parallel == 4

    def test_custom_max_parallel(self) -> None:
        """Test with custom max_parallel."""
        config = SubagentConfig(enabled=True, max_parallel=8)
        assert config.max_parallel == 8

    @pytest.mark.parametrize(
        "max_parallel",
        [0, 17],
        ids=["below_min", "above_max"],
    )
    def test_max_parallel_out_of_range(self, max_parallel: int) -> None:
        """Test that max_parallel outside 1-16 is rejected."""
        with pytest.raises(ValidationError):
            SubagentConfig(max_parallel=max_parallel)

    @pytest.mark.parametrize(
        "max_parallel",
        [1, 16],
        ids=["at_min", "at_max"],
    )
    def test_max_parallel_at_boundaries(self, max_parallel: int) -> None:
        """Test that max_parallel at boundaries is accepted."""
        config = SubagentConfig(max_parallel=max_parallel)
        assert config.max_parallel == max_parallel

    def test_extra_fields_rejected(self) -> None:
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            SubagentConfig(unknown="value")


class TestClaudeConfig:
    """Tests for ClaudeConfig model."""

    def test_minimal_defaults(self) -> None:
        """Test that ClaudeConfig can be created with all defaults."""
        config = ClaudeConfig()
        assert config.working_directory is None
        assert config.permission_mode == PermissionMode.manual
        assert config.max_turns is None
        assert config.extended_thinking is None
        assert config.web_search is False
        assert config.bash is None
        assert config.file_system is None
        assert config.subagents is None
        assert config.allowed_tools is None

    def test_with_extended_thinking(self) -> None:
        """Test ClaudeConfig with extended thinking configured."""
        config = ClaudeConfig(
            extended_thinking=ExtendedThinkingConfig(
                enabled=True, budget_tokens=20_000
            ),
        )
        assert config.extended_thinking is not None
        assert config.extended_thinking.enabled is True
        assert config.extended_thinking.budget_tokens == 20_000

    def test_with_bash(self) -> None:
        """Test ClaudeConfig with bash configured."""
        config = ClaudeConfig(
            bash=BashConfig(enabled=True, excluded_commands=["rm -rf"]),
        )
        assert config.bash is not None
        assert config.bash.enabled is True
        assert config.bash.excluded_commands == ["rm -rf"]

    def test_with_file_system(self) -> None:
        """Test ClaudeConfig with file system configured."""
        config = ClaudeConfig(
            file_system=FileSystemConfig(read=True, write=True, edit=True),
        )
        assert config.file_system is not None
        assert config.file_system.read is True
        assert config.file_system.write is True
        assert config.file_system.edit is True

    def test_with_subagents(self) -> None:
        """Test ClaudeConfig with subagents configured."""
        config = ClaudeConfig(
            subagents=SubagentConfig(enabled=True, max_parallel=8),
        )
        assert config.subagents is not None
        assert config.subagents.enabled is True
        assert config.subagents.max_parallel == 8

    def test_with_all_fields(self) -> None:
        """Test ClaudeConfig with all fields populated."""
        config = ClaudeConfig(
            working_directory="/home/user/workspace",
            permission_mode=PermissionMode.acceptAll,
            max_turns=10,
            extended_thinking=ExtendedThinkingConfig(
                enabled=True, budget_tokens=50_000
            ),
            web_search=True,
            bash=BashConfig(enabled=True, excluded_commands=["sudo"]),
            file_system=FileSystemConfig(read=True, write=True, edit=True),
            subagents=SubagentConfig(enabled=True, max_parallel=16),
            allowed_tools=["bash", "file_read", "web_search"],
        )
        assert config.working_directory == "/home/user/workspace"
        assert config.permission_mode == PermissionMode.acceptAll
        assert config.max_turns == 10
        assert config.extended_thinking.enabled is True
        assert config.web_search is True
        assert config.bash.enabled is True
        assert config.file_system.read is True
        assert config.subagents.max_parallel == 16
        assert len(config.allowed_tools) == 3

    def test_max_turns_must_be_positive(self) -> None:
        """Test that max_turns must be >= 1."""
        with pytest.raises(ValidationError):
            ClaudeConfig(max_turns=0)

    def test_max_turns_at_minimum(self) -> None:
        """Test that max_turns=1 is accepted."""
        config = ClaudeConfig(max_turns=1)
        assert config.max_turns == 1

    def test_permission_mode_default_is_manual(self) -> None:
        """Test that permission_mode defaults to manual (least-privilege)."""
        config = ClaudeConfig()
        assert config.permission_mode == PermissionMode.manual

    def test_extra_fields_rejected(self) -> None:
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            ClaudeConfig(unknown_setting="value")
