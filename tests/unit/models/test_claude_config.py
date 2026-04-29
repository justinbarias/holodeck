"""Tests for Claude-specific configuration models in holodeck.models.claude_config."""

import warnings
from pathlib import Path

import pytest
from pydantic import ValidationError

from holodeck.config.context import agent_base_dir
from holodeck.models.claude_config import (
    KNOWN_BUILTIN_TOOLS,
    AuthProvider,
    BashConfig,
    ClaudeConfig,
    ExtendedThinkingConfig,
    FileSystemConfig,
    PermissionMode,
    SubagentSpec,
)

# ---------------------------------------------------------------------------
# Foundational tests — T002 and T003 (written first, must FAIL before T008/T007)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFoundationalSubagentMigration:
    """Foundational tests for SubagentSpec / ClaudeConfig.agents scaffolding."""

    @pytest.mark.parametrize("value", ["", "   "], ids=["empty", "whitespace"])
    def test_subagent_description_required_non_empty(self, value: str) -> None:
        """T003a: description must be non-empty after strip with the spec'd message.

        Traces to FR-004, SC-004, quickstart.md §5.
        """
        with pytest.raises(ValidationError, match="subagent requires description"):
            SubagentSpec(description=value, prompt="x")

    def test_legacy_subagents_block_rejected(self) -> None:
        """T002: Loading claude.subagents produces a targeted migration error.

        Traces to SC-005, FR-011.
        """
        with pytest.raises(ValidationError) as exc_info:
            ClaudeConfig(**{"subagents": {"enabled": True}})
        error_text = str(exc_info.value)
        assert "claude.subagents" in error_text
        assert "no longer supported" in error_text
        assert "claude.agents" in error_text
        assert "execution.parallel_test_cases" in error_text

    def test_claude_config_agents_empty_map_normalized_to_none(self) -> None:
        """T003: ClaudeConfig(agents={}).agents is None after normalization.

        Traces to FR-010, edge case "empty agents map", research.md §5.
        """
        config = ClaudeConfig(agents={})
        assert config.agents is None


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
        assert config.agents is None
        assert config.allowed_tools is None
        assert config.effort is None
        assert config.max_budget_usd is None
        assert config.fallback_model is None
        assert config.disallowed_tools is None

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
            allowed_tools=["bash", "file_read", "web_search"],
        )
        assert config.working_directory == "/home/user/workspace"
        assert config.permission_mode == PermissionMode.acceptAll
        assert config.max_turns == 10
        assert config.extended_thinking.enabled is True
        assert config.web_search is True
        assert config.bash.enabled is True
        assert config.file_system.read is True
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

    @pytest.mark.unit
    def test_max_concurrent_sessions_default_is_10(self) -> None:
        """T012: Test that max_concurrent_sessions defaults to 10."""
        config = ClaudeConfig()
        assert config.max_concurrent_sessions == 10

    @pytest.mark.unit
    def test_max_concurrent_sessions_valid_value(self) -> None:
        """T012: Test that max_concurrent_sessions accepts a valid value."""
        config = ClaudeConfig(max_concurrent_sessions=50)
        assert config.max_concurrent_sessions == 50

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("value", "reason"),
        [
            (0, "below_min"),
            (101, "above_max"),
        ],
        ids=["below_min_ge1", "above_max_le100"],
    )
    def test_max_concurrent_sessions_out_of_range(
        self, value: int, reason: str
    ) -> None:
        """T012: Test that max_concurrent_sessions outside 1-100 is rejected."""
        with pytest.raises(ValidationError):
            ClaudeConfig(max_concurrent_sessions=value)

    @pytest.mark.unit
    def test_max_concurrent_sessions_none_is_valid(self) -> None:
        """T012: Test that max_concurrent_sessions accepts None."""
        config = ClaudeConfig(max_concurrent_sessions=None)
        assert config.max_concurrent_sessions is None

    def test_extra_fields_rejected(self) -> None:
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            ClaudeConfig(unknown_setting="value")

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "value", ["low", "medium", "high", "max"], ids=["low", "medium", "high", "max"]
    )
    def test_effort_valid_values(self, value: str) -> None:
        """Test effort accepts the four SDK literal values."""
        config = ClaudeConfig(effort=value)
        assert config.effort == value

    @pytest.mark.unit
    def test_effort_invalid_value_rejected(self) -> None:
        """Test effort rejects values outside the SDK literal set."""
        with pytest.raises(ValidationError):
            ClaudeConfig(effort="extreme")

    @pytest.mark.unit
    def test_max_budget_usd_positive_accepted(self) -> None:
        """Test max_budget_usd accepts positive floats."""
        config = ClaudeConfig(max_budget_usd=5.0)
        assert config.max_budget_usd == 5.0

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "value", [0, -1.0, -0.01], ids=["zero", "negative_one", "negative_small"]
    )
    def test_max_budget_usd_non_positive_rejected(self, value: float) -> None:
        """Test max_budget_usd rejects 0 and negative values."""
        with pytest.raises(ValidationError):
            ClaudeConfig(max_budget_usd=value)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "value",
        ["haiku", "sonnet", "claude-haiku-4-5"],
        ids=["literal_haiku", "literal_sonnet", "full_model_id"],
    )
    def test_fallback_model_accepts_any_string(self, value: str) -> None:
        """Test fallback_model has no SDK-side literal restriction."""
        config = ClaudeConfig(fallback_model=value)
        assert config.fallback_model == value

    @pytest.mark.unit
    def test_disallowed_tools_list_accepted(self) -> None:
        """Test disallowed_tools accepts a list of tool names."""
        config = ClaudeConfig(disallowed_tools=["Bash", "Write"])
        assert config.disallowed_tools == ["Bash", "Write"]

    @pytest.mark.unit
    def test_disallowed_tools_empty_list_accepted(self) -> None:
        """Test disallowed_tools accepts [] (equivalent to omitted at SDK layer)."""
        config = ClaudeConfig(disallowed_tools=[])
        assert config.disallowed_tools == []

    @pytest.mark.unit
    def test_effort_with_extended_thinking_warns(self) -> None:
        """FR-009: Setting both effort and extended_thinking emits a warning."""
        with pytest.warns(UserWarning, match=r"effort.*extended_thinking"):
            ClaudeConfig(
                effort="high",
                extended_thinking=ExtendedThinkingConfig(
                    enabled=True, budget_tokens=20_000
                ),
            )

    @pytest.mark.unit
    def test_effort_alone_does_not_warn(self) -> None:
        """effort without extended_thinking should not warn."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Any warning becomes an error
            ClaudeConfig(effort="high")

    @pytest.mark.unit
    def test_extended_thinking_disabled_does_not_warn(self) -> None:
        """extended_thinking with enabled=False should not warn even with effort."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ClaudeConfig(
                effort="high",
                extended_thinking=ExtendedThinkingConfig(enabled=False),
            )


# ---------------------------------------------------------------------------
# US1 tests — T012 and T013 (SubagentSpec model round-trip + literal validation)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSubagentSpec:
    """US1 tests for SubagentSpec model fields and validation."""

    def test_subagent_spec_round_trip_minimal(self) -> None:
        """T012 (US1): SubagentSpec round-trips description and prompt correctly.

        Traces to FR-001, FR-002, US1 acceptance scenario 1.
        """
        spec = SubagentSpec(description="x", prompt="y")
        assert spec.description == "x"
        assert spec.prompt == "y"
        assert spec.tools is None
        assert spec.model is None

    @pytest.mark.parametrize(
        "model_value",
        ["sonnet", "opus", "haiku", "inherit", "claude-3-5-sonnet-20241022"],
        ids=["sonnet", "opus", "haiku", "inherit", "full_model_id"],
    )
    def test_subagent_spec_model_passes_through(self, model_value: str) -> None:
        """T013 (US1): SubagentSpec accepts any string for model and passes it through.

        The SDK is the source of truth for valid model values; HoloDeck does not
        gate on a fixed literal set so the surface stays compatible as the SDK
        adds new model aliases. Traces to FR-008, US1 acceptance scenario 2.
        """
        spec = SubagentSpec(
            description="agent", prompt="Do something.", model=model_value
        )
        assert spec.model == model_value

    def test_subagent_spec_description_empty_rejected(self) -> None:
        """description must be non-empty after strip (data-model.md rule 4)."""
        with pytest.raises(ValidationError):
            SubagentSpec(description="   ", prompt="Some prompt.")

    def test_subagent_spec_extra_fields_rejected(self) -> None:
        """SubagentSpec rejects unknown extra fields (extra='forbid')."""
        with pytest.raises(ValidationError):
            SubagentSpec(description="agent", prompt="Hello.", unknown_field="oops")

    def test_subagent_spec_tools_list_accepted(self) -> None:
        """SubagentSpec accepts a tools allowlist."""
        spec = SubagentSpec(
            description="researcher",
            prompt="Search the web.",
            tools=["WebSearch", "WebFetch"],
        )
        assert spec.tools == ["WebSearch", "WebFetch"]

    def test_subagent_spec_tools_none_means_inherit(self) -> None:
        """Omitting tools defaults to None (subagent inherits all parent tools).

        Traces to FR-007.
        """
        spec = SubagentSpec(description="writer", prompt="Write the report.")
        assert spec.tools is None


# ---------------------------------------------------------------------------
# KNOWN_BUILTIN_TOOLS constant (foundational — needed by US2 tool-name warnings)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKnownBuiltinTools:
    """KNOWN_BUILTIN_TOOLS module constant (data-model.md rule #5)."""

    def test_known_builtin_tools_is_frozenset(self) -> None:
        """KNOWN_BUILTIN_TOOLS must be a frozenset."""
        assert isinstance(KNOWN_BUILTIN_TOOLS, frozenset)

    def test_known_builtin_tools_exact_members(self) -> None:
        """KNOWN_BUILTIN_TOOLS must match the canonical set from data-model.md."""
        expected = frozenset(
            {
                "Read",
                "Write",
                "Edit",
                "Bash",
                "Glob",
                "Grep",
                "WebSearch",
                "WebFetch",
                "Task",
                "TodoWrite",
                "NotebookEdit",
            }
        )
        assert expected == KNOWN_BUILTIN_TOOLS


# ---------------------------------------------------------------------------
# T008 — SubagentSpec tool-name typo warning (US2)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSubagentSpecToolWarning:
    """T008: SubagentSpec emits UserWarning for unknown tool names.

    Traces to data-model.md rule #5, research.md §3.
    """

    def test_no_warning_for_known_builtins(self) -> None:
        """No warning when tools are valid built-in names (e.g. Read, WebSearch)."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            SubagentSpec(
                description="analyst",
                prompt="You are an analyst.",
                tools=["Read", "WebSearch"],
            )

    def test_no_warning_for_mcp_prefix(self) -> None:
        """No warning when all tools have the mcp__ prefix."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            SubagentSpec(
                description="db analyst",
                prompt="Query the database.",
                tools=["mcp__db__query", "mcp__db__describe"],
            )

    def test_warning_for_unknown_bare_name(self) -> None:
        """UserWarning is emitted for an unknown bare name (e.g. WebSerach — typo)."""
        with pytest.warns(UserWarning) as rec:
            SubagentSpec(
                description="researcher",
                prompt="Research the topic.",
                tools=["WebSerach"],
            )
        messages = [str(w.message) for w in rec.list]
        assert any("WebSerach" in m for m in messages)
        # Warning must reference all three accepted patterns
        combined = " ".join(messages)
        assert "mcp__" in combined or "MCP" in combined or "mcp" in combined
        assert any(
            builtin in combined
            for builtin in ("WebSearch", "Read", "Write", "Bash", "Glob")
        )

    def test_no_warning_when_tools_is_none(self) -> None:
        """No warning when tools is None (FR-007 inheritance path)."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            SubagentSpec(
                description="inheriting subagent",
                prompt="Inherit all parent tools.",
                tools=None,
            )

    def test_no_warning_when_tools_is_empty_list(self) -> None:
        """No warning when tools is [] (pure-reasoning subagent per quickstart §7)."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            SubagentSpec(
                description="pure reasoner",
                prompt="Think carefully without tools.",
                tools=[],
            )


# ---------------------------------------------------------------------------
# US3 tests — T010-T016 (prompt sourcing & validation)
# ---------------------------------------------------------------------------


_FIXTURE_PROMPTS_DIR = Path(__file__).parent / "fixtures" / "subagent_prompts"


@pytest.mark.unit
class TestSubagentPromptSourcing:
    """US3: prompt / prompt_file sourcing and validation rules."""

    def test_subagent_inline_prompt_loads(self) -> None:
        """T010: Inline prompt round-trips and prompt_file stays None."""
        spec = SubagentSpec(description="x", prompt="You are a financial analyst.")
        assert spec.prompt == "You are a financial analyst."
        assert spec.prompt_file is None

    def test_subagent_inline_prompt_empty_after_strip_rejected(self) -> None:
        """T011: Whitespace-only inline prompt is rejected."""
        with pytest.raises(ValidationError):
            SubagentSpec(description="x", prompt="   ")

    def test_subagent_prompt_file_inlined(self) -> None:
        """T012: prompt_file contents inline into prompt; prompt_file becomes None."""
        token = agent_base_dir.set(str(_FIXTURE_PROMPTS_DIR))
        try:
            spec = SubagentSpec(description="x", prompt_file="./analyst.md")
        finally:
            agent_base_dir.reset(token)
        expected = (_FIXTURE_PROMPTS_DIR / "analyst.md").read_text(encoding="utf-8")
        assert spec.prompt == expected
        assert spec.prompt_file is None

    def test_subagent_prompt_file_resolves_relative_to_agent_base_dir(
        self, tmp_path: Path
    ) -> None:
        """T013: relative paths resolve under agent_base_dir; absolute paths pass."""
        nested = tmp_path / "subdir"
        nested.mkdir()
        prompt_file = nested / "analyst.md"
        prompt_file.write_text("nested prompt body", encoding="utf-8")

        # Relative path under base_dir
        token = agent_base_dir.set(str(tmp_path))
        try:
            spec_rel = SubagentSpec(description="x", prompt_file="subdir/analyst.md")
        finally:
            agent_base_dir.reset(token)
        assert spec_rel.prompt == "nested prompt body"
        assert spec_rel.prompt_file is None

        # Absolute path is honoured even if base_dir is unrelated
        unrelated_base = tmp_path / "unrelated"
        unrelated_base.mkdir()
        token = agent_base_dir.set(str(unrelated_base))
        try:
            spec_abs = SubagentSpec(description="x", prompt_file=str(prompt_file))
        finally:
            agent_base_dir.reset(token)
        assert spec_abs.prompt == "nested prompt body"
        assert spec_abs.prompt_file is None

    def test_subagent_prompt_file_not_found_rejected(self, tmp_path: Path) -> None:
        """T014: Missing prompt_file produces a ValidationError naming the path."""
        token = agent_base_dir.set(str(tmp_path))
        try:
            with pytest.raises(ValidationError) as exc_info:
                SubagentSpec(description="x", prompt_file="./does-not-exist.md")
        finally:
            agent_base_dir.reset(token)
        message = str(exc_info.value)
        assert "prompt_file not found" in message
        assert "does-not-exist.md" in message

    def test_subagent_prompt_and_prompt_file_mutually_exclusive(self) -> None:
        """T015: Setting both prompt and prompt_file raises the documented error."""
        with pytest.raises(
            ValidationError, match="prompt and prompt_file are mutually exclusive"
        ):
            SubagentSpec(description="x", prompt="hi", prompt_file="./analyst.md")

    def test_subagent_neither_prompt_nor_prompt_file_rejected(self) -> None:
        """T016: Omitting both prompt and prompt_file raises the documented error."""
        with pytest.raises(
            ValidationError,
            match="subagent requires either prompt or prompt_file",
        ):
            SubagentSpec(description="x")
