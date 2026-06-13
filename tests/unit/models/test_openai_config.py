"""Tests for OpenAI Agents SDK configuration models in holodeck.models.openai_config."""

import pytest
from pydantic import ValidationError

from holodeck.models.agent import Agent
from holodeck.models.openai_config import OpenAIConfig, OpenAIPermissionsConfig


@pytest.mark.unit
class TestOpenAIConfigDefaults:
    """Field defaults must match the spec (A1)."""

    def test_empty_config_uses_spec_defaults(self) -> None:
        """An empty `openai:` block carries the spec'd defaults."""
        config = OpenAIConfig()
        assert config.session_memory_estimate_mib == 100
        assert config.max_turns == 20
        assert config.i_understand_this_is_unsafe is False
        assert config.disable_default_hooks is False
        assert config.disable_subprocess_env_scrub is False
        assert config.max_concurrent_sessions is None
        assert config.permissions is None
        assert config.effort is None
        assert config.max_budget_usd is None
        assert config.fallback_model is None
        assert config.disallowed_tools is None

    def test_all_fields_settable(self) -> None:
        """Each declared field accepts a value."""
        config = OpenAIConfig(
            max_concurrent_sessions=10,
            session_memory_estimate_mib=200,
            max_turns=5,
            i_understand_this_is_unsafe=True,
            disable_default_hooks=True,
            disable_subprocess_env_scrub=True,
            permissions=OpenAIPermissionsConfig(
                allowed_tools=["a"], disallowed_tools=["b"]
            ),
            effort="high",
            max_budget_usd=1.5,
            fallback_model="gpt-4o-mini",
            disallowed_tools=["c"],
        )
        assert config.max_concurrent_sessions == 10
        assert config.session_memory_estimate_mib == 200
        assert config.max_turns == 5
        assert config.permissions is not None
        assert config.permissions.allowed_tools == ["a"]
        assert config.effort == "high"
        assert config.max_budget_usd == 1.5
        assert config.fallback_model == "gpt-4o-mini"
        assert config.disallowed_tools == ["c"]


@pytest.mark.unit
class TestOpenAIConfigValidation:
    """Validation rules (extra=forbid, bounds, literals)."""

    def test_unknown_key_rejected(self) -> None:
        """`extra="forbid"` rejects unknown keys."""
        with pytest.raises(ValidationError):
            OpenAIConfig(not_a_field=True)

    def test_permissions_unknown_key_rejected(self) -> None:
        """The permissions sub-block also forbids extras."""
        with pytest.raises(ValidationError):
            OpenAIPermissionsConfig(nope=1)

    @pytest.mark.parametrize("value", ["low", "medium", "high", "max"])
    def test_effort_literals_accepted(self, value: str) -> None:
        """`effort` accepts the spec'd literal set."""
        assert OpenAIConfig(effort=value).effort == value

    def test_effort_invalid_rejected(self) -> None:
        """An out-of-set effort value is rejected."""
        with pytest.raises(ValidationError):
            OpenAIConfig(effort="xhigh")

    def test_max_budget_usd_must_be_positive(self) -> None:
        """`max_budget_usd` must be > 0."""
        with pytest.raises(ValidationError):
            OpenAIConfig(max_budget_usd=0)

    def test_max_turns_must_be_positive(self) -> None:
        """`max_turns` must be >= 1."""
        with pytest.raises(ValidationError):
            OpenAIConfig(max_turns=0)


@pytest.mark.unit
class TestAgentOpenAIBlock:
    """The `openai:` block on the Agent model."""

    def _agent_kwargs(self) -> dict:
        return {
            "name": "a",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "hi"},
        }

    def test_agent_accepts_openai_block(self) -> None:
        """An Agent with an `openai:` block validates."""
        agent = Agent(**self._agent_kwargs(), openai={"max_turns": 7})
        assert agent.openai is not None
        assert agent.openai.max_turns == 7

    def test_agent_openai_defaults_to_none(self) -> None:
        """`openai` is None when omitted."""
        agent = Agent(**self._agent_kwargs())
        assert agent.openai is None

    def test_agent_openai_unknown_key_rejected(self) -> None:
        """Unknown keys inside `openai:` are rejected."""
        with pytest.raises(ValidationError):
            Agent(**self._agent_kwargs(), openai={"bogus": 1})
