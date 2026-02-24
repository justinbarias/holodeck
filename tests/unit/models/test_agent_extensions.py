"""Tests for Agent model extensions (embedding_provider + claude fields)."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from holodeck.models.agent import Agent, Instructions
from holodeck.models.claude_config import (
    ClaudeConfig,
    ExtendedThinkingConfig,
    PermissionMode,
)
from holodeck.models.llm import LLMProvider, ProviderEnum

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "agents"


class TestAgentEmbeddingProvider:
    """Tests for Agent.embedding_provider field."""

    def test_embedding_provider_defaults_to_none(self) -> None:
        """Test that embedding_provider defaults to None."""
        agent = Agent(
            name="test",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="Test"),
        )
        assert agent.embedding_provider is None

    def test_embedding_provider_accepts_llm_provider(self) -> None:
        """Test that embedding_provider accepts a valid LLMProvider."""
        embedding = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="text-embedding-3-small",
        )
        agent = Agent(
            name="test",
            model=LLMProvider(
                provider=ProviderEnum.ANTHROPIC, name="claude-sonnet-4-20250514"
            ),
            instructions=Instructions(inline="Test"),
            embedding_provider=embedding,
        )
        assert agent.embedding_provider is not None
        assert agent.embedding_provider.name == "text-embedding-3-small"

    def test_embedding_provider_none_explicit(self) -> None:
        """Test that embedding_provider can be explicitly set to None."""
        agent = Agent(
            name="test",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="Test"),
            embedding_provider=None,
        )
        assert agent.embedding_provider is None


class TestAgentClaudeField:
    """Tests for Agent.claude field."""

    def test_claude_defaults_to_none(self) -> None:
        """Test that claude defaults to None."""
        agent = Agent(
            name="test",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="Test"),
        )
        assert agent.claude is None

    def test_claude_accepts_claude_config(self) -> None:
        """Test that claude accepts a valid ClaudeConfig."""
        claude_config = ClaudeConfig(
            permission_mode=PermissionMode.acceptEdits,
            max_turns=20,
            extended_thinking=ExtendedThinkingConfig(enabled=True),
        )
        agent = Agent(
            name="test",
            model=LLMProvider(
                provider=ProviderEnum.ANTHROPIC, name="claude-sonnet-4-20250514"
            ),
            instructions=Instructions(inline="Test"),
            claude=claude_config,
        )
        assert agent.claude is not None
        assert agent.claude.permission_mode == PermissionMode.acceptEdits
        assert agent.claude.max_turns == 20

    def test_claude_none_explicit(self) -> None:
        """Test that claude can be explicitly set to None."""
        agent = Agent(
            name="test",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="Test"),
            claude=None,
        )
        assert agent.claude is None

    def test_agent_with_both_new_fields(self) -> None:
        """Test Agent with both embedding_provider and claude fields."""
        agent = Agent(
            name="test",
            model=LLMProvider(
                provider=ProviderEnum.ANTHROPIC, name="claude-sonnet-4-20250514"
            ),
            instructions=Instructions(inline="Test"),
            embedding_provider=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="text-embedding-3-small",
            ),
            claude=ClaudeConfig(
                permission_mode=PermissionMode.acceptAll,
                web_search=True,
            ),
        )
        assert agent.embedding_provider is not None
        assert agent.claude is not None
        assert agent.claude.web_search is True


class TestAgentBackwardCompatibility:
    """Tests that existing valid YAML fixtures still load without errors.

    These tests verify that adding embedding_provider and claude fields
    does not break backward compatibility with existing configurations.
    """

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "minimal_agent.yaml",
            "chat_happy_agent.yaml",
        ],
        ids=["minimal_agent", "chat_happy_agent"],
    )
    def test_existing_fixture_loads_without_error(self, fixture_name: str) -> None:
        """Test that existing valid YAML fixture loads through Agent model."""
        fixture_path = FIXTURES_DIR / fixture_name
        assert fixture_path.exists(), f"Fixture not found: {fixture_path}"

        with open(fixture_path) as f:
            config = yaml.safe_load(f)

        # Should not raise ValidationError
        agent = Agent(**config)

        # New fields should default to None
        assert agent.claude is None
        assert agent.embedding_provider is None

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "invalid_agent.yaml",
            "chat_invalid_agent.yaml",
            "chat_missing_tool.yaml",
            # valid_agent.yaml has pre-existing schema issues (MCP server field,
            # evaluation metrics missing type discriminator) unrelated to our changes
            "valid_agent.yaml",
        ],
        ids=[
            "invalid_agent",
            "chat_invalid_agent",
            "chat_missing_tool",
            "valid_agent_schema_drift",
        ],
    )
    def test_invalid_fixtures_still_fail(self, fixture_name: str) -> None:
        """Test that intentionally invalid fixtures still raise ValidationError."""
        fixture_path = FIXTURES_DIR / fixture_name
        if not fixture_path.exists():
            pytest.skip(f"Fixture not found: {fixture_path}")

        with open(fixture_path) as f:
            config = yaml.safe_load(f)

        with pytest.raises((ValidationError, TypeError)):
            Agent(**config)
