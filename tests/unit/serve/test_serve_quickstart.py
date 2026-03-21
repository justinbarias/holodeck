"""Quickstart validation tests for US1 (T023).

Validates that the quickstart YAML from the spec parses correctly
through the Agent Pydantic model.
"""

import pytest
import yaml

from holodeck.models.agent import Agent
from holodeck.models.llm import ProviderEnum


@pytest.mark.unit
class TestServeQuickstart:
    """T023: Validate quickstart Scenario 1 config."""

    def test_quickstart_yaml_parses_through_agent_model(self) -> None:
        """T023: Quickstart YAML validates through Agent Pydantic model."""
        # The quickstart YAML from specs/024-claude-serve-deploy/quickstart.md
        quickstart_yaml = """
name: claude-assistant
model:
  provider: anthropic
  name: claude-sonnet-4-20250514
instructions:
  inline: "You are a helpful assistant."
claude:
  permission_mode: acceptAll
  max_concurrent_sessions: 5
"""
        config = yaml.safe_load(quickstart_yaml)
        agent = Agent(**config)

        assert agent.model.provider == ProviderEnum.ANTHROPIC
        assert agent.claude is not None
        assert agent.claude.max_concurrent_sessions == 5
        assert agent.model.name == "claude-sonnet-4-20250514"
