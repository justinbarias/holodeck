"""Shared fixtures for the memory unit tests."""

from __future__ import annotations

import pytest

# Variables that change which provider or credential a model client uses.
# A developer's shell often has some of them set; no unit test may depend on
# that, and the OAuth path refuses to start when one is present.
_ROUTING_VARIABLES = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_VERTEX",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "AZURE_OPENAI_API_KEY",
)


@pytest.fixture(autouse=True)
def _model_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ROUTING_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-unit-not-real")
