"""Keep credential values out of text that leaves the Graphiti backend.

The values come from where the clients really get them: the YAML, the
environment variable a client falls back to, and a password written inside a
connection URI. They are removed from error text the backend returns, and
from log records that Graphiti's own modules emit before the backend sees the
exception.
"""

from __future__ import annotations

import contextlib
import logging
import os
from urllib.parse import urlsplit

from holodeck.models.claude_config import AuthProvider
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import GraphBackendConfig

REDACTED = "***"
MIN_SECRET_LENGTH = 4
"""Shorter values are not replaced: they would blank ordinary words."""

GRAPHITI_LOGGER_PREFIX = "graphiti_core"


def credential_env_var(model: LLMProvider) -> str | None:
    """The environment variable the extraction client reads its credential from."""
    if model.provider == ProviderEnum.OPENAI:
        return "OPENAI_API_KEY"
    if model.provider == ProviderEnum.AZURE_OPENAI:
        return "AZURE_OPENAI_API_KEY"
    if model.provider == ProviderEnum.ANTHROPIC:
        if model.auth_provider == AuthProvider.oauth_token:
            return "CLAUDE_CODE_OAUTH_TOKEN"
        return "ANTHROPIC_API_KEY"
    return None


def credential_values(model: LLMProvider, config: GraphBackendConfig) -> frozenset[str]:
    """Every credential value the backend's clients can use."""
    candidates: list[str | None] = [
        model.api_key.get_secret_value() if model.api_key is not None else None,
        config.password.get_secret_value() if config.password is not None else None,
    ]
    env_var = credential_env_var(model)
    if env_var is not None:
        candidates.append(os.environ.get(env_var))
    if config.uri:
        with contextlib.suppress(ValueError):
            candidates.append(urlsplit(config.uri).password)
    return frozenset(
        value for value in candidates if value and len(value) >= MIN_SECRET_LENGTH
    )


def redact(text: str, secrets: frozenset[str]) -> str:
    """Return ``text`` with every secret replaced. Longest first."""
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, REDACTED)
    return text


class SecretRedactingFilter(logging.Filter):
    """Replace credential values in a log record's message and traceback."""

    def __init__(self, secrets: frozenset[str]) -> None:
        super().__init__()
        self.secrets = secrets

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.secrets:
            return True
        record.msg = redact(record.getMessage(), self.secrets)
        record.args = None
        if record.exc_info:
            formatted = logging.Formatter().formatException(record.exc_info)
            record.exc_text = redact(formatted, self.secrets)
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = redact(record.exc_text, self.secrets)
        return True


def install_log_redaction(secrets: frozenset[str]) -> None:
    """Attach the filter to every Graphiti logger that exists now.

    A logger's filters do not apply to its children, and Graphiti logs from
    one logger per module, so each one gets the filter. Call this after the
    Graphiti modules are imported. A second call adds the new secrets.
    """
    manager = logging.root.manager
    names = [GRAPHITI_LOGGER_PREFIX] + [
        name
        for name in manager.loggerDict
        if name.startswith(f"{GRAPHITI_LOGGER_PREFIX}.")
    ]
    for name in names:
        target = logging.getLogger(name)
        existing = next(
            (f for f in target.filters if isinstance(f, SecretRedactingFilter)), None
        )
        if existing is None:
            target.addFilter(SecretRedactingFilter(secrets))
        else:
            existing.secrets = existing.secrets | secrets
