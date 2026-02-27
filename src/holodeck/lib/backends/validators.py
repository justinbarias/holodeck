"""Startup validators for Claude backend initialization.

These validators are called by ClaudeBackend.initialize() before spawning
the Claude subprocess to surface configuration errors at startup, not at
runtime.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from holodeck.config.env_loader import get_env_var
from holodeck.lib.errors import ConfigError
from holodeck.models.agent import Agent
from holodeck.models.claude_config import AuthProvider
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import HierarchicalDocumentToolConfig, VectorstoreTool

logger = logging.getLogger(__name__)

_BEDROCK_REGION_ENV_CANDIDATES = ("AWS_REGION", "AWS_DEFAULT_REGION")
_VERTEX_PROJECT_ENV_CANDIDATES = (
    "ANTHROPIC_VERTEX_PROJECT_ID",
    "GCLOUD_PROJECT",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_APPLICATION_CREDENTIALS",
)
_FOUNDRY_TARGET_ENV_CANDIDATES = (
    "ANTHROPIC_FOUNDRY_RESOURCE",
    "ANTHROPIC_FOUNDRY_BASE_URL",
)


def _get_required_env_var(name: str, error_message: str) -> str:
    """Return a required env var value or raise ConfigError."""
    value = get_env_var(name)
    if value is None or not str(value).strip():
        raise ConfigError(name, error_message)
    return str(value)


def _get_first_present_env_var(candidates: tuple[str, ...]) -> tuple[str, str] | None:
    """Return first non-empty env var value from candidates."""
    for key in candidates:
        value = get_env_var(key)
        if value is not None and str(value).strip():
            return key, str(value)
    return None


def validate_nodejs() -> None:
    """Validate that Node.js is available on PATH.

    Claude Agent SDK requires Node.js to spawn its subprocess.

    Raises:
        ConfigError: If node is not found on PATH.
    """
    if shutil.which("node") is None:
        raise ConfigError(
            "nodejs",
            "Node.js is required to run Claude Agent SDK but was not found on "
            "PATH. Install Node.js from https://nodejs.org/ and ensure it is "
            "on your PATH.",
        )


def validate_credentials(model: LLMProvider) -> dict[str, str]:
    """Validate authentication credentials for the LLM provider.

    Checks that the required environment variables are present for the
    configured auth_provider, including cloud routing context for Bedrock,
    Vertex, and Foundry. Returns a dict of environment variables to inject
    into the Claude subprocess.

    Args:
        model: LLM provider configuration.

    Returns:
        Dict of environment variables to set for the subprocess.

    Raises:
        ConfigError: If required credentials are absent.
    """
    auth = model.auth_provider or AuthProvider.api_key

    if auth == AuthProvider.api_key:
        key = _get_required_env_var(
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it with: export ANTHROPIC_API_KEY=sk-ant-...",
        )
        return {"ANTHROPIC_API_KEY": key}

    if auth == AuthProvider.oauth_token:
        token = _get_required_env_var(
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN environment variable is not set. "
            "Run `claude setup-token` to authenticate with OAuth.",
        )
        return {"CLAUDE_CODE_OAUTH_TOKEN": token}

    if auth == AuthProvider.bedrock:
        bedrock_region = _get_first_present_env_var(_BEDROCK_REGION_ENV_CANDIDATES)
        if bedrock_region is None:
            raise ConfigError(
                "AWS_REGION",
                "Missing AWS region for auth_provider: bedrock. Set either "
                "AWS_REGION or AWS_DEFAULT_REGION (for example: "
                "export AWS_REGION=us-east-1).",
            )
        return {
            "CLAUDE_CODE_USE_BEDROCK": "1",
            bedrock_region[0]: bedrock_region[1],
        }

    if auth == AuthProvider.vertex:
        region = _get_required_env_var(
            "CLOUD_ML_REGION",
            "CLOUD_ML_REGION environment variable is not set for "
            "auth_provider: vertex. Set it with: "
            "export CLOUD_ML_REGION=us-east5",
        )
        project_context = _get_first_present_env_var(_VERTEX_PROJECT_ENV_CANDIDATES)
        if project_context is None:
            raise ConfigError(
                "ANTHROPIC_VERTEX_PROJECT_ID",
                "Missing Vertex project context for auth_provider: vertex. "
                "Set one of: ANTHROPIC_VERTEX_PROJECT_ID, GCLOUD_PROJECT, "
                "GOOGLE_CLOUD_PROJECT, or GOOGLE_APPLICATION_CREDENTIALS.",
            )
        return {
            "CLAUDE_CODE_USE_VERTEX": "1",
            "CLOUD_ML_REGION": region,
            project_context[0]: project_context[1],
        }

    # AuthProvider.foundry
    foundry_target = _get_first_present_env_var(_FOUNDRY_TARGET_ENV_CANDIDATES)
    if foundry_target is None:
        raise ConfigError(
            "ANTHROPIC_FOUNDRY_RESOURCE",
            "Missing Foundry target for auth_provider: foundry. "
            "Set one of: ANTHROPIC_FOUNDRY_RESOURCE or "
            "ANTHROPIC_FOUNDRY_BASE_URL.",
        )
    return {"CLAUDE_CODE_USE_FOUNDRY": "1", foundry_target[0]: foundry_target[1]}


def validate_embedding_provider(agent: Agent) -> None:
    """Validate embedding provider configuration for vectorstore tools.

    Anthropic does not support generating embeddings, so an external
    embedding_provider must be specified when using vectorstore tools
    with the Anthropic LLM provider.

    Args:
        agent: Agent configuration to validate.

    Raises:
        ConfigError: If embedding configuration is invalid for the provider.
    """
    if not agent.tools:
        return

    has_vectorstore = any(
        isinstance(tool, VectorstoreTool | HierarchicalDocumentToolConfig)
        for tool in agent.tools
    )
    if not has_vectorstore:
        return

    # Anthropic cannot be used as an embedding provider
    if agent.embedding_provider is not None:
        if agent.embedding_provider.provider == ProviderEnum.ANTHROPIC:
            raise ConfigError(
                "embedding_provider",
                "Anthropic cannot generate embeddings. Use a different provider "
                "such as openai or azure_openai for embedding_provider when "
                "using vectorstore tools.",
            )
        return

    # Anthropic LLM + vectorstore tool + no embedding_provider
    if agent.model.provider == ProviderEnum.ANTHROPIC:
        raise ConfigError(
            "embedding_provider",
            "embedding_provider is required when using vectorstore tools with "
            "provider: anthropic. Add an embedding_provider using openai or "
            "azure_openai.",
        )


def validate_tool_filtering(agent: Agent) -> None:
    """Warn if tool_filtering is configured for Anthropic provider.

    Claude Agent SDK manages tool selection natively; tool_filtering is a
    Semantic Kernel feature that is not supported by the Claude backend.

    This validator never raises — it only emits a warning. The
    tool_filtering field is not mutated.

    Args:
        agent: Agent configuration to validate.
    """
    if agent.tool_filtering is None:
        return

    if agent.model.provider == ProviderEnum.ANTHROPIC:
        logger.warning(
            "tool_filtering is configured but will be ignored when using "
            "provider: anthropic with the Claude Agent SDK backend. "
            "Claude manages tool selection natively.",
        )


def validate_working_directory(path: str | None) -> None:
    """Warn if CLAUDE.md in working directory may conflict with agent instructions.

    Detects a CLAUDE.md file that contains a '# CLAUDE.md' header, which
    is the standard format used by Claude Code project instructions. Such
    a file may override or conflict with the agent's configured instructions.

    Args:
        path: Working directory path, or None to skip validation.
    """
    if path is None:
        return

    claude_md = Path(path) / "CLAUDE.md"
    if not claude_md.exists():
        return

    content = claude_md.read_text()
    if "# CLAUDE.md" in content:
        logger.warning(
            "A CLAUDE.md file with a '# CLAUDE.md' header was found in the "
            "working directory '%s'. This may conflict with the agent's system "
            "instructions. Review the file to avoid unexpected behavior.",
            path,
        )


def validate_response_format(response_format: dict[str, Any] | str | None) -> None:
    """Validate response format schema is serializable and accessible.

    Args:
        response_format: Inline schema dict, file path string, or None.

    Raises:
        ConfigError: If the schema is not JSON-serializable or file not found.
    """
    if response_format is None:
        return

    if isinstance(response_format, str):
        schema_path = Path(response_format)
        if not schema_path.exists():
            raise ConfigError(
                "response_format",
                f"response_format file not found: {response_format}",
            )
        try:
            json.loads(schema_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            raise ConfigError(
                "response_format",
                f"response_format file is not valid JSON: {e}",
            ) from e
        return

    # Must be a dict — verify it is JSON-serializable
    try:
        json.dumps(response_format)
    except (TypeError, ValueError) as e:
        raise ConfigError(
            "response_format",
            f"response_format contains non-JSON-serializable values: {e}",
        ) from e
