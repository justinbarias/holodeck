"""Tests for Phase 3 startup validators (T017-T022)."""

import json
import logging
from unittest.mock import patch

import pytest

from holodeck.lib.backends.validators import (
    validate_credentials,
    validate_embedding_provider,
    validate_nodejs,
    validate_response_format,
    validate_tool_filtering,
    validate_working_directory,
)
from holodeck.lib.errors import ConfigError
from holodeck.lib.tool_filter.models import ToolFilterConfig
from holodeck.models.agent import Agent, Instructions
from holodeck.models.claude_config import AuthProvider
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import VectorstoreTool


def _make_agent(**kwargs: object) -> Agent:
    """Build an Agent with minimal required fields."""
    defaults: dict[str, object] = {
        "name": "test",
        "model": LLMProvider(provider=ProviderEnum.ANTHROPIC, name="claude-sonnet-4-5"),
        "instructions": Instructions(inline="Test instructions"),
    }
    defaults.update(kwargs)
    return Agent(**defaults)  # type: ignore[arg-type]


@pytest.mark.unit
class TestValidateNodejs:
    """Tests for validate_nodejs (T017)."""

    @patch("holodeck.lib.backends.validators.shutil.which")
    def test_passes_when_node_found(self, mock_which: object) -> None:
        """No exception raised when Node.js is found on PATH."""
        mock_which.return_value = "/usr/local/bin/node"  # type: ignore[union-attr]
        validate_nodejs()  # Should not raise

    @patch("holodeck.lib.backends.validators.shutil.which")
    def test_raises_config_error_when_node_missing(self, mock_which: object) -> None:
        """ConfigError raised with 'nodejs' field when Node.js is not found."""
        mock_which.return_value = None  # type: ignore[union-attr]
        with pytest.raises(ConfigError) as exc_info:
            validate_nodejs()
        assert exc_info.value.field == "nodejs"


@pytest.mark.unit
class TestValidateCredentials:
    """Tests for validate_credentials (T018)."""

    @patch("holodeck.lib.backends.validators.get_env_var")
    def test_api_key_present(self, mock_get_env: object) -> None:
        """Returns dict with ANTHROPIC_API_KEY when set."""
        mock_get_env.return_value = "sk-ant-test-key"  # type: ignore[union-attr]
        model = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-sonnet-4-5",
            auth_provider=AuthProvider.api_key,
        )
        result = validate_credentials(model)
        assert result == {"ANTHROPIC_API_KEY": "sk-ant-test-key"}

    @patch("holodeck.lib.backends.validators.get_env_var")
    def test_api_key_missing_raises(self, mock_get_env: object) -> None:
        """ConfigError raised with ANTHROPIC_API_KEY field when key is absent."""
        mock_get_env.return_value = None  # type: ignore[union-attr]
        model = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-sonnet-4-5",
            auth_provider=AuthProvider.api_key,
        )
        with pytest.raises(ConfigError) as exc_info:
            validate_credentials(model)
        assert exc_info.value.field == "ANTHROPIC_API_KEY"

    @patch("holodeck.lib.backends.validators.get_env_var")
    def test_api_key_default_when_auth_none(self, mock_get_env: object) -> None:
        """auth_provider=None defaults to api_key behavior."""
        mock_get_env.return_value = "sk-ant-test-key"  # type: ignore[union-attr]
        model = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-sonnet-4-5",
            auth_provider=None,
        )
        result = validate_credentials(model)
        assert result == {"ANTHROPIC_API_KEY": "sk-ant-test-key"}

    @patch("holodeck.lib.backends.validators.get_env_var")
    def test_oauth_token_present(self, mock_get_env: object) -> None:
        """Returns dict with CLAUDE_CODE_OAUTH_TOKEN when set."""
        mock_get_env.return_value = "oauth-test-token"  # type: ignore[union-attr]
        model = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-sonnet-4-5",
            auth_provider=AuthProvider.oauth_token,
        )
        result = validate_credentials(model)
        assert result == {"CLAUDE_CODE_OAUTH_TOKEN": "oauth-test-token"}

    @patch("holodeck.lib.backends.validators.get_env_var")
    def test_oauth_token_missing_raises(self, mock_get_env: object) -> None:
        """ConfigError raised with CLAUDE_CODE_OAUTH_TOKEN when token is absent."""
        mock_get_env.return_value = None  # type: ignore[union-attr]
        model = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-sonnet-4-5",
            auth_provider=AuthProvider.oauth_token,
        )
        with pytest.raises(ConfigError) as exc_info:
            validate_credentials(model)
        assert exc_info.value.field == "CLAUDE_CODE_OAUTH_TOKEN"
        assert "setup-token" in exc_info.value.message

    @patch("holodeck.lib.backends.validators.get_env_var")
    def test_bedrock_returns_env_dict(self, mock_get_env: object) -> None:
        """Bedrock auth returns env dict with routing vars."""
        mock_get_env.side_effect = lambda key, default=None: {  # type: ignore[union-attr]
            "AWS_REGION": "us-east-1",
        }.get(
            key, default
        )
        model = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-sonnet-4-5",
            auth_provider=AuthProvider.bedrock,
        )
        result = validate_credentials(model)
        assert result == {
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "AWS_REGION": "us-east-1",
        }

    @patch("holodeck.lib.backends.validators.get_env_var")
    def test_bedrock_missing_region_raises(self, mock_get_env: object) -> None:
        """ConfigError raised when AWS_REGION is missing for Bedrock."""
        mock_get_env.return_value = None  # type: ignore[union-attr]
        model = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-sonnet-4-5",
            auth_provider=AuthProvider.bedrock,
        )
        with pytest.raises(ConfigError) as exc_info:
            validate_credentials(model)
        assert exc_info.value.field == "AWS_REGION"

    @patch("holodeck.lib.backends.validators.get_env_var")
    def test_bedrock_accepts_aws_default_region(self, mock_get_env: object) -> None:
        """Bedrock accepts AWS_DEFAULT_REGION when AWS_REGION is unset."""
        mock_get_env.side_effect = lambda key, default=None: {  # type: ignore[union-attr]
            "AWS_DEFAULT_REGION": "us-west-2",
        }.get(
            key, default
        )
        model = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-sonnet-4-5",
            auth_provider=AuthProvider.bedrock,
        )
        result = validate_credentials(model)
        assert result == {
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "AWS_DEFAULT_REGION": "us-west-2",
        }

    @patch("holodeck.lib.backends.validators.get_env_var")
    def test_vertex_returns_env_dict(self, mock_get_env: object) -> None:
        """Vertex auth returns env dict with region and project context."""
        mock_get_env.side_effect = lambda key, default=None: {  # type: ignore[union-attr]
            "CLOUD_ML_REGION": "us-east5",
            "ANTHROPIC_VERTEX_PROJECT_ID": "my-gcp-project",
        }.get(
            key, default
        )
        model = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-sonnet-4-5",
            auth_provider=AuthProvider.vertex,
        )
        result = validate_credentials(model)
        assert result == {
            "CLAUDE_CODE_USE_VERTEX": "1",
            "CLOUD_ML_REGION": "us-east5",
            "ANTHROPIC_VERTEX_PROJECT_ID": "my-gcp-project",
        }

    @patch("holodeck.lib.backends.validators.get_env_var")
    def test_vertex_accepts_gcloud_project_override(self, mock_get_env: object) -> None:
        """Vertex accepts GCLOUD_PROJECT as project context."""
        mock_get_env.side_effect = lambda key, default=None: {  # type: ignore[union-attr]
            "CLOUD_ML_REGION": "us-east5",
            "GCLOUD_PROJECT": "my-gcp-project",
        }.get(
            key, default
        )
        model = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-sonnet-4-5",
            auth_provider=AuthProvider.vertex,
        )
        result = validate_credentials(model)
        assert result == {
            "CLAUDE_CODE_USE_VERTEX": "1",
            "CLOUD_ML_REGION": "us-east5",
            "GCLOUD_PROJECT": "my-gcp-project",
        }

    @patch("holodeck.lib.backends.validators.get_env_var")
    def test_vertex_missing_region_raises(self, mock_get_env: object) -> None:
        """ConfigError raised when CLOUD_ML_REGION is missing for Vertex."""
        mock_get_env.side_effect = lambda key, default=None: {  # type: ignore[union-attr]
            "ANTHROPIC_VERTEX_PROJECT_ID": "my-gcp-project",
        }.get(
            key, default
        )
        model = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-sonnet-4-5",
            auth_provider=AuthProvider.vertex,
        )
        with pytest.raises(ConfigError) as exc_info:
            validate_credentials(model)
        assert exc_info.value.field == "CLOUD_ML_REGION"

    @patch("holodeck.lib.backends.validators.get_env_var")
    def test_vertex_missing_project_context_raises(self, mock_get_env: object) -> None:
        """ConfigError raised when Vertex project context is missing."""
        mock_get_env.side_effect = lambda key, default=None: {  # type: ignore[union-attr]
            "CLOUD_ML_REGION": "us-east5",
        }.get(
            key, default
        )
        model = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-sonnet-4-5",
            auth_provider=AuthProvider.vertex,
        )
        with pytest.raises(ConfigError) as exc_info:
            validate_credentials(model)
        assert exc_info.value.field == "ANTHROPIC_VERTEX_PROJECT_ID"

    @patch("holodeck.lib.backends.validators.get_env_var")
    def test_foundry_returns_env_dict(self, mock_get_env: object) -> None:
        """Foundry auth returns env dict with target resource."""
        mock_get_env.side_effect = lambda key, default=None: {  # type: ignore[union-attr]
            "ANTHROPIC_FOUNDRY_RESOURCE": "my-foundry-resource",
        }.get(
            key, default
        )
        model = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-sonnet-4-5",
            auth_provider=AuthProvider.foundry,
        )
        result = validate_credentials(model)
        assert result == {
            "CLAUDE_CODE_USE_FOUNDRY": "1",
            "ANTHROPIC_FOUNDRY_RESOURCE": "my-foundry-resource",
        }

    @patch("holodeck.lib.backends.validators.get_env_var")
    def test_foundry_accepts_base_url(self, mock_get_env: object) -> None:
        """Foundry accepts ANTHROPIC_FOUNDRY_BASE_URL target."""
        mock_get_env.side_effect = lambda key, default=None: {  # type: ignore[union-attr]
            "ANTHROPIC_FOUNDRY_BASE_URL": "https://example.services.ai.azure.com",
        }.get(
            key, default
        )
        model = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-sonnet-4-5",
            auth_provider=AuthProvider.foundry,
        )
        result = validate_credentials(model)
        assert result == {
            "CLAUDE_CODE_USE_FOUNDRY": "1",
            "ANTHROPIC_FOUNDRY_BASE_URL": "https://example.services.ai.azure.com",
        }

    @patch("holodeck.lib.backends.validators.get_env_var")
    def test_foundry_missing_target_raises(self, mock_get_env: object) -> None:
        """ConfigError raised when Foundry target env vars are missing."""
        mock_get_env.return_value = None  # type: ignore[union-attr]
        model = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-sonnet-4-5",
            auth_provider=AuthProvider.foundry,
        )
        with pytest.raises(ConfigError) as exc_info:
            validate_credentials(model)
        assert exc_info.value.field == "ANTHROPIC_FOUNDRY_RESOURCE"


@pytest.mark.unit
class TestValidateEmbeddingProvider:
    """Tests for validate_embedding_provider (T019)."""

    def test_passes_no_vectorstore_tools(self) -> None:
        """No exception when tools list is None."""
        agent = _make_agent(tools=None)
        validate_embedding_provider(agent)  # Should not raise

    def test_passes_vectorstore_with_valid_embedding(self) -> None:
        """No exception when vectorstore tool paired with non-Anthropic embedding."""
        agent = _make_agent(
            tools=[
                VectorstoreTool(name="kb", description="Knowledge base", source="data/")
            ],
            embedding_provider=LLMProvider(
                provider=ProviderEnum.OPENAI, name="text-embedding-3-small"
            ),
        )
        validate_embedding_provider(agent)  # Should not raise

    def test_raises_anthropic_no_embedding(self) -> None:
        """ConfigError when Anthropic provider + vectorstore tool + no embedding."""
        agent = _make_agent(
            tools=[
                VectorstoreTool(name="kb", description="Knowledge base", source="data/")
            ],
            embedding_provider=None,
        )
        with pytest.raises(ConfigError) as exc_info:
            validate_embedding_provider(agent)
        assert exc_info.value.field == "embedding_provider"

    def test_raises_anthropic_as_embedding(self) -> None:
        """ConfigError when embedding_provider.provider is anthropic."""
        agent = _make_agent(
            tools=[
                VectorstoreTool(name="kb", description="Knowledge base", source="data/")
            ],
            embedding_provider=LLMProvider(
                provider=ProviderEnum.ANTHROPIC, name="claude-sonnet-4-5"
            ),
        )
        with pytest.raises(ConfigError) as exc_info:
            validate_embedding_provider(agent)
        assert exc_info.value.field == "embedding_provider"
        assert "cannot generate embeddings" in exc_info.value.message


@pytest.mark.unit
class TestValidateToolFiltering:
    """Tests for validate_tool_filtering (T020)."""

    def test_no_warning_when_none(self, caplog: pytest.LogCaptureFixture) -> None:
        """No warning logged when tool_filtering is None."""
        agent = _make_agent(tool_filtering=None)
        with caplog.at_level(logging.WARNING):
            validate_tool_filtering(agent)
        assert not caplog.records

    def test_warning_when_anthropic_with_filtering(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning logged when Anthropic provider with tool_filtering configured."""
        agent = _make_agent(tool_filtering=ToolFilterConfig())
        with caplog.at_level(logging.WARNING):
            validate_tool_filtering(agent)
        assert caplog.records

    def test_does_not_mutate_tool_filtering(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """tool_filtering field is not cleared after validation."""
        tf = ToolFilterConfig()
        agent = _make_agent(tool_filtering=tf)
        with caplog.at_level(logging.WARNING):
            validate_tool_filtering(agent)
        assert agent.tool_filtering is not None


@pytest.mark.unit
class TestValidateWorkingDirectory:
    """Tests for validate_working_directory (T021)."""

    def test_passes_when_none(self, caplog: pytest.LogCaptureFixture) -> None:
        """No warning when path is None."""
        with caplog.at_level(logging.WARNING):
            validate_working_directory(None)
        assert not caplog.records

    def test_passes_no_claude_md(
        self, tmp_path: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No warning when directory exists but has no CLAUDE.md."""
        with caplog.at_level(logging.WARNING):
            validate_working_directory(str(tmp_path))  # type: ignore[arg-type]
        assert not caplog.records

    def test_warning_when_claude_md_with_header(
        self, tmp_path: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning logged when CLAUDE.md contains '# CLAUDE.md' header."""
        from pathlib import Path

        claude_md = Path(str(tmp_path)) / "CLAUDE.md"  # type: ignore[arg-type]
        claude_md.write_text("# CLAUDE.md\nSome agent instructions here.")
        with caplog.at_level(logging.WARNING):
            validate_working_directory(str(tmp_path))  # type: ignore[arg-type]
        assert caplog.records


@pytest.mark.unit
class TestValidateResponseFormat:
    """Tests for validate_response_format (T022)."""

    def test_passes_when_none(self) -> None:
        """No exception when response_format is None."""
        validate_response_format(None)  # Should not raise

    def test_passes_valid_dict(self) -> None:
        """No exception for a valid JSON-serializable dict."""
        fmt = {"type": "object", "properties": {"name": {"type": "string"}}}
        validate_response_format(fmt)  # Should not raise

    def test_raises_non_serializable_dict(self) -> None:
        """ConfigError when dict contains non-JSON-serializable values."""
        fmt = {"key": {1, 2, 3}}  # set is not JSON-serializable
        with pytest.raises(ConfigError) as exc_info:
            validate_response_format(fmt)
        assert exc_info.value.field == "response_format"

    def test_str_valid_file(self, tmp_path: object) -> None:
        """No exception when str points to a valid JSON file."""
        from pathlib import Path

        schema_file = Path(str(tmp_path)) / "schema.json"  # type: ignore[arg-type]
        schema_file.write_text(json.dumps({"type": "object"}))
        validate_response_format(str(schema_file))  # Should not raise

    def test_str_missing_file(self, tmp_path: object) -> None:
        """ConfigError when str points to a nonexistent file."""
        from pathlib import Path

        missing = str(Path(str(tmp_path)) / "nonexistent.json")  # type: ignore[arg-type]
        with pytest.raises(ConfigError) as exc_info:
            validate_response_format(missing)
        assert exc_info.value.field == "response_format"
