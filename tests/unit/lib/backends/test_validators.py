"""Tests for Phase 3 startup validators (T017-T022) and version checks (T013)."""

import json
import logging
import subprocess
from unittest.mock import patch

import pytest

from holodeck.lib.backends.validators import (
    agent_needs_nodejs,
    validate_credentials,
    validate_embedding_provider,
    validate_nodejs,
    validate_openai_agents,
    validate_response_format,
    validate_working_directory,
)
from holodeck.lib.errors import ConfigError
from holodeck.models.agent import Agent, Instructions
from holodeck.models.claude_config import AuthProvider
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import CommandType, MCPTool, VectorstoreTool


def _make_agent(**kwargs: object) -> Agent:
    """Build an Agent with minimal required fields."""
    defaults: dict[str, object] = {
        "name": "test",
        "model": LLMProvider(provider=ProviderEnum.ANTHROPIC, name="claude-sonnet-4-5"),
        "instructions": Instructions(inline="Test instructions"),
    }
    defaults.update(kwargs)
    return Agent(**defaults)  # type: ignore[arg-type]


def _make_npx_mcp_tool() -> MCPTool:
    """Build a minimal MCPTool that spawns Node via npx."""
    return MCPTool(
        name="npxtool", description="npx-based MCP tool", command=CommandType.NPX
    )


def _make_node_mcp_tool() -> MCPTool:
    """Build a minimal MCPTool that spawns Node directly."""
    return MCPTool(
        name="nodetool", description="node-based MCP tool", command=CommandType.NODE
    )


def _make_uvx_mcp_tool() -> MCPTool:
    """Build a minimal MCPTool that spawns uvx (not Node)."""
    return MCPTool(
        name="uvxtool", description="uvx-based MCP tool", command=CommandType.UVX
    )


@pytest.mark.unit
class TestAgentNeedsNodejs:
    """Tests for agent_needs_nodejs helper."""

    def test_false_when_no_tools(self) -> None:
        """Returns False when agent has no tools."""
        agent = _make_agent(tools=None)
        assert agent_needs_nodejs(agent) is False

    def test_false_when_uvx_tool_only(self) -> None:
        """Returns False when only MCP tools use uvx (not Node)."""
        agent = _make_agent(tools=[_make_uvx_mcp_tool()])
        assert agent_needs_nodejs(agent) is False

    def test_true_when_npx_tool_present(self) -> None:
        """Returns True when an npx MCP tool is present."""
        agent = _make_agent(tools=[_make_npx_mcp_tool()])
        assert agent_needs_nodejs(agent) is True

    def test_true_when_node_tool_present(self) -> None:
        """Returns True when a node MCP tool is present."""
        agent = _make_agent(tools=[_make_node_mcp_tool()])
        assert agent_needs_nodejs(agent) is True

    def test_true_when_mixed_tools_contain_npx(self) -> None:
        """Returns True when at least one tool among many uses npx."""
        agent = _make_agent(tools=[_make_uvx_mcp_tool(), _make_npx_mcp_tool()])
        assert agent_needs_nodejs(agent) is True


@pytest.mark.unit
class TestValidateNodejs:
    """Tests for validate_nodejs (T017)."""

    @patch("holodeck.lib.backends.validators.subprocess.run")
    @patch("holodeck.lib.backends.validators.shutil.which")
    def test_passes_when_node_found(self, mock_which: object, mock_run: object) -> None:
        """No exception raised when Node.js is found on PATH."""
        mock_which.return_value = "/usr/local/bin/node"  # type: ignore[union-attr]
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[union-attr]
            args=["node", "--version"], returncode=0, stdout="v22.0.0\n", stderr=""
        )
        agent = _make_agent(tools=[_make_npx_mcp_tool()])
        validate_nodejs(agent)  # Should not raise

    @patch("holodeck.lib.backends.validators.shutil.which")
    def test_raises_config_error_when_node_missing(self, mock_which: object) -> None:
        """ConfigError raised with 'nodejs' field when Node.js is not found."""
        mock_which.return_value = None  # type: ignore[union-attr]
        agent = _make_agent(tools=[_make_npx_mcp_tool()])
        with pytest.raises(ConfigError) as exc_info:
            validate_nodejs(agent)
        assert exc_info.value.field == "nodejs"

    def test_skips_when_no_node_mcp_tools(self) -> None:
        """No exception raised for agents without Node-spawning MCP tools.

        This is the regression fix: a deployed agent without Node MCP tools
        must not fail startup when Node is absent from PATH.
        """
        agent = _make_agent(tools=None)
        with patch("holodeck.lib.backends.validators.shutil.which", return_value=None):
            validate_nodejs(agent)  # Should not raise

    def test_skips_when_uvx_mcp_tool_only(self) -> None:
        """No Node check when the only MCP tool uses uvx."""
        agent = _make_agent(tools=[_make_uvx_mcp_tool()])
        with patch("holodeck.lib.backends.validators.shutil.which", return_value=None):
            validate_nodejs(agent)  # Should not raise

    def test_required_when_npx_tool_present(self) -> None:
        """ConfigError raised when npx tool present and Node absent from PATH."""
        agent = _make_agent(tools=[_make_npx_mcp_tool()])
        with (
            patch("holodeck.lib.backends.validators.shutil.which", return_value=None),
            pytest.raises(ConfigError, match="Node.js is required"),
        ):
            validate_nodejs(agent)

    # -- Version check tests (T013) ------------------------------------------

    @patch("holodeck.lib.backends.validators.subprocess.run")
    @patch("holodeck.lib.backends.validators.shutil.which")
    def test_v22_passes(self, mock_which: object, mock_run: object) -> None:
        """T013: v22.1.0 passes version check without raising."""
        mock_which.return_value = "/usr/bin/node"  # type: ignore[union-attr]
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[union-attr]
            args=["node", "--version"], returncode=0, stdout="v22.1.0\n", stderr=""
        )
        agent = _make_agent(tools=[_make_npx_mcp_tool()])
        validate_nodejs(agent)  # Should not raise

    @patch("holodeck.lib.backends.validators.subprocess.run")
    @patch("holodeck.lib.backends.validators.shutil.which")
    def test_v18_boundary_passes(self, mock_which: object, mock_run: object) -> None:
        """T013: v18.0.0 passes at the minimum version boundary."""
        mock_which.return_value = "/usr/bin/node"  # type: ignore[union-attr]
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[union-attr]
            args=["node", "--version"], returncode=0, stdout="v18.0.0\n", stderr=""
        )
        agent = _make_agent(tools=[_make_npx_mcp_tool()])
        validate_nodejs(agent)  # Should not raise

    @patch("holodeck.lib.backends.validators.subprocess.run")
    @patch("holodeck.lib.backends.validators.shutil.which")
    def test_v16_fails(self, mock_which: object, mock_run: object) -> None:
        """T013: v16.20.0 fails version check with ConfigError."""
        mock_which.return_value = "/usr/bin/node"  # type: ignore[union-attr]
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[union-attr]
            args=["node", "--version"], returncode=0, stdout="v16.20.0\n", stderr=""
        )
        agent = _make_agent(tools=[_make_npx_mcp_tool()])
        with pytest.raises(ConfigError) as exc_info:
            validate_nodejs(agent)
        assert exc_info.value.field == "nodejs"
        assert "16.20.0" in exc_info.value.message
        assert "18" in exc_info.value.message

    @patch("holodeck.lib.backends.validators.subprocess.run")
    @patch("holodeck.lib.backends.validators.shutil.which")
    def test_timeout_raises(self, mock_which: object, mock_run: object) -> None:
        """T013: Timeout during version check raises ConfigError."""
        mock_which.return_value = "/usr/bin/node"  # type: ignore[union-attr]
        mock_run.side_effect = subprocess.TimeoutExpired(  # type: ignore[union-attr]
            cmd=["node", "--version"], timeout=5
        )
        agent = _make_agent(tools=[_make_npx_mcp_tool()])
        with pytest.raises(ConfigError) as exc_info:
            validate_nodejs(agent)
        assert exc_info.value.field == "nodejs"

    @patch("holodeck.lib.backends.validators.subprocess.run")
    @patch("holodeck.lib.backends.validators.shutil.which")
    def test_nonzero_exit_raises(self, mock_which: object, mock_run: object) -> None:
        """T013: Non-zero exit code raises ConfigError."""
        mock_which.return_value = "/usr/bin/node"  # type: ignore[union-attr]
        mock_run.side_effect = subprocess.CalledProcessError(  # type: ignore[union-attr]
            returncode=1, cmd=["node", "--version"]
        )
        agent = _make_agent(tools=[_make_npx_mcp_tool()])
        with pytest.raises(ConfigError) as exc_info:
            validate_nodejs(agent)
        assert exc_info.value.field == "nodejs"

    @patch("holodeck.lib.backends.validators.shutil.which")
    def test_node_not_found_does_not_call_subprocess(self, mock_which: object) -> None:
        """T013: Node not found raises ConfigError without subprocess."""
        mock_which.return_value = None  # type: ignore[union-attr]
        agent = _make_agent(tools=[_make_npx_mcp_tool()])
        with pytest.raises(ConfigError) as exc_info:
            validate_nodejs(agent)
        assert exc_info.value.field == "nodejs"

    @patch("holodeck.lib.backends.validators.subprocess.run")
    @patch("holodeck.lib.backends.validators.shutil.which")
    def test_unparseable_version_raises(
        self, mock_which: object, mock_run: object
    ) -> None:
        """T013: Unparseable version output raises ConfigError."""
        mock_which.return_value = "/usr/bin/node"  # type: ignore[union-attr]
        mock_run.return_value = subprocess.CompletedProcess(  # type: ignore[union-attr]
            args=["node", "--version"], returncode=0, stdout="garbage", stderr=""
        )
        agent = _make_agent(tools=[_make_npx_mcp_tool()])
        with pytest.raises(ConfigError) as exc_info:
            validate_nodejs(agent)
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

    @patch("holodeck.lib.backends.validators.get_env_var")
    def test_custom_auth_token_present(self, mock_get_env: object) -> None:
        """Returns dict with ANTHROPIC_AUTH_TOKEN when set."""
        mock_get_env.return_value = "ollama"  # type: ignore[union-attr]
        model = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="llama3.1",
            endpoint="http://localhost:11434/v1",
            auth_provider=AuthProvider.custom,
        )
        result = validate_credentials(model)
        assert result == {"ANTHROPIC_AUTH_TOKEN": "ollama"}

    @patch("holodeck.lib.backends.validators.get_env_var")
    def test_custom_auth_token_missing_raises(self, mock_get_env: object) -> None:
        """ConfigError raised with ANTHROPIC_AUTH_TOKEN when token is absent."""
        mock_get_env.return_value = None  # type: ignore[union-attr]
        model = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="llama3.1",
            endpoint="http://localhost:11434/v1",
            auth_provider=AuthProvider.custom,
        )
        with pytest.raises(ConfigError) as exc_info:
            validate_credentials(model)
        assert exc_info.value.field == "ANTHROPIC_AUTH_TOKEN"


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


@pytest.mark.unit
class TestValidateOpenAIAgents:
    """A2 — collect-all-errors, side-effect-free openai_agents preflight."""

    def _openai_agent(self, **openai: object) -> Agent:
        kwargs: dict = {
            "name": "a",
            "model": LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            "instructions": Instructions(inline="hi"),
        }
        if openai:
            kwargs["openai"] = openai
        return Agent(**kwargs)

    def test_passes_with_valid_creds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        validate_openai_agents(self._openai_agent())  # must not raise

    def test_missing_credential_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
            validate_openai_agents(self._openai_agent())

    def test_collects_missing_creds_and_tool_conflict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        agent = self._openai_agent(
            permissions={"allowed_tools": ["x", "y"], "disallowed_tools": ["y"]}
        )
        with pytest.raises(ConfigError) as exc_info:
            validate_openai_agents(agent)
        message = str(exc_info.value)
        assert "OPENAI_API_KEY" in message  # credential error
        assert "y" in message  # conflict error — both surfaced together

    def test_tool_conflict_alone_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        agent = self._openai_agent(
            permissions={"allowed_tools": ["z"], "disallowed_tools": ["z"]}
        )
        with pytest.raises(ConfigError, match="z"):
            validate_openai_agents(agent)

    def test_no_sdk_side_effects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with (
            patch("agents.set_default_openai_key") as set_key,
            patch("agents.set_tracing_disabled") as disable_tracing,
        ):
            validate_openai_agents(self._openai_agent())
        set_key.assert_not_called()
        disable_tracing.assert_not_called()
