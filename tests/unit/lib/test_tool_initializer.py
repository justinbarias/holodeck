"""Unit tests for the shared tool_initializer module.

Tests cover:
- T001: create_embedding_service for various providers
- T002: resolve_embedding_model defaults and overrides
- T003: initialize vectorstore tools
- T004: initialize hierarchical document tools
- T005: initialize_tools convenience (combined)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from holodeck.lib.tool_initializer import (
    ToolInitializerError,
    _create_chat_service_from_config,
    _resolve_context_model_config,
    create_embedding_service,
    initialize_tools,
    resolve_embedding_model,
)
from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_agent(
    provider: ProviderEnum = ProviderEnum.OPENAI,
    tools: list[Any] | None = None,
    embedding_provider: LLMProvider | None = None,
) -> Agent:
    """Create an Agent fixture with sensible defaults."""
    return Agent(
        name="test_agent",
        model=LLMProvider(
            provider=provider,
            name="gpt-4o" if provider != ProviderEnum.ANTHROPIC else "claude-3-opus",
            api_key="test-key",
            endpoint=(
                "https://test.openai.azure.com"
                if provider == ProviderEnum.AZURE_OPENAI
                else None
            ),
        ),
        instructions=Instructions(inline="You are a test agent."),
        tools=tools,
        embedding_provider=embedding_provider,
    )


def _make_vectorstore_tool(
    name: str = "knowledge_base",
    embedding_model: str | None = None,
) -> Any:
    """Create a VectorstoreTool config fixture."""
    from holodeck.models.tool import VectorstoreTool

    return VectorstoreTool(
        name=name,
        description=f"Search {name}",
        source="./data/docs",
        embedding_model=embedding_model,
    )


def _make_hierarchical_doc_tool(
    name: str = "doc_search",
    context_model: LLMProvider | None = None,
) -> Any:
    """Create a HierarchicalDocumentToolConfig fixture."""
    from holodeck.models.tool import HierarchicalDocumentToolConfig

    return HierarchicalDocumentToolConfig(
        name=name,
        description=f"Search {name}",
        source="./data/docs",
        context_model=context_model,
    )


# ===================================================================
# T001: TestCreateEmbeddingService
# ===================================================================


class TestCreateEmbeddingService:
    """Tests for create_embedding_service()."""

    @patch("holodeck.lib.tool_initializer.OpenAITextEmbedding", create=True)
    def test_openai_provider(self, mock_cls: MagicMock) -> None:
        """OpenAI provider → returns OpenAITextEmbedding instance."""
        with patch(
            "semantic_kernel.connectors.ai.open_ai.OpenAITextEmbedding",
            mock_cls,
        ):
            agent = _make_agent(ProviderEnum.OPENAI)
            result = create_embedding_service(agent)
            assert result is not None

    @patch("semantic_kernel.connectors.ai.open_ai.AzureTextEmbedding")
    def test_azure_openai_provider(self, mock_cls: MagicMock) -> None:
        """Azure OpenAI provider → returns AzureTextEmbedding instance."""
        agent = _make_agent(ProviderEnum.AZURE_OPENAI)
        result = create_embedding_service(agent)
        assert result is not None
        mock_cls.assert_called_once()

    def test_ollama_provider(self) -> None:
        """Ollama provider → returns OllamaTextEmbedding instance."""
        mock_cls = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "semantic_kernel.connectors.ai.ollama": MagicMock(
                    OllamaTextEmbedding=mock_cls
                )
            },
        ):
            agent = _make_agent(ProviderEnum.OLLAMA)
            result = create_embedding_service(agent)
            assert result is not None

    @patch("semantic_kernel.connectors.ai.open_ai.OpenAITextEmbedding")
    def test_anthropic_with_embedding_provider(self, mock_cls: MagicMock) -> None:
        """Anthropic provider with embedding_provider → uses that provider."""
        embedding_provider = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="text-embedding-3-small",
            api_key="embed-key",
        )
        agent = _make_agent(
            ProviderEnum.ANTHROPIC,
            embedding_provider=embedding_provider,
        )
        result = create_embedding_service(agent)
        assert result is not None

    def test_anthropic_without_embedding_provider(self) -> None:
        """Anthropic provider without embedding_provider → raises error."""
        agent = _make_agent(ProviderEnum.ANTHROPIC)
        with pytest.raises(ToolInitializerError, match="does not support embeddings"):
            create_embedding_service(agent)

    def test_unsupported_provider_via_anthropic_no_embed(self) -> None:
        """Anthropic without embedding_provider raises ToolInitializerError."""
        agent = _make_agent(ProviderEnum.ANTHROPIC, embedding_provider=None)
        with pytest.raises(ToolInitializerError):
            create_embedding_service(agent)


# ===================================================================
# T002: TestResolveEmbeddingModel
# ===================================================================


class TestResolveEmbeddingModel:
    """Tests for resolve_embedding_model()."""

    def test_explicit_model_on_vectorstore_tool(self) -> None:
        """Explicit embedding_model on tool config → returned as-is."""
        tool = _make_vectorstore_tool(embedding_model="my-custom-model")
        agent = _make_agent(tools=[tool])
        assert resolve_embedding_model(agent) == "my-custom-model"

    def test_openai_default(self) -> None:
        """No explicit model + OpenAI provider → text-embedding-3-small."""
        agent = _make_agent(ProviderEnum.OPENAI)
        assert resolve_embedding_model(agent) == "text-embedding-3-small"

    def test_azure_openai_default(self) -> None:
        """No explicit model + Azure OpenAI provider → text-embedding-3-small."""
        agent = _make_agent(ProviderEnum.AZURE_OPENAI)
        assert resolve_embedding_model(agent) == "text-embedding-3-small"

    def test_ollama_default(self) -> None:
        """No explicit model + Ollama provider → nomic-embed-text:latest."""
        agent = _make_agent(ProviderEnum.OLLAMA)
        assert resolve_embedding_model(agent) == "nomic-embed-text:latest"

    def test_anthropic_with_openai_embedding_provider(self) -> None:
        """Anthropic + OpenAI embedding_provider → text-embedding-3-small."""
        embedding_provider = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="text-embedding-3-small",
            api_key="key",
        )
        agent = _make_agent(
            ProviderEnum.ANTHROPIC,
            embedding_provider=embedding_provider,
        )
        assert resolve_embedding_model(agent) == "text-embedding-3-small"

    def test_anthropic_with_ollama_embedding_provider(self) -> None:
        """Anthropic + Ollama embedding_provider → nomic-embed-text:latest."""
        embedding_provider = LLMProvider(
            provider=ProviderEnum.OLLAMA,
            name="nomic-embed-text:latest",
            api_key="key",
        )
        agent = _make_agent(
            ProviderEnum.ANTHROPIC,
            embedding_provider=embedding_provider,
        )
        assert resolve_embedding_model(agent) == "nomic-embed-text:latest"


# ===================================================================
# T003: TestInitializeVectorstoreTools
# ===================================================================


class TestInitializeVectorstoreTools:
    """Tests for vectorstore tool initialization via initialize_tools()."""

    @pytest.mark.asyncio
    async def test_two_vectorstore_tools(self) -> None:
        """Agent with 2 vectorstore tools → returns dict with 2 instances."""
        tool1 = _make_vectorstore_tool(name="tool_a")
        tool2 = _make_vectorstore_tool(name="tool_b")
        agent = _make_agent(tools=[tool1, tool2])

        mock_vs = MagicMock()
        mock_vs.return_value = MagicMock()
        mock_vs.return_value.set_embedding_service = MagicMock()
        mock_vs.return_value.initialize = AsyncMock()

        with (
            patch("holodeck.lib.tool_initializer.create_embedding_service"),
            patch(
                "holodeck.tools.vectorstore_tool.VectorStoreTool",
                mock_vs,
            ),
        ):
            result = await initialize_tools(agent)
            assert len(result) == 2
            assert "tool_a" in result
            assert "tool_b" in result

    @pytest.mark.asyncio
    async def test_no_tools(self) -> None:
        """Agent with no tools → returns empty dict."""
        agent = _make_agent(tools=None)
        result = await initialize_tools(agent)
        assert result == {}

    @pytest.mark.asyncio
    async def test_empty_tools(self) -> None:
        """Agent with empty tools list → returns empty dict."""
        agent = _make_agent(tools=[])
        result = await initialize_tools(agent)
        assert result == {}

    @pytest.mark.asyncio
    async def test_tool_init_failure(self) -> None:
        """Tool initialization failure → raises ToolInitializerError."""
        tool = _make_vectorstore_tool(name="failing_tool")
        agent = _make_agent(tools=[tool])

        mock_vs = MagicMock()
        mock_vs.return_value = MagicMock()
        mock_vs.return_value.set_embedding_service = MagicMock()
        mock_vs.return_value.initialize = AsyncMock(
            side_effect=RuntimeError("init boom")
        )

        with (
            patch("holodeck.lib.tool_initializer.create_embedding_service"),
            patch(
                "holodeck.tools.vectorstore_tool.VectorStoreTool",
                mock_vs,
            ),
            pytest.raises(ToolInitializerError, match="failing_tool"),
        ):
            await initialize_tools(agent)


# ===================================================================
# T004: TestInitializeHierarchicalDocTools
# ===================================================================


class TestInitializeHierarchicalDocTools:
    """Tests for hierarchical document tool initialization."""

    @pytest.mark.asyncio
    async def test_one_hierarchical_doc_tool(self) -> None:
        """Agent with 1 hierarchical doc tool → returns dict with 1 instance."""
        tool = _make_hierarchical_doc_tool(name="doc_tool")
        agent = _make_agent(tools=[tool])

        mock_hd = MagicMock()
        mock_hd.return_value = MagicMock()
        mock_hd.return_value.set_embedding_service = MagicMock()
        mock_hd.return_value.set_chat_service = MagicMock()
        mock_hd.return_value.initialize = AsyncMock()

        with (
            patch("holodeck.lib.tool_initializer.create_embedding_service"),
            patch(
                "holodeck.tools.hierarchical_document_tool.HierarchicalDocumentTool",
                mock_hd,
            ),
        ):
            result = await initialize_tools(agent)
            assert len(result) == 1
            assert "doc_tool" in result

    @pytest.mark.asyncio
    async def test_chat_service_injected(self) -> None:
        """With chat_service provided → sets it on tool."""
        tool = _make_hierarchical_doc_tool(name="doc_tool")
        agent = _make_agent(tools=[tool])

        mock_hd = MagicMock()
        mock_instance = MagicMock()
        mock_instance.set_embedding_service = MagicMock()
        mock_instance.set_chat_service = MagicMock()
        mock_instance.initialize = AsyncMock()
        mock_hd.return_value = mock_instance

        mock_chat = MagicMock()

        with (
            patch("holodeck.lib.tool_initializer.create_embedding_service"),
            patch(
                "holodeck.tools.hierarchical_document_tool.HierarchicalDocumentTool",
                mock_hd,
            ),
        ):
            await initialize_tools(agent, chat_service=mock_chat)
            mock_instance.set_chat_service.assert_called_once_with(mock_chat)


# ===================================================================
# T005: TestInitializeAllTools
# ===================================================================


class TestInitializeAllTools:
    """Tests for initialize_tools() combining vectorstore + hierarchical-doc."""

    @pytest.mark.asyncio
    async def test_mixed_tools(self) -> None:
        """Agent with mixed tools → returns unified dict."""
        vs_tool = _make_vectorstore_tool(name="vs_tool")
        hd_tool = _make_hierarchical_doc_tool(name="hd_tool")
        agent = _make_agent(tools=[vs_tool, hd_tool])

        mock_vs = MagicMock()
        mock_vs.return_value = MagicMock()
        mock_vs.return_value.set_embedding_service = MagicMock()
        mock_vs.return_value.initialize = AsyncMock()

        mock_hd = MagicMock()
        mock_hd.return_value = MagicMock()
        mock_hd.return_value.set_embedding_service = MagicMock()
        mock_hd.return_value.set_chat_service = MagicMock()
        mock_hd.return_value.initialize = AsyncMock()

        with (
            patch("holodeck.lib.tool_initializer.create_embedding_service"),
            patch(
                "holodeck.tools.vectorstore_tool.VectorStoreTool",
                mock_vs,
            ),
            patch(
                "holodeck.tools.hierarchical_document_tool.HierarchicalDocumentTool",
                mock_hd,
            ),
        ):
            result = await initialize_tools(agent)
            assert len(result) == 2
            assert "vs_tool" in result
            assert "hd_tool" in result

    @pytest.mark.asyncio
    async def test_anthropic_agent_uses_embedding_provider_for_dimensions(self) -> None:
        """Anthropic + Ollama embedding_provider → uses 'ollama' for dimensions."""
        embedding_provider = LLMProvider(
            provider=ProviderEnum.OLLAMA,
            name="nomic-embed-text:latest",
            api_key="key",
        )
        vs_tool = _make_vectorstore_tool(name="vs_tool")
        agent = _make_agent(
            ProviderEnum.ANTHROPIC,
            tools=[vs_tool],
            embedding_provider=embedding_provider,
        )

        mock_vs = MagicMock()
        mock_vs_instance = MagicMock()
        mock_vs_instance.set_embedding_service = MagicMock()
        mock_vs_instance.initialize = AsyncMock()
        mock_vs.return_value = mock_vs_instance

        with (
            patch("holodeck.lib.tool_initializer.create_embedding_service"),
            patch(
                "holodeck.tools.vectorstore_tool.VectorStoreTool",
                mock_vs,
            ),
        ):
            await initialize_tools(agent)
            # The provider_type passed to tool.initialize() should be "ollama",
            # not "anthropic", so dimensions resolve correctly (768 vs 1536).
            mock_vs_instance.initialize.assert_called_once()
            _, kwargs = mock_vs_instance.initialize.call_args
            assert kwargs["provider_type"] == "ollama"

    @pytest.mark.asyncio
    async def test_embedding_service_creation_failure(self) -> None:
        """Embedding service failure → raises ToolInitializerError."""
        vs_tool = _make_vectorstore_tool(name="vs_tool")
        agent = _make_agent(tools=[vs_tool])

        with (
            patch(
                "holodeck.lib.tool_initializer.create_embedding_service",
                side_effect=RuntimeError("no embedding"),
            ),
            pytest.raises(ToolInitializerError, match="embedding service"),
        ):
            await initialize_tools(agent)


# ===================================================================
# T006: TestResolveContextModelConfig
# ===================================================================


class TestResolveContextModelConfig:
    """Tests for _resolve_context_model_config() resolution chain."""

    def test_tool_context_model_takes_priority(self) -> None:
        """Per-tool context_model wins over agent-level configs."""
        tool_context = LLMProvider(
            provider=ProviderEnum.OLLAMA,
            name="llama3",
            api_key="key",
        )
        embedding_provider = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="text-embedding-3-small",
            api_key="key",
        )
        tool = _make_hierarchical_doc_tool(context_model=tool_context)
        agent = _make_agent(
            ProviderEnum.ANTHROPIC,
            tools=[tool],
            embedding_provider=embedding_provider,
        )
        result = _resolve_context_model_config(agent, tool)
        assert result.provider == ProviderEnum.OLLAMA
        assert result.name == "llama3"

    def test_falls_back_to_embedding_provider(self) -> None:
        """No tool context_model → uses agent.embedding_provider."""
        embedding_provider = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-4o-mini",
            api_key="key",
        )
        tool = _make_hierarchical_doc_tool()  # no context_model
        agent = _make_agent(
            ProviderEnum.ANTHROPIC,
            tools=[tool],
            embedding_provider=embedding_provider,
        )
        result = _resolve_context_model_config(agent, tool)
        assert result.provider == ProviderEnum.OPENAI
        assert result.name == "gpt-4o-mini"

    def test_falls_back_to_agent_model(self) -> None:
        """No tool context_model, no embedding_provider → uses agent.model."""
        tool = _make_hierarchical_doc_tool()  # no context_model
        agent = _make_agent(ProviderEnum.OPENAI, tools=[tool])
        result = _resolve_context_model_config(agent, tool)
        assert result.provider == ProviderEnum.OPENAI
        assert result.name == "gpt-4o"


# ===================================================================
# T007: TestContextModelWiring
# ===================================================================


class TestContextModelWiring:
    """Tests that context_model override is wired into tool initialization."""

    @pytest.mark.asyncio
    async def test_context_model_creates_chat_service(self) -> None:
        """Tool with context_model and no caller chat_service → auto-creates one."""
        tool_context = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-4o-mini",
            api_key="test-key",
        )
        tool = _make_hierarchical_doc_tool(name="ctx_tool", context_model=tool_context)
        agent = _make_agent(tools=[tool])

        mock_hd = MagicMock()
        mock_instance = MagicMock()
        mock_instance.set_embedding_service = MagicMock()
        mock_instance.set_chat_service = MagicMock()
        mock_instance.initialize = AsyncMock()
        mock_hd.return_value = mock_instance

        mock_chat_cls = MagicMock()

        with (
            patch("holodeck.lib.tool_initializer.create_embedding_service"),
            patch(
                "holodeck.tools.hierarchical_document_tool.HierarchicalDocumentTool",
                mock_hd,
            ),
            patch(
                "semantic_kernel.connectors.ai.open_ai.OpenAIChatCompletion",
                mock_chat_cls,
            ),
        ):
            result = await initialize_tools(agent, chat_service=None)
            assert "ctx_tool" in result
            mock_instance.set_chat_service.assert_called_once()

    @pytest.mark.asyncio
    async def test_caller_chat_service_used_when_provided(self) -> None:
        """Caller-provided chat_service takes precedence even with context_model."""
        tool_context = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-4o-mini",
            api_key="test-key",
        )
        tool = _make_hierarchical_doc_tool(name="ctx_tool", context_model=tool_context)
        agent = _make_agent(tools=[tool])

        mock_hd = MagicMock()
        mock_instance = MagicMock()
        mock_instance.set_embedding_service = MagicMock()
        mock_instance.set_chat_service = MagicMock()
        mock_instance.initialize = AsyncMock()
        mock_hd.return_value = mock_instance

        caller_chat = MagicMock()

        with (
            patch("holodeck.lib.tool_initializer.create_embedding_service"),
            patch(
                "holodeck.tools.hierarchical_document_tool.HierarchicalDocumentTool",
                mock_hd,
            ),
        ):
            await initialize_tools(agent, chat_service=caller_chat)
            mock_instance.set_chat_service.assert_called_once_with(caller_chat)

    @pytest.mark.asyncio
    async def test_no_context_model_no_chat_service(self) -> None:
        """No context_model and no chat_service → set_chat_service not called."""
        tool = _make_hierarchical_doc_tool(name="plain_tool")
        agent = _make_agent(tools=[tool])

        mock_hd = MagicMock()
        mock_instance = MagicMock()
        mock_instance.set_embedding_service = MagicMock()
        mock_instance.set_chat_service = MagicMock()
        mock_instance.initialize = AsyncMock()
        mock_hd.return_value = mock_instance

        with (
            patch("holodeck.lib.tool_initializer.create_embedding_service"),
            patch(
                "holodeck.tools.hierarchical_document_tool.HierarchicalDocumentTool",
                mock_hd,
            ),
        ):
            await initialize_tools(agent, chat_service=None)
            mock_instance.set_chat_service.assert_not_called()


# ===================================================================
# T008: TestCreateChatServiceFromConfig
# ===================================================================


class TestCreateChatServiceFromConfig:
    """Tests for _create_chat_service_from_config()."""

    @patch("semantic_kernel.connectors.ai.open_ai.OpenAIChatCompletion")
    def test_openai_provider(self, mock_cls: MagicMock) -> None:
        """OpenAI provider → returns OpenAIChatCompletion."""
        config = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-4o-mini",
            api_key="key",
        )
        result = _create_chat_service_from_config(config)
        assert result is not None
        mock_cls.assert_called_once()

    @patch("semantic_kernel.connectors.ai.open_ai.AzureChatCompletion")
    def test_azure_openai_provider(self, mock_cls: MagicMock) -> None:
        """Azure OpenAI provider → returns AzureChatCompletion."""
        config = LLMProvider(
            provider=ProviderEnum.AZURE_OPENAI,
            name="gpt-4o",
            api_key="key",
            endpoint="https://test.openai.azure.com",
        )
        result = _create_chat_service_from_config(config)
        assert result is not None
        mock_cls.assert_called_once()
