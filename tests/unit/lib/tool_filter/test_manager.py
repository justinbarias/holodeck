"""Unit tests for tool_filter manager."""

from unittest.mock import MagicMock

import pytest
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)

from holodeck.lib.tool_filter.manager import ToolFilterManager
from holodeck.lib.tool_filter.models import ToolFilterConfig, ToolMetadata


class TestToolFilterManager:
    """Tests for ToolFilterManager class."""

    @pytest.fixture
    def mock_kernel(self) -> MagicMock:
        """Create a mock Semantic Kernel."""
        kernel = MagicMock(spec=Kernel)

        # Create mock plugins with functions
        mock_func = MagicMock()
        mock_func.description = "Search the knowledge base"
        mock_func.parameters = []

        mock_plugin = MagicMock()
        mock_plugin.functions = {"search": mock_func}

        kernel.plugins = {"vectorstore": mock_plugin}

        return kernel

    @pytest.fixture
    def sample_config(self) -> ToolFilterConfig:
        """Create sample filter config."""
        return ToolFilterConfig(
            enabled=True,
            top_k=5,
            similarity_threshold=0.3,
            always_include=["get_user"],
            always_include_top_n_used=2,
            search_method="bm25",
        )

    @pytest.fixture
    def disabled_config(self) -> ToolFilterConfig:
        """Create disabled filter config."""
        return ToolFilterConfig(enabled=False)

    def test_initialization(
        self, sample_config: ToolFilterConfig, mock_kernel: MagicMock
    ) -> None:
        """Test manager initialization."""
        manager = ToolFilterManager(
            config=sample_config,
            kernel=mock_kernel,
        )

        assert manager.config == sample_config
        assert manager.kernel == mock_kernel
        assert manager.embedding_service is None
        assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_initialize(
        self, sample_config: ToolFilterConfig, mock_kernel: MagicMock
    ) -> None:
        """Test manager initialization builds index."""
        manager = ToolFilterManager(
            config=sample_config,
            kernel=mock_kernel,
        )

        await manager.initialize()

        assert manager._initialized is True
        assert len(manager.index.tools) > 0

    @pytest.mark.asyncio
    async def test_initialize_idempotent(
        self, sample_config: ToolFilterConfig, mock_kernel: MagicMock
    ) -> None:
        """Test that multiple initialize calls are safe."""
        manager = ToolFilterManager(
            config=sample_config,
            kernel=mock_kernel,
        )

        await manager.initialize()
        initial_tool_count = len(manager.index.tools)

        await manager.initialize()  # Should be no-op
        assert len(manager.index.tools) == initial_tool_count

    @pytest.mark.asyncio
    async def test_filter_tools_returns_all_when_not_initialized(
        self, sample_config: ToolFilterConfig, mock_kernel: MagicMock
    ) -> None:
        """Test filter_tools returns all tools when not initialized."""
        manager = ToolFilterManager(
            config=sample_config,
            kernel=mock_kernel,
        )

        # Manually add some tools to the index without initializing
        manager.index.tools["test-func"] = ToolMetadata(
            name="func",
            full_name="test-func",
            description="Test function",
        )

        results = await manager.filter_tools("test query")
        assert "test-func" in results

    @pytest.mark.asyncio
    async def test_filter_tools_includes_always_include(
        self, sample_config: ToolFilterConfig, mock_kernel: MagicMock
    ) -> None:
        """Test filter_tools always includes specified tools."""
        manager = ToolFilterManager(
            config=sample_config,
            kernel=mock_kernel,
        )

        # Manually populate index
        manager.index.tools["plugin-get_user"] = ToolMetadata(
            name="get_user",
            full_name="plugin-get_user",
            description="Get user information",
        )
        manager.index.tools["plugin-search"] = ToolMetadata(
            name="search",
            full_name="plugin-search",
            description="Search for items",
        )
        manager._initialized = True

        # Build BM25 index
        docs = [
            (t.full_name, manager.index._create_searchable_text(t))
            for t in manager.index.tools.values()
        ]
        manager.index._build_bm25_index(docs)

        results = await manager.filter_tools("unrelated query")

        # get_user should be included (in always_include)
        assert "plugin-get_user" in results

    @pytest.mark.asyncio
    async def test_filter_tools_includes_top_used(
        self, sample_config: ToolFilterConfig, mock_kernel: MagicMock
    ) -> None:
        """Test filter_tools includes most-used tools."""
        manager = ToolFilterManager(
            config=sample_config,
            kernel=mock_kernel,
        )

        # Manually populate index with usage counts
        manager.index.tools["plugin-popular"] = ToolMetadata(
            name="popular",
            full_name="plugin-popular",
            description="Popular function",
            usage_count=100,
        )
        manager.index.tools["plugin-rare"] = ToolMetadata(
            name="rare",
            full_name="plugin-rare",
            description="Rarely used function",
            usage_count=1,
        )
        manager._initialized = True

        # Build BM25 index
        docs = [
            (t.full_name, manager.index._create_searchable_text(t))
            for t in manager.index.tools.values()
        ]
        manager.index._build_bm25_index(docs)

        results = await manager.filter_tools("unrelated query")

        # popular should be included (top used)
        assert "plugin-popular" in results

    def test_create_function_choice_behavior(
        self, sample_config: ToolFilterConfig, mock_kernel: MagicMock
    ) -> None:
        """Test creating FunctionChoiceBehavior with filtered tools."""
        manager = ToolFilterManager(
            config=sample_config,
            kernel=mock_kernel,
        )

        filtered_tools = ["tool1", "tool2", "tool3"]
        behavior = manager.create_function_choice_behavior(filtered_tools)

        assert isinstance(behavior, FunctionChoiceBehavior)

    def test_record_tool_usage(
        self, sample_config: ToolFilterConfig, mock_kernel: MagicMock
    ) -> None:
        """Test recording tool usage updates index."""
        manager = ToolFilterManager(
            config=sample_config,
            kernel=mock_kernel,
        )

        # Add tool to index
        manager.index.tools["plugin-search"] = ToolMetadata(
            name="search",
            full_name="plugin-search",
            description="Search function",
            usage_count=0,
        )

        tool_calls = [{"name": "plugin-search", "arguments": {}}]
        manager.record_tool_usage(tool_calls)

        assert manager.index.tools["plugin-search"].usage_count == 1

    def test_record_tool_usage_multiple(
        self, sample_config: ToolFilterConfig, mock_kernel: MagicMock
    ) -> None:
        """Test recording multiple tool usages."""
        manager = ToolFilterManager(
            config=sample_config,
            kernel=mock_kernel,
        )

        # Add tools to index
        manager.index.tools["tool1"] = ToolMetadata(
            name="tool1",
            full_name="tool1",
            description="Tool 1",
            usage_count=0,
        )
        manager.index.tools["tool2"] = ToolMetadata(
            name="tool2",
            full_name="tool2",
            description="Tool 2",
            usage_count=0,
        )

        tool_calls = [
            {"name": "tool1", "arguments": {}},
            {"name": "tool2", "arguments": {}},
            {"name": "tool1", "arguments": {}},  # Called twice
        ]
        manager.record_tool_usage(tool_calls)

        assert manager.index.tools["tool1"].usage_count == 2
        assert manager.index.tools["tool2"].usage_count == 1

    def test_get_filter_stats(
        self, sample_config: ToolFilterConfig, mock_kernel: MagicMock
    ) -> None:
        """Test getting filter statistics."""
        manager = ToolFilterManager(
            config=sample_config,
            kernel=mock_kernel,
        )

        stats = manager.get_filter_stats()

        assert stats["enabled"] is True
        assert stats["top_k"] == 5
        assert stats["similarity_threshold"] == 0.3
        assert stats["search_method"] == "bm25"
        assert stats["always_include"] == ["get_user"]
        assert stats["always_include_top_n_used"] == 2

    @pytest.mark.asyncio
    async def test_prepare_execution_settings_disabled(
        self, disabled_config: ToolFilterConfig, mock_kernel: MagicMock
    ) -> None:
        """Test prepare_execution_settings returns base settings when disabled."""
        manager = ToolFilterManager(
            config=disabled_config,
            kernel=mock_kernel,
        )

        mock_settings = MagicMock()
        result = await manager.prepare_execution_settings(
            query="test",
            base_settings=mock_settings,
        )

        assert result is mock_settings  # Should return unchanged

    @pytest.mark.asyncio
    async def test_prepare_execution_settings_enabled(
        self, sample_config: ToolFilterConfig, mock_kernel: MagicMock
    ) -> None:
        """Test prepare_execution_settings modifies settings when enabled."""
        manager = ToolFilterManager(
            config=sample_config,
            kernel=mock_kernel,
        )

        # Add tool to index
        manager.index.tools["plugin-search"] = ToolMetadata(
            name="search",
            full_name="plugin-search",
            description="Search function",
        )
        manager._initialized = True

        # Build BM25 index
        docs = [
            (t.full_name, manager.index._create_searchable_text(t))
            for t in manager.index.tools.values()
        ]
        manager.index._build_bm25_index(docs)

        # Create mock settings with model_copy
        mock_settings = MagicMock()
        mock_settings.function_choice_behavior = None
        mock_settings.model_copy = MagicMock(return_value=MagicMock())
        mock_settings.model_copy.return_value.function_choice_behavior = None

        await manager.prepare_execution_settings(
            query="test search",
            base_settings=mock_settings,
        )

        # Should have called model_copy
        mock_settings.model_copy.assert_called_once()
