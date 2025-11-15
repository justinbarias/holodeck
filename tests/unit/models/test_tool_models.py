"""Tests for Tool models in holodeck.models.tool."""

import pytest
from pydantic import ValidationError

from holodeck.models.tool import (
    FunctionTool,
    MCPTool,
    PromptTool,
    Tool,
    VectorstoreTool,
)


class TestToolBase:
    """Tests for Tool base model."""

    def test_tool_type_field_required(self) -> None:
        """Test that type field is required."""
        with pytest.raises(ValidationError) as exc_info:
            Tool(name="test_tool", description="A test tool")
        assert "type" in str(exc_info.value).lower()

    def test_tool_type_discriminator_validates_tool_type(self) -> None:
        """Test that type field accepts any string value."""
        # Note: The base Tool class accepts any type string.
        # Type validation happens at the concrete implementation level.
        tool = Tool(
            name="test_tool",
            description="A test tool",
            type="invalid_type",
        )
        assert tool.type == "invalid_type"

    def test_tool_concrete_implementations_required(self) -> None:
        """Test that concrete tool implementations are used for specific types."""
        # For vectorstore type, use VectorstoreTool directly
        tool = VectorstoreTool(
            name="test_tool",
            description="A test tool",
            type="vectorstore",
            source="data.txt",
        )
        assert tool.type == "vectorstore"
        assert tool.source == "data.txt"

    def test_tool_name_required(self) -> None:
        """Test that name is required."""
        with pytest.raises(ValidationError) as exc_info:
            VectorstoreTool(
                description="Test vectorstore",
                type="vectorstore",
                source="data.txt",
            )
        assert "name" in str(exc_info.value).lower()

    def test_tool_description_required(self) -> None:
        """Test that description is required."""
        with pytest.raises(ValidationError) as exc_info:
            VectorstoreTool(
                name="my_tool",
                type="vectorstore",
                source="data.txt",
            )
        assert "description" in str(exc_info.value).lower()


class TestVectorstoreTool:
    """Tests for VectorstoreTool model."""

    def test_vectorstore_tool_valid_creation(self) -> None:
        """Test creating a valid VectorstoreTool."""
        tool = VectorstoreTool(
            name="my_vectorstore",
            description="Search knowledge base",
            type="vectorstore",
            source="data/documents",
        )
        assert tool.name == "my_vectorstore"
        assert tool.description == "Search knowledge base"
        assert tool.type == "vectorstore"
        assert tool.source == "data/documents"

    def test_vectorstore_source_required(self) -> None:
        """Test that source field is required."""
        with pytest.raises(ValidationError) as exc_info:
            VectorstoreTool(
                name="test",
                description="Test",
                type="vectorstore",
            )
        assert "source" in str(exc_info.value).lower()

    def test_vectorstore_source_not_empty(self) -> None:
        """Test that source cannot be empty string."""
        with pytest.raises(ValidationError):
            VectorstoreTool(
                name="test",
                description="Test",
                type="vectorstore",
                source="",
            )

    def test_vectorstore_chunk_size_optional(self) -> None:
        """Test that chunk_size is optional with default."""
        tool = VectorstoreTool(
            name="test",
            description="Test",
            type="vectorstore",
            source="data.txt",
        )
        assert tool.chunk_size is None or isinstance(tool.chunk_size, int)

    def test_vectorstore_chunk_size_positive(self) -> None:
        """Test that chunk_size must be positive."""
        with pytest.raises(ValidationError):
            VectorstoreTool(
                name="test",
                description="Test",
                type="vectorstore",
                source="data.txt",
                chunk_size=-1,
            )

    def test_vectorstore_chunk_overlap_optional(self) -> None:
        """Test that chunk_overlap is optional."""
        tool = VectorstoreTool(
            name="test",
            description="Test",
            type="vectorstore",
            source="data.txt",
            chunk_overlap=50,
        )
        assert tool.chunk_overlap == 50

    def test_vectorstore_embedding_model_optional(self) -> None:
        """Test that embedding_model is optional."""
        tool = VectorstoreTool(
            name="test",
            description="Test",
            type="vectorstore",
            source="data.txt",
            embedding_model="text-embedding-ada-002",
        )
        assert tool.embedding_model == "text-embedding-ada-002"

    def test_vectorstore_vector_field_optional(self) -> None:
        """Test that vector_field is optional."""
        tool = VectorstoreTool(
            name="test",
            description="Test",
            type="vectorstore",
            source="data.txt",
            vector_field="content",
        )
        assert tool.vector_field == "content"

    def test_vectorstore_vector_field_can_be_list(self) -> None:
        """Test that vector_field can be a list of fields."""
        tool = VectorstoreTool(
            name="test",
            description="Test",
            type="vectorstore",
            source="data.txt",
            vector_field=["title", "content"],
        )
        assert tool.vector_field == ["title", "content"]


class TestFunctionTool:
    """Tests for FunctionTool model."""

    def test_function_tool_valid_creation(self) -> None:
        """Test creating a valid FunctionTool."""
        tool = FunctionTool(
            name="my_function",
            description="Call a Python function",
            type="function",
            file="tools/search.py",
            function="search_documents",
        )
        assert tool.name == "my_function"
        assert tool.file == "tools/search.py"
        assert tool.function == "search_documents"

    def test_function_tool_file_required(self) -> None:
        """Test that file field is required."""
        with pytest.raises(ValidationError) as exc_info:
            FunctionTool(
                name="test",
                description="Test",
                type="function",
                function="my_func",
            )
        assert "file" in str(exc_info.value).lower()

    def test_function_tool_function_required(self) -> None:
        """Test that function field is required."""
        with pytest.raises(ValidationError) as exc_info:
            FunctionTool(
                name="test",
                description="Test",
                type="function",
                file="tools.py",
            )
        assert "function" in str(exc_info.value).lower()

    def test_function_tool_file_not_empty(self) -> None:
        """Test that file cannot be empty string."""
        with pytest.raises(ValidationError):
            FunctionTool(
                name="test",
                description="Test",
                type="function",
                file="",
                function="my_func",
            )

    def test_function_tool_function_not_empty(self) -> None:
        """Test that function name cannot be empty string."""
        with pytest.raises(ValidationError):
            FunctionTool(
                name="test",
                description="Test",
                type="function",
                file="tools.py",
                function="",
            )

    def test_function_tool_parameters_optional(self) -> None:
        """Test that parameters schema is optional."""
        tool = FunctionTool(
            name="test",
            description="Test",
            type="function",
            file="tools.py",
            function="my_func",
        )
        assert tool.parameters is None or isinstance(tool.parameters, dict)

    def test_function_tool_with_parameters(self) -> None:
        """Test FunctionTool with parameters."""
        tool = FunctionTool(
            name="test",
            description="Test",
            type="function",
            file="tools.py",
            function="my_func",
            parameters={
                "query": {"type": "string", "description": "Search query"},
            },
        )
        assert tool.parameters is not None
        assert "query" in tool.parameters


class TestMCPTool:
    """Tests for MCPTool model."""

    def test_mcp_tool_valid_creation(self) -> None:
        """Test creating a valid MCPTool."""
        tool = MCPTool(
            name="filesystem",
            description="Filesystem access via MCP",
            type="mcp",
            server="@modelcontextprotocol/server-filesystem",
        )
        assert tool.name == "filesystem"
        assert tool.server == "@modelcontextprotocol/server-filesystem"
        assert tool.type == "mcp"

    def test_mcp_server_required(self) -> None:
        """Test that server field is required."""
        with pytest.raises(ValidationError) as exc_info:
            MCPTool(
                name="test",
                description="Test",
                type="mcp",
            )
        assert "server" in str(exc_info.value).lower()

    def test_mcp_server_not_empty(self) -> None:
        """Test that server cannot be empty string."""
        with pytest.raises(ValidationError):
            MCPTool(
                name="test",
                description="Test",
                type="mcp",
                server="",
            )

    def test_mcp_config_optional(self) -> None:
        """Test that config dict is optional."""
        tool = MCPTool(
            name="test",
            description="Test",
            type="mcp",
            server="my_server",
        )
        assert tool.config is None or isinstance(tool.config, dict)

    def test_mcp_config_accepts_any_dict(self) -> None:
        """Test that config accepts arbitrary MCP configuration."""
        config = {"root_dir": "/data", "permissions": ["read", "write"]}
        tool = MCPTool(
            name="test",
            description="Test",
            type="mcp",
            server="filesystem",
            config=config,
        )
        assert tool.config == config


class TestPromptTool:
    """Tests for PromptTool model."""

    def test_prompt_tool_with_inline_template(self) -> None:
        """Test creating PromptTool with inline template."""
        tool = PromptTool(
            name="classifier",
            description="Classify text",
            type="prompt",
            template="Classify the following text: {text}",
            parameters={"text": {"type": "string", "description": "Text to classify"}},
        )
        assert tool.name == "classifier"
        assert tool.template == "Classify the following text: {text}"
        assert tool.type == "prompt"

    def test_prompt_tool_with_file(self) -> None:
        """Test creating PromptTool with file."""
        tool = PromptTool(
            name="classifier",
            description="Classify text",
            type="prompt",
            file="prompts/classifier.txt",
            parameters={"text": {"type": "string"}},
        )
        assert tool.file == "prompts/classifier.txt"
        assert tool.template is None

    def test_prompt_tool_template_and_file_mutually_exclusive(self) -> None:
        """Test that template and file are mutually exclusive."""
        with pytest.raises(ValidationError):
            PromptTool(
                name="test",
                description="Test",
                type="prompt",
                template="Some template",
                file="prompts/template.txt",
                parameters={"x": {"type": "string"}},
            )

    def test_prompt_tool_template_required_if_no_file(self) -> None:
        """Test that either template or file is required."""
        with pytest.raises(ValidationError):
            PromptTool(
                name="test",
                description="Test",
                type="prompt",
                parameters={"x": {"type": "string"}},
            )

    def test_prompt_tool_file_required_if_no_template(self) -> None:
        """Test that file is validated if template is not provided."""
        # This will fail because neither template nor file is provided
        with pytest.raises(ValidationError):
            PromptTool(
                name="test",
                description="Test",
                type="prompt",
                parameters={"x": {"type": "string"}},
            )

    def test_prompt_tool_template_not_empty(self) -> None:
        """Test that template cannot be empty string."""
        with pytest.raises(ValidationError):
            PromptTool(
                name="test",
                description="Test",
                type="prompt",
                template="",
                parameters={"x": {"type": "string"}},
            )

    def test_prompt_tool_file_not_empty(self) -> None:
        """Test that file cannot be empty string."""
        with pytest.raises(ValidationError):
            PromptTool(
                name="test",
                description="Test",
                type="prompt",
                file="",
                parameters={"x": {"type": "string"}},
            )

    def test_prompt_tool_parameters_required(self) -> None:
        """Test that parameters field is required."""
        with pytest.raises(ValidationError):
            PromptTool(
                name="test",
                description="Test",
                type="prompt",
                template="Test template",
            )

    def test_prompt_tool_parameters_not_empty(self) -> None:
        """Test that parameters cannot be empty."""
        with pytest.raises(ValidationError):
            PromptTool(
                name="test",
                description="Test",
                type="prompt",
                template="Test template",
                parameters={},
            )

    def test_prompt_tool_model_optional(self) -> None:
        """Test that model config is optional."""
        tool = PromptTool(
            name="test",
            description="Test",
            type="prompt",
            template="Test",
            parameters={"x": {"type": "string"}},
        )
        assert tool.model is None

    def test_prompt_tool_description_optional(self) -> None:
        """Test that description can be provided or omitted."""
        tool = PromptTool(
            name="test",
            description="Prompt tool description",
            type="prompt",
            template="Test",
            parameters={"x": {"type": "string"}},
        )
        assert tool.description == "Prompt tool description"
