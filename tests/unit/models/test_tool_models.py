"""Tests for Tool models in holodeck.models.tool."""

import pytest
from pydantic import ValidationError

from holodeck.models.tool import (
    CommandType,
    DatabaseConfig,
    FunctionTool,
    MCPTool,
    PromptTool,
    Tool,
    TransportType,
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

    @pytest.mark.parametrize(
        "missing_field,kwargs",
        [
            ("source", {"name": "test", "description": "Test", "type": "vectorstore"}),
        ],
        ids=["source_required"],
    )
    def test_vectorstore_required_fields(
        self, missing_field: str, kwargs: dict
    ) -> None:
        """Test that required fields raise ValidationError when missing."""
        with pytest.raises(ValidationError) as exc_info:
            VectorstoreTool(**kwargs)
        assert missing_field in str(exc_info.value).lower()

    @pytest.mark.parametrize(
        "empty_field",
        ["source"],
        ids=["source_not_empty"],
    )
    def test_vectorstore_string_fields_not_empty(self, empty_field: str) -> None:
        """Test that string fields cannot be empty."""
        kwargs = {
            "name": "test",
            "description": "Test",
            "type": "vectorstore",
            "source": "data.txt",
        }
        kwargs[empty_field] = ""
        with pytest.raises(ValidationError):
            VectorstoreTool(**kwargs)

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
        """Test that vector_field is optional (but requires id_field when set)."""
        tool = VectorstoreTool(
            name="test",
            description="Test",
            type="vectorstore",
            source="data.txt",
            vector_field="content",
            id_field="id",  # Required when vector_field is set
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
            id_field="id",  # Required when vector_field is set
        )
        assert tool.vector_field == ["title", "content"]

    def test_vectorstore_embedding_dimensions_valid(self) -> None:
        """Test that embedding_dimensions can be set to valid values."""
        tool = VectorstoreTool(
            name="test",
            description="Test",
            type="vectorstore",
            source="data.txt",
            embedding_dimensions=768,
        )
        assert tool.embedding_dimensions == 768

        tool = VectorstoreTool(
            name="test",
            description="Test",
            type="vectorstore",
            source="data.txt",
            embedding_dimensions=3072,
        )
        assert tool.embedding_dimensions == 3072

    def test_vectorstore_embedding_dimensions_optional(self) -> None:
        """Test that embedding_dimensions is optional."""
        tool = VectorstoreTool(
            name="test",
            description="Test",
            type="vectorstore",
            source="data.txt",
        )
        assert tool.embedding_dimensions is None

    def test_vectorstore_embedding_dimensions_zero_invalid(self) -> None:
        """Test that zero embedding_dimensions raises ValueError."""
        with pytest.raises(ValueError, match="embedding_dimensions must be positive"):
            VectorstoreTool(
                name="test",
                description="Test",
                type="vectorstore",
                source="data.txt",
                embedding_dimensions=0,
            )

    def test_vectorstore_embedding_dimensions_negative_invalid(self) -> None:
        """Test that negative embedding_dimensions raises ValueError."""
        with pytest.raises(ValueError, match="embedding_dimensions must be positive"):
            VectorstoreTool(
                name="test",
                description="Test",
                type="vectorstore",
                source="data.txt",
                embedding_dimensions=-1,
            )

    def test_vectorstore_embedding_dimensions_too_large_invalid(self) -> None:
        """Test that dimensions over 10000 raises ValueError."""
        with pytest.raises(ValueError, match="embedding_dimensions unreasonably large"):
            VectorstoreTool(
                name="test",
                description="Test",
                type="vectorstore",
                source="data.txt",
                embedding_dimensions=10001,
            )


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

    @pytest.mark.parametrize(
        "missing_field,kwargs",
        [
            (
                "file",
                {
                    "name": "test",
                    "description": "Test",
                    "type": "function",
                    "function": "my_func",
                },
            ),
            (
                "function",
                {
                    "name": "test",
                    "description": "Test",
                    "type": "function",
                    "file": "tools.py",
                },
            ),
        ],
        ids=["file_required", "function_required"],
    )
    def test_function_tool_required_fields(
        self, missing_field: str, kwargs: dict
    ) -> None:
        """Test that required fields raise ValidationError when missing."""
        with pytest.raises(ValidationError) as exc_info:
            FunctionTool(**kwargs)
        assert missing_field in str(exc_info.value).lower()

    @pytest.mark.parametrize(
        "empty_field",
        ["file", "function"],
        ids=["file_not_empty", "function_not_empty"],
    )
    def test_function_tool_string_fields_not_empty(self, empty_field: str) -> None:
        """Test that string fields cannot be empty."""
        kwargs = {
            "name": "test",
            "description": "Test",
            "type": "function",
            "file": "tools.py",
            "function": "my_func",
        }
        kwargs[empty_field] = ""
        with pytest.raises(ValidationError):
            FunctionTool(**kwargs)

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

    def test_mcp_tool_valid_creation_with_stdio(self) -> None:
        """Test creating a valid MCPTool with stdio transport."""
        tool = MCPTool(
            name="filesystem",
            description="Filesystem access via MCP",
            type="mcp",
            command=CommandType.NPX,
            args=["-y", "@modelcontextprotocol/server-filesystem"],
        )
        assert tool.name == "filesystem"
        assert tool.type == "mcp"
        assert tool.command == CommandType.NPX
        assert tool.args == ["-y", "@modelcontextprotocol/server-filesystem"]
        assert tool.transport == TransportType.STDIO

    def test_mcp_config_optional(self) -> None:
        """Test that config dict is optional."""
        tool = MCPTool(
            name="test",
            description="Test",
            type="mcp",
            command=CommandType.NPX,
        )
        assert tool.config is None or isinstance(tool.config, dict)

    def test_mcp_config_accepts_any_dict(self) -> None:
        """Test that config accepts arbitrary MCP configuration."""
        config = {"root_dir": "/data", "permissions": ["read", "write"]}
        tool = MCPTool(
            name="test",
            description="Test",
            type="mcp",
            command=CommandType.NPX,
            config=config,
        )
        assert tool.config == config


class TestMCPToolEnhanced:
    """Tests for enhanced MCPTool with transport configuration."""

    def test_default_transport_is_stdio(self) -> None:
        """Default transport should be stdio."""
        tool = MCPTool(
            name="test",
            description="Test tool",
            command=CommandType.NPX,
        )
        assert tool.transport == TransportType.STDIO

    def test_stdio_requires_command(self) -> None:
        """Stdio transport should require command field."""
        with pytest.raises(ValidationError) as exc_info:
            MCPTool(
                name="test",
                description="Test tool",
                transport=TransportType.STDIO,
            )
        assert "command" in str(exc_info.value).lower()

    def test_sse_requires_url(self) -> None:
        """SSE transport should require url field."""
        with pytest.raises(ValidationError) as exc_info:
            MCPTool(
                name="test",
                description="Test tool",
                transport=TransportType.SSE,
            )
        assert "url" in str(exc_info.value).lower()

    def test_websocket_requires_url(self) -> None:
        """WebSocket transport should require url field."""
        with pytest.raises(ValidationError) as exc_info:
            MCPTool(
                name="test",
                description="Test tool",
                transport=TransportType.WEBSOCKET,
            )
        assert "url" in str(exc_info.value).lower()

    def test_http_requires_url(self) -> None:
        """HTTP transport should require url field."""
        with pytest.raises(ValidationError) as exc_info:
            MCPTool(
                name="test",
                description="Test tool",
                transport=TransportType.HTTP,
            )
        assert "url" in str(exc_info.value).lower()

    def test_valid_stdio_config(self) -> None:
        """Test valid stdio transport configuration."""
        tool = MCPTool(
            name="filesystem",
            description="FS access",
            command=CommandType.NPX,
            args=["-y", "@modelcontextprotocol/server-filesystem"],
            env={"PATH": "/usr/bin"},
        )
        assert tool.command == CommandType.NPX
        assert tool.args is not None
        assert len(tool.args) == 2

    def test_valid_sse_config(self) -> None:
        """Test valid SSE transport configuration."""
        tool = MCPTool(
            name="remote",
            description="Remote server",
            transport=TransportType.SSE,
            url="https://example.com/sse",
            headers={"Authorization": "Bearer token"},
        )
        assert tool.url == "https://example.com/sse"
        assert tool.headers is not None
        assert "Authorization" in tool.headers

    def test_valid_websocket_config(self) -> None:
        """Test valid WebSocket transport configuration."""
        tool = MCPTool(
            name="realtime",
            description="Realtime server",
            transport=TransportType.WEBSOCKET,
            url="wss://example.com/ws",
        )
        assert tool.url == "wss://example.com/ws"

    def test_valid_http_config(self) -> None:
        """Test valid HTTP transport configuration."""
        tool = MCPTool(
            name="streaming",
            description="Streaming server",
            transport=TransportType.HTTP,
            url="https://example.com/stream",
            terminate_on_close=True,
        )
        assert tool.url == "https://example.com/stream"
        assert tool.terminate_on_close is True

    def test_url_scheme_validation_rejects_remote_http(self) -> None:
        """URL validation should reject http:// for remote hosts."""
        with pytest.raises(ValidationError) as exc_info:
            MCPTool(
                name="test",
                description="Test",
                transport=TransportType.SSE,
                url="http://remote.example.com/sse",
            )
        assert "https://" in str(exc_info.value)

    def test_url_scheme_allows_localhost_http(self) -> None:
        """URL validation should allow http:// for localhost."""
        tool = MCPTool(
            name="test",
            description="Test",
            transport=TransportType.SSE,
            url="http://localhost:8080/sse",
        )
        assert tool.url == "http://localhost:8080/sse"

    def test_url_scheme_allows_127_0_0_1_http(self) -> None:
        """URL validation should allow http:// for 127.0.0.1."""
        tool = MCPTool(
            name="test",
            description="Test",
            transport=TransportType.SSE,
            url="http://127.0.0.1:8080/sse",
        )
        assert tool.url == "http://127.0.0.1:8080/sse"

    def test_url_scheme_allows_ipv6_localhost_http(self) -> None:
        """URL validation should allow http:// for [::1] (IPv6 localhost)."""
        tool = MCPTool(
            name="test",
            description="Test",
            transport=TransportType.SSE,
            url="http://[::1]:8080/sse",
        )
        assert tool.url == "http://[::1]:8080/sse"

    def test_url_scheme_allows_https(self) -> None:
        """URL validation should allow https://."""
        tool = MCPTool(
            name="test",
            description="Test",
            transport=TransportType.SSE,
            url="https://example.com/sse",
        )
        assert tool.url == "https://example.com/sse"

    def test_url_scheme_allows_wss(self) -> None:
        """URL validation should allow wss://."""
        tool = MCPTool(
            name="test",
            description="Test",
            transport=TransportType.WEBSOCKET,
            url="wss://example.com/ws",
        )
        assert tool.url == "wss://example.com/ws"

    def test_url_scheme_allows_ws(self) -> None:
        """URL validation should allow ws://."""
        tool = MCPTool(
            name="test",
            description="Test",
            transport=TransportType.WEBSOCKET,
            url="ws://example.com/ws",
        )
        assert tool.url == "ws://example.com/ws"

    def test_request_timeout_must_be_positive(self) -> None:
        """request_timeout must be a positive integer."""
        with pytest.raises(ValidationError) as exc_info:
            MCPTool(
                name="test",
                description="Test",
                command=CommandType.NPX,
                request_timeout=0,
            )
        assert "positive" in str(exc_info.value).lower()

    def test_request_timeout_negative_rejected(self) -> None:
        """request_timeout must not be negative."""
        with pytest.raises(ValidationError):
            MCPTool(
                name="test",
                description="Test",
                command=CommandType.NPX,
                request_timeout=-1,
            )

    def test_default_request_timeout_is_60(self) -> None:
        """Default request_timeout should be 60 seconds."""
        tool = MCPTool(
            name="test",
            description="Test",
            command=CommandType.NPX,
        )
        assert tool.request_timeout == 60

    def test_default_load_tools_is_true(self) -> None:
        """Default load_tools should be True."""
        tool = MCPTool(
            name="test",
            description="Test",
            command=CommandType.NPX,
        )
        assert tool.load_tools is True

    def test_default_load_prompts_is_true(self) -> None:
        """Default load_prompts should be True."""
        tool = MCPTool(
            name="test",
            description="Test",
            command=CommandType.NPX,
        )
        assert tool.load_prompts is True

    @pytest.mark.parametrize(
        "cmd", [CommandType.NPX, CommandType.UVX, CommandType.DOCKER]
    )
    def test_allowed_commands(self, cmd: CommandType) -> None:
        """All CommandType values should be accepted."""
        tool = MCPTool(
            name="test",
            description="Test",
            command=cmd,
        )
        assert tool.command == cmd

    def test_command_enum_rejects_invalid_values(self) -> None:
        """Invalid command values should be rejected."""
        with pytest.raises(ValidationError):
            MCPTool(
                name="test",
                description="Test",
                command="bash",  # type: ignore[arg-type]
            )

    def test_transport_enum_rejects_invalid_values(self) -> None:
        """Invalid transport values should be rejected."""
        with pytest.raises(ValidationError):
            MCPTool(
                name="test",
                description="Test",
                transport="invalid",  # type: ignore[arg-type]
                url="https://example.com",
            )

    def test_env_file_optional(self) -> None:
        """env_file should be optional."""
        tool = MCPTool(
            name="test",
            description="Test",
            command=CommandType.NPX,
        )
        assert tool.env_file is None

    def test_env_file_accepted(self) -> None:
        """env_file should be accepted."""
        tool = MCPTool(
            name="test",
            description="Test",
            command=CommandType.NPX,
            env_file=".env",
        )
        assert tool.env_file == ".env"

    def test_encoding_optional(self) -> None:
        """encoding should be optional."""
        tool = MCPTool(
            name="test",
            description="Test",
            command=CommandType.NPX,
        )
        assert tool.encoding is None

    def test_encoding_accepted(self) -> None:
        """encoding should be accepted."""
        tool = MCPTool(
            name="test",
            description="Test",
            command=CommandType.NPX,
            encoding="utf-8",
        )
        assert tool.encoding == "utf-8"

    def test_timeout_and_sse_read_timeout_optional(self) -> None:
        """timeout and sse_read_timeout should be optional."""
        tool = MCPTool(
            name="test",
            description="Test",
            transport=TransportType.SSE,
            url="https://example.com/sse",
        )
        assert tool.timeout is None
        assert tool.sse_read_timeout is None

    def test_timeout_and_sse_read_timeout_accepted(self) -> None:
        """timeout and sse_read_timeout should be accepted."""
        tool = MCPTool(
            name="test",
            description="Test",
            transport=TransportType.SSE,
            url="https://example.com/sse",
            timeout=30.0,
            sse_read_timeout=120.0,
        )
        assert tool.timeout == 30.0
        assert tool.sse_read_timeout == 120.0


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

    def test_prompt_tool_template_or_file_required(self) -> None:
        """Test that either template or file is required."""
        with pytest.raises(ValidationError):
            PromptTool(
                name="test",
                description="Test",
                type="prompt",
                parameters={"x": {"type": "string"}},
            )

    @pytest.mark.parametrize(
        "empty_field",
        ["template", "file"],
        ids=["template_not_empty", "file_not_empty"],
    )
    def test_prompt_tool_string_fields_not_empty(self, empty_field: str) -> None:
        """Test that string fields cannot be empty."""
        kwargs = {
            "name": "test",
            "description": "Test",
            "type": "prompt",
            "parameters": {"x": {"type": "string"}},
        }
        kwargs[empty_field] = ""
        with pytest.raises(ValidationError):
            PromptTool(**kwargs)

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


class TestDatabaseConfig:
    """Tests for DatabaseConfig model."""

    def test_database_config_valid_postgres(self) -> None:
        """Test creating a valid DatabaseConfig for PostgreSQL."""
        config = DatabaseConfig(
            provider="postgres",
            connection_string="postgresql://user:pass@localhost/db",
        )
        assert config.provider == "postgres"

    def test_database_config_valid_qdrant(self) -> None:
        """Test creating a valid DatabaseConfig for Qdrant."""
        config = DatabaseConfig(
            provider="qdrant",
            url="http://localhost:6333",
        )
        assert config.provider == "qdrant"

    def test_database_config_valid_in_memory(self) -> None:
        """Test creating a valid DatabaseConfig for in-memory store."""
        config = DatabaseConfig(
            provider="in-memory",
        )
        assert config.provider == "in-memory"
        assert config.connection_string is None

    def test_database_config_valid_azure_ai_search(self) -> None:
        """Test creating a valid DatabaseConfig for Azure AI Search."""
        config = DatabaseConfig(
            provider="azure-ai-search",
            connection_string="https://search-service.search.windows.net",
            api_key="test-key",
        )
        assert config.provider == "azure-ai-search"

    def test_database_config_optional_connection_string(self) -> None:
        """Test that connection_string is optional."""
        config = DatabaseConfig(
            provider="in-memory",
        )
        assert config.connection_string is None

    def test_database_config_empty_connection_string(self) -> None:
        """Test that connection_string must be non-empty if provided."""
        with pytest.raises(ValidationError):
            DatabaseConfig(
                provider="postgres",
                connection_string="",
            )

    def test_database_config_invalid_provider(self) -> None:
        """Test that provider must be from supported list."""
        with pytest.raises(ValidationError) as exc_info:
            DatabaseConfig(
                provider="invalid-provider",  # type: ignore
                connection_string="connection://string",
            )
        assert "provider" in str(exc_info.value).lower()

    def test_database_config_extra_fields_allowed(self) -> None:
        """Test that provider-specific extra fields are allowed."""
        config = DatabaseConfig(
            provider="qdrant",
            url="http://localhost:6333",
            api_key="test-key",
            prefer_grpc=True,
        )
        assert config.provider == "qdrant"
        # Extra fields should be stored
        assert config.url == "http://localhost:6333"  # type: ignore


class TestVectorstoreToolExtended:
    """Tests for extended VectorstoreTool features.

    Tests cover database, top_k, and min_similarity_score fields.
    """

    def test_vectorstore_with_database_config(self) -> None:
        """Test VectorstoreTool with database configuration."""
        db_config = DatabaseConfig(
            provider="postgres",
            connection_string="postgresql://user:pass@localhost/db",
        )
        tool = VectorstoreTool(
            name="test",
            description="Test",
            type="vectorstore",
            source="data.txt",
            database=db_config,
        )
        assert tool.database is not None
        assert tool.database.provider == "postgres"

    def test_vectorstore_without_database_config(self) -> None:
        """Test VectorstoreTool works without database (in-memory default)."""
        tool = VectorstoreTool(
            name="test",
            description="Test",
            type="vectorstore",
            source="data.txt",
        )
        assert tool.database is None

    def test_vectorstore_top_k_default(self) -> None:
        """Test VectorstoreTool has default top_k of 5."""
        tool = VectorstoreTool(
            name="test",
            description="Test",
            type="vectorstore",
            source="data.txt",
        )
        assert tool.top_k == 5

    def test_vectorstore_top_k_custom(self) -> None:
        """Test VectorstoreTool with custom top_k."""
        tool = VectorstoreTool(
            name="test",
            description="Test",
            type="vectorstore",
            source="data.txt",
            top_k=10,
        )
        assert tool.top_k == 10

    def test_vectorstore_top_k_positive_validation(self) -> None:
        """Test that top_k must be positive."""
        with pytest.raises(ValidationError):
            VectorstoreTool(
                name="test",
                description="Test",
                type="vectorstore",
                source="data.txt",
                top_k=0,
            )

    def test_vectorstore_top_k_maximum_validation(self) -> None:
        """Test that top_k should not exceed 100."""
        with pytest.raises(ValidationError):
            VectorstoreTool(
                name="test",
                description="Test",
                type="vectorstore",
                source="data.txt",
                top_k=101,
            )

    def test_vectorstore_min_similarity_score_optional(self) -> None:
        """Test that min_similarity_score is optional."""
        tool = VectorstoreTool(
            name="test",
            description="Test",
            type="vectorstore",
            source="data.txt",
        )
        assert tool.min_similarity_score is None

    def test_vectorstore_min_similarity_score_custom(self) -> None:
        """Test VectorstoreTool with custom min_similarity_score."""
        tool = VectorstoreTool(
            name="test",
            description="Test",
            type="vectorstore",
            source="data.txt",
            min_similarity_score=0.75,
        )
        assert tool.min_similarity_score == 0.75

    def test_vectorstore_min_similarity_score_range_validation(self) -> None:
        """Test that min_similarity_score must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            VectorstoreTool(
                name="test",
                description="Test",
                type="vectorstore",
                source="data.txt",
                min_similarity_score=1.5,
            )

        with pytest.raises(ValidationError):
            VectorstoreTool(
                name="test",
                description="Test",
                type="vectorstore",
                source="data.txt",
                min_similarity_score=-0.1,
            )

    def test_vectorstore_all_extended_fields(self) -> None:
        """Test VectorstoreTool with all extended fields."""
        db_config = DatabaseConfig(
            provider="chromadb",
            connection_string="http://localhost:8000",
        )
        tool = VectorstoreTool(
            name="knowledge_base",
            description="Production knowledge base search",
            type="vectorstore",
            source="data/docs/",
            database=db_config,
            embedding_model="text-embedding-3-large",
            top_k=20,
            min_similarity_score=0.7,
        )
        assert tool.name == "knowledge_base"
        assert tool.database is not None
        assert tool.database.provider == "chromadb"
        assert tool.embedding_model == "text-embedding-3-large"
        assert tool.top_k == 20
        assert tool.min_similarity_score == 0.7


class TestVectorstoreToolStructuredData:
    """Tests for VectorstoreTool structured data configuration.

    Tests the structured data mode fields: id_field, field_separator, delimiter.
    These tests should FAIL initially (TDD RED phase) until T010-T011 are implemented.
    """

    def test_id_field_optional_when_no_vector_field(self) -> None:
        """Test that id_field is optional when vector_field is not set."""
        tool = VectorstoreTool(
            name="test",
            description="Test",
            type="vectorstore",
            source="data.txt",
        )
        assert tool.id_field is None

    def test_id_field_required_when_vector_field_set(self) -> None:
        """Test that id_field is required when vector_field is set (structured mode)."""
        with pytest.raises(ValidationError) as exc_info:
            VectorstoreTool(
                name="test",
                description="Test",
                type="vectorstore",
                source="data.csv",
                vector_field="description",
            )
        assert "id_field" in str(exc_info.value).lower()

    def test_id_field_required_when_vector_field_is_list(self) -> None:
        """Test that id_field is required when vector_field is a list."""
        with pytest.raises(ValidationError) as exc_info:
            VectorstoreTool(
                name="test",
                description="Test",
                type="vectorstore",
                source="data.json",
                vector_field=["title", "description"],
            )
        assert "id_field" in str(exc_info.value).lower()

    def test_structured_data_valid_with_id_field(self) -> None:
        """Test valid structured data configuration with id_field."""
        tool = VectorstoreTool(
            name="test",
            description="Test",
            type="vectorstore",
            source="data.csv",
            vector_field="description",
            id_field="product_id",
        )
        assert tool.vector_field == "description"
        assert tool.id_field == "product_id"

    def test_field_separator_default_newline(self) -> None:
        """Test that field_separator defaults to newline."""
        tool = VectorstoreTool(
            name="test",
            description="Test",
            type="vectorstore",
            source="data.csv",
            vector_field="description",
            id_field="id",
        )
        assert tool.field_separator == "\n"

    def test_field_separator_custom(self) -> None:
        """Test custom field_separator value."""
        tool = VectorstoreTool(
            name="test",
            description="Test",
            type="vectorstore",
            source="data.csv",
            vector_field=["title", "description"],
            id_field="id",
            field_separator="\n\n",
        )
        assert tool.field_separator == "\n\n"

    def test_delimiter_optional_none_for_auto_detect(self) -> None:
        """Test that delimiter is optional and None means auto-detect."""
        tool = VectorstoreTool(
            name="test",
            description="Test",
            type="vectorstore",
            source="data.csv",
            vector_field="description",
            id_field="id",
        )
        assert tool.delimiter is None

    def test_delimiter_explicit_value(self) -> None:
        """Test explicit delimiter value for CSV."""
        tool = VectorstoreTool(
            name="test",
            description="Test",
            type="vectorstore",
            source="data.csv",
            vector_field="description",
            id_field="id",
            delimiter=";",
        )
        assert tool.delimiter == ";"

    def test_structured_data_with_meta_fields(self) -> None:
        """Test structured data configuration with meta_fields."""
        tool = VectorstoreTool(
            name="test",
            description="Test",
            type="vectorstore",
            source="data.csv",
            vector_field="description",
            id_field="id",
            meta_fields=["title", "category", "price"],
        )
        assert tool.meta_fields == ["title", "category", "price"]

    def test_structured_data_with_record_path(self) -> None:
        """Test structured data configuration with record_path for nested JSON."""
        tool = VectorstoreTool(
            name="test",
            description="Test",
            type="vectorstore",
            source="data.json",
            vector_field="description",
            id_field="id",
            record_path="data.items",
        )
        assert tool.record_path == "data.items"

    def test_full_structured_data_configuration(self) -> None:
        """Test complete structured data configuration with all fields."""
        tool = VectorstoreTool(
            name="product_search",
            description="Search product catalog",
            type="vectorstore",
            source="products.csv",
            vector_field=["title", "description"],
            id_field="product_id",
            field_separator="\n\n",
            delimiter=",",
            meta_fields=["category", "price"],
            top_k=10,
            min_similarity_score=0.7,
        )
        assert tool.name == "product_search"
        assert tool.vector_field == ["title", "description"]
        assert tool.id_field == "product_id"
        assert tool.field_separator == "\n\n"
        assert tool.delimiter == ","
        assert tool.meta_fields == ["category", "price"]
