"""Content block models for MCP responses.

These models represent the different content types that MCP servers can return:
- TextContent: Plain text content
- ImageContent: Base64-encoded or URL-referenced images
- AudioContent: Base64-encoded audio data
- BinaryContent: Binary data from embedded resources

The MCPToolResult model provides a standardized wrapper for MCP tool responses.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from holodeck.tools.mcp.errors import MCPError


class TextContent(BaseModel):
    """Plain text content from MCP response."""

    model_config = ConfigDict(frozen=True)

    type: Literal["text"] = "text"
    text: str = Field(..., description="Text content")


class ImageContent(BaseModel):
    """Image content from MCP response.

    Contains base64-encoded image data and its MIME type.
    Common MIME types: image/png, image/jpeg, image/gif, image/webp
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["image"] = "image"
    data: str = Field(..., description="Base64-encoded image data")
    mime_type: str = Field(..., description="MIME type (e.g., image/png)")


class AudioContent(BaseModel):
    """Audio content from MCP response.

    Contains base64-encoded audio data and its MIME type.
    Common MIME types: audio/wav, audio/mp3, audio/ogg, audio/webm
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["audio"] = "audio"
    data: str = Field(..., description="Base64-encoded audio data")
    mime_type: str = Field(..., description="MIME type (e.g., audio/wav)")


class BinaryContent(BaseModel):
    """Binary content from MCP embedded resources.

    Used for EmbeddedResource and ResourceLink responses from MCP servers.
    Contains raw binary data with optional MIME type and resource URI.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["binary"] = "binary"
    data: bytes = Field(..., description="Binary data")
    mime_type: str | None = Field(None, description="MIME type if known")
    uri: str | None = Field(None, description="Resource URI if applicable")


# Union type for all content blocks - enables type-safe content handling
ContentBlock = TextContent | ImageContent | AudioContent | BinaryContent


class MCPToolResult(BaseModel):
    """Standardized result from MCP tool invocation.

    Provides a consistent interface for handling MCP tool responses,
    whether successful or failed. The result contains:
    - success: Boolean indicating operation success
    - content: List of content blocks (may be empty)
    - error: MCPError instance if failed, None if success
    - metadata: Additional response metadata (timing, request_id, etc.)
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    success: bool = Field(..., description="Whether the operation succeeded")
    content: list[ContentBlock] = Field(
        default_factory=list, description="Response content blocks"
    )
    error: MCPError | None = Field(None, description="Error details if failed")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional response metadata"
    )

    @classmethod
    def success_result(
        cls,
        content: list[ContentBlock],
        metadata: dict[str, Any] | None = None,
    ) -> "MCPToolResult":
        """Create a successful result.

        Args:
            content: List of content blocks from the MCP response
            metadata: Optional metadata dictionary

        Returns:
            MCPToolResult with success=True
        """
        return cls(
            success=True,
            content=content,
            error=None,
            metadata=metadata or {},
        )

    @classmethod
    def error_result(
        cls,
        error: MCPError,
        metadata: dict[str, Any] | None = None,
    ) -> "MCPToolResult":
        """Create an error result.

        Args:
            error: The MCPError that occurred
            metadata: Optional metadata dictionary

        Returns:
            MCPToolResult with success=False and the error
        """
        return cls(
            success=False,
            error=error,
            metadata=metadata or {},
        )
