"""Tests for MCP content models."""

import pytest
from pydantic import ValidationError

from holodeck.tools.mcp.content import (
    AudioContent,
    BinaryContent,
    ImageContent,
    MCPToolResult,
    TextContent,
)
from holodeck.tools.mcp.errors import MCPError


class TestTextContent:
    """Tests for TextContent model."""

    def test_create_text_content(self) -> None:
        """TextContent should be created with text."""
        content = TextContent(text="Hello, world!")
        assert content.type == "text"
        assert content.text == "Hello, world!"

    def test_text_content_is_frozen(self) -> None:
        """TextContent should be immutable (frozen)."""
        content = TextContent(text="test")
        with pytest.raises(ValidationError):
            content.text = "modified"  # type: ignore[misc]

    def test_text_content_empty_text(self) -> None:
        """TextContent should allow empty text."""
        content = TextContent(text="")
        assert content.text == ""


class TestImageContent:
    """Tests for ImageContent model."""

    def test_create_image_content(self) -> None:
        """ImageContent should be created with data and mime_type."""
        content = ImageContent(data="base64data==", mime_type="image/png")
        assert content.type == "image"
        assert content.data == "base64data=="
        assert content.mime_type == "image/png"

    def test_image_content_various_mime_types(self) -> None:
        """ImageContent should accept various image MIME types."""
        for mime_type in ["image/png", "image/jpeg", "image/gif", "image/webp"]:
            content = ImageContent(data="data", mime_type=mime_type)
            assert content.mime_type == mime_type


class TestAudioContent:
    """Tests for AudioContent model."""

    def test_create_audio_content(self) -> None:
        """AudioContent should be created with data and mime_type."""
        content = AudioContent(data="base64audio==", mime_type="audio/wav")
        assert content.type == "audio"
        assert content.data == "base64audio=="
        assert content.mime_type == "audio/wav"

    def test_audio_content_various_mime_types(self) -> None:
        """AudioContent should accept various audio MIME types."""
        for mime_type in ["audio/wav", "audio/mp3", "audio/ogg", "audio/webm"]:
            content = AudioContent(data="data", mime_type=mime_type)
            assert content.mime_type == mime_type


class TestBinaryContent:
    """Tests for BinaryContent model."""

    def test_create_binary_content(self) -> None:
        """BinaryContent should be created with binary data."""
        content = BinaryContent(
            data=b"\x00\x01\x02", mime_type="application/octet-stream"
        )
        assert content.type == "binary"
        assert content.data == b"\x00\x01\x02"
        assert content.mime_type == "application/octet-stream"

    def test_binary_content_optional_fields(self) -> None:
        """BinaryContent should have optional mime_type and uri."""
        content = BinaryContent(data=b"test")
        assert content.mime_type is None
        assert content.uri is None

    def test_binary_content_with_uri(self) -> None:
        """BinaryContent should accept uri."""
        content = BinaryContent(
            data=b"test", uri="file:///path/to/resource", mime_type="text/plain"
        )
        assert content.uri == "file:///path/to/resource"


class TestMCPToolResult:
    """Tests for MCPToolResult model."""

    def test_success_result_factory(self) -> None:
        """success_result should create successful MCPToolResult."""
        content = [TextContent(text="Result")]
        result = MCPToolResult.success_result(content, metadata={"timing": 1.5})
        assert result.success is True
        assert len(result.content) == 1
        assert result.error is None
        assert result.metadata["timing"] == 1.5

    def test_success_result_without_metadata(self) -> None:
        """success_result should work without metadata."""
        content = [TextContent(text="Result")]
        result = MCPToolResult.success_result(content)
        assert result.success is True
        assert result.metadata == {}

    def test_success_result_empty_content(self) -> None:
        """success_result should work with empty content list."""
        result = MCPToolResult.success_result([])
        assert result.success is True
        assert result.content == []

    def test_error_result_factory(self) -> None:
        """error_result should create failed MCPToolResult."""
        error = MCPError(message="Failed", server="test")
        result = MCPToolResult.error_result(error)
        assert result.success is False
        assert result.error is error
        assert len(result.content) == 0

    def test_error_result_with_metadata(self) -> None:
        """error_result should accept metadata."""
        error = MCPError(message="Failed")
        result = MCPToolResult.error_result(error, metadata={"request_id": "123"})
        assert result.success is False
        assert result.metadata["request_id"] == "123"

    def test_multiple_content_blocks(self) -> None:
        """MCPToolResult should support multiple content blocks."""
        content = [
            TextContent(text="Line 1"),
            TextContent(text="Line 2"),
            ImageContent(data="imgdata", mime_type="image/png"),
        ]
        result = MCPToolResult.success_result(content)
        assert len(result.content) == 3
        assert result.content[0].type == "text"
        assert result.content[2].type == "image"

    def test_result_is_frozen(self) -> None:
        """MCPToolResult should be immutable (frozen)."""
        result = MCPToolResult.success_result([])
        with pytest.raises(ValidationError):
            result.success = False  # type: ignore[misc]
