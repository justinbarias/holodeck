"""REST protocol adapter for Agent Local Server.

Implements the REST protocol with:
- Synchronous chat endpoint: POST /agent/{name}/chat → ChatResponse JSON
- Streaming chat endpoint: POST /agent/{name}/chat/stream → SSE events

See: specs/017-agent-local-server/contracts/openapi.yaml for API specification.
See: specs/017-agent-local-server/contracts/sse-events.md for SSE event format.
"""

from __future__ import annotations

import base64
import json
import tempfile
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ulid import ULID

from holodeck.lib.logging_config import get_logger
from holodeck.serve.models import ChatRequest, ChatResponse, FileContent, ToolCallInfo
from holodeck.serve.protocols.base import Protocol

if TYPE_CHECKING:
    from fastapi import UploadFile

    from holodeck.models.config import ExecutionConfig
    from holodeck.models.test_case import FileInput
    from holodeck.serve.session_store import ServerSession

logger = get_logger(__name__)


# =============================================================================
# SSE Event Serialization (per sse-events.md contract)
# =============================================================================


class SSEEvent:
    """SSE event serializer following sse-events.md specification.

    Event format:
        event: {type}
        data: {json}

    Keepalive format:
        : keepalive
    """

    @staticmethod
    def format(event_type: str, data: dict[str, Any]) -> str:
        """Format an SSE event with type and JSON data.

        Args:
            event_type: The event type name.
            data: Dictionary to serialize as JSON data.

        Returns:
            SSE formatted string: "event: {type}\\ndata: {json}\\n\\n"
        """
        json_data = json.dumps(data, separators=(",", ":"))
        return f"event: {event_type}\ndata: {json_data}\n\n"

    @staticmethod
    def stream_start(session_id: str, message_id: str) -> str:
        """Create stream_start event.

        Args:
            session_id: Session identifier (ULID).
            message_id: Message identifier (ULID).

        Returns:
            SSE formatted stream_start event.
        """
        return SSEEvent.format(
            "stream_start",
            {
                "session_id": session_id,
                "message_id": message_id,
            },
        )

    @staticmethod
    def message_delta(delta: str, message_id: str) -> str:
        """Create message_delta event with text chunk.

        Args:
            delta: Text content chunk.
            message_id: Message identifier for correlation.

        Returns:
            SSE formatted message_delta event.
        """
        return SSEEvent.format(
            "message_delta",
            {
                "delta": delta,
                "message_id": message_id,
            },
        )

    @staticmethod
    def tool_call_start(tool_call_id: str, name: str, message_id: str) -> str:
        """Create tool_call_start event.

        Args:
            tool_call_id: Unique tool call identifier.
            name: Tool name being called.
            message_id: Parent message identifier.

        Returns:
            SSE formatted tool_call_start event.
        """
        return SSEEvent.format(
            "tool_call_start",
            {
                "tool_call_id": tool_call_id,
                "name": name,
                "message_id": message_id,
            },
        )

    @staticmethod
    def tool_call_args(tool_call_id: str, args_delta: str) -> str:
        """Create tool_call_args event with argument fragment.

        Args:
            tool_call_id: Tool call identifier for correlation.
            args_delta: JSON fragment of tool arguments.

        Returns:
            SSE formatted tool_call_args event.
        """
        return SSEEvent.format(
            "tool_call_args",
            {
                "tool_call_id": tool_call_id,
                "args_delta": args_delta,
            },
        )

    @staticmethod
    def tool_call_end(tool_call_id: str, status: str) -> str:
        """Create tool_call_end event.

        Args:
            tool_call_id: Tool call identifier.
            status: Execution status ("success" or "error").

        Returns:
            SSE formatted tool_call_end event.
        """
        return SSEEvent.format(
            "tool_call_end",
            {
                "tool_call_id": tool_call_id,
                "status": status,
            },
        )

    @staticmethod
    def stream_end(
        message_id: str,
        tokens_used: dict[str, int] | None,
        execution_time_ms: int,
    ) -> str:
        """Create stream_end event.

        Args:
            message_id: Message identifier.
            tokens_used: Token consumption statistics (may be None).
            execution_time_ms: Total execution time in milliseconds.

        Returns:
            SSE formatted stream_end event.
        """
        return SSEEvent.format(
            "stream_end",
            {
                "message_id": message_id,
                "tokens_used": tokens_used,
                "execution_time_ms": execution_time_ms,
            },
        )

    @staticmethod
    def error(
        type: str,
        title: str,
        status: int,
        detail: str | None = None,
    ) -> str:
        """Create error event following RFC 7807 ProblemDetail.

        Args:
            type: Error type URI.
            title: Short human-readable description.
            status: HTTP status code.
            detail: Detailed error message (optional).

        Returns:
            SSE formatted error event.
        """
        data: dict[str, Any] = {
            "type": type,
            "title": title,
            "status": status,
        }
        if detail is not None:
            data["detail"] = detail

        return SSEEvent.format("error", data)

    @staticmethod
    def keepalive() -> str:
        """Create keepalive comment.

        Returns:
            SSE comment format: ": keepalive\\n"
        """
        return ": keepalive\n"


# =============================================================================
# File Content to FileInput Conversion
# =============================================================================


def convert_file_content_to_file_input(file_content: FileContent) -> FileInput:
    """Convert REST FileContent (base64) to FileProcessor-compatible FileInput.

    Creates a temporary file with the decoded content and returns a FileInput
    pointing to it.

    Args:
        file_content: FileContent with base64-encoded data and MIME type.

    Returns:
        FileInput suitable for FileProcessor.process_file().
    """
    from holodeck.models.test_case import FileInput

    # Decode base64 content
    content_bytes = base64.b64decode(file_content.content)

    # MIME type constants for Office documents
    docx_mime = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    xlsx_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    pptx_mime = (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )

    # Map MIME type to file type
    mime_to_type: dict[str, str] = {
        "image/png": "image",
        "image/jpeg": "image",
        "image/gif": "image",
        "image/webp": "image",
        "application/pdf": "pdf",
        docx_mime: "word",
        xlsx_mime: "excel",
        pptx_mime: "powerpoint",
        "text/plain": "text",
        "text/csv": "csv",
        "text/markdown": "text",
    }
    file_type = mime_to_type.get(file_content.mime_type, "text")

    # Determine file extension from MIME type
    mime_to_ext: dict[str, str] = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "application/pdf": ".pdf",
        docx_mime: ".docx",
        xlsx_mime: ".xlsx",
        pptx_mime: ".pptx",
        "text/plain": ".txt",
        "text/csv": ".csv",
        "text/markdown": ".md",
    }
    extension = mime_to_ext.get(file_content.mime_type, ".bin")

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=extension, delete=False, mode="wb") as tmp:
        tmp.write(content_bytes)
        tmp_path = tmp.name

    logger.debug(
        f"Converted FileContent to temp file: {tmp_path} "
        f"(type={file_type}, size={len(content_bytes)} bytes)"
    )

    return FileInput(
        path=tmp_path,
        url=None,
        type=file_type,
        description=file_content.filename,
        pages=None,
        sheet=None,
        range=None,
        cache=None,
    )


def cleanup_temp_file(file_input: FileInput) -> None:
    """Clean up temporary file created by convert_file_content_to_file_input.

    Args:
        file_input: FileInput with path to temporary file.
    """
    if file_input.path:
        try:
            Path(file_input.path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {file_input.path}: {e}")


# =============================================================================
# Multipart File Upload Processing
# =============================================================================


async def convert_upload_file_to_file_content(
    upload_file: UploadFile,
) -> FileContent:
    """Convert FastAPI UploadFile to FileContent model.

    Reads the uploaded file content and encodes it as base64.

    Args:
        upload_file: FastAPI UploadFile from multipart form-data.

    Returns:
        FileContent with base64-encoded content.

    Raises:
        ValueError: If file content type is not supported.
    """
    # Read file content
    content = await upload_file.read()

    # Encode as base64
    content_b64 = base64.b64encode(content).decode("utf-8")

    # Get MIME type from content_type or filename
    mime_type = upload_file.content_type or "application/octet-stream"

    # Validate MIME type is supported
    supported_types = {
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "text/plain",
        "text/csv",
        "text/markdown",
    }

    if mime_type not in supported_types:
        raise ValueError(f"Unsupported MIME type: {mime_type}")

    return FileContent(
        content=content_b64,
        mime_type=mime_type,
        filename=upload_file.filename,
    )


async def process_multipart_files(
    files: list[UploadFile],
) -> list[FileContent]:
    """Process multipart file uploads and convert to FileContent list.

    Args:
        files: List of FastAPI UploadFile objects.

    Returns:
        List of FileContent objects with base64-encoded content.

    Raises:
        ValueError: If any file has unsupported MIME type or exceeds size limits.
    """
    # Size limits per spec
    max_file_size = 50 * 1024 * 1024  # 50 MB per file
    max_total_size = 100 * 1024 * 1024  # 100 MB total

    file_contents: list[FileContent] = []
    total_size = 0

    for upload_file in files:
        # Check file size
        content = await upload_file.read()
        file_size = len(content)

        if file_size > max_file_size:
            raise ValueError(
                f"File '{upload_file.filename}' exceeds maximum size of 50MB"
            )

        total_size += file_size
        if total_size > max_total_size:
            raise ValueError("Total file size exceeds maximum of 100MB")

        # Reset file position for re-reading
        await upload_file.seek(0)

        # Convert to FileContent
        file_content = await convert_upload_file_to_file_content(upload_file)
        file_contents.append(file_content)

    return file_contents


# =============================================================================
# RESTProtocol Implementation
# =============================================================================


class RESTProtocol(Protocol):
    """REST protocol implementation with sync and streaming endpoints.

    Handles:
    - Synchronous requests: handle_sync_request() → ChatResponse
    - Streaming requests: handle_request() → AsyncGenerator[bytes, None] (SSE)
    """

    @property
    def name(self) -> str:
        """Return the protocol name.

        Returns:
            Protocol identifier string.
        """
        return "rest"

    @property
    def content_type(self) -> str:
        """Return the content type for streaming responses.

        Returns:
            MIME type string for response Content-Type header.
        """
        return "text/event-stream"

    async def handle_request(
        self,
        request: Any,
        session: ServerSession,
    ) -> AsyncGenerator[bytes, None]:
        """Handle streaming request and generate SSE events.

        Processes the ChatRequest, executes the agent, and yields
        encoded SSE events for streaming to the client.

        Args:
            request: ChatRequest from client.
            session: Server session with AgentExecutor.

        Yields:
            Encoded SSE events as bytes.
        """
        start_time = time.time()
        message_id = str(ULID())
        chat_request: ChatRequest = request

        try:
            # 1. Emit stream_start event
            yield SSEEvent.stream_start(session.session_id, message_id).encode("utf-8")

            # 2. Execute agent
            logger.debug(f"Executing agent for session {session.session_id}")
            response = await session.agent_executor.execute_turn(chat_request.message)

            # 3. Emit tool call events (if any)
            for tool_exec in response.tool_executions:
                tool_call_id = str(ULID())
                logger.debug(f"Emitting tool call events for: {tool_exec.tool_name}")

                # Tool call start
                yield SSEEvent.tool_call_start(
                    tool_call_id=tool_call_id,
                    name=tool_exec.tool_name,
                    message_id=message_id,
                ).encode("utf-8")

                # Tool call args (send full args at once for non-streaming)
                args_json = json.dumps(tool_exec.parameters)
                yield SSEEvent.tool_call_args(
                    tool_call_id=tool_call_id,
                    args_delta=args_json,
                ).encode("utf-8")

                # Tool call end
                status = "success" if tool_exec.status.value == "success" else "error"
                yield SSEEvent.tool_call_end(
                    tool_call_id=tool_call_id,
                    status=status,
                ).encode("utf-8")

            # 4. Emit message delta (send full content at once for non-streaming)
            if response.content:
                yield SSEEvent.message_delta(
                    delta=response.content,
                    message_id=message_id,
                ).encode("utf-8")

            # 5. Calculate execution time and tokens
            execution_time_ms = int((time.time() - start_time) * 1000)
            tokens_used = None
            if response.tokens_used:
                tokens_used = {
                    "prompt_tokens": response.tokens_used.prompt_tokens,
                    "completion_tokens": response.tokens_used.completion_tokens,
                    "total_tokens": response.tokens_used.total_tokens,
                }

            # 6. Emit stream_end event
            yield SSEEvent.stream_end(
                message_id=message_id,
                tokens_used=tokens_used,
                execution_time_ms=execution_time_ms,
            ).encode("utf-8")

            logger.debug(
                f"Completed streaming request for session {session.session_id}"
            )

        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            # Emit error event
            yield SSEEvent.error(
                type="about:blank",
                title="Agent Error",
                status=500,
                detail=str(e),
            ).encode("utf-8")

    async def handle_sync_request(
        self,
        request: ChatRequest,
        session: ServerSession,
    ) -> ChatResponse:
        """Handle synchronous request and return complete response.

        Args:
            request: ChatRequest from client.
            session: Server session with AgentExecutor.

        Returns:
            ChatResponse with agent's response.

        Raises:
            Exception: If agent execution fails.
        """
        start_time = time.time()
        message_id = str(ULID())

        logger.debug(f"Executing sync request for session {session.session_id}")

        # Execute agent
        response = await session.agent_executor.execute_turn(request.message)

        # Calculate execution time
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Build tool_calls list
        tool_calls = []
        for tool_exec in response.tool_executions:
            status = "success" if tool_exec.status.value == "success" else "error"
            tool_calls.append(
                ToolCallInfo(
                    name=tool_exec.tool_name,
                    arguments=tool_exec.parameters,
                    status=status,
                )
            )

        # Build tokens_used dict
        tokens_used = None
        if response.tokens_used:
            tokens_used = {
                "prompt_tokens": response.tokens_used.prompt_tokens,
                "completion_tokens": response.tokens_used.completion_tokens,
                "total_tokens": response.tokens_used.total_tokens,
            }

        logger.debug(
            f"Completed sync request for session {session.session_id} "
            f"in {execution_time_ms}ms"
        )

        return ChatResponse(
            message_id=message_id,
            content=response.content,
            session_id=session.session_id,
            tool_calls=tool_calls,
            tokens_used=tokens_used,
            execution_time_ms=execution_time_ms,
        )

    async def process_files(
        self,
        files: list[FileContent],
        execution_config: ExecutionConfig | None = None,
    ) -> str:
        """Process base64 files through FileProcessor and return combined text.

        Args:
            files: List of FileContent with base64-encoded data.
            execution_config: Optional execution configuration for timeouts.

        Returns:
            Combined markdown content from all processed files.
        """
        from holodeck.lib.file_processor import FileProcessor

        if not files:
            return ""

        # Create FileProcessor
        if execution_config:
            processor = FileProcessor.from_execution_config(execution_config)
        else:
            processor = FileProcessor()

        # Process each file
        file_inputs = []
        combined_content = []

        try:
            for file_content in files:
                file_input = convert_file_content_to_file_input(file_content)
                file_inputs.append(file_input)

                # Process file
                result = processor.process_file(file_input)

                if result.error:
                    logger.warning(f"File processing error: {result.error}")
                    combined_content.append(f"[File processing error: {result.error}]")
                elif result.markdown_content:
                    combined_content.append(result.markdown_content)

        finally:
            # Clean up temporary files
            for file_input in file_inputs:
                cleanup_temp_file(file_input)

        return "\n\n".join(combined_content)
