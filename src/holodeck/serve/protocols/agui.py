"""AG-UI protocol adapter for Agent Local Server.

Implements the AG-UI (Agent User Interface) protocol using the ag-ui-protocol SDK.
This module maps between AG-UI events and HoloDeck's AgentExecutor.

See: https://github.com/ag-ui-protocol/ag-ui for protocol specification.
"""

from __future__ import annotations

import base64
import json
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from ag_ui.core.events import (
    BaseEvent,
    EventType,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from ag_ui.encoder import EventEncoder
from ulid import ULID

from holodeck.lib.logging_config import get_logger
from holodeck.models.test_case import FileInput
from holodeck.serve.models import SUPPORTED_MIME_TYPES
from holodeck.serve.protocols.base import Protocol

if TYPE_CHECKING:
    from holodeck.models.tool_execution import ToolExecution
    from holodeck.serve.session_store import ServerSession

logger = get_logger(__name__)


# =============================================================================
# T029e-h: AG-UI Multimodal Support
# =============================================================================

# MIME type to file type mapping (same as REST protocol)
# Long MIME type keys for Office formats
_WORD_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_EXCEL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

_MIME_TO_TYPE: dict[str, str] = {
    "image/png": "image",
    "image/jpeg": "image",
    "image/gif": "image",
    "image/webp": "image",
    "application/pdf": "pdf",
    _WORD_MIME: "word",
    _EXCEL_MIME: "excel",
    _PPTX_MIME: "powerpoint",
    "text/plain": "text",
    "text/csv": "csv",
    "text/markdown": "text",
}

# MIME type to file extension mapping (same as REST protocol)
_MIME_TO_EXT: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
    _WORD_MIME: ".docx",
    _EXCEL_MIME: ".xlsx",
    _PPTX_MIME: ".pptx",
    "text/plain": ".txt",
    "text/csv": ".csv",
    "text/markdown": ".md",
}


def extract_binary_parts_from_content(
    content: list[dict[str, Any] | Any],
) -> list[dict[str, Any]]:
    """Extract binary content parts from AG-UI message content list.

    Filters the content list for binary type parts and validates MIME types.
    Handles both dict format and AG-UI Pydantic objects (BinaryInputContent).

    Args:
        content: List of content parts (text, binary, or strings).

    Returns:
        List of binary content dicts with type, mimeType, and data/url/id fields.
    """
    binary_parts: list[dict[str, Any]] = []

    logger.info(
        "[Multimodal] Scanning content list for binary parts (total items: %d)",
        len(content),
    )

    for idx, part in enumerate(content):
        part_type_name = type(part).__name__
        logger.debug("[Multimodal] Content item %d: type=%s", idx, part_type_name)

        # Handle dict format
        if isinstance(part, dict):
            part_type = part.get("type", "")
            if part_type != "binary":
                logger.debug(
                    "[Multimodal] Item %d is dict with type='%s', skipping",
                    idx,
                    part_type,
                )
                continue

            mime_type = part.get("mimeType", "")
            if mime_type not in SUPPORTED_MIME_TYPES:
                logger.warning(
                    "[Multimodal] Skipping binary content with unsupported "
                    "MIME type: %s. Supported types: %s",
                    mime_type,
                    ", ".join(sorted(SUPPORTED_MIME_TYPES)),
                )
                continue

            has_data = "data" in part and part["data"]
            has_url = "url" in part and part["url"]
            has_id = "id" in part and part["id"]
            data_len = len(part.get("data", "") or "") if has_data else 0
            logger.info(
                "[Multimodal] Found binary content (dict): mime=%s, "
                "has_data=%s (len=%d), has_url=%s, has_id=%s, filename=%s",
                mime_type,
                has_data,
                data_len,
                has_url,
                has_id,
                part.get("filename"),
            )
            binary_parts.append(part)

        # Handle AG-UI Pydantic object (BinaryInputContent)
        elif hasattr(part, "type") and getattr(part, "type", None) == "binary":
            # AG-UI uses 'mime_type' attribute (snake_case), not 'mimeType'
            mime_type = getattr(part, "mime_type", "")
            if mime_type not in SUPPORTED_MIME_TYPES:
                logger.warning(
                    "[Multimodal] Skipping binary content with unsupported "
                    "MIME type: %s. Supported types: %s",
                    mime_type,
                    ", ".join(sorted(SUPPORTED_MIME_TYPES)),
                )
                continue

            data_val = getattr(part, "data", None)
            url_val = getattr(part, "url", None)
            id_val = getattr(part, "id", None)
            filename_val = getattr(part, "filename", None)
            data_len = len(data_val or "") if data_val else 0

            logger.info(
                "[Multimodal] Found binary content (Pydantic): mime=%s, "
                "has_data=%s (len=%d), has_url=%s, has_id=%s, filename=%s",
                mime_type,
                bool(data_val),
                data_len,
                bool(url_val),
                bool(id_val),
                filename_val,
            )

            # Convert Pydantic object to dict for consistent handling
            binary_parts.append(
                {
                    "type": "binary",
                    "mimeType": mime_type,
                    "data": data_val,
                    "url": url_val,
                    "id": id_val,
                    "filename": filename_val,
                }
            )

    logger.info(
        "[Multimodal] Extracted %d binary parts from content", len(binary_parts)
    )
    return binary_parts


def convert_agui_binary_to_file_input(
    binary_content: dict[str, Any],
) -> FileInput | None:
    """Convert AG-UI binary content dict to FileProcessor-compatible FileInput.

    Handles three transport options:
    - data: Inline base64-encoded content
    - url: Remote URL to download
    - id: File ID reference (not supported, returns None)

    Args:
        binary_content: Dict with type, mimeType, and data/url/id fields.

    Returns:
        FileInput suitable for FileProcessor, or None if not processable.

    Raises:
        ValueError: If base64 decoding fails.
    """
    mime_type = binary_content.get("mimeType", "")
    filename = binary_content.get("filename")

    logger.info(
        "[Multimodal] Converting binary content to FileInput: mime=%s, filename=%s",
        mime_type,
        filename,
    )

    # Determine file type and extension from MIME type
    file_type = _MIME_TO_TYPE.get(mime_type, "text")
    extension = _MIME_TO_EXT.get(mime_type, ".bin")
    logger.debug(
        "[Multimodal] Mapped MIME type %s -> file_type=%s, extension=%s",
        mime_type,
        file_type,
        extension,
    )

    # Handle inline base64 data
    if "data" in binary_content:
        logger.info(
            "[Multimodal] Processing base64 data (length=%d chars)",
            len(binary_content["data"]),
        )
        try:
            content_bytes = base64.b64decode(binary_content["data"])
            logger.debug(
                "[Multimodal] Base64 decoded successfully: %d bytes", len(content_bytes)
            )
        except Exception as e:
            logger.error("[Multimodal] Base64 decode failed: %s", e)
            raise ValueError(f"Invalid base64 data: {e}") from e

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            suffix=extension, delete=False, mode="wb"
        ) as tmp:
            tmp.write(content_bytes)
            tmp_path = tmp.name

        logger.info(
            "[Multimodal] Created temp file from base64: "
            "path=%s, type=%s, size=%d bytes",
            tmp_path,
            file_type,
            len(content_bytes),
        )

        return FileInput(
            path=tmp_path,
            url=None,
            type=file_type,
            description=filename,
            pages=None,
            sheet=None,
            range=None,
            cache=None,
        )

    # Handle URL reference
    if "url" in binary_content:
        url = binary_content["url"]
        logger.info("[Multimodal] Downloading file from URL: %s", url)
        try:
            response = httpx.get(url, timeout=30.0)
            response.raise_for_status()
            content_bytes = response.content
            logger.debug(
                "[Multimodal] URL download successful: status=%d, size=%d bytes",
                response.status_code,
                len(content_bytes),
            )
        except Exception as e:
            logger.error("[Multimodal] URL download failed for %s: %s", url, e)
            raise ValueError(f"Failed to download file from {url}: {e}") from e

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            suffix=extension, delete=False, mode="wb"
        ) as tmp:
            tmp.write(content_bytes)
            tmp_path = tmp.name

        logger.info(
            "[Multimodal] Created temp file from URL: path=%s, type=%s, size=%d bytes",
            tmp_path,
            file_type,
            len(content_bytes),
        )

        return FileInput(
            path=tmp_path,
            url=None,
            type=file_type,
            description=filename or url.split("/")[-1],
            pages=None,
            sheet=None,
            range=None,
            cache=None,
        )

    # Handle file ID reference (not supported in MVP)
    if "id" in binary_content:
        file_id = binary_content["id"]
        logger.warning(
            "[Multimodal] File ID references are not supported (MVP limitation). "
            "Use inline base64 or URL instead. id=%s",
            file_id,
        )
        return None

    logger.warning(
        "[Multimodal] Binary content has no data, url, or id field - skipping. "
        "Content keys: %s",
        list(binary_content.keys()),
    )
    return None


def cleanup_temp_file(file_input: FileInput) -> None:
    """Clean up temporary file created by convert_agui_binary_to_file_input.

    Args:
        file_input: FileInput with path to temporary file.
    """
    if file_input.path:
        try:
            Path(file_input.path).unlink(missing_ok=True)
            logger.debug("[Multimodal] Cleaned up temp file: %s", file_input.path)
        except Exception as e:
            logger.warning(
                "[Multimodal] Failed to cleanup temp file %s: %s", file_input.path, e
            )


def process_binary_files_for_message(
    binary_parts: list[dict[str, Any]],
) -> tuple[str, list[FileInput]]:
    """Process binary content parts and return combined file content.

    Converts binary parts to FileInputs, processes through FileProcessor,
    and returns the combined markdown content.

    Args:
        binary_parts: List of binary content dicts from message.

    Returns:
        Tuple of (combined_file_content, list_of_file_inputs_for_cleanup).
    """
    from holodeck.lib.file_processor import FileProcessor

    if not binary_parts:
        logger.debug("[Multimodal] No binary parts to process")
        return "", []

    logger.info(
        "[Multimodal] Processing %d binary parts through FileProcessor",
        len(binary_parts),
    )

    file_inputs: list[FileInput] = []
    file_contents: list[str] = []
    processor = FileProcessor()

    for idx, binary in enumerate(binary_parts):
        mime_type = binary.get("mimeType", "unknown")
        filename = binary.get("filename", "unnamed")
        logger.info(
            "[Multimodal] Processing file %d/%d: mime=%s, filename=%s",
            idx + 1,
            len(binary_parts),
            mime_type,
            filename,
        )

        try:
            file_input = convert_agui_binary_to_file_input(binary)
            if file_input is None:
                logger.warning(
                    "[Multimodal] File %d/%d: conversion returned None, skipping",
                    idx + 1,
                    len(binary_parts),
                )
                continue

            file_inputs.append(file_input)

            # Process through FileProcessor
            logger.debug(
                "[Multimodal] File %d/%d: invoking FileProcessor on %s",
                idx + 1,
                len(binary_parts),
                file_input.path,
            )
            result = processor.process_file(file_input)

            if result.error:
                logger.warning(
                    "[Multimodal] File %d/%d: FileProcessor error: %s",
                    idx + 1,
                    len(binary_parts),
                    result.error,
                )
                file_contents.append(f"[File processing error: {result.error}]")
            elif result.markdown_content:
                content_len = len(result.markdown_content)
                logger.info(
                    "[Multimodal] File %d/%d: FileProcessor success, "
                    "extracted %d chars of markdown",
                    idx + 1,
                    len(binary_parts),
                    content_len,
                )
                # Add filename header if available
                display_name = (
                    binary.get("filename") or file_input.description or "file"
                )
                file_contents.append(
                    f"## File: {display_name}\n\n{result.markdown_content}"
                )
            else:
                logger.warning(
                    "[Multimodal] File %d/%d: FileProcessor returned no content",
                    idx + 1,
                    len(binary_parts),
                )

        except Exception as e:
            logger.error(
                "[Multimodal] File %d/%d: exception during processing: %s",
                idx + 1,
                len(binary_parts),
                e,
                exc_info=True,
            )
            file_contents.append(f"[Error processing file: {e}]")

    combined_content = "\n\n".join(file_contents)
    logger.info(
        "[Multimodal] Completed processing: %d files processed, "
        "%d FileInputs for cleanup, %d chars of combined content",
        len(binary_parts),
        len(file_inputs),
        len(combined_content),
    )
    return combined_content, file_inputs


def extract_message_and_files_from_input(
    input_data: RunAgentInput,
) -> tuple[str, list[dict[str, Any]]]:
    """Extract text message and binary content parts from RunAgentInput.

    Args:
        input_data: AG-UI input containing messages list.

    Returns:
        Tuple of (text_message, binary_parts_list).

    Raises:
        ValueError: If no user messages found.
    """
    messages = input_data.messages or []
    logger.debug(
        "[Multimodal] Extracting message and files from input (total messages: %d)",
        len(messages),
    )

    # Find the last user message
    for message in reversed(messages):
        # Messages can be dicts or Message objects
        if isinstance(message, dict):  # type: ignore[unreachable]
            role = message.get("role", "")  # type: ignore[unreachable]
            content = message.get("content", "")
        else:
            role = getattr(message, "role", "")
            content = getattr(message, "content", "")

        if role == "user":
            logger.debug(
                "[Multimodal] Found user message, content type: %s",
                type(content).__name__,
            )

            # Content can be a string or a list of content parts
            if isinstance(content, str):
                logger.info(
                    "[Multimodal] User message is plain string (%d chars), no files",
                    len(content),
                )
                return content, []
            elif isinstance(content, list):
                logger.debug(
                    "[Multimodal] User message has list content (%d parts)",
                    len(content),
                )

                # Extract text and binary parts
                text_parts: list[str] = []
                binary_parts = extract_binary_parts_from_content(content)

                for part in content:
                    # Handle dict format
                    if isinstance(part, dict):
                        part_type = part.get("type", "unknown")
                        if part_type == "text":
                            text_parts.append(part.get("text", ""))
                    # Handle AG-UI Pydantic object (TextInputContent)
                    elif (
                        hasattr(part, "type") and getattr(part, "type", None) == "text"
                    ):
                        text_parts.append(getattr(part, "text", ""))
                    # Handle plain string
                    elif isinstance(part, str):
                        text_parts.append(part)

                # Validate that we have at least some content (text or binary)
                if not text_parts and not binary_parts:
                    logger.error(
                        "[Multimodal] No content found in user message - "
                        "neither text nor binary parts"
                    )
                    raise ValueError(
                        "No content found in user message. "
                        "Message contained no text or binary content parts."
                    )

                text_message = " ".join(text_parts) if text_parts else ""
                logger.info(
                    "[Multimodal] Extracted: text=%d chars, binary_parts=%d",
                    len(text_message),
                    len(binary_parts),
                )
                return text_message, binary_parts

    logger.error("[Multimodal] No user messages found in input")
    raise ValueError("No user messages found in input")


# =============================================================================
# T019: RunAgentInput to HoloDeck request mapping
# =============================================================================


def extract_message_from_input(input_data: RunAgentInput) -> str:
    """Extract the last user message from RunAgentInput.

    Args:
        input_data: AG-UI input containing messages list.

    Returns:
        The text content of the last user message.

    Raises:
        ValueError: If no user messages found.
    """
    messages = input_data.messages or []

    # Find the last user message
    for message in reversed(messages):
        # Messages can be dicts or Message objects
        # Note: At runtime, JSON deserialization may produce dicts
        if isinstance(message, dict):  # type: ignore[unreachable]
            role = message.get("role", "")  # type: ignore[unreachable]
            content = message.get("content", "")
        else:
            role = getattr(message, "role", "")
            content = getattr(message, "content", "")

        if role == "user":
            # Content can be a string or a list of content parts
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Extract text from content parts
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        part_type = part.get("type", "unknown")
                        if part_type == "text":
                            text_parts.append(part.get("text", ""))
                        else:
                            # Non-text content (image, file, etc) not supported
                            logger.warning(
                                "Skipping non-text content part (type: %s). "
                                "Only 'text' content parts are supported.",
                                part_type,
                            )
                    elif isinstance(part, str):
                        text_parts.append(part)

                # Validate that we have at least some text content
                if not text_parts:
                    raise ValueError(
                        "No text content found in user message. "
                        "Message contained only non-text content parts."
                    )
                return " ".join(text_parts)

    raise ValueError("No user messages found in input")


def map_session_id(thread_id: str) -> str:
    """Map AG-UI thread_id to HoloDeck session_id.

    The thread_id from AG-UI is used directly as the session_id.

    Args:
        thread_id: AG-UI conversation thread identifier.

    Returns:
        Session ID (uses thread_id directly).
    """
    return thread_id


def generate_run_id() -> str:
    """Generate a unique run ID for this request.

    Returns:
        New ULID string for the run.
    """
    return str(ULID())


# =============================================================================
# T020: AG-UI EventEncoder wrapper
# =============================================================================


class AGUIEventStream:
    """Wrapper for AG-UI event encoding and streaming.

    Handles format negotiation based on HTTP Accept header and
    encodes events for streaming to clients.
    """

    def __init__(self, accept_header: str | None = None) -> None:
        """Initialize event stream with format negotiation.

        Args:
            accept_header: HTTP Accept header for format selection.
                         Defaults to text/event-stream (SSE).
        """
        # EventEncoder requires a string, default to SSE format
        self.encoder = EventEncoder(accept=accept_header or "text/event-stream")

    @property
    def content_type(self) -> str:
        """Get the content type for the streaming response.

        Returns:
            MIME type string for response Content-Type header.
        """
        content_type: str = self.encoder.get_content_type()
        return content_type

    def encode(self, event: BaseEvent) -> bytes:
        """Encode a single AG-UI event.

        Args:
            event: AG-UI event to encode.

        Returns:
            Encoded bytes for SSE or binary format.
        """
        encoded = self.encoder.encode(event)
        # Ensure we return bytes for streaming
        # EventEncoder returns str for SSE format, bytes for binary
        # Note: mypy doesn't know EventEncoder can return bytes
        if isinstance(encoded, bytes):  # type: ignore[unreachable]
            return encoded  # type: ignore[unreachable]
        result: bytes = encoded.encode("utf-8")
        return result


# =============================================================================
# T021: Lifecycle events implementation
# =============================================================================


def create_run_started_event(thread_id: str, run_id: str) -> RunStartedEvent:
    """Create RunStartedEvent for stream beginning.

    Args:
        thread_id: Conversation thread identifier.
        run_id: Unique run identifier.

    Returns:
        RunStartedEvent instance.
    """
    return RunStartedEvent(
        type=EventType.RUN_STARTED,
        thread_id=thread_id,
        run_id=run_id,
    )


def create_run_finished_event(thread_id: str, run_id: str) -> RunFinishedEvent:
    """Create RunFinishedEvent for successful completion.

    Args:
        thread_id: Conversation thread identifier.
        run_id: Unique run identifier.

    Returns:
        RunFinishedEvent instance.
    """
    return RunFinishedEvent(
        type=EventType.RUN_FINISHED,
        thread_id=thread_id,
        run_id=run_id,
    )


def create_run_error_event(message: str, code: str | None = None) -> RunErrorEvent:
    """Create RunErrorEvent for failure.

    Args:
        message: Error message describing the failure.
        code: Optional error code for categorization.

    Returns:
        RunErrorEvent instance.
    """
    return RunErrorEvent(
        type=EventType.RUN_ERROR,
        message=message,
        code=code,
    )


# =============================================================================
# T022: Text message events implementation
# =============================================================================


def create_text_message_start(message_id: str) -> TextMessageStartEvent:
    """Create TextMessageStartEvent to open message stream.

    Args:
        message_id: Unique message identifier.

    Returns:
        TextMessageStartEvent instance.
    """
    return TextMessageStartEvent(
        type=EventType.TEXT_MESSAGE_START,
        message_id=message_id,
        role="assistant",
    )


def create_text_message_content(message_id: str, delta: str) -> TextMessageContentEvent:
    """Create TextMessageContentEvent with text chunk.

    Args:
        message_id: Message identifier for correlation.
        delta: Text chunk to stream.

    Returns:
        TextMessageContentEvent instance.
    """
    return TextMessageContentEvent(
        type=EventType.TEXT_MESSAGE_CONTENT,
        message_id=message_id,
        delta=delta,
    )


def create_text_message_end(message_id: str) -> TextMessageEndEvent:
    """Create TextMessageEndEvent to close message stream.

    Args:
        message_id: Message identifier for correlation.

    Returns:
        TextMessageEndEvent instance.
    """
    return TextMessageEndEvent(
        type=EventType.TEXT_MESSAGE_END,
        message_id=message_id,
    )


# =============================================================================
# T023: Tool call events implementation
# =============================================================================


def create_tool_call_start(
    tool_call_id: str,
    tool_call_name: str,
    parent_message_id: str | None = None,
) -> ToolCallStartEvent:
    """Create ToolCallStartEvent to initiate tool execution.

    Args:
        tool_call_id: Unique tool call identifier.
        tool_call_name: Name of the tool being called.
        parent_message_id: Optional parent message identifier.

    Returns:
        ToolCallStartEvent instance.
    """
    return ToolCallStartEvent(
        type=EventType.TOOL_CALL_START,
        tool_call_id=tool_call_id,
        tool_call_name=tool_call_name,
        parent_message_id=parent_message_id,
    )


def create_tool_call_args(tool_call_id: str, delta: str) -> ToolCallArgsEvent:
    """Create ToolCallArgsEvent with argument fragment.

    Args:
        tool_call_id: Tool call identifier for correlation.
        delta: JSON fragment of arguments.

    Returns:
        ToolCallArgsEvent instance.
    """
    return ToolCallArgsEvent(
        type=EventType.TOOL_CALL_ARGS,
        tool_call_id=tool_call_id,
        delta=delta,
    )


def create_tool_call_end(tool_call_id: str) -> ToolCallEndEvent:
    """Create ToolCallEndEvent to complete argument transmission.

    Args:
        tool_call_id: Tool call identifier for correlation.

    Returns:
        ToolCallEndEvent instance.
    """
    return ToolCallEndEvent(
        type=EventType.TOOL_CALL_END,
        tool_call_id=tool_call_id,
    )


def create_tool_call_events(
    tool_execution: ToolExecution,
    message_id: str,
) -> list[BaseEvent]:
    """Create complete tool call event sequence from ToolExecution.

    Generates the sequence: ToolCallStart -> ToolCallArgs -> ToolCallEnd

    Args:
        tool_execution: HoloDeck tool execution result.
        message_id: Parent message identifier.

    Returns:
        List of [ToolCallStart, ToolCallArgs, ToolCallEnd] events.
    """
    tool_call_id = str(ULID())

    events: list[BaseEvent] = [
        create_tool_call_start(
            tool_call_id=tool_call_id,
            tool_call_name=tool_execution.tool_name,
            parent_message_id=message_id,
        ),
        create_tool_call_args(
            tool_call_id=tool_call_id,
            delta=json.dumps(tool_execution.parameters),
        ),
        create_tool_call_end(tool_call_id=tool_call_id),
    ]

    return events


# =============================================================================
# T024-T025: AGUIProtocol class with AgentExecutor integration
# =============================================================================


class AGUIProtocol(Protocol):
    """AG-UI protocol implementation.

    Handles /awp endpoint requests, converting between AG-UI events
    and HoloDeck's AgentExecutor.

    The AG-UI protocol follows an event-driven streaming pattern:
    1. RunStartedEvent - Signals execution start
    2. TextMessageStartEvent - Opens message stream
    3. ToolCall* events - For any tool invocations
    4. TextMessageContentEvent - Streams response text
    5. TextMessageEndEvent - Closes message stream
    6. RunFinishedEvent/RunErrorEvent - Signals completion
    """

    def __init__(self, accept_header: str | None = None) -> None:
        """Initialize the AG-UI protocol.

        Args:
            accept_header: HTTP Accept header for format negotiation.
        """
        self._accept_header = accept_header

    @property
    def name(self) -> str:
        """Return the protocol name.

        Returns:
            Protocol identifier string.
        """
        return "ag-ui"

    @property
    def content_type(self) -> str:
        """Return the content type for responses.

        Returns:
            MIME type string for response Content-Type header.
        """
        return "text/event-stream"

    async def handle_request(
        self,
        request: Any,
        session: ServerSession,
    ) -> AsyncGenerator[bytes, None]:
        """Handle AG-UI request and generate event stream.

        Processes the RunAgentInput, executes the agent, and yields
        encoded AG-UI events for streaming to the client.

        Args:
            request: RunAgentInput from client.
            session: Server session with AgentExecutor.

        Yields:
            Encoded AG-UI events as bytes.
        """
        # Extract components from RunAgentInput
        input_data: RunAgentInput = request
        thread_id = input_data.thread_id
        run_id = input_data.run_id

        # Create event encoder
        encoder = AGUIEventStream(accept_header=self._accept_header)
        message_id = str(ULID())

        # Track file inputs for cleanup
        file_inputs_to_cleanup: list[FileInput] = []

        try:
            # Extract user message and binary content from input
            logger.info(
                "[Multimodal] Starting request processing for run_id=%s, thread_id=%s",
                run_id,
                thread_id,
            )
            text_message, binary_parts = extract_message_and_files_from_input(
                input_data
            )
            logger.info(
                "[Multimodal] Request contains: text=%d chars, binary_parts=%d",
                len(text_message) if text_message else 0,
                len(binary_parts),
            )
            if text_message:
                logger.debug(
                    "[Multimodal] Text message preview: %s...",
                    text_message[:100],
                )

            # Process binary files if present
            file_content = ""
            if binary_parts:
                logger.info(
                    "[Multimodal] Processing %d binary parts for message context",
                    len(binary_parts),
                )
                file_content, file_inputs_to_cleanup = process_binary_files_for_message(
                    binary_parts
                )
                if file_content:
                    logger.info(
                        "[Multimodal] File processing complete: %d files -> %d chars "
                        "of context",
                        len(file_inputs_to_cleanup),
                        len(file_content),
                    )
                else:
                    logger.warning(
                        "[Multimodal] File processing returned no content for %d files",
                        len(binary_parts),
                    )
            else:
                logger.debug("[Multimodal] No binary parts in request")

            # Combine text message with file content
            if file_content and text_message:
                full_message = f"{text_message}\n\n{file_content}"
                logger.info(
                    "[Multimodal] Combined message: text + files = %d chars total",
                    len(full_message),
                )
            elif file_content:
                full_message = file_content
                logger.info(
                    "[Multimodal] Message is file content only: %d chars",
                    len(full_message),
                )
            else:
                full_message = text_message
                logger.debug("[Multimodal] Message is text only (no file content)")

            # 1. Emit RunStartedEvent
            yield encoder.encode(create_run_started_event(thread_id, run_id))

            # 2. Emit TextMessageStartEvent
            yield encoder.encode(create_text_message_start(message_id))

            # 3. Execute agent with combined message
            logger.debug("Executing agent for session %s", session.session_id)
            response = await session.agent_executor.execute_turn(full_message)

            # 4. Emit tool call events (if any)
            for tool_exec in response.tool_executions:
                logger.debug("Emitting tool call events for: %s", tool_exec.tool_name)
                for event in create_tool_call_events(tool_exec, message_id):
                    yield encoder.encode(event)

            # 5. Emit text content
            yield encoder.encode(
                create_text_message_content(message_id, response.content)
            )

            # 6. Emit TextMessageEndEvent
            yield encoder.encode(create_text_message_end(message_id))

            # 7. Emit RunFinishedEvent
            yield encoder.encode(create_run_finished_event(thread_id, run_id))

            logger.debug("Completed request for run %s", run_id)

        except Exception as e:
            logger.error("Error processing request: %s", e, exc_info=True)
            # Emit error event
            yield encoder.encode(create_run_error_event(str(e)))

        finally:
            # Clean up temporary files
            if file_inputs_to_cleanup:
                logger.info(
                    "[Multimodal] Cleaning up %d temporary files",
                    len(file_inputs_to_cleanup),
                )
                for file_input in file_inputs_to_cleanup:
                    cleanup_temp_file(file_input)
                logger.debug("[Multimodal] Cleanup complete")
