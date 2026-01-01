"""AG-UI protocol adapter for Agent Local Server.

Implements the AG-UI (Agent User Interface) protocol using the ag-ui-protocol SDK.
This module maps between AG-UI events and HoloDeck's AgentExecutor.

See: https://github.com/ag-ui-protocol/ag-ui for protocol specification.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

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
from holodeck.serve.protocols.base import Protocol

if TYPE_CHECKING:
    from holodeck.models.tool_execution import ToolExecution
    from holodeck.serve.session_store import ServerSession

logger = get_logger(__name__)


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
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
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

        try:
            # Extract user message from input
            message = extract_message_from_input(input_data)
            logger.debug(f"Processing message: {message[:100]}...")

            # 1. Emit RunStartedEvent
            yield encoder.encode(create_run_started_event(thread_id, run_id))

            # 2. Emit TextMessageStartEvent
            yield encoder.encode(create_text_message_start(message_id))

            # 3. Execute agent
            logger.debug(f"Executing agent for session {session.session_id}")
            response = await session.agent_executor.execute_turn(message)

            # 4. Emit tool call events (if any)
            for tool_exec in response.tool_executions:
                logger.debug(f"Emitting tool call events for: {tool_exec.tool_name}")
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

            logger.debug(f"Completed request for run {run_id}")

        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            # Emit error event
            yield encoder.encode(create_run_error_event(str(e)))
