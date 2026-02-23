"""Provider-agnostic backend interfaces for HoloDeck agent execution.

This module defines the core abstractions that all agent execution backends
(Semantic Kernel, Claude Agent SDK) must implement. Downstream consumers
(TestExecutor, ChatSessionManager, AgentExecutor) depend only on these
interfaces — no provider-specific types leak through.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from holodeck.lib.errors import HoloDeckError
from holodeck.models.token_usage import TokenUsage

if TYPE_CHECKING:
    from holodeck.lib.structured_chunker import DocumentChunk


@dataclass
class ExecutionResult:
    """Provider-agnostic result of a single agent turn.

    Attributes:
        response: The text response from the agent.
        tool_calls: List of tool call records made during execution.
        tool_results: List of tool result records returned during execution.
        token_usage: Token consumption metadata for this turn.
        structured_output: Optional structured output from the agent.
        num_turns: Number of turns taken to produce this result.
        is_error: Whether the execution ended in an error state.
        error_reason: Human-readable reason for the error, if any.
    """

    response: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage.zero)
    structured_output: Any | None = None
    num_turns: int = 1
    is_error: bool = False
    error_reason: str | None = None


@runtime_checkable
class AgentSession(Protocol):
    """Stateful multi-turn conversation session.

    Implementations maintain conversation history across multiple ``send``
    calls. Callers must invoke ``close`` when the session is no longer needed
    to release any held resources (connections, subprocesses, etc.).
    """

    async def send(self, message: str) -> ExecutionResult:
        """Send a message and receive a single-turn result.

        Args:
            message: The user message to send to the agent.

        Returns:
            ExecutionResult containing the agent response and metadata.
        """
        ...

    async def send_streaming(self, message: str) -> AsyncGenerator[str, None]:
        """Send a message and stream the agent response token by token.

        Args:
            message: The user message to send to the agent.

        Yields:
            Successive string chunks of the agent response.
        """
        # Protocol stub — concrete implementations use `yield`
        yield ""  # pragma: no cover

    async def close(self) -> None:
        """Release session resources (connections, subprocesses, etc.)."""
        ...


@runtime_checkable
class AgentBackend(Protocol):
    """Provider backend factory.

    Each backend encapsulates provider-specific initialisation logic and
    exposes a uniform surface for single-turn invocations (``invoke_once``)
    and stateful sessions (``create_session``). Callers must call
    ``initialize`` before any other method and ``teardown`` when done.
    """

    async def initialize(self) -> None:
        """Prepare the backend for use.

        Raises:
            BackendInitError: If the backend cannot be initialised (e.g.
                missing API key, unavailable subprocess).
        """
        ...

    async def invoke_once(
        self,
        message: str,
        context: list[dict[str, Any]] | None = None,
    ) -> ExecutionResult:
        """Execute a single stateless agent turn.

        Args:
            message: The user message to send to the agent.
            context: Optional list of prior conversation turns.

        Returns:
            ExecutionResult containing the agent response and metadata.

        Raises:
            BackendSessionError: If the invocation fails at runtime.
            BackendTimeoutError: If the invocation exceeds configured timeout.
        """
        ...

    async def create_session(self) -> AgentSession:
        """Create a new stateful multi-turn session.

        Returns:
            A fresh AgentSession instance bound to this backend.

        Raises:
            BackendInitError: If the backend was not initialised before calling.
            BackendSessionError: If the session cannot be created.
        """
        ...

    async def teardown(self) -> None:
        """Release all backend resources."""
        ...


class BackendError(HoloDeckError):
    """Base exception for all backend errors.

    Catch this to handle any backend-related failure without needing to know
    the specific subtype.
    """

    pass


class BackendInitError(BackendError):
    """Raised during ``initialize()`` — startup validation failures.

    Examples include a missing API key, an unreachable subprocess, or an
    incompatible runtime environment.
    """

    pass


class BackendSessionError(BackendError):
    """Raised during ``send()`` — session-level failures.

    Examples include unexpected disconnections, malformed responses, or
    provider-reported errors during an active session.
    """

    pass


class BackendTimeoutError(BackendError):
    """Raised when a single invocation exceeds the configured timeout.

    Callers may choose to retry with a longer timeout or surface this as a
    user-visible error.
    """

    pass


@runtime_checkable
class ContextGenerator(Protocol):
    """Backend-agnostic contextual embedding generation.

    Implementations produce situating context for document chunks by
    summarising each chunk's role within the larger document. Both the
    existing Semantic Kernel generator and future Claude SDK generator
    should satisfy this protocol.
    """

    async def contextualize_batch(
        self,
        chunks: list[DocumentChunk],
        document_text: str,
        concurrency: int | None = None,
    ) -> list[str]:
        """Generate contextual descriptions for a batch of chunks.

        Args:
            chunks: Document chunks to contextualize.
            document_text: Full text of the source document.
            concurrency: Maximum number of concurrent LLM calls.

        Returns:
            A list of contextual description strings, one per chunk.
        """
        ...
