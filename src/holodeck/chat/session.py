"""Chat session state management."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from semantic_kernel.contents import ChatHistory

from holodeck.chat.executor import AgentExecutor, AgentResponse
from holodeck.chat.message import MessageValidator
from holodeck.lib.logging_config import get_logger
from holodeck.models.agent import Agent
from holodeck.models.chat import ChatConfig, ChatSession, SessionState
from holodeck.models.token_usage import TokenUsage

logger = get_logger(__name__)


class ChatSessionManager:
    """Maintains chat session lifecycle and state management.

    Coordinates between message validation, agent execution,
    and session state tracking.
    """

    def __init__(self, agent_config: Agent, config: ChatConfig) -> None:
        """Initialize session manager with configuration.

        Args:
            agent_config: Agent configuration to use for execution.
            config: Chat runtime configuration.
        """
        self.agent_config = agent_config
        self.config = config
        self.session: ChatSession | None = None
        self._executor: AgentExecutor | None = None
        self._validator = MessageValidator(max_length=10_000)

        # Token tracking
        self.total_tokens = TokenUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
        self.session_start_time = datetime.now()

        logger.debug(f"ChatSessionManager initialized for agent: {agent_config.name}")

    async def start(self) -> None:
        """Start a new chat session.

        Initializes the agent executor, creates a chat session,
        and transitions state to ACTIVE.

        Raises:
            RuntimeError: If session initialization fails.
        """
        try:
            logger.info(f"Starting chat session for agent: {self.agent_config.name}")

            # Create executor with resolved timeout
            timeout = self.config.llm_timeout if self.config.llm_timeout else 60
            self._executor = AgentExecutor(
                self.agent_config,
                enable_observability=self.config.enable_observability,
                force_ingest=self.config.force_ingest,
                timeout=timeout,
            )

            # Create chat session with empty history
            history = ChatHistory()
            self.session = ChatSession(
                agent_config=self.agent_config,
                history=history,
                state=SessionState.ACTIVE,
            )

            logger.info(f"Chat session started: session_id={self.session.session_id}")

        except Exception as e:
            logger.error(f"Failed to start chat session: {e}", exc_info=True)
            raise RuntimeError(f"Failed to start chat session: {e}") from e

    async def process_message(self, message: str) -> AgentResponse:
        """Process a user message through validation and execution.

        Args:
            message: User message to process.

        Returns:
            AgentResponse from the agent.

        Raises:
            RuntimeError: If session not started or execution fails.
            ValueError: If message validation fails.
        """
        if self.session is None or self._executor is None:
            raise RuntimeError("Session not started. Call start() first.")

        # Validate message
        is_valid, error = self._validator.validate(message)
        if not is_valid:
            raise ValueError(f"Invalid message: {error}")

        # Execute turn
        try:
            logger.debug(f"Processing message ({len(message)} chars)")
            response = await self._executor.execute_turn(message)

            # Increment message count
            self.session.message_count += 1

            # Accumulate token usage
            if response.tokens_used:
                self.total_tokens = TokenUsage(
                    prompt_tokens=self.total_tokens.prompt_tokens
                    + response.tokens_used.prompt_tokens,
                    completion_tokens=self.total_tokens.completion_tokens
                    + response.tokens_used.completion_tokens,
                    total_tokens=self.total_tokens.total_tokens
                    + response.tokens_used.total_tokens,
                )

            logger.debug(
                f"Message processed: count={self.session.message_count}, "
                f"tools={len(response.tool_executions)}"
            )

            return response

        except Exception as e:
            logger.error(f"Message processing failed: {e}", exc_info=True)
            raise RuntimeError(f"Message processing failed: {e}") from e

    def get_session(self) -> ChatSession | None:
        """Get current chat session.

        Returns:
            ChatSession instance, or None if not started.
        """
        return self.session

    def get_session_stats(self) -> dict[str, Any]:
        """Get current session statistics.

        Returns:
            Dict with message_count, total_tokens, and session_duration.
        """
        session_duration = (datetime.now() - self.session_start_time).total_seconds()
        return {
            "message_count": self.session.message_count if self.session else 0,
            "total_tokens": self.total_tokens,
            "session_duration": session_duration,
        }

    def should_warn_context_limit(self) -> bool:
        """Check if conversation is approaching context limit.

        Returns:
            True if message count >= 80% of max_messages, False otherwise.
        """
        if self.session is None:
            return False

        threshold = int(self.config.max_messages * 0.8)
        should_warn = self.session.message_count >= threshold
        return should_warn

    async def terminate(self) -> None:
        """Terminate the chat session.

        Cleans up resources and transitions state to TERMINATED.
        """
        try:
            if self.session is not None:
                self.session.state = SessionState.TERMINATED
                logger.info(
                    f"Chat session terminated: session_id={self.session.session_id}"
                )

            if self._executor is not None:
                await self._executor.shutdown()

        except Exception as e:
            logger.error(f"Error during session termination: {e}", exc_info=True)
