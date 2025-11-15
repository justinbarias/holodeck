"""
Logging utilities for common patterns in HoloDeck.

This module provides context managers and helpers for structured logging,
including operation timing, structured context, and common logging patterns.
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def log_operation(
    logger: logging.Logger,
    operation: str,
    level: int = logging.INFO,
    context: dict[str, Any] | None = None,
) -> Iterator[None]:
    """
    Context manager for logging operation start, completion, and timing.

    Logs the start of an operation, then logs its completion with elapsed time.
    If an exception occurs, logs the error with the operation context.

    Parameters:
        logger (logging.Logger): Logger to use for logging.
        operation (str): Name/description of the operation.
        level (int): Log level to use (default: INFO).
        context (dict, optional): Additional context to include in logs.

    Yields:
        None

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> with log_operation(logger, "Processing file", context={"file": "test.txt"}):
        ...     # Do work here
        ...     process_file("test.txt")
    """
    context_str = _format_context(context) if context else ""
    start_time = time.time()

    logger.log(level, f"{operation} started{context_str}")

    try:
        yield
        elapsed = time.time() - start_time
        logger.log(level, f"{operation} completed in {elapsed:.2f}s{context_str}")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            f"{operation} failed after {elapsed:.2f}s{context_str}: {e}",
            exc_info=True,
        )
        raise


@contextmanager
def log_context(
    logger: logging.Logger,
    **kwargs: Any,
) -> Iterator[None]:
    """
    Context manager for adding structured context to log messages.

    This is useful for adding contextual information that should be included
    in all log messages within a specific scope.

    Parameters:
        logger (logging.Logger): Logger to use.
        **kwargs: Key-value pairs to add to log context.

    Yields:
        None

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> with log_context(logger, test_id="test-001", attempt=1):
        ...     logger.info("Processing test")  # Includes test_id and attempt
    """
    # Store original context
    original_extra = getattr(logger, "_holodeck_context", {})

    # Add new context
    new_context = {**original_extra, **kwargs}
    logger._holodeck_context = new_context  # type: ignore

    try:
        yield
    finally:
        # Restore original context
        logger._holodeck_context = original_extra  # type: ignore


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    **context: Any,
) -> None:
    """
    Log a message with structured context.

    Parameters:
        logger (logging.Logger): Logger to use.
        level (int): Log level.
        message (str): Log message.
        **context: Additional context key-value pairs.

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> log_with_context(
        ...     logger,
        ...     logging.INFO,
        ...     "Test passed",
        ...     test_id="test-001",
        ...     duration=1.23
        ... )
    """
    context_str = _format_context(context) if context else ""
    logger.log(level, f"{message}{context_str}")


def _format_context(context: dict[str, Any]) -> str:
    """
    Format context dictionary as a string for log messages.

    Parameters:
        context (dict): Context dictionary.

    Returns:
        str: Formatted context string.

    Example:
        >>> _format_context({"test_id": "test-001", "attempt": 1})
        ' [test_id=test-001, attempt=1]'
    """
    if not context:
        return ""

    items = [f"{k}={v}" for k, v in context.items()]
    return f" [{', '.join(items)}]"


class LogTimer:
    """
    Timer utility for logging operation durations.

    This class provides a simple way to measure and log operation durations.

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> timer = LogTimer(logger)
        >>> timer.start("Processing batch")
        >>> # Do work...
        >>> timer.stop()  # Logs: "Processing batch completed in X.XXs"
    """

    def __init__(self, logger: logging.Logger, level: int = logging.INFO):
        """
        Initialize the timer.

        Parameters:
            logger (logging.Logger): Logger to use for timing messages.
            level (int): Log level to use (default: INFO).
        """
        self.logger = logger
        self.level = level
        self.start_time: float | None = None
        self.operation: str | None = None

    def start(self, operation: str) -> None:
        """
        Start timing an operation.

        Parameters:
            operation (str): Name/description of the operation.
        """
        self.operation = operation
        self.start_time = time.time()
        self.logger.log(self.level, f"{operation} started")

    def stop(self, context: dict[str, Any] | None = None) -> float:
        """
        Stop timing and log the elapsed time.

        Parameters:
            context (dict, optional): Additional context to include in log.

        Returns:
            float: Elapsed time in seconds.

        Raises:
            ValueError: If start() was not called first.
        """
        if self.start_time is None or self.operation is None:
            raise ValueError("Timer not started. Call start() first.")

        elapsed = time.time() - self.start_time
        context_str = _format_context(context) if context else ""
        self.logger.log(
            self.level,
            f"{self.operation} completed in {elapsed:.2f}s{context_str}",
        )

        self.start_time = None
        self.operation = None
        return elapsed


def log_exception(
    logger: logging.Logger,
    message: str,
    exc: Exception,
    level: int = logging.ERROR,
    context: dict[str, Any] | None = None,
) -> None:
    """
    Log an exception with context and stack trace.

    Parameters:
        logger (logging.Logger): Logger to use.
        message (str): Error message describing what failed.
        exc (Exception): The exception that occurred.
        level (int): Log level (default: ERROR).
        context (dict, optional): Additional context information.

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     log_exception(logger, "Operation failed", e, context={"id": "123"})
    """
    context_str = _format_context(context) if context else ""
    logger.log(
        level,
        f"{message}{context_str}: {type(exc).__name__}: {exc}",
        exc_info=True,
    )


def log_retry(
    logger: logging.Logger,
    operation: str,
    attempt: int,
    max_attempts: int,
    delay: float,
    error: Exception | None = None,
) -> None:
    """
    Log a retry attempt with structured context.

    Parameters:
        logger (logging.Logger): Logger to use.
        operation (str): Name of the operation being retried.
        attempt (int): Current attempt number.
        max_attempts (int): Maximum number of attempts.
        delay (float): Delay before next retry in seconds.
        error (Exception, optional): The error that caused the retry.

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> log_retry(logger, "API call", attempt=2, max_attempts=3, delay=5.0)
    """
    error_msg = f" (error: {error})" if error else ""
    logger.warning(
        f"{operation} retry {attempt}/{max_attempts}, "
        f"waiting {delay:.1f}s before next attempt{error_msg}"
    )
