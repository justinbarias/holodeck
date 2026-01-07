"""
Centralized logging configuration for HoloDeck.

This module provides structured logging setup with support for:
- Console and file handlers
- Environment variable configuration
- Integration with ExecutionConfig (verbose/quiet modes)
- Log rotation for production use
- Structured logging format with timestamps, levels, and module names
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Default log format with timestamp, level, module, and message
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Environment variable names
ENV_LOG_LEVEL = "HOLODECK_LOG_LEVEL"
ENV_LOG_FILE = "HOLODECK_LOG_FILE"
ENV_LOG_FORMAT = "HOLODECK_LOG_FORMAT"

# Default log file settings
DEFAULT_LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10MB
DEFAULT_LOG_FILE_BACKUP_COUNT = 5

# Third-party loggers to configure (these can be noisy at INFO level)
THIRD_PARTY_LOGGERS = [
    "httpx",
    "httpcore",
    "semantic_kernel",
    "openai",
    "anthropic",
    "ollama",
    "urllib3",
    "asyncio",
    "redis",
    "azure",
    "chromadb",
]


def configure_third_party_loggers(log_level: int) -> None:
    """Configure known third-party loggers to respect the specified log level.

    This suppresses noisy INFO logs from libraries like httpx, chromadb, etc.
    Can be called from both traditional logging and OTel logging setup.

    Setting level on parent logger (e.g., "chromadb") also affects child
    loggers (e.g., "chromadb.telemetry.product.posthog").

    Args:
        log_level: The logging level to apply (e.g., logging.WARNING)
    """
    for logger_name in THIRD_PARTY_LOGGERS:
        third_party_logger = logging.getLogger(logger_name)
        third_party_logger.setLevel(log_level)
        third_party_logger.handlers.clear()
        third_party_logger.propagate = True


def setup_logging(
    level: str | None = None,
    log_file: str | None = None,
    log_format: str | None = None,
    verbose: bool = False,
    quiet: bool = False,
) -> None:
    """
    Configure logging for HoloDeck application.

    This function sets up the root logger with appropriate handlers and formatters.
    It respects environment variables and command-line flags for configuration.
    It also configures third-party library loggers to respect the quiet flag.

    Parameters:
        level (str, optional): Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            If not provided, uses HOLODECK_LOG_LEVEL env var or defaults to INFO.
        log_file (str, optional): Path to log file. If not provided, uses
            HOLODECK_LOG_FILE env var. If neither is set, only console logging is used.
        log_format (str, optional): Log format string. If not provided, uses
            HOLODECK_LOG_FORMAT env var or default format.
        verbose (bool): If True, sets log level to DEBUG. Overrides level parameter.
        quiet (bool): If True, sets log level to ERROR. Overrides verbose and level.

    Returns:
        None

    Example:
        >>> setup_logging(verbose=True)  # Enable DEBUG logging
        >>> setup_logging(quiet=True)    # Only show ERROR and above
        >>> setup_logging(log_file="/var/log/holodeck.log")  # Enable file logging
    """
    # Determine log level based on flags and configuration
    if quiet:
        log_level = logging.WARNING
    elif verbose:
        log_level = logging.DEBUG
    elif level:
        log_level = getattr(logging, level.upper(), logging.INFO)
    else:
        env_level = os.getenv(ENV_LOG_LEVEL, "INFO")
        log_level = getattr(logging, env_level.upper(), logging.INFO)

    # Determine log format
    if log_format is None:
        log_format = os.getenv(ENV_LOG_FORMAT, DEFAULT_LOG_FORMAT)

    # Determine log file path
    if log_file is None:
        log_file = os.getenv(ENV_LOG_FILE)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=DEFAULT_DATE_FORMAT)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if log file is specified
    if log_file:
        _add_file_handler(root_logger, log_file, log_level, formatter)

    # Configure all existing loggers to respect the log level
    # This ensures quiet mode works even for loggers created during imports
    _configure_all_loggers(log_level)

    # Log initial setup message at DEBUG level
    logger = logging.getLogger(__name__)
    logger.debug(
        f"Logging configured: level={logging.getLevelName(log_level)}, "
        f"file={'enabled' if log_file else 'disabled'}"
    )


def _add_file_handler(
    logger: logging.Logger,
    log_file: str,
    log_level: int,
    formatter: logging.Formatter,
) -> None:
    """
    Add a rotating file handler to the logger.

    Parameters:
        logger (logging.Logger): Logger to add handler to.
        log_file (str): Path to log file.
        log_level (int): Log level for the handler.
        formatter (logging.Formatter): Formatter for log messages.

    Returns:
        None
    """
    try:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=DEFAULT_LOG_FILE_MAX_BYTES,
            backupCount=DEFAULT_LOG_FILE_BACKUP_COUNT,
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    except OSError as e:
        # If file logging fails, log a warning but continue with console logging
        console_logger = logging.getLogger(__name__)
        console_logger.warning(f"Failed to setup file logging to {log_file}: {e}")


def _configure_all_loggers(log_level: int) -> None:
    """
    Configure holodeck and known third-party loggers to respect the log level.

    This function configures:
    1. Known third-party loggers (THIRD_PARTY_LOGGERS) - sets level and clears
       handlers to ensure they propagate to root and respect quiet mode.
    2. HoloDeck loggers (holodeck.*) - clears handlers and sets appropriate level.

    Other third-party loggers are left untouched to avoid interfering with
    libraries that configure their own logging handlers.

    Parameters:
        log_level (int): The log level to apply to configured loggers.

    Returns:
        None
    """
    # Configure known third-party loggers using the shared function
    configure_third_party_loggers(log_level)

    # Configure holodeck loggers only (not all loggers system-wide)
    # This avoids interfering with third-party libraries that configure
    # their own logging handlers
    for name in list(logging.Logger.manager.loggerDict.keys()):
        # Only configure holodeck.* loggers
        if not name.startswith("holodeck"):
            continue

        logger = logging.getLogger(name)
        # Clear any handlers the logger might have
        logger.handlers.clear()
        # Set level to NOTSET so it inherits from root, OR set explicitly
        # For quiet mode, we set explicitly to ensure no messages get through
        if log_level >= logging.WARNING:
            logger.setLevel(log_level)
        else:
            # For non-quiet modes, let loggers inherit from root
            logger.setLevel(logging.NOTSET)
        # Ensure propagation to root logger
        logger.propagate = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module.

    This is a convenience wrapper around logging.getLogger() that ensures
    consistent logger naming across the application.

    Parameters:
        name (str): Name of the logger, typically __name__ of the calling module.

    Returns:
        logging.Logger: Logger instance for the specified name.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return logging.getLogger(name)


def set_log_level(level: str) -> None:
    """
    Dynamically change the log level for all loggers.

    Parameters:
        level (str): New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        None

    Example:
        >>> set_log_level("DEBUG")  # Enable debug logging
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Update all handlers
    for handler in root_logger.handlers:
        handler.setLevel(log_level)

    # Also update all existing loggers
    _configure_all_loggers(log_level)

    logger = get_logger(__name__)
    logger.debug(f"Log level changed to {logging.getLevelName(log_level)}")
