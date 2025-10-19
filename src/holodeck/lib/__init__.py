"""Shared utilities and error handling for HoloDeck."""

from holodeck.lib.errors import ConfigError, HoloDeckError, ValidationError
from holodeck.lib.errors import FileNotFoundError as HoloDeckFileNotFoundError

__all__ = [
    "HoloDeckError",
    "ConfigError",
    "ValidationError",
    "HoloDeckFileNotFoundError",
]
