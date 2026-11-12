"""Errors raised by the temporal memory layer."""

from __future__ import annotations

from holodeck.lib.errors import HoloDeckError, UserFixableError


class TemporalMemoryError(HoloDeckError):
    """Base class for every error raised by the temporal memory layer."""


class MemoryNotSupportedError(TemporalMemoryError):
    """The backend cannot do the requested operation.

    Backends raise this before any data is read or written. An unsupported
    operation never returns a partial or empty result.
    """

    def __init__(self, backend: str, operation: str, hint: str = "") -> None:
        self.backend = backend
        self.operation = operation
        message = f"Memory backend '{backend}' does not support {operation}."
        if hint:
            message = f"{message} {hint}"
        super().__init__(message)


class MemoryBackendUnavailableError(TemporalMemoryError):
    """The backend's store could not be reached."""

    def __init__(self, backend: str, detail: str) -> None:
        self.backend = backend
        super().__init__(f"Memory backend '{backend}' is unavailable: {detail}")


class MemoryConfigError(TemporalMemoryError, UserFixableError):
    """The memory configuration is wrong. The user fixes the YAML or install."""


class MemoryValueError(TemporalMemoryError, ValueError):
    """A value crossing the memory boundary breaks the contract.

    Raised for a naive or non-UTC datetime, a hop count other than 1 or 2,
    a non-positive limit, a namespace with characters a backend rejects, and
    an episode that lists itself as context.
    """
