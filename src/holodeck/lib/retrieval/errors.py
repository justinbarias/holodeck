"""Errors raised by vector store adapters."""

from __future__ import annotations

from holodeck.lib.errors import HoloDeckError


class RetrievalError(HoloDeckError):
    """Base class for every error raised by the retrieval layer."""


class UnsupportedOperationError(RetrievalError):
    """The provider cannot do the requested operation.

    Adapters raise this before any data is written or read. An unsupported
    operation never returns an empty result.
    """

    def __init__(self, provider: str, operation: str, hint: str = "") -> None:
        self.provider = provider
        self.operation = operation
        message = f"Vector store provider '{provider}' does not support {operation}."
        if hint:
            message = f"{message} {hint}"
        super().__init__(message)


class MissingClientError(RetrievalError):
    """The provider's client library is not installed."""

    def __init__(self, provider: str, extra: str) -> None:
        self.provider = provider
        self.extra = extra
        self.install_command = f"pip install 'holodeck-ai[{extra}]'"
        super().__init__(
            f"Vector store provider '{provider}' needs a client library that is "
            f"not installed. Install it with: {self.install_command}"
        )


class RetiredProviderError(RetrievalError):
    """The provider name was supported once and has been retired."""

    def __init__(self, provider: str, migration: str) -> None:
        self.provider = provider
        super().__init__(
            f"Vector store provider '{provider}' is no longer supported. {migration}"
        )


class UnknownProviderError(RetrievalError):
    """The provider name is not known."""

    def __init__(self, provider: str, supported: list[str]) -> None:
        self.provider = provider
        super().__init__(
            f"Unsupported vector store provider: {provider}. "
            f"Supported providers: {', '.join(sorted(supported))}"
        )


class SchemaError(RetrievalError):
    """A record or query does not match the collection schema."""


class IncompatibleCollectionError(RetrievalError):
    """An existing collection cannot hold records written with this schema."""


class StoreConnectionError(RetrievalError):
    """The store could not be reached or rejected the connection."""

    def __init__(self, provider: str, detail: str) -> None:
        self.provider = provider
        super().__init__(f"Vector store provider '{provider}': {detail}")
