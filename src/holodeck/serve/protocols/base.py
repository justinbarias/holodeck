"""Base protocol interface for Agent Local Server.

Defines the abstract base class that both AG-UI and REST protocols implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from holodeck.serve.session_store import ServerSession


class Protocol(ABC):
    """Abstract base class for server protocols.

    Both AG-UI and REST protocols implement this interface to handle
    incoming requests and stream responses back to clients.
    """

    @abstractmethod
    def handle_request(
        self,
        request: Any,
        session: ServerSession,
    ) -> AsyncGenerator[bytes, None]:
        """Handle incoming request and yield response chunks.

        Args:
            request: The incoming request (format depends on protocol).
            session: The server session for this request.

        Yields:
            Response chunks as bytes for streaming to the client.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the protocol name.

        Returns:
            Protocol identifier string (e.g., 'ag-ui', 'rest').
        """
        ...

    @property
    @abstractmethod
    def content_type(self) -> str:
        """Return the content type for responses.

        Returns:
            MIME type string for response Content-Type header.
        """
        ...
