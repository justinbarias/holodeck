"""Tool execution streaming utilities (placeholder)."""

from collections.abc import Iterable
from typing import Any


class ToolExecutionStream:
    """Streams tool execution events to the caller."""

    def __init__(self, verbose: bool = False) -> None:
        """Initialize the stream with verbosity preference."""
        self.verbose = verbose

    def stream(self, events: Iterable[Any]) -> None:
        """Iterate over events and handle them (not yet implemented)."""
        for _ in events:
            continue
        raise NotImplementedError("ToolExecutionStream.stream is not implemented.")
