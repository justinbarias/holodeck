"""HoloDeck's temporal memory contract (spec 043).

Tools and the CLI depend on ``MemoryBackend`` and the types here. Each
backend's client stays inside its module under
``holodeck.lib.memory.backends``.
"""

from holodeck.lib.memory.errors import (
    MemoryBackendUnavailableError,
    MemoryConfigError,
    MemoryNotSupportedError,
    MemoryValueError,
    TemporalMemoryError,
)
from holodeck.lib.memory.ports import MemoryBackend
from holodeck.lib.memory.types import (
    EpisodeIn,
    EpisodeRef,
    FactHit,
    HitKind,
    IngestReport,
    MemoryCapabilities,
    MemoryHit,
    MemoryOperation,
    MemoryQuery,
    Provenance,
    ReferenceTimeSource,
    TimelineEntry,
    require_utc,
    validate_namespace,
)

__all__ = [
    "EpisodeIn",
    "EpisodeRef",
    "FactHit",
    "HitKind",
    "IngestReport",
    "MemoryBackend",
    "MemoryBackendUnavailableError",
    "MemoryCapabilities",
    "MemoryConfigError",
    "MemoryHit",
    "MemoryNotSupportedError",
    "MemoryOperation",
    "MemoryQuery",
    "MemoryValueError",
    "Provenance",
    "ReferenceTimeSource",
    "TemporalMemoryError",
    "TimelineEntry",
    "require_utc",
    "validate_namespace",
]
