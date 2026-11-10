"""Vector store provider names that HoloDeck no longer supports.

This module has no imports so config models and the adapter registry can
share one list without an import cycle or heavy imports at model load.
"""

from __future__ import annotations

from holodeck.lib.removed_in import REMOVED_IN_VERSION

RETIRED_PROVIDERS: dict[str, str] = {
    "faiss": (
        "Use 'chromadb' (embedded, local files) or 'in-memory' for local development."
    ),
    "sql-server": "Use 'postgres' (pgvector) for a SQL database.",
    "azure-cosmos-mongo": "Use 'azure-cosmos-nosql'.",
}


def retired_provider_message(provider: str) -> str | None:
    """Return the config error text for a retired provider, else None."""
    migration = RETIRED_PROVIDERS.get(provider)
    if migration is None:
        return None
    return (
        f"Vector store provider '{provider}' was removed in HoloDeck "
        f"{REMOVED_IN_VERSION}. {migration}"
    )
