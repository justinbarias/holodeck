# Legacy store layouts (frozen)

These files hold the raw store state that HoloDeck's Semantic Kernel (SK) path
wrote on Qdrant, Chroma and Postgres, and what that path returned for each
read the tools make. They replace the SK bridge in the compatibility tests
(`tests/integration/retrieval/test_{qdrant,chroma,postgres}_compat.py`) after
042 T16 removed Semantic Kernel (AC-05, FR-017).

## How they were made

- Generator: `scripts/freeze_legacy_layouts.py`, run with
  `uv run python -m scripts.freeze_legacy_layouts` on base commit
  `62a2af5f9a5347fb677da9c689af8ba836422ed8`, Semantic Kernel 1.47.0, through
  the SK bridge (`holodeck.lib.retrieval.adapters.sk_bridge`). The generator
  needed SK, so T16 removed it with SK. Commit `5da0f55`
  (`test(retrieval): freeze legacy store layouts as fixtures`) still has the
  generator and the bridge; check it out to read or rerun them.
- Stores: Qdrant 1.17.1 (qdrant-client), Chroma 1.0.0 over HTTP (chromadb),
  PostgreSQL 17.11 with pgvector 0.8.7 (psycopg 3). Each file's `meta` key
  records the exact versions.
- Postgres scores went through the bridge with `SCORE_IS_DISTANCE = True`
  (SK's Postgres cosine search returns a distance), so they are similarities.

## What each file holds

`<store>/<schema>.json`, for the schemas `all_types` (one field of every type
and index kind), `document`, `hierarchical`, and for Postgres also
`structured`:

| Key | Content |
| --- | --- |
| `meta` | Generator, base commit, SK version, store and client versions |
| `collection` | Collection or table name SK wrote (Qdrant point IDs depend on it) |
| `schema` | Field names, types and index flags; a test fails if the code drifts |
| `layout` | Qdrant: collection params, HNSW, optimizer, quantization, payload indexes. Chroma: configuration and metadata. Postgres: `information_schema` columns, `format_type` types, indexes, constraints (table name as `<table>`) |
| `rows` | Every point or row: ID, payload or metadata, document, vector |
| `expected` | What the SK path returned for `get`, `find`, `list_page`, `count`, `vector_search` (and Qdrant `hybrid_search`) |
| `after_writes` | `all_types` only: what the SK path read back after the native adapter's writes |

The tests rebuild each legacy collection with the store's own client only
(`tests/integration/retrieval/legacy_layout.py`), check the rebuild equals the
frozen layout and rows, then check the native adapter's reads, fresh layout,
stored rows and writes against these files.

## When a store default changes

The layout comparison includes server defaults (for example Qdrant's HNSW and
optimizer settings). If a new store version changes a default, compare the
fresh native collection with a rebuilt legacy one on the same server instead
of updating the frozen layout; the frozen file records what SK wrote and
cannot be regenerated without SK.
