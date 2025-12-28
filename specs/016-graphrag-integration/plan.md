# Implementation Plan: GraphRAG Engine Integration

**Branch**: `016-graphrag-integration` | **Date**: 2025-12-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/016-graphrag-integration/spec.md`

---

## Summary

Integrate Microsoft GraphRAG as an `engine: graphrag` option within HoloDeck's existing vectorstore tool type. This enables knowledge graph-based retrieval with:

- **Local Search**: Entity-centric queries with relationship context
- **Global Search**: Dataset-wide analytical queries using community hierarchies
- **Automatic Indexing**: On-demand index building during tool initialization
- **Incremental Updates**: Source file change detection for efficient re-indexing

**Technical Approach**: Extend `VectorstoreTool` configuration to support `engine` field, add nested `graphrag` configuration, and create a new `lib/graphrag/` module that wraps the Microsoft GraphRAG library APIs.

---

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**:
- graphrag>=0.5.0,<1.0.0 (optional)
- pandas>=2.0 (for parquet loading)
- pydantic>=2.0 (existing)

**Storage**: Local filesystem (parquet files in `.holodeck/graphrag/`)
**Testing**: pytest with async support, unit + integration markers
**Target Platform**: Linux/macOS/Windows (same as HoloDeck)
**Project Type**: Single project (extends existing codebase)
**Performance Goals**:
- Index cached: <5s initialization
- Local search: 1-3s response time
- Global search: 5-15s response time

**Constraints**:
- Optional dependency (graphrag not required unless engine="graphrag")
- Backward compatible (existing vectorstore configs work unchanged)
- LLM cost awareness (recommend cheaper models for indexing)

**Scale/Scope**:
- Support document corpora up to ~1000 files
- Entities up to ~100k per index
- Single-user/single-agent context

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. No-Code-First Agent Definition
- [x] **PASS**: GraphRAG is configured entirely via YAML (`engine: graphrag`, `graphrag:` section)
- [x] **PASS**: No Python code required by users

### II. MCP for API Integrations
- [x] **N/A**: GraphRAG is not an external API integration; it's an embedded library
- [x] **PASS**: LLM API calls use existing provider infrastructure

### III. Test-First with Multimodal Support
- [x] **PASS**: Unit tests for config, storage, indexer modules
- [x] **PASS**: Integration tests for end-to-end indexing and search
- [ ] **PENDING**: Tests to be created during implementation

### IV. OpenTelemetry-Native Observability
- [x] **PASS**: Uses existing logging infrastructure
- [ ] **FUTURE**: Tracing for indexing pipeline (not in scope for MVP)

### V. Evaluation Flexibility
- [x] **PASS**: GraphRAG tools work with existing evaluation framework
- [x] **PASS**: Search results can be evaluated with standard metrics

---

## Project Structure

### Documentation (this feature)

```text
specs/016-graphrag-integration/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # GraphRAG API research findings
├── data-model.md        # Configuration and entity models
├── quickstart.md        # User-facing quickstart guide
├── contracts/           # Internal API contracts
│   └── graphrag-engine.md
├── checklists/          # Quality checklists
│   └── requirements.md
└── tasks.md             # Implementation tasks (to be created)
```

### Source Code (repository root)

```text
src/holodeck/
├── models/
│   └── tool.py                    # MODIFY: Add engine, graphrag fields
│
├── tools/
│   └── vectorstore_tool.py        # MODIFY: Add engine branching
│
└── lib/
    └── graphrag/                  # NEW: GraphRAG integration module
        ├── __init__.py            # Lazy imports, GRAPHRAG_AVAILABLE
        ├── config.py              # build_graphrag_config()
        ├── storage.py             # GraphRAGStorage class
        ├── indexer.py             # GraphRAGIndexer class
        ├── search.py              # Search engine wrappers
        └── engine.py              # GraphRAGEngine class

tests/
├── unit/
│   └── lib/
│       └── graphrag/              # NEW: Unit tests
│           ├── test_config.py
│           ├── test_storage.py
│           └── test_engine.py
│
└── integration/
    └── test_graphrag_integration.py  # NEW: E2E tests

pyproject.toml                     # MODIFY: Add graphrag optional dep
```

**Structure Decision**: Single project pattern. New code goes in `src/holodeck/lib/graphrag/` following existing patterns like `lib/evaluators/`. Tests follow existing `tests/unit/lib/` and `tests/integration/` structure.

---

## Files to Create

| File | Purpose | Lines (est.) |
|------|---------|--------------|
| `src/holodeck/lib/graphrag/__init__.py` | Package exports, availability check | ~50 |
| `src/holodeck/lib/graphrag/config.py` | Config generation | ~150 |
| `src/holodeck/lib/graphrag/storage.py` | Artifact management | ~150 |
| `src/holodeck/lib/graphrag/indexer.py` | Indexing orchestration | ~150 |
| `src/holodeck/lib/graphrag/search.py` | Search wrappers | ~150 |
| `src/holodeck/lib/graphrag/engine.py` | Main engine class | ~150 |
| `tests/unit/lib/graphrag/test_config.py` | Config tests | ~80 |
| `tests/unit/lib/graphrag/test_storage.py` | Storage tests | ~100 |
| `tests/unit/lib/graphrag/test_engine.py` | Engine tests | ~100 |
| `tests/integration/test_graphrag_integration.py` | E2E tests | ~100 |

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/holodeck/models/tool.py` | Add `engine`, `GraphRAGConfig`, `GraphRAGModelConfig` |
| `src/holodeck/tools/vectorstore_tool.py` | Add engine branching in `initialize()` and `search()` |
| `pyproject.toml` | Add `graphrag` optional dependency group |

---

## Key Implementation Notes

### 1. GraphRAG API Usage

Based on research, use the direct API approach:

```python
from graphrag.api.index import build_index
from graphrag.api.query import local_search, global_search
from graphrag.config.create_graphrag_config import create_graphrag_config
```

### 2. Config Generation

Build `GraphRagConfig` programmatically (not from YAML file):

```python
config_dict = {
    "root_dir": str(storage_dir),
    "models": {
        "default_chat_model": {...},
        "default_embedding_model": {...},
    },
    "input": {...},
    "output": {...},
    # ... other sections
}
config = create_graphrag_config(values=config_dict, root_dir=str(storage_dir))
```

### 3. Data Loading for Search

Load parquet files using pandas, then convert via adapters:

```python
entities_df = pd.read_parquet(output_dir / "create_final_entities.parquet")
# ... load other tables

from graphrag.query.indexer_adapters import read_indexer_entities
entities = read_indexer_entities(entities_df, communities_df, community_level)
```

### 4. Search Response Format

Return markdown-formatted response matching existing VectorStoreTool pattern:

```python
def _format_result(result: SearchResult) -> str:
    return f"""**GraphRAG {result.search_mode.title()} Search Result**

{result.response}

---
*Mode: {result.search_mode} | Entities: {result.entities_used}*
"""
```

---

## Complexity Tracking

> No constitution violations requiring justification.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| - | - | - |

---

## Related Documents

- [research.md](./research.md) - GraphRAG API research findings
- [data-model.md](./data-model.md) - Configuration and entity models
- [quickstart.md](./quickstart.md) - User-facing quickstart guide
- [contracts/graphrag-engine.md](./contracts/graphrag-engine.md) - Internal API contract
- [claude-plan.md](../graph-rag-integration/claude-plan.md) - Original planning reference
