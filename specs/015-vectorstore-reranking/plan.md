# Implementation Plan: Vectorstore Reranking Extension

**Branch**: `015-vectorstore-reranking` | **Date**: 2025-12-23 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/015-vectorstore-reranking/spec.md`

## Summary

Add opt-in reranking capability to vectorstore-type tools to improve search result relevance. When enabled via `rerank: true`, initial vector search results are passed through a reranking model (Cohere or vLLM) before being returned. The feature supports configurable `rerank_top_n` (default: `top_k * 3`) to control how many candidates are reranked, with graceful fallback to original results on failure.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**:
- cohere (for both Cohere cloud and vLLM - vLLM is Cohere API compatible)
- Semantic Kernel (existing agent framework)
**Storage**: N/A (uses existing vector store infrastructure)
**Testing**: pytest with async support (`@pytest.mark.unit`, `@pytest.mark.integration`)
**Target Platform**: Linux server / cross-platform CLI
**Project Type**: single
**Performance Goals**: Reranking adds <500ms latency for typical batch sizes (20-30 candidates)
**Constraints**: Graceful degradation on reranker failure; backward compatibility required
**Scale/Scope**: Supports same scale as existing vectorstore tools

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Justification |
|-----------|--------|---------------|
| I. No-Code-First Agent Definition | ✅ PASS | Reranking is configured via YAML (`rerank: true`, `reranker:` block) |
| II. MCP for API Integrations | ✅ PASS | Cohere and vLLM are reranking-specific services, not general API integrations. They are tightly coupled to the vectorstore search flow and don't benefit from MCP's tool discovery pattern. |
| III. Test-First with Multimodal Support | ✅ PASS | Will include unit tests for reranker classes and integration tests for search flow |
| IV. OpenTelemetry-Native Observability | ✅ PASS | Debug-level logging for reranker calls with latency and result counts (per FR-011) |
| V. Evaluation Flexibility | ✅ N/A | Feature does not involve evaluation metrics |

## Project Structure

### Documentation (this feature)

```text
specs/015-vectorstore-reranking/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
src/holodeck/
├── models/
│   └── tool.py                    # Extend VectorstoreTool model with rerank fields
├── lib/
│   └── rerankers/                 # NEW: Reranker implementations
│       ├── __init__.py            # Factory function and exports
│       ├── base.py                # Abstract BaseReranker class
│       └── cohere.py              # CohereReranker (works for both Cohere & vLLM)
└── tools/
    └── vectorstore_tool.py        # Integrate reranking into search flow

tests/
├── unit/
│   └── lib/
│       └── rerankers/             # NEW: Reranker unit tests
│           ├── test_base.py
│           └── test_cohere.py     # Tests both Cohere and vLLM modes
└── integration/
    └── tools/
        └── test_vectorstore_reranking.py  # NEW: Integration tests
```

**Structure Decision**: Single project structure with new `lib/rerankers/` module. The `CohereReranker` class handles both Cohere cloud and vLLM servers since vLLM's rerank endpoints are fully Cohere API compatible. This reduces code duplication while maintaining separate YAML config models for clarity.

## Complexity Tracking

> No Constitution violations requiring justification.
