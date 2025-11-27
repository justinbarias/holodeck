# Implementation Plan: Unstructured Vector Ingestion and Search

**Branch**: `008-unstructured-vector-ingestion-search` | **Date**: 2025-11-23 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/008-unstructured-vector-ingestion-search/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement vectorstore tool support enabling agents to perform semantic search over unstructured text data from single files or directories. The system will automatically ingest content from multiple file formats (.txt, .md, .pdf, .csv, .json), generate embeddings using configurable models, and enable query-based search with relevance ranking. The implementation will use Semantic Kernel's vector store abstractions with support for Redis persistence and in-memory storage.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: Semantic Kernel (vector stores, text chunking, embeddings), markitdown (file-to-markdown conversion - already implemented in src/holodeck/lib/file_processor.py), Pydantic (config models)
**Storage**: Redis (via Semantic Kernel connector, optional), In-memory (default fallback)
**Testing**: pytest with markers (@pytest.mark.unit, @pytest.mark.integration), pytest-cov (80% minimum coverage)
**Target Platform**: Linux/macOS/Windows (cross-platform CLI)
**Project Type**: Single (existing holodeck project structure)
**Performance Goals**: <2s search queries for 1000 documents, support up to 10,000 documents with Redis
**Constraints**: Top 5 results default (configurable via top_k), min_similarity_score threshold filter, UTF-8 encoding normalization via existing FileProcessor
**Scale/Scope**: Single vectorstore tool type within existing tool system (5 types total), 3 user stories (P1-P3), integration with existing agent configuration system and FileProcessor

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-check after Phase 1 design._

### Principle I: No-Code-First Agent Definition ✅ PASS

- **Requirement**: Agent configuration must be YAML-only, no Python code required
- **Status**: Compliant - vectorstore tools are configured via `type: vectorstore` in agent.yaml with declarative parameters (source, embedding_model, top_k, min_similarity_score)
- **Evidence**: Feature spec FR-001 through FR-013 define pure YAML configuration interface

### Principle II: MCP for API Integrations ✅ PASS

- **Requirement**: External APIs must use MCP servers
- **Status**: Not applicable - this feature uses Semantic Kernel for vector operations and file processing, not external API integrations
- **Evidence**: No external API integrations required; uses local/embedded libraries

### Principle III: Test-First with Multimodal Support ✅ PASS

- **Requirement**: Test cases with multimodal inputs, expected_tools validation, ground truth
- **Status**: Compliant - feature leverages existing FileProcessor (src/holodeck/lib/file_processor.py) which supports multimodal files (.txt, .md, .pdf, .csv, .json, images, Office docs)
- **Evidence**: User Story 1 acceptance scenarios define testable behavior with file/directory sources; existing test infrastructure supports multimodal validation

### Principle IV: OpenTelemetry-Native Observability ⚠️ DEFERRED

- **Requirement**: OTel instrumentation with GenAI semantic conventions from day one
- **Status**: Deferred to project-wide observability implementation (not specific to this feature)
- **Evidence**: CLAUDE.md Phase 3 lists "OpenTelemetry instrumentation" as future work; will be applied across all features consistently

### Principle V: Evaluation Flexibility with Model Overrides ✅ PASS

- **Requirement**: Support global/run/metric-level model configuration
- **Status**: Compliant - embedding model configuration supports default (provider-based) and per-tool override via `embedding_model` parameter
- **Evidence**: FR-008, FR-009, User Story 2 define flexible embedding model configuration

### Architecture Constraints ✅ PASS

- **Requirement**: Three decoupled engines (Agent, Evaluation, Deployment)
- **Status**: Compliant - vectorstore tool is part of Agent Engine tool execution system
- **Evidence**: Fits within existing architecture defined in CLAUDE.md (Agent Engine section)

### Code Quality & Testing Discipline ✅ PASS

- **Requirement**: Python 3.10+, Google Style Guide, MyPy strict, 80% coverage, pre-commit hooks
- **Status**: Compliant - follows existing project standards
- **Evidence**: Will use existing Makefile commands (format, lint, type-check, security), pytest infrastructure with unit/integration markers

## Project Structure

### Documentation (this feature)

```
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```
src/holodeck/
├── models/
│   ├── tool.py              # Existing - add VectorStoreConfig model
│   └── config.py            # Existing - referenced by VectorStoreConfig
├── tools/
│   └── vectorstore_tool.py  # NEW - VectorStore tool implementation
├── lib/
│   ├── file_processor.py    # EXISTING - reuse for file ingestion
│   ├── vector_store.py      # NEW - Vector store abstraction layer
│   └── text_chunker.py      # NEW - Text chunking using Semantic Kernel
└── cli/
    └── commands/
        ├── chat.py          # MODIFY - add --force-ingest flag
        └── test.py          # MODIFY - add --force-ingest flag

tests/
├── unit/
│   ├── test_vectorstore_tool.py       # NEW - Tool unit tests
│   ├── test_vector_store.py           # NEW - Vector store tests
│   └── test_text_chunker.py           # NEW - Text chunking tests
└── integration/
    └── test_vectorstore_integration.py # NEW - End-to-end tests
```

**Structure Decision**: Single project structure (Option 1). This feature extends the existing holodeck agent tool system by adding vectorstore capabilities. Key integration points:

- Reuses existing FileProcessor (src/holodeck/lib/file_processor.py) for file-to-markdown conversion
- Extends existing tool configuration models in src/holodeck/models/tool.py
- Integrates with agent execution system via new VectorStoreTool class
- Follows established testing patterns with unit/integration separation

## Complexity Tracking

_No constitutional violations - table not required._

---

## Post-Design Constitution Re-Evaluation

**Date**: 2025-11-23
**Status**: ✅ ALL GATES PASS

After completing Phase 0 (research.md) and Phase 1 (data-model.md, contracts/, quickstart.md), the constitutional compliance has been re-evaluated:

### Design Artifacts Verification

1. **research.md** ✅

   - Technical decisions documented for: Semantic Kernel abstractions, text chunking, embedding generation, Redis integration, file modification tracking, in-memory fallback
   - All "NEEDS CLARIFICATION" items from Technical Context resolved
   - No violations introduced

2. **data-model.md** ✅

   - Defines 7 core entities with validation rules and relationships
   - All entities align with no-code principle (YAML-configurable)
   - No custom code required for agent definition

3. **contracts/vectorstore-tool-interface.md** ✅

   - Tool invocation contract documented
   - Configuration precedence: tool-specific → project config → user config → in-memory
   - Supports multimodal file inputs via existing FileProcessor
   - No external API integrations (MCP principle N/A)

4. **quickstart.md** ✅
   - User-facing documentation demonstrates pure YAML configuration
   - No Python code required for basic or advanced usage
   - Test cases use YAML expected_tools validation

### Constitutional Principles - Post-Design Status

| Principle                  | Pre-Design    | Post-Design   | Notes                                  |
| -------------------------- | ------------- | ------------- | -------------------------------------- |
| I. No-Code-First           | ✅ PASS       | ✅ PASS       | Quickstart confirms pure YAML workflow |
| II. MCP for APIs           | ✅ PASS (N/A) | ✅ PASS (N/A) | No API integrations added              |
| III. Test-First Multimodal | ✅ PASS       | ✅ PASS       | Reuses existing FileProcessor          |
| IV. OTel Observability     | ⚠️ DEFERRED   | ⚠️ DEFERRED   | Project-wide feature                   |
| V. Eval Flexibility        | ✅ PASS       | ✅ PASS       | Embedding model overrides confirmed    |
| Architecture Constraints   | ✅ PASS       | ✅ PASS       | Agent Engine integration               |
| Code Quality               | ✅ PASS       | ✅ PASS       | Standards maintained                   |

### New Dependencies Added

- **Semantic Kernel**: Vector store abstractions, text chunking, embedding services
- **Redis (optional)**: Vector storage backend via Semantic Kernel connector
- **markitdown**: Already existing in FileProcessor - reused, not new

All dependencies align with constitution (no violations).

### Conclusion

**Gate Status**: ✅ **APPROVED FOR IMPLEMENTATION**

All constitutional gates pass. The design maintains:

- Pure YAML configuration (no Python required)
- Reuse of existing multimodal file processing
- Integration within Agent Engine architecture
- No new API integration complexity (uses embedded libraries)

Ready to proceed to Phase 2: Task generation (`/speckit.tasks`).
