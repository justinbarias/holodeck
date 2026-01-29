# Tasks: HierarchicalDocumentTool

**Input**: Design documents from `/specs/020-structured-document-tool/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, dependency installation, and base structure

**References**:
- plan.md:18-25 (Technical Context - Python 3.10+, dependencies)
- plan.md:189-239 (Project Structure)
- plan.md:887-895 (Dependencies to Add - rank-bm25)
- research.md:487-494 (Technology Stack Summary)

- [ ] T001 Add `rank-bm25 = "^0.2.2"` to pyproject.toml dependencies per plan.md:887-895
- [ ] T002 [P] Create directory structure: `src/holodeck/lib/structured_chunker.py`, `keyword_search.py`, `hybrid_search.py`, `definition_extractor.py`, `llm_context_generator.py` per plan.md:216-221
- [ ] T003 [P] Create directory structure: `src/holodeck/tools/hierarchical_document_tool.py` per plan.md:212
- [ ] T004 [P] Create test fixtures directory: `tests/fixtures/hierarchical_documents/` with sample files per plan.md:236-239
- [ ] T005 [P] Create sample_legislation.md fixture with definitions section and cross-references
- [ ] T006 [P] Create sample_technical_doc.md fixture with nested headings (H1-H4)
- [ ] T007 [P] Create sample_flat_text.txt fixture without structure for fallback testing

**Checkpoint**: Project structure created, dependencies installed, test fixtures ready

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**References**:
- plan.md:749-772 (HierarchicalDocumentChunk Record Class)
- data-model.md:101-312 (Pydantic Configuration Models)
- data-model.md:420-502 (Vector Store Record Schema)
- research.md:311-357 (Document Record Model Pattern)

**CRITICAL**: No user story work can begin until this phase is complete

### Pydantic Configuration Models

- [ ] T008 Create `SearchMode` enum (semantic, keyword, exact, hybrid) in src/holodeck/models/tool.py per data-model.md:113-119
- [ ] T009 [P] Create `ChunkingStrategy` enum (structure, token) in src/holodeck/models/tool.py per data-model.md:121-125
- [ ] T010 [P] Create `ChunkType` enum (content, definition, requirement, reference, header) in src/holodeck/lib/structured_chunker.py per data-model.md:323-330
- [ ] T011 Create `HierarchicalDocumentToolConfig` model in src/holodeck/models/tool.py with all fields per data-model.md:127-312 and contracts/hierarchical_document_tool_config.yaml:8-58
- [ ] T012 Add `HierarchicalDocumentToolConfig` to `ToolUnion` discriminated union in src/holodeck/models/tool.py

### Vector Store Record Class

- [ ] T013 Create `create_hierarchical_document_record_class()` factory function in src/holodeck/lib/vector_store.py per data-model.md:434-502 and research.md:334-357
- [ ] T014 Ensure record class has `is_full_text_indexed=True` on `content` field for native hybrid search per plan.md:477-479

### Runtime Data Classes

- [ ] T015 [P] Create `DocumentChunk` dataclass in src/holodeck/lib/structured_chunker.py per data-model.md:332-369
- [ ] T016 [P] Create `DefinitionEntry` dataclass in src/holodeck/lib/definition_extractor.py per data-model.md:372-383
- [ ] T017 [P] Create `SearchResult` dataclass in src/holodeck/lib/hybrid_search.py per data-model.md:385-416

**Checkpoint**: Foundation ready - all configuration models and data classes available for user story implementation

---

## Phase 3: User Story 1 - Semantic Search Across Documents (Priority: P1) MVP

**Goal**: Enable natural language queries that find conceptually related content regardless of exact wording

**Independent Test**: Ingest sample_technical_doc.md, query "What are the reporting requirements?", verify relevant sections returned with source citations

**References**:
- spec.md:20-33 (User Story 1 - Semantic Search)
- plan.md:64-167 (Preprocessing Pipeline)
- plan.md:330-442 (LLM-Based Contextual Embedding)
- research.md:8-105 (LLM-Based Contextual Embeddings)
- data-model.md:506-583 (Document Lifecycle)

### Implementation for User Story 1

#### StructuredChunker Module

- [ ] T018 [US1] Create `StructuredChunker` class in src/holodeck/lib/structured_chunker.py per plan.md:82-101 and research.md:369-411
- [ ] T019 [US1] Implement `_extract_headings()` method using regex `^(#{1,6})\s+(.+)$` per research.md:369-373 and research.md:407-410
- [ ] T020 [US1] Implement `_split_by_headings()` to create sections with parent_chain metadata per plan.md:264-267
- [ ] T021 [US1] Implement `_split_section()` for sentences boundary splitting when exceeding max_tokens per plan.md:99-101
- [ ] T022 [US1] Implement `parse()` method returning `list[DocumentChunk]` with parent_chain, section_id per research.md:384-405
- [ ] T023 [US1] Add token counting via tiktoken (or fallback to word count / 0.75) per plan.md:95-96

#### LLM Context Generator (Anthropic Approach)

- [ ] T024 [US1] Create `LLMContextGenerator` class in src/holodeck/lib/llm_context_generator.py per plan.md:356-418
- [ ] T025 [US1] Implement `CONTEXT_PROMPT_TEMPLATE` constant per plan.md:343-351 and research.md:17-27
- [ ] T026 [US1] Implement `generate_context()` async method for single chunk per plan.md:368-384
- [ ] T027 [US1] Implement `contextualize_chunk()` returning `"{context}\n\n{chunk.content}"` per plan.md:385-393
- [ ] T028 [US1] Implement `contextualize_batch()` with semaphore for concurrency control (default 10) per plan.md:394-418 and research.md:52-55

#### Dense Index Integration

- [ ] T029 [US1] Create `HierarchicalDocumentTool` class skeleton in src/holodeck/tools/hierarchical_document_tool.py
- [ ] T030 [US1] Implement `_ingest_documents()` method orchestrating: convert → parse → contextualize → embed → store per plan.md:64-167
- [ ] T031 [US1] Implement `_embed_chunks()` using existing embedding patterns from src/holodeck/lib/vector_store.py per plan.md:129-135
- [ ] T032 [US1] Implement `_store_chunks()` using Semantic Kernel vector store collection per research.md:311-357
- [ ] T033 [US1] Implement `search()` method for semantic-only mode per spec.md:28-33

#### Search Result Formatting

- [ ] T034 [US1] Implement `SearchResult.format()` method with source attribution per data-model.md:400-415 and spec.md:30-31
- [ ] T035 [US1] Add confidence indication when no matches found per spec.md:32-33

**Checkpoint**: Semantic search working - can ingest documents and query with natural language

---

## Phase 4: User Story 2 - Exact Match and Keyword Search (Priority: P1)

**Goal**: Enable precise searches for exact phrases, section numbers, and specific terminology

**Independent Test**: Ingest sample_legislation.md with "Section 203(a)(1)", query that exact section, verify it returns as top result

**References**:
- spec.md:36-49 (User Story 2 - Exact Match and Keyword Search)
- plan.md:449-614 (Tiered Keyword Search Strategy)
- research.md:108-234 (Keyword Search Strategy - Tiered)
- plan.md:299-326 (Keyword Index Strategy - Tiered)

### Implementation for User Story 2

#### Keyword Search Module (Tiered Strategy)

- [ ] T036 [US2] Create `KeywordSearchStrategy` enum (NATIVE_HYBRID, FALLBACK_BM25) in src/holodeck/lib/keyword_search.py per plan.md:485-491
- [ ] T037 [US2] Define `NATIVE_HYBRID_PROVIDERS` set (azure-ai-search, weaviate, qdrant, mongodb, azure-cosmos-nosql) per plan.md:495-501 and research.md:113-119
- [ ] T038 [US2] Define `FALLBACK_BM25_PROVIDERS` set (postgres, pinecone, chromadb, faiss, in-memory, sql-server, azure-cosmos-mongo) per plan.md:503-511 and research.md:120-124
- [ ] T039 [US2] Implement `get_keyword_search_strategy()` factory function per plan.md:514-518
- [ ] T040 [US2] Create `BM25FallbackProvider` class using rank_bm25.BM25Okapi per plan.md:588-614 and research.md:137-166
- [ ] T041 [US2] Implement `BM25FallbackProvider.build()` from (doc_id, contextualized_text) tuples per plan.md:597-602
- [ ] T042 [US2] Implement `BM25FallbackProvider.search()` returning (doc_id, score) tuples per plan.md:604-609
- [ ] T043 [US2] Implement `_tokenize()` using regex `[a-zA-Z0-9]+` lowercase per plan.md:611-613

#### Exact Match Index

- [ ] T044 [US2] Create exact match index (dict[str, list[str]]) mapping section_id → chunk_ids in src/holodeck/lib/hybrid_search.py per data-model.md:66-69
- [ ] T045 [US2] Implement section ID pattern detection regex for queries (e.g., "Section 203(a)", "§4.2") per research.md:434-438
- [ ] T046 [US2] Implement exact phrase detection for quoted strings per spec.md:47

#### Hybrid Search Executor

- [ ] T047 [US2] Create `HybridSearchExecutor` class in src/holodeck/lib/keyword_search.py per plan.md:520-586
- [ ] T048 [US2] Implement `_native_hybrid_search()` using SK `collection.hybrid_search()` per plan.md:543-562 and research.md:173-216
- [ ] T049 [US2] Implement `_fallback_hybrid_search()` running vector + BM25 separately per plan.md:563-586

#### Tool Integration

- [ ] T050 [US2] Update `HierarchicalDocumentTool._ingest_documents()` to build BM25 index for fallback providers per plan.md:161-164
- [ ] T051 [US2] Update `HierarchicalDocumentTool.search()` to support keyword and exact modes per spec.md:46-49

**Checkpoint**: Exact match and keyword search working - can find precise section references and technical terms

---

## Phase 5: User Story 3 - Structure-Aware Document Ingestion (Priority: P1)

**Goal**: Preserve hierarchical structure during ingestion so search results include context about document location

**Independent Test**: Ingest sample_legislation.md, retrieve a chunk from "Title I > Chapter 2 > Section 203", verify parent_chain is accurate

**References**:
- spec.md:52-65 (User Story 3 - Structure-Aware Document Ingestion)
- plan.md:82-101 (Stage 2: Structure Parsing)
- plan.md:262-268 (StructuredChunker in Architecture)
- data-model.md:506-583 (Document Lifecycle)

### Implementation for User Story 3

#### Enhanced Structure Parsing

- [ ] T052 [US3] Enhance `StructuredChunker` to generate normalized section_id from heading text (e.g., "sec_1_2_3") per data-model.md:36-37
- [ ] T053 [US3] Implement heading level tracking (1-6 for H1-H6, 0 for body content) per data-model.md:39
- [ ] T054 [US3] Implement chunk_type classification based on heading keywords and content patterns per data-model.md:37-38

#### Markitdown Integration

- [ ] T055 [US3] Integrate with existing markitdown for PDF/Word/HTML → Markdown conversion per plan.md:73-79
- [ ] T056 [US3] Ensure heading levels from converted documents map correctly to hierarchical depth per spec.md:63

#### Parent Chain Metadata

- [ ] T057 [US3] Store parent_chain as JSON string in vector store record per data-model.md:479
- [ ] T058 [US3] Include parent_chain in SearchResult for display per data-model.md:95

**Checkpoint**: Structure-aware ingestion working - chunks retain full hierarchical context

---

## Phase 6: User Story 4 - Hybrid Search with Result Fusion (Priority: P2)

**Goal**: Combine semantic, keyword, and exact match searches with RRF fusion for optimal retrieval

**Independent Test**: Query with mixed intent ("reporting requirements in Section 403"), verify results come from multiple modalities with sensible ranking

**References**:
- spec.md:68-82 (User Story 4 - Hybrid Search with Result Fusion)
- plan.md:624-648 (Reciprocal Rank Fusion)
- research.md:236-308 (RRF Algorithm)
- data-model.md:625-664 (Search Flow)

### Implementation for User Story 4

#### RRF Fusion Module

- [ ] T059 [US4] Create `reciprocal_rank_fusion()` function in src/holodeck/lib/hybrid_search.py per plan.md:626-648 and research.md:262-292
- [ ] T060 [US4] Implement weighted RRF with configurable k parameter (default 60) per research.md:251-259
- [ ] T061 [US4] Handle imbalanced result sets (many keyword results, few semantic) per spec.md:154-155

#### Hybrid Orchestration

- [ ] T062 [US4] Update `HierarchicalDocumentTool.search()` to run all three search modalities in parallel per data-model.md:636-641
- [ ] T063 [US4] Implement fusion of native hybrid results with exact match results via app-level RRF per research.md:240-244
- [ ] T064 [US4] Apply configurable weights (semantic_weight, keyword_weight, exact_weight) per spec.md:79 and data-model.md:187-205
- [ ] T065 [US4] Ensure strong matches are not diluted by weak cross-modality matches per spec.md:81

**Checkpoint**: Hybrid search working - results combine all modalities with intelligent fusion

---

## Phase 7: User Story 5 - Contextual Embeddings for Improved Retrieval (Priority: P2)

**Goal**: Prepend LLM-generated structural context before embedding for better semantic understanding

**Independent Test**: Compare retrieval accuracy with and without contextual embeddings on same query set

**References**:
- spec.md:84-97 (User Story 5 - Contextual Embeddings)
- plan.md:330-442 (LLM-Based Contextual Embedding)
- research.md:8-105 (LLM-Based Contextual Embeddings with Preprocessing Pipeline)
- quickstart.md:149-178 (Preprocessing Pipeline explanation)

### Implementation for User Story 5

#### Context Application to BM25

- [ ] T066 [US5] Ensure BM25 index uses contextualized_content (not raw content) per plan.md:170-171 and research.md:88-93
- [ ] T067 [US5] Store both original `content` and `contextualized_content` in vector store record per research.md:54-55

#### Configurable Context Generation

- [ ] T068 [US5] Make contextual_embeddings configurable (default: true) per data-model.md:231-235 and spec.md:96
- [ ] T069 [US5] Support context_model override in configuration per data-model.md:236-243 and contracts/hierarchical_document_tool_config.yaml:41-46
- [ ] T070 [US5] Handle large documents that exceed LLM context limits (truncate document, preserve chunk) per quickstart.md:236

**Checkpoint**: Contextual embeddings configurable and improving retrieval per Anthropic research (49% better)

---

## Phase 8: User Story 6 - YAML Configuration (Priority: P2)

**Goal**: Full YAML-based configuration without requiring code

**Independent Test**: Create agent.yaml with HierarchicalDocumentTool, run holodeck chat, verify all configuration options respected

**References**:
- spec.md:100-113 (User Story 6 - YAML Configuration)
- contracts/hierarchical_document_tool_config.yaml (Full configuration contract)
- data-model.md:127-312 (HierarchicalDocumentToolConfig model)
- quickstart.md:14-36 (Quick Start configuration)

### Implementation for User Story 6

#### Tool Factory Integration

- [ ] T071 [US6] Register `hierarchical_document` tool type in tool factory (src/holodeck/lib/test_runner/agent_factory.py or equivalent) per spec.md:111
- [ ] T072 [US6] Implement tool instantiation from HierarchicalDocumentToolConfig per plan.md:59-60
- [ ] T073 [US6] Validate configuration on load with Pydantic validators per data-model.md:284-311

#### Configuration Features

- [ ] T074 [US6] Support database configuration for persistence per contracts/hierarchical_document_tool_config.yaml:108-111
- [ ] T075 [US6] Support defer_loading option per data-model.md:279-282
- [ ] T076 [US6] Implement weight validation (warn if sum != 1.0) per data-model.md:291-302

**Checkpoint**: Tool fully configurable via YAML - no code required

---

## Phase 9: User Story 7 - Definition and Cross-Reference Extraction (Priority: P3)

**Goal**: Auto-extract definitions and cross-references for enhanced document understanding

**Independent Test**: Ingest sample_legislation.md with definitions section, query containing a defined term, verify definition included in response context

**References**:
- spec.md:116-129 (User Story 7 - Definition and Cross-Reference Extraction)
- plan.md:651-747 (Definition Detection and Persistence)
- research.md:415-460 (Definition Detection Patterns)
- data-model.md:47-59 (DefinitionEntry entity)

### Implementation for User Story 7

#### Definition Extractor Module

- [ ] T077 [US7] Create `DefinitionExtractor` class in src/holodeck/lib/definition_extractor.py per plan.md:669-717
- [ ] T078 [US7] Define `SECTION_KEYWORDS` set ("definitions", "glossary", "terms", "interpretation", "defined terms") per plan.md:672
- [ ] T079 [US7] Implement `is_definitions_section()` checking heading keywords per plan.md:683-685
- [ ] T080 [US7] Implement `DEFINITION_PATTERNS` regex list per plan.md:675-681 and research.md:426-431:
  - `"X" means/shall mean Y`
  - `X: Y` (at paragraph start)
  - `X - Y` (capitalized term with dash)
- [ ] T081 [US7] Implement `extract_definitions()` returning list[DefinitionEntry] per plan.md:686-712
- [ ] T082 [US7] Implement `_normalize()` for term lookup (lowercase, spaces → underscores) per plan.md:714-716

#### Cross-Reference Detection

- [ ] T083 [US7] Implement `XREF_PATTERNS` regex list per research.md:434-438:
  - Section number patterns (e.g., "Section 203", "§4.2")
  - Natural language patterns ("see X", "as defined in", "pursuant to")
- [ ] T084 [US7] Extract cross_references during chunking and store in chunk metadata per data-model.md:38

#### Definition Persistence

- [ ] T085 [US7] Store `defined_term` and `defined_term_normalized` on definition chunks per data-model.md:41-44 and research.md:446-459
- [ ] T086 [US7] Implement `rebuild_definitions_index()` from stored chunks on startup per plan.md:723-746
- [ ] T087 [US7] Include relevant definitions in SearchResult.definitions_context per data-model.md:97-98

**Checkpoint**: Definitions automatically extracted and surfaced in search results

---

## Phase 10: User Story 8 - Optional Reranking (Priority: P3)

**Goal**: Allow optional reranking pass to improve result quality for precision-focused use cases

**Independent Test**: Run same query with and without reranking, verify ordering improves with reranking enabled

**References**:
- spec.md:132-145 (User Story 8 - Optional Reranking)
- data-model.md:265-272 (Reranking configuration fields)
- data-model.md:650-654 (Reranking in Search Flow)

### Implementation for User Story 8

#### Reranker Integration

- [ ] T088 [US8] Add reranking step after RRF fusion in search flow per data-model.md:650-654
- [ ] T089 [US8] Support `reranker_model` configuration (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2") per contracts/hierarchical_document_tool_config.yaml:138
- [ ] T090 [US8] Validate reranker_model required when enable_reranking=True per data-model.md:304-311
- [ ] T091 [US8] Implement bypass when reranking disabled per spec.md:142-143

**Checkpoint**: Reranking optionally available for high-precision use cases

---

## Phase 11: tool_filter Refactoring (Technical Debt)

**Purpose**: Unify BM25 implementation across codebase using shared modules

**References**:
- plan.md:800-885 (Refactoring: Unify tool_filter with Shared Modules)
- plan.md:870-885 (Migration Tasks)

### Implementation

- [ ] T092 Refactor src/holodeck/lib/tool_filter/index.py to use `BM25FallbackProvider` from keyword_search.py per plan.md:837-846
- [ ] T093 Refactor src/holodeck/lib/tool_filter/index.py to use `reciprocal_rank_fusion()` from hybrid_search.py per plan.md:848-859
- [ ] T094 Remove deprecated manual BM25 methods (lines 207-428) from tool_filter/index.py per plan.md:810-824
- [ ] T095 Remove inline RRF implementation (lines 430-501) from tool_filter/index.py per plan.md:810-824
- [ ] T096 Update tool_filter tests to verify equivalent behavior with shared modules per plan.md:884

**Checkpoint**: BM25 and RRF unified across HierarchicalDocumentTool and tool_filter

---

## Phase 12: Polish & Cross-Cutting Concerns

**Purpose**: Final improvements affecting multiple user stories

**References**:
- plan.md:897-903 (Test Strategy)
- spec.md:216-228 (Success Criteria)
- quickstart.md:240-246 (Testing Your Configuration)

### Unit Tests

- [ ] T097 [P] Create tests/unit/lib/test_structured_chunker.py per plan.md:227
- [ ] T098 [P] Create tests/unit/lib/test_keyword_search.py (BM25 fallback + provider detection) per plan.md:228
- [ ] T099 [P] Create tests/unit/lib/test_hybrid_search.py (RRF fusion logic) per plan.md:229
- [ ] T100 [P] Create tests/unit/lib/test_definition_extractor.py per plan.md:230
- [ ] T101 [P] Create tests/unit/lib/test_llm_context_generator.py
- [ ] T102 [P] Create tests/unit/tools/test_hierarchical_document_tool.py per plan.md:232

### Integration Tests

- [ ] T103 Create tests/integration/tools/test_hierarchical_document_integration.py per plan.md:234
- [ ] T104 Test full ingestion → search flow with sample_legislation.md fixture
- [ ] T105 Test with persistent storage (postgres) if configured
- [ ] T106 Verify RRF improves over semantic-only (SC-003: 20% improvement) per spec.md:223 and plan.md:902

### Documentation

- [ ] T107 [P] Add HierarchicalDocumentTool to CLI help/docs
- [ ] T108 [P] Add inline code documentation per Google Python Style Guide

### Performance Validation

- [ ] T109 Verify ingestion <30 seconds per 100 pages (SC-004) per spec.md:224
- [ ] T110 Verify search <2 seconds for up to 10,000 chunks (SC-005) per spec.md:225

**Checkpoint**: Feature complete with tests, documentation, and performance validation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Story 1-3 (Phase 3-5)**: All P1 priority, can run in sequence or parallel after Phase 2
- **User Story 4-6 (Phase 6-8)**: P2 priority, depend on P1 stories being complete
- **User Story 7-8 (Phase 9-10)**: P3 priority, depend on P2 stories being complete
- **tool_filter Refactor (Phase 11)**: Can run anytime after Phase 4 (when keyword_search.py and hybrid_search.py exist)
- **Polish (Phase 12)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (Semantic Search)**: Foundation for all other stories
- **User Story 2 (Keyword/Exact)**: Builds on US1 infrastructure
- **User Story 3 (Structure-Aware)**: Enhances US1 with better metadata
- **User Story 4 (Hybrid Fusion)**: Combines US1 + US2 capabilities
- **User Story 5 (Contextual Embeddings)**: Enhances US1's embedding quality
- **User Story 6 (YAML Config)**: Exposes all prior features via configuration
- **User Story 7 (Definitions)**: Advanced feature on top of US1-3
- **User Story 8 (Reranking)**: Advanced feature on top of US4

### Within Each User Story

- Models before services
- Services before tool methods
- Core implementation before integration
- Tests can be written in parallel with implementation

### Parallel Opportunities

#### Phase 1 (Setup)
```
T002, T003, T004 can run in parallel (different directories)
T005, T006, T007 can run in parallel (different fixture files)
```

#### Phase 2 (Foundation)
```
T008, T009, T010 can run in parallel (different enums)
T015, T016, T017 can run in parallel (different dataclasses)
```

#### Phase 3-5 (P1 Stories) - Can run sequentially or in parallel
```
US1: T018-T035 (StructuredChunker + LLMContextGenerator + Dense Index)
US2: T036-T051 (Keyword Search + Exact Match)
US3: T052-T058 (Enhanced Structure Parsing)
```

#### Phase 12 (Polish)
```
T097, T098, T099, T100, T101, T102 can run in parallel (different test files)
T107, T108 can run in parallel (different documentation areas)
```

---

## Implementation Strategy

### MVP First (User Stories 1-3 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1 (Semantic Search)
4. Complete Phase 4: User Story 2 (Keyword/Exact Match)
5. Complete Phase 5: User Story 3 (Structure-Aware Ingestion)
6. **STOP and VALIDATE**: Test all P1 stories independently
7. Deploy/demo if ready

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add US1 → Semantic search working (MVP!)
3. Add US2 → Exact match and keyword working
4. Add US3 → Full structure awareness
5. Add US4 → Hybrid fusion (major quality improvement)
6. Add US5-6 → Production-ready configuration
7. Add US7-8 → Advanced features

### Key Milestones

| Milestone | Tasks Complete | Capability |
|-----------|----------------|------------|
| Foundation | T001-T017 | Models and infrastructure ready |
| Semantic MVP | T018-T035 | Natural language search works |
| Full P1 | T036-T058 | All search modes work |
| Production | T059-T076 | Hybrid fusion + YAML config |
| Advanced | T077-T091 | Definitions + reranking |
| Complete | T092-T110 | All features + tests + refactoring |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Reference lines in spec documents when implementing for context
