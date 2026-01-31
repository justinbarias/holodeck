# Tasks: HierarchicalDocumentTool

**Input**: Design documents from `/specs/020-structured-document-tool/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

**Methodology**: TDD (Test-Driven Development) - Unit tests are written BEFORE implementation code in each phase.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- **[TDD]**: Test task that must be completed before implementation
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, dependency installation, and base structure

**References**:

- plan.md:18-25 (Technical Context - Python 3.10+, dependencies)
- plan.md:189-239 (Project Structure)
- plan.md:887-895 (Dependencies to Add - rank-bm25)
- research.md:487-494 (Technology Stack Summary)

- [x] T001 Add `rank-bm25 = "^0.2.2"` to pyproject.toml dependencies per plan.md:887-895
- [x] T002 [P] Create directory structure: `src/holodeck/lib/structured_chunker.py`, `keyword_search.py`, `hybrid_search.py`, `definition_extractor.py`, `llm_context_generator.py` per plan.md:216-221
- [x] T003 [P] Create directory structure: `src/holodeck/tools/hierarchical_document_tool.py` per plan.md:212
- [x] T004 [P] Create test fixtures directory: `tests/fixtures/hierarchical_documents/` with sample files per plan.md:236-239
- [x] T005 [P] Create sample_legislation.md fixture with definitions section and cross-references
- [x] T006 [P] Create sample_technical_doc.md fixture with nested headings (H1-H4)
- [x] T007 [P] Create sample_flat_text.txt fixture without structure for fallback testing

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

- [x] T008 Create `SearchMode` enum (semantic, keyword, exact, hybrid) in src/holodeck/models/tool.py per data-model.md:113-119
- [x] T009 [P] Create `ChunkingStrategy` enum (structure, token) in src/holodeck/models/tool.py per data-model.md:121-125
- [x] T010 [P] Create `ChunkType` enum (content, definition, requirement, reference, header) in src/holodeck/lib/structured_chunker.py per data-model.md:323-330
- [x] T011 Create `HierarchicalDocumentToolConfig` model in src/holodeck/models/tool.py with all fields per data-model.md:127-312 and contracts/hierarchical_document_tool_config.yaml:8-58
- [x] T012 Add `HierarchicalDocumentToolConfig` to `ToolUnion` discriminated union in src/holodeck/models/tool.py

### Vector Store Record Class

- [x] T013 Create `create_hierarchical_document_record_class()` factory function in src/holodeck/lib/vector_store.py per data-model.md:434-502 and research.md:334-357
- [x] T014 Ensure record class has `is_full_text_indexed=True` on `content` field for native hybrid search per plan.md:477-479

### Runtime Data Classes

- [x] T015 [P] Create `DocumentChunk` dataclass in src/holodeck/lib/structured_chunker.py per data-model.md:332-369
- [x] T016 [P] Create `DefinitionEntry` dataclass in src/holodeck/lib/definition_extractor.py per data-model.md:372-383
- [x] T017 [P] Create `SearchResult` dataclass in src/holodeck/lib/hybrid_search.py per data-model.md:385-416

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

### TDD: Write Tests First

- [x] T018 [TDD][US1] Create tests/unit/lib/test_structured_chunker.py with test cases per plan.md:227:
  - Test `_extract_headings()` extracts H1-H6 with correct levels
  - Test `_split_by_headings()` preserves parent_chain
  - Test `_split_section()` respects max_tokens boundary
  - Test `parse()` returns correct DocumentChunk list
  - Test token counting fallback when tiktoken unavailable
  - Test flat text fallback (no headings)

- [x] T019 [TDD][US1] Create tests/unit/lib/test_llm_context_generator.py with test cases:
  - Test `CONTEXT_PROMPT_TEMPLATE` format
  - Test `generate_context()` returns context string
  - Test `contextualize_chunk()` prepends context correctly
  - Test `contextualize_batch()` respects concurrency limit
  - Test exponential backoff on LLM failures
  - Test fallback to no-context on final retry failure
  - Test adaptive concurrency reduction on rate limits

- [x] T020 [TDD][US1] Create tests/unit/tools/test_hierarchical_document_tool.py skeleton with test cases per plan.md:232:
  - Test tool initialization from config
  - Test `_ingest_documents()` orchestration
  - Test `search()` semantic mode returns results with scores
  - Test `SearchResult.format()` includes source attribution
  - Test confidence indication when no matches found

### Implementation: StructuredChunker Module

- [x] T021 [US1] Create `StructuredChunker` class in src/holodeck/lib/structured_chunker.py per plan.md:82-101 and research.md:369-411
- [x] T022 [US1] Implement `_extract_headings()` method using regex `^(#{1,6})\s+(.+)$` per research.md:369-373 and research.md:407-410
- [x] T023 [US1] Implement `_split_by_headings()` to create sections with parent_chain metadata per plan.md:264-267
- [x] T024 [US1] Implement `_split_section()` for sentences boundary splitting when exceeding max_tokens per plan.md:99-101
- [x] T025 [US1] Implement `parse()` method returning `list[DocumentChunk]` with parent_chain, section_id per research.md:384-405
- [x] T026 [US1] Add token counting via tiktoken (or fallback to word count / 0.75) per plan.md:95-96

### Implementation: LLM Context Generator (Anthropic Approach)

- [x] T027 [US1] Create `LLMContextGenerator` class in src/holodeck/lib/llm_context_generator.py per plan.md:356-418
- [x] T028 [US1] Implement `CONTEXT_PROMPT_TEMPLATE` constant per plan.md:343-351 and research.md:17-27
- [x] T029 [US1] Implement `generate_context()` async method for single chunk per plan.md:368-384
- [x] T030 [US1] Implement `contextualize_chunk()` returning `"{context}\n\n{chunk.content}"` per plan.md:385-393
- [x] T031 [US1] Implement `contextualize_batch()` with semaphore for concurrency control (default 10) per plan.md:394-418 and research.md:52-55

### Implementation: Context Generation Error Handling

- [x] T032 [US1] Implement exponential backoff retry for LLM failures (3 attempts: 1s, 2s, 4s delays) per spec.md:163-166
- [x] T033 [US1] Implement fallback to no-context on final retry failure (use original chunk content) per spec.md:165-166
- [x] T034 [US1] Implement adaptive concurrency reduction on rate limit errors (halve on 429, respect Retry-After) per spec.md:167-170
- [x] T035 [US1] Implement document truncation when exceeding LLM context window (prioritize beginning/end) per spec.md:171-174

### Implementation: Dense Index Integration

- [x] T036 [US1] Create `HierarchicalDocumentTool` class skeleton in src/holodeck/tools/hierarchical_document_tool.py
- [x] T037 [US1] Implement `_ingest_documents()` method orchestrating: convert → parse → contextualize → embed → store per plan.md:64-167
- [x] T038 [US1] Implement `_embed_chunks()` using existing embedding patterns from src/holodeck/lib/vector_store.py per plan.md:129-135
- [x] T039 [US1] Implement `_store_chunks()` using Semantic Kernel vector store collection per research.md:311-357
- [x] T040 [US1] Implement `search()` method for semantic-only mode per spec.md:28-33

### Implementation: Search Result Formatting

- [x] T041 [US1] Implement `SearchResult.format()` method with source attribution per data-model.md:400-415 and spec.md:30-31
- [x] T042 [US1] Add confidence indication when no matches found per spec.md:32-33

**Checkpoint**: Semantic search working - can ingest documents and query with natural language. All US1 unit tests pass.

---

## Phase 4: User Story 2 - Exact Match and Keyword Search (Priority: P1)

**Goal**: Enable precise searches for exact phrases, section numbers, and specific terminology

**Independent Test**: Ingest sample_legislation.md with "Section 203(a)(1)", query that exact section, verify it returns as top result

**References**:

- spec.md:36-49 (User Story 2 - Exact Match and Keyword Search)
- plan.md:449-614 (Tiered Keyword Search Strategy)
- research.md:108-234 (Keyword Search Strategy - Tiered)
- plan.md:299-326 (Keyword Index Strategy - Tiered)

### TDD: Write Tests First

- [ ] T043 [TDD][US2] Create tests/unit/lib/test_keyword_search.py with test cases per plan.md:228:
  - Test `KeywordSearchStrategy` enum values
  - Test `get_keyword_search_strategy()` returns correct strategy per provider
  - Test `NATIVE_HYBRID_PROVIDERS` contains expected providers
  - Test `FALLBACK_BM25_PROVIDERS` contains expected providers
  - Test `BM25FallbackProvider.build()` indexes documents
  - Test `BM25FallbackProvider.search()` returns ranked results
  - Test `_tokenize()` handles edge cases (empty, special chars)
  - Test `HybridSearchExecutor` routing to correct strategy

### Implementation: Keyword Search Module (Tiered Strategy)

- [ ] T044 [US2] Create `KeywordSearchStrategy` enum (NATIVE_HYBRID, FALLBACK_BM25) in src/holodeck/lib/keyword_search.py per plan.md:485-491
- [ ] T045 [US2] Define `NATIVE_HYBRID_PROVIDERS` set (azure-ai-search, weaviate, qdrant, mongodb, azure-cosmos-nosql) per plan.md:495-501 and research.md:113-119
- [ ] T046 [US2] Define `FALLBACK_BM25_PROVIDERS` set (postgres, pinecone, chromadb, faiss, in-memory, sql-server, azure-cosmos-mongo) per plan.md:503-511 and research.md:120-124
- [ ] T047 [US2] Implement `get_keyword_search_strategy()` factory function per plan.md:514-518
- [ ] T048 [US2] Create `BM25FallbackProvider` class using rank_bm25.BM25Okapi per plan.md:588-614 and research.md:137-166
- [ ] T049 [US2] Implement `BM25FallbackProvider.build()` from (doc_id, contextualized_text) tuples per plan.md:597-602
- [ ] T050 [US2] Implement `BM25FallbackProvider.search()` returning (doc_id, score) tuples per plan.md:604-609
- [ ] T051 [US2] Implement `_tokenize()` using regex `[a-zA-Z0-9]+` lowercase per plan.md:611-613

### Implementation: Exact Match Index

- [ ] T052 [US2] Create exact match index (dict[str, list[str]]) mapping section_id → chunk_ids in src/holodeck/lib/hybrid_search.py per data-model.md:66-69
- [ ] T053 [US2] Implement section ID pattern detection regex for queries (e.g., "Section 203(a)", "§4.2") per research.md:434-438
- [ ] T054 [US2] Implement exact phrase detection for quoted strings per spec.md:47

### Implementation: Hybrid Search Executor

- [ ] T055 [US2] Create `HybridSearchExecutor` class in src/holodeck/lib/keyword_search.py per plan.md:520-586
- [ ] T056 [US2] Implement `_native_hybrid_search()` using SK `collection.hybrid_search()` per plan.md:543-562 and research.md:173-216
- [ ] T057 [US2] Implement `_fallback_hybrid_search()` running vector + BM25 separately per plan.md:563-586

### Implementation: Tool Integration

- [ ] T058 [US2] Update `HierarchicalDocumentTool._ingest_documents()` to build BM25 index for fallback providers per plan.md:161-164
- [ ] T059 [US2] Update `HierarchicalDocumentTool.search()` to support keyword and exact modes per spec.md:46-49

**Checkpoint**: Exact match and keyword search working - can find precise section references and technical terms. All US2 unit tests pass.

---

## Phase 5: User Story 3 - Structure-Aware Document Ingestion (Priority: P1)

**Goal**: Preserve hierarchical structure during ingestion so search results include context about document location

**Independent Test**: Ingest sample_legislation.md, retrieve a chunk from "Title I > Chapter 2 > Section 203", verify parent_chain is accurate

**References**:

- spec.md:52-65 (User Story 3 - Structure-Aware Document Ingestion)
- plan.md:82-101 (Stage 2: Structure Parsing)
- plan.md:262-268 (StructuredChunker in Architecture)
- data-model.md:506-583 (Document Lifecycle)

### TDD: Write Tests First

- [ ] T060 [TDD][US3] Add test cases to tests/unit/lib/test_structured_chunker.py for structure enhancements:
  - Test normalized section_id generation (e.g., "sec_1_2_3")
  - Test heading level tracking (1-6 for H1-H6, 0 for body)
  - Test chunk_type classification from heading keywords
  - Test parent_chain JSON serialization/deserialization

### Implementation: Enhanced Structure Parsing

- [ ] T061 [US3] Enhance `StructuredChunker` to generate normalized section_id from heading text (e.g., "sec_1_2_3") per data-model.md:36-37
- [ ] T062 [US3] Implement heading level tracking (1-6 for H1-H6, 0 for body content) per data-model.md:39
- [ ] T063 [US3] Implement chunk_type classification based on heading keywords and content patterns per data-model.md:37-38

### Implementation: Markitdown Integration

- [ ] T064 [US3] Integrate with existing markitdown for PDF/Word/HTML → Markdown conversion per plan.md:73-79
- [ ] T065 [US3] Ensure heading levels from converted documents map correctly to hierarchical depth per spec.md:63

### Implementation: Parent Chain Metadata

- [ ] T066 [US3] Store parent_chain as JSON string in vector store record per data-model.md:479
- [ ] T067 [US3] Include parent_chain in SearchResult for display per data-model.md:95

**Checkpoint**: Structure-aware ingestion working - chunks retain full hierarchical context. All US3 unit tests pass.

---

## Phase 6: User Story 4 - Hybrid Search with Result Fusion (Priority: P2)

**Goal**: Combine semantic, keyword, and exact match searches with RRF fusion for optimal retrieval

**Independent Test**: Query with mixed intent ("reporting requirements in Section 403"), verify results come from multiple modalities with sensible ranking

**References**:

- spec.md:68-82 (User Story 4 - Hybrid Search with Result Fusion)
- plan.md:624-648 (Reciprocal Rank Fusion)
- research.md:236-308 (RRF Algorithm)
- data-model.md:625-664 (Search Flow)

### TDD: Write Tests First

- [ ] T068 [TDD][US4] Create tests/unit/lib/test_hybrid_search.py with test cases per plan.md:229:
  - Test `reciprocal_rank_fusion()` basic merge of two lists
  - Test RRF with k=60 produces expected scores
  - Test weighted RRF with custom weights
  - Test handling of imbalanced result sets
  - Test empty result list handling
  - Test exact match index lookup
  - Test fusion of native hybrid with exact match results

### Implementation: RRF Fusion Module

- [ ] T069 [US4] Create `reciprocal_rank_fusion()` function in src/holodeck/lib/hybrid_search.py per plan.md:626-648 and research.md:262-292
- [ ] T070 [US4] Implement weighted RRF with configurable k parameter (default 60) per research.md:251-259
- [ ] T071 [US4] Handle imbalanced result sets (many keyword results, few semantic) per spec.md:154-155

### Implementation: Hybrid Orchestration

- [ ] T072 [US4] Update `HierarchicalDocumentTool.search()` to run all three search modalities in parallel per data-model.md:636-641
- [ ] T073 [US4] Implement fusion of native hybrid results with exact match results via app-level RRF per research.md:240-244
- [ ] T074 [US4] Apply configurable weights (semantic_weight, keyword_weight, exact_weight) per spec.md:79 and data-model.md:187-205
- [ ] T075 [US4] Ensure strong matches are not diluted by weak cross-modality matches per spec.md:81

**Checkpoint**: Hybrid search working - results combine all modalities with intelligent fusion. All US4 unit tests pass.

---

## Phase 7: User Story 5 - Contextual Embeddings for Improved Retrieval (Priority: P2)

**Goal**: Prepend LLM-generated structural context before embedding for better semantic understanding

**Independent Test**: Compare retrieval accuracy with and without contextual embeddings on same query set

**References**:

- spec.md:84-97 (User Story 5 - Contextual Embeddings)
- plan.md:330-442 (LLM-Based Contextual Embedding)
- research.md:8-105 (LLM-Based Contextual Embeddings with Preprocessing Pipeline)
- quickstart.md:149-178 (Preprocessing Pipeline explanation)

### TDD: Write Tests First

- [ ] T076 [TDD][US5] Add test cases to tests/unit/lib/test_llm_context_generator.py for configuration:
  - Test contextual_embeddings toggle (on/off)
  - Test context_model override
  - Test document truncation for large documents

### Implementation: Context Application to BM25

- [ ] T077 [US5] Ensure BM25 index uses contextualized_content (not raw content) per plan.md:170-171 and research.md:88-93
- [ ] T078 [US5] Store both original `content` and `contextualized_content` in vector store record per research.md:54-55

### Implementation: Configurable Context Generation

- [ ] T079 [US5] Make contextual_embeddings configurable (default: true) per data-model.md:231-235 and spec.md:96
- [ ] T080 [US5] Support context_model override in configuration per data-model.md:236-243 and contracts/hierarchical_document_tool_config.yaml:41-46
- [ ] T081 [US5] Handle large documents that exceed LLM context limits (truncate document, preserve chunk) per quickstart.md:236

**Checkpoint**: Contextual embeddings configurable and improving retrieval per Anthropic research (49% better). All US5 unit tests pass.

---

## Phase 8: User Story 6 - YAML Configuration (Priority: P2)

**Goal**: Full YAML-based configuration without requiring code

**Independent Test**: Create agent.yaml with HierarchicalDocumentTool, run holodeck chat, verify all configuration options respected

**References**:

- spec.md:100-113 (User Story 6 - YAML Configuration)
- contracts/hierarchical_document_tool_config.yaml (Full configuration contract)
- data-model.md:127-312 (HierarchicalDocumentToolConfig model)
- quickstart.md:14-36 (Quick Start configuration)

### TDD: Write Tests First

- [ ] T082 [TDD][US6] Add test cases to tests/unit/tools/test_hierarchical_document_tool.py for configuration:
  - Test tool factory registration for `hierarchical_document` type
  - Test tool instantiation from HierarchicalDocumentToolConfig
  - Test Pydantic validation errors for invalid config
  - Test weight validation warning (sum != 1.0)
  - Test defer_loading behavior
  - Test database configuration parsing

### Implementation: Tool Factory Integration

- [ ] T083 [US6] Register `hierarchical_document` tool type in tool factory (src/holodeck/lib/test_runner/agent_factory.py or equivalent) per spec.md:111
- [ ] T084 [US6] Implement tool instantiation from HierarchicalDocumentToolConfig per plan.md:59-60
- [ ] T085 [US6] Validate configuration on load with Pydantic validators per data-model.md:284-311

### Implementation: Configuration Features

- [ ] T086 [US6] Support database configuration for persistence per contracts/hierarchical_document_tool_config.yaml:108-111
- [ ] T087 [US6] Support defer_loading option per data-model.md:279-282
- [ ] T088 [US6] Implement weight validation (warn if sum != 1.0) per data-model.md:291-302

**Checkpoint**: Tool fully configurable via YAML - no code required. All US6 unit tests pass.

---

## Phase 9: User Story 7 - Definition and Cross-Reference Extraction (Priority: P3)

**Goal**: Auto-extract definitions and cross-references for enhanced document understanding

**Independent Test**: Ingest sample_legislation.md with definitions section, query containing a defined term, verify definition included in response context

**References**:

- spec.md:116-129 (User Story 7 - Definition and Cross-Reference Extraction)
- plan.md:651-747 (Definition Detection and Persistence)
- research.md:415-460 (Definition Detection Patterns)
- data-model.md:47-59 (DefinitionEntry entity)

### TDD: Write Tests First

- [ ] T089 [TDD][US7] Create tests/unit/lib/test_definition_extractor.py with test cases per plan.md:230:
  - Test `SECTION_KEYWORDS` contains expected keywords
  - Test `is_definitions_section()` for various headings
  - Test `DEFINITION_PATTERNS` match expected formats:
    - `"X" means Y`
    - `X: Y` at paragraph start
    - `X - Y` with capitalized term
  - Test `extract_definitions()` returns DefinitionEntry list
  - Test `_normalize()` produces correct lookup keys
  - Test `XREF_PATTERNS` match section references
  - Test `rebuild_definitions_index()` from stored chunks

### Implementation: Definition Extractor Module

- [ ] T090 [US7] Create `DefinitionExtractor` class in src/holodeck/lib/definition_extractor.py per plan.md:669-717
- [ ] T091 [US7] Define `SECTION_KEYWORDS` set ("definitions", "glossary", "terms", "interpretation", "defined terms") per plan.md:672
- [ ] T092 [US7] Implement `is_definitions_section()` checking heading keywords per plan.md:683-685
- [ ] T093 [US7] Implement `DEFINITION_PATTERNS` regex list per plan.md:675-681 and research.md:426-431:
  - `"X" means/shall mean Y`
  - `X: Y` (at paragraph start)
  - `X - Y` (capitalized term with dash)
- [ ] T094 [US7] Implement `extract_definitions()` returning list[DefinitionEntry] per plan.md:686-712
- [ ] T095 [US7] Implement `_normalize()` for term lookup (lowercase, spaces → underscores) per plan.md:714-716

### Implementation: Cross-Reference Detection

- [ ] T096 [US7] Implement `XREF_PATTERNS` regex list per research.md:434-438:
  - Section number patterns (e.g., "Section 203", "§4.2")
  - Natural language patterns ("see X", "as defined in", "pursuant to")
- [ ] T097 [US7] Extract cross_references during chunking and store in chunk metadata per data-model.md:38

### Implementation: Definition Persistence

- [ ] T098 [US7] Store `defined_term` and `defined_term_normalized` on definition chunks per data-model.md:41-44 and research.md:446-459
- [ ] T099 [US7] Implement `rebuild_definitions_index()` from stored chunks on startup per plan.md:723-746
- [ ] T100 [US7] Include relevant definitions in SearchResult.definitions_context per data-model.md:97-98

**Checkpoint**: Definitions automatically extracted and surfaced in search results. All US7 unit tests pass.

---

## Phase 10: User Story 8 - Optional Reranking (Priority: P3)

**Goal**: Allow optional reranking pass to improve result quality for precision-focused use cases

**Independent Test**: Run same query with and without reranking, verify ordering improves with reranking enabled

**References**:

- spec.md:132-145 (User Story 8 - Optional Reranking)
- data-model.md:265-272 (Reranking configuration fields)
- data-model.md:650-654 (Reranking in Search Flow)

### TDD: Write Tests First

- [ ] T101 [TDD][US8] Add test cases to tests/unit/tools/test_hierarchical_document_tool.py for reranking:
  - Test reranking step added after RRF fusion
  - Test reranker_model configuration validation
  - Test bypass when reranking disabled

### Implementation: Reranker Integration

- [ ] T102 [US8] Add reranking step after RRF fusion in search flow per data-model.md:650-654
- [ ] T103 [US8] Support `reranker_model` configuration (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2") per contracts/hierarchical_document_tool_config.yaml:138
- [ ] T104 [US8] Validate reranker_model required when enable_reranking=True per data-model.md:304-311
- [ ] T105 [US8] Implement bypass when reranking disabled per spec.md:142-143

**Checkpoint**: Reranking optionally available for high-precision use cases. All US8 unit tests pass.

---

## Phase 11: tool_filter Refactoring (Technical Debt)

**Purpose**: Unify BM25 implementation across codebase using shared modules

**References**:

- plan.md:800-885 (Refactoring: Unify tool_filter with Shared Modules)
- plan.md:870-885 (Migration Tasks)

### TDD: Write Tests First

- [ ] T106 [TDD] Add test cases to existing tool_filter tests verifying equivalent behavior with shared modules per plan.md:884

### Implementation

- [ ] T107 Refactor src/holodeck/lib/tool_filter/index.py to use `BM25FallbackProvider` from keyword_search.py per plan.md:837-846
- [ ] T108 Refactor src/holodeck/lib/tool_filter/index.py to use `reciprocal_rank_fusion()` from hybrid_search.py per plan.md:848-859
- [ ] T109 Remove deprecated manual BM25 methods (lines 207-428) from tool_filter/index.py per plan.md:810-824
- [ ] T110 Remove inline RRF implementation (lines 430-501) from tool_filter/index.py per plan.md:810-824

**Checkpoint**: BM25 and RRF unified across HierarchicalDocumentTool and tool_filter. All refactored tests pass.

---

## Phase 12: Integration Tests & Cross-Cutting Concerns

**Purpose**: End-to-end validation and final polish

**References**:

- plan.md:897-903 (Test Strategy)
- spec.md:216-228 (Success Criteria)
- quickstart.md:240-246 (Testing Your Configuration)

### Integration Tests

- [ ] T111 Create tests/integration/tools/test_hierarchical_document_integration.py per plan.md:234
- [ ] T112 Test full ingestion → search flow with sample_legislation.md fixture
- [ ] T113 Test with persistent storage (postgres) if configured
- [ ] T114 Verify RRF improves over semantic-only (SC-003: 20% improvement) per spec.md:223 and plan.md:902

### Documentation

- [ ] T115 [P] Add HierarchicalDocumentTool to CLI help/docs
- [ ] T116 [P] Add inline code documentation per Google Python Style Guide

### Performance Validation

- [ ] T117 Verify ingestion <30 seconds per 100 pages (SC-004) per spec.md:224
- [ ] T118 Verify search <2 seconds for up to 10,000 chunks (SC-005) per spec.md:225

**Checkpoint**: Feature complete with integration tests, documentation, and performance validation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Story 1-3 (Phase 3-5)**: All P1 priority, can run in sequence or parallel after Phase 2
- **User Story 4-6 (Phase 6-8)**: P2 priority, depend on P1 stories being complete
- **User Story 7-8 (Phase 9-10)**: P3 priority, depend on P2 stories being complete
- **tool_filter Refactor (Phase 11)**: Can run anytime after Phase 4 (when keyword_search.py and hybrid_search.py exist)
- **Integration & Polish (Phase 12)**: Depends on all user stories being complete

### TDD Workflow Within Each Phase

```
1. Write failing tests [TDD] tasks
2. Run tests - verify they fail
3. Implement code to make tests pass
4. Refactor while keeping tests green
5. Checkpoint: All tests pass
```

### User Story Dependencies

- **User Story 1 (Semantic Search)**: Foundation for all other stories
- **User Story 2 (Keyword/Exact)**: Builds on US1 infrastructure
- **User Story 3 (Structure-Aware)**: Enhances US1 with better metadata
- **User Story 4 (Hybrid Fusion)**: Combines US1 + US2 capabilities
- **User Story 5 (Contextual Embeddings)**: Enhances US1's embedding quality
- **User Story 6 (YAML Config)**: Exposes all prior features via configuration
- **User Story 7 (Definitions)**: Advanced feature on top of US1-3
- **User Story 8 (Reranking)**: Advanced feature on top of US4

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

#### Phase 3 (US1) - TDD then Implementation

```
TDD: T018, T019, T020 can run in parallel (different test files)
Implementation: Sequential within each module
```

#### Phase 12 (Integration)

```
T115, T116 can run in parallel (different documentation areas)
```

---

## Implementation Strategy

### TDD-First Approach

Each phase follows the Red-Green-Refactor cycle:

1. **Red**: Write tests first (all [TDD] tasks)
2. **Green**: Implement code to pass tests
3. **Refactor**: Clean up while maintaining passing tests

### MVP First (User Stories 1-3 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1 (Semantic Search) - TDD first
4. Complete Phase 4: User Story 2 (Keyword/Exact Match) - TDD first
5. Complete Phase 5: User Story 3 (Structure-Aware Ingestion) - TDD first
6. **STOP and VALIDATE**: All P1 unit tests pass, integration test passes
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

| Milestone    | Tasks Complete | Capability                       | Tests                 |
| ------------ | -------------- | -------------------------------- | --------------------- |
| Foundation   | T001-T017      | Models and infrastructure ready  | N/A (no behavior)     |
| Semantic MVP | T018-T042      | Natural language search works    | US1 unit tests pass   |
| Full P1      | T043-T067      | All search modes work            | US1-3 unit tests pass |
| Production   | T068-T088      | Hybrid fusion + YAML config      | US1-6 unit tests pass |
| Advanced     | T089-T105      | Definitions + reranking          | US1-8 unit tests pass |
| Complete     | T106-T118      | All features + integration tests | All tests pass        |

**Note**: Total task count is 118

---

## Notes

- **[TDD]** tasks = Write tests BEFORE implementation
- **[P]** tasks = Different files, no dependencies (can run in parallel)
- **[Story]** label maps task to specific user story for traceability
- Each user story follows: Write Tests → Implement → Verify → Checkpoint
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Reference lines in spec documents when implementing for context
