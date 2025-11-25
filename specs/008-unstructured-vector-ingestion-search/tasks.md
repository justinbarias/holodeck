# Tasks: Unstructured Vector Ingestion and Search

**Input**: Design documents from `/specs/008-unstructured-vector-ingestion-search/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Tests are NOT explicitly requested in the feature specification. Test tasks are included for validation but can be removed if not desired.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/holodeck/`, `tests/` at repository root
- Existing project structure is extended

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add dependencies and prepare infrastructure for vectorstore tools

- [x] T001 Add Semantic Kernel dependency to pyproject.toml (vector stores, text chunking, embeddings)
- [x] T002 [P] Add redis-py async dependency to pyproject.toml for Redis vector store support
- [x] T003 [P] Update pyproject.toml with tiktoken dependency for token-based text chunking
- [x] T004 Install all new dependencies via poetry and verify no conflicts

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core abstractions and utilities that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T005 [P] Extend VectorStoreConfig model in src/holodeck/models/tool.py with all configuration fields (source, embedding_model, database, top_k, min_similarity_score)
- [x] T006 [P] Add DatabaseConfig model in src/holodeck/models/tool.py for Vector store (redis, postgres, pinecone, chroma, faiss) connection configuration (provider, connection_string, index_name, vector_algorithm, distance_metric)
- [x] T007 [P] Create DocumentRecord dataclass in src/holodeck/lib/vector_store.py with Semantic Kernel annotations (VectorStoreRecordKeyField, VectorStoreRecordDataField, VectorStoreRecordVectorField)
- [x] T008 [P] Create SourceFile dataclass in src/holodeck/lib/file_processor.py (path, content, mtime, size_bytes, file_type, chunks)
- [x] T009 [P] Create QueryResult dataclass in src/holodeck/lib/vector_store.py (content, score, source_path, chunk_index, metadata)
- [x] T010 Implement VectorStore abstraction class in src/holodeck/lib/vector_store.py with methods: upsert, get, delete, search, delete_by_source (depends on T007, T009)
- [x] T011 Implement Redis backend initialization in VectorStore using RedisStore from Semantic Kernel in src/holodeck/lib/vector_store.py (depends on T010)
- [x] T012 Implement in-memory backend initialization in VectorStore using VolatileVectorStore from Semantic Kernel in src/holodeck/lib/vector_store.py (depends on T010)
- [x] T013 [P] Create TextChunker wrapper class in src/holodeck/lib/text_chunker.py using RecursiveCharacterTextSplitter from Semantic Kernel with token-based chunking (512 tokens, 50 overlap)
- [x] T014 Implement configuration resolution logic for database config (tool-specific → project config → user config → in-memory) in src/holodeck/config/loader.py
- [x] T015 [P] Update ConfigValidator in src/holodeck/config/validator.py to validate VectorStoreConfig fields (type, source, top_k range, min_similarity_score range, database config)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Semantic Search Over Unstructured Data (Priority: P1) 🎯 MVP

**Goal**: Enable agents to perform semantic search over single files or directories with automatic ingestion, embedding generation, and relevance-ranked results

**Independent Test**: Configure a vectorstore tool pointing to either a single markdown file or a directory of mixed file types, run queries, and verify semantically relevant results are returned with source references

### Unit Tests for User Story 1

**NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T016 [P] [US1] Unit test for VectorStoreTool initialization with valid config in tests/unit/test_vectorstore_tool.py
- [x] T017 [P] [US1] Unit test for VectorStoreTool initialization with missing source path in tests/unit/test_vectorstore_tool.py
- [x] T018 [P] [US1] Unit test for VectorStoreTool file discovery (single file) in tests/unit/test_vectorstore_tool.py
- [x] T019 [P] [US1] Unit test for VectorStoreTool file discovery (directory with nested subdirectories) in tests/unit/test_vectorstore_tool.py
- [x] T020 [P] [US1] Unit test for VectorStoreTool search result formatting in tests/unit/test_vectorstore_tool.py
- [x] T021 [P] [US1] Unit test for text chunking with markdown content in tests/unit/test_text_chunker.py
- [x] T022 [P] [US1] Unit test for VectorStore upsert and get operations in tests/unit/test_vector_store.py
- [x] T023 [P] [US1] Unit test for VectorStore search with top_k filtering in tests/unit/test_vector_store.py

### Implementation for User Story 1

- [x] T024 [P] [US1] Create VectorStoreTool class skeleton in src/holodeck/tools/vectorstore_tool.py with **init**, initialize, and search methods (depends on T005)
- [x] T025 [US1] Implement VectorStoreTool.**init** to accept VectorStoreConfig and initialize instance variables in src/holodeck/tools/vectorstore_tool.py (depends on T024, T010, T013)
- [x] T026 [US1] Implement file/directory discovery logic in VectorStoreTool.\_discover_files method in src/holodeck/tools/vectorstore_tool.py supporting recursive traversal (depends on T025)
- [x] T027 [US1] Implement file filtering by supported extensions (.txt, .md, .pdf, .csv, .json) in VectorStoreTool.\_discover_files in src/holodeck/tools/vectorstore_tool.py (depends on T026)
- [x] T028 [US1] Implement VectorStoreTool.\_process_file method to convert file to markdown using existing FileProcessor in src/holodeck/tools/vectorstore_tool.py (depends on T025, T008)
- [x] T029 [US1] Implement text chunking in VectorStoreTool.\_process_file using TextChunker in src/holodeck/tools/vectorstore_tool.py (depends on T028, T013)
- [x] T030 [US1] Implement batch embedding generation in VectorStoreTool.\_embed_chunks method using Semantic Kernel TextEmbedding service in src/holodeck/tools/vectorstore_tool.py (depends on T025)
- [x] T031 [US1] Implement DocumentRecord creation and storage in VectorStoreTool.\_store_chunks method in src/holodeck/tools/vectorstore_tool.py (depends on T029, T030, T010)
- [x] T032 [US1] Implement VectorStoreTool.initialize method to orchestrate file discovery, processing, and storage in src/holodeck/tools/vectorstore_tool.py (depends on T026, T027, T031)
- [x] T033 [US1] Implement VectorStoreTool.search method to generate query embedding and execute vector search in src/holodeck/tools/vectorstore_tool.py (depends on T032, T010)
- [x] T034 [US1] Implement search result formatting in VectorStoreTool.\_format_results method in src/holodeck/tools/vectorstore_tool.py (depends on T033, T009)
- [x] T035 [US1] Add error handling for FileNotFoundError when source path doesn't exist in VectorStoreTool.initialize in src/holodeck/tools/vectorstore_tool.py (depends on T032)
- [x] T036 [US1] Add warning logs for skipped files (unsupported types, empty files, processing errors) in VectorStoreTool.\_process_file in src/holodeck/tools/vectorstore_tool.py (depends on T028)
- [x] T037 [US1] Implement top_k result limiting in VectorStoreTool.search in src/holodeck/tools/vectorstore_tool.py (depends on T033)

### Agent Integration for User Story 1

**Purpose**: Integrate VectorStoreTool with AgentFactory so agents can use vectorstore tools during execution

- [x] T037a [US1] Add TextEmbedding service registration method to AgentFactory in src/holodeck/lib/test_runner/agent_factory.py (OpenAI/Azure text-embedding-3-small) following Semantic Kernel pattern (depends on T024)
- [x] T037b [US1] Create KernelFunction for VectorStoreTool.search using kernel.add_function() in AgentFactory (NOT decorator) in src/holodeck/lib/test_runner/agent_factory.py (depends on T033)
- [x] T037c [US1] Add \_register_vectorstore_tools method to AgentFactory to discover vectorstore tools from agent config, initialize them, and register search functions via kernel.add_function() in src/holodeck/lib/test_runner/agent_factory.py (depends on T037a, T037b)
- [x] T037d [US1] Update AgentFactory.\_create_kernel to call embedding service registration when vectorstore tools are configured in src/holodeck/lib/test_runner/agent_factory.py (depends on T037a)
- [x] T037e [US1] Update AgentFactory.\_create_agent to call \_register_vectorstore_tools before agent creation in src/holodeck/lib/test_runner/agent_factory.py (depends on T037c)
- [x] T037f [P] [US1] Unit test for AgentFactory embedding service registration in tests/unit/test_agent_factory.py
- [x] T037g [P] [US1] Unit test for VectorStoreTool kernel function registration via add_function in tests/unit/test_agent_factory.py
- [x] T037h [P] [US1] Unit test for AgentFactory vectorstore tool discovery and initialization in tests/unit/test_agent_factory.py

### Integration Tests for User Story 1

- [ ] T038 [P] [US1] Integration test for end-to-end ingestion and search with single markdown file in tests/integration/test_vectorstore_integration.py
- [ ] T039 [P] [US1] Integration test for end-to-end ingestion and search with directory of mixed file types in tests/integration/test_vectorstore_integration.py
- [ ] T040 [P] [US1] Integration test for nested subdirectory traversal in tests/integration/test_vectorstore_integration.py
- [ ] T041 [P] [US1] Integration test for search result ordering by relevance score in tests/integration/test_vectorstore_integration.py

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently - agents can perform semantic search over files and directories

---

## Phase 4: User Story 2 - Custom Embedding Model Configuration (Priority: P2)

**Goal**: Allow developers to specify custom embedding models for optimization (domain-specific embeddings, cost, performance)

**Independent Test**: Configure embedding_model parameter in vectorstore tool, verify embeddings are generated using the specified model, compare against default model behavior

### Unit Tests for User Story 2

- [x] T042 [P] [US2] Unit test for embedding model parameter parsing from config in tests/unit/test_vectorstore_tool.py
- [x] T043 [P] [US2] Unit test for default embedding model selection based on LLM provider in tests/unit/test_vectorstore_tool.py
- [x] T044 [P] [US2] Unit test for custom embedding model initialization in tests/unit/test_vectorstore_tool.py

### Implementation for User Story 2

- [x] T045 [P] [US2] Implement embedding model resolution logic in VectorStoreTool.**init** (custom model vs provider default) in src/holodeck/tools/vectorstore_tool.py (depends on T025)
- [x] T046 [US2] Implement TextEmbedding service initialization with configurable model in VectorStoreTool.**init** in src/holodeck/tools/vectorstore_tool.py (depends on T045)
- [x] T047 [US2] Add provider-specific default embedding model mapping (OpenAI → text-embedding-3-small, Azure → text-embedding-ada-002) in src/holodeck/tools/vectorstore_tool.py (depends on T046)
- [x] T048 [US2] Add validation for embedding model availability based on configured LLM provider in src/holodeck/config/validator.py (depends on T015)
- [x] T049 [US2] Update embedding dimension detection to match configured model in DocumentRecord creation in src/holodeck/tools/vectorstore_tool.py (depends on T031, T046)

### Integration Tests for User Story 2

- [ ] T050 [P] [US2] Integration test for custom embedding model usage (text-embedding-3-large) in tests/integration/test_vectorstore_integration.py
- [ ] T051 [P] [US2] Integration test for default embedding model when no explicit model specified in tests/integration/test_vectorstore_integration.py

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently - custom embedding models are supported

---

## Phase 5: User Story 3 - Vector Database Persistence (Priority: P3)

**Goal**: Enable persistent vector storage using Redis for production deployments with large knowledge bases and multi-session persistence

**Independent Test**: Configure Redis vector database connection, ingest test data, verify data persists across agent restarts, measure search performance with large datasets

### Unit Tests for User Story 3

- [x] T052 [P] [US3] Unit test for file modification timestamp tracking in tests/unit/test_vectorstore_tool.py
- [x] T053 [P] [US3] Unit test for mtime comparison logic (needs re-ingestion vs up-to-date) in tests/unit/test_vectorstore_tool.py
- [x] T054 [P] [US3] Unit test for --force-ingest flag bypassing mtime checks in tests/unit/test_vectorstore_tool.py
- [x] T055 [P] [US3] Unit test for Redis connection fallback to in-memory on connection failure in tests/unit/test_vector_store.py

### Implementation for User Story 3

- [x] T056 [P] [US3] Implement file modification time tracking in VectorStoreTool.\_process_file (store mtime in DocumentRecord) in src/holodeck/tools/vectorstore_tool.py (depends on T028, T007)
- [x] T057 [US3] Implement VectorStoreTool.\_needs_reingest method to compare file mtime with stored DocumentRecord mtime in src/holodeck/tools/vectorstore_tool.py (depends on T056)
- [x] T058 [US3] Integrate mtime checking into VectorStoreTool.\_ingest_source to skip unchanged files in src/holodeck/tools/vectorstore_tool.py (depends on T057, T032)
- [x] T059 [US3] Implement force_ingest parameter support in VectorStoreTool.initialize to bypass mtime checks in src/holodeck/tools/vectorstore_tool.py (depends on T058)
- [x] T060 [US3] Implement VectorStore.delete_by_source method to remove old records when file is modified in src/holodeck/lib/vector_store.py (depends on T010, T058)
- [x] T061 [US3] Add Redis connection error handling with fallback to in-memory storage in VectorStore.**init** in src/holodeck/lib/vector_store.py (depends on T011, T012)
- [x] T062 [US3] Add --force-ingest flag to holodeck chat command in src/holodeck/cli/commands/chat.py
- [x] T063 [US3] Add --force-ingest flag to holodeck test command in src/holodeck/cli/commands/test.py
- [x] T064 [US3] Pass --force-ingest flag to VectorStoreTool.initialize in agent execution flow in src/holodeck/cli/commands/chat.py (depends on T062, T059)
- [x] T065 [US3] Pass --force-ingest flag to VectorStoreTool.initialize in test execution flow in src/holodeck/cli/commands/test.py (depends on T063, T059)
- [x] T066 [US3] Add logging for Redis connection success/failure and in-memory fallback in VectorStore.**init** in src/holodeck/lib/vector_store.py (depends on T061)

### Integration Tests for User Story 3

- [ ] T067 [P] [US3] Integration test for Redis vector store persistence across agent restarts in tests/integration/test_vectorstore_integration.py
- [ ] T068 [P] [US3] Integration test for automatic re-ingestion when source file modified in tests/integration/test_vectorstore_integration.py
- [ ] T069 [P] [US3] Integration test for --force-ingest flag forcing full re-ingestion in tests/integration/test_vectorstore_integration.py
- [ ] T070 [P] [US3] Integration test for Redis connection failure fallback to in-memory in tests/integration/test_vectorstore_integration.py

**Checkpoint**: All user stories should now be independently functional - Redis persistence and modification tracking complete

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T071 [P] Add min_similarity_score filtering in VectorStoreTool.search in src/holodeck/tools/vectorstore_tool.py (depends on T033)
- [ ] T072 [P] Add comprehensive docstrings to VectorStoreTool class and methods in src/holodeck/tools/vectorstore_tool.py following Google Python Style Guide (depends on T037, T049, T066)
- [ ] T073 [P] Add comprehensive docstrings to VectorStore class and methods in src/holodeck/lib/vector_store.py following Google Python Style Guide (depends on T060, T066)
- [ ] T074 [P] Add comprehensive docstrings to TextChunker class in src/holodeck/lib/text_chunker.py following Google Python Style Guide (depends on T013)
- [ ] T075 [P] Add type hints to all VectorStoreTool methods in src/holodeck/tools/vectorstore_tool.py (depends on T072)
- [ ] T076 [P] Add type hints to all VectorStore methods in src/holodeck/lib/vector_store.py (depends on T073)
- [ ] T077 [P] Add type hints to all TextChunker methods in src/holodeck/lib/text_chunker.py (depends on T074)
- [ ] T078 Run make format to format all new code with Black and Ruff
- [ ] T079 Run make lint to check code quality with Ruff and Bandit (depends on T078)
- [ ] T080 Run make type-check to verify MyPy type checking passes (depends on T075, T076, T077)
- [ ] T081 Run make test-coverage to verify 80% minimum test coverage (depends on T041, T051, T070)
- [ ] T082 [P] Update quickstart.md with any corrections based on implementation in specs/008-unstructured-vector-ingestion-search/quickstart.md
- [ ] T083 [P] Add edge case handling for empty query string in VectorStoreTool.search in src/holodeck/tools/vectorstore_tool.py (depends on T033)
- [ ] T084 [P] Add edge case handling for no results found in VectorStoreTool.\_format_results in src/holodeck/tools/vectorstore_tool.py (depends on T034)
- [ ] T085 Run make security to check for security vulnerabilities with Bandit and Safety (depends on T079)
- [ ] T086 Validate quickstart.md examples work end-to-end (depends on T082)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion (T001-T004) - BLOCKS all user stories
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion (T005-T015)
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Extends US1 but independently testable (depends on T025, T031)
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Extends US1 but independently testable (depends on T028, T032, T033)

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Data models before tool implementation
- Core functionality before edge case handling
- Implementation before integration tests
- Story complete before moving to next priority

### Parallel Opportunities

**Phase 1 (Setup)**:

- T002, T003 can run in parallel

**Phase 2 (Foundational)**:

- T005, T006, T007, T008, T009, T013, T015 can all run in parallel (different files, no dependencies)

**User Story 1**:

- All unit tests (T016-T023) can run in parallel
- T024 and T025 must complete before parallel work begins
- T028, T030 can run in parallel after T025
- Agent integration: T037a-T037e must complete sequentially (depends on T033)
- Agent integration unit tests (T037f, T037g, T037h) can run in parallel
- All integration tests (T038-T041) can run in parallel after implementation and agent integration

**User Story 2**:

- All unit tests (T042-T044) can run in parallel
- T045 and T046 must complete before T047 and T049
- All integration tests (T050-T051) can run in parallel after implementation

**User Story 3**:

- All unit tests (T052-T055) can run in parallel
- T056 and T057 must complete before parallel work begins
- T062, T063, T066 can run in parallel
- All integration tests (T067-T070) can run in parallel after implementation

**Phase 6 (Polish)**:

- T071, T072, T073, T074, T075, T076, T077, T082, T083, T084 can all run in parallel

---

## Parallel Example: User Story 1

```bash
# Launch all unit tests for User Story 1 together:
Task T016: "Unit test for VectorStoreTool initialization with valid config"
Task T017: "Unit test for VectorStoreTool initialization with missing source path"
Task T018: "Unit test for VectorStoreTool file discovery (single file)"
Task T019: "Unit test for VectorStoreTool file discovery (directory with nested subdirectories)"
Task T020: "Unit test for VectorStoreTool search result formatting"
Task T021: "Unit test for text chunking with markdown content"
Task T022: "Unit test for VectorStore upsert and get operations"
Task T023: "Unit test for VectorStore search with top_k filtering"

# Launch foundational abstractions together (after Phase 2):
Task T007: "Create DocumentRecord dataclass"
Task T008: "Create SourceFile dataclass"
Task T009: "Create QueryResult dataclass"
Task T013: "Create TextChunker wrapper class"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T004)
2. Complete Phase 2: Foundational (T005-T015) - CRITICAL
3. Complete Phase 3: User Story 1 (T016-T041)
4. **STOP and VALIDATE**: Test semantic search over single files and directories
5. Deploy/demo basic vectorstore functionality

**Deliverable**: Agents can perform semantic search over documentation files with in-memory storage

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready (T001-T015)
2. Add User Story 1 → Test independently → Deploy/Demo (T016-T041) - **MVP!**
3. Add User Story 2 → Test independently → Deploy/Demo (T042-T051) - Custom embedding models
4. Add User Story 3 → Test independently → Deploy/Demo (T052-T070) - Redis persistence
5. Polish → Run validation (T071-T086)

Each story adds value without breaking previous stories.

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together (T001-T015)
2. Once Foundational is done:
   - Developer A: User Story 1 (T016-T041 + T037a-T037h agent integration)
   - Developer B: User Story 2 (T042-T051) - Can start in parallel
   - Developer C: User Story 3 (T052-T070) - Can start in parallel
3. Stories complete and integrate independently

---

## Summary

**Total Tasks**: 94 tasks

- **Phase 1 (Setup)**: 4 tasks
- **Phase 2 (Foundational)**: 11 tasks (BLOCKS all user stories)
- **Phase 3 (User Story 1)**: 34 tasks (8 unit tests + 14 implementation + 8 agent integration + 4 integration tests)
- **Phase 4 (User Story 2)**: 10 tasks (3 unit tests + 5 implementation + 2 integration tests)
- **Phase 5 (User Story 3)**: 19 tasks (4 unit tests + 11 implementation + 4 integration tests)
- **Phase 6 (Polish)**: 16 tasks

**Parallel Opportunities**: 41 tasks marked with [P] can run in parallel within their phases

**MVP Scope**: Phases 1-3 (T001-T041 + T037a-T037h) - 49 tasks for basic semantic search functionality with agent integration

**Independent Test Criteria**:

- **US1**: Configure vectorstore with file/directory, run queries, verify relevant results with source attribution
- **US2**: Configure custom embedding model, verify model is used, compare results to default
- **US3**: Configure Redis, verify persistence across restarts, test modification tracking with --force-ingest

**Suggested Implementation Order**: P1 (US1) → P2 (US2) → P3 (US3) → Polish

---

## Notes

- [P] tasks = different files, no dependencies within phase
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing (TDD approach)
- Run code quality commands (format, lint, type-check) after each phase
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
