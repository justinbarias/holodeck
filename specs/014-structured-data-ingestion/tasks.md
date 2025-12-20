# Tasks: Structured Data Field Mapping and Ingestion

**Input**: Design documents from `/specs/014-structured-data-ingestion/`
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, quickstart.md ✓

**Approach**: TDD (Test-Driven Development) - Write tests FIRST, ensure they FAIL, then implement to make them pass.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup ✅

**Purpose**: Project structure and test fixtures for structured data

- [x] T001 [P] Create test fixture directory at tests/fixtures/structured/
- [x] T002 [P] Create products.csv test fixture at tests/fixtures/structured/products.csv
- [x] T003 [P] Create faqs.json test fixture at tests/fixtures/structured/faqs.json
- [x] T004 [P] Create nested_data.json test fixture at tests/fixtures/structured/nested_data.json (for record_path testing)
- [x] T005 [P] Create products.jsonl test fixture at tests/fixtures/structured/products.jsonl (for JSONL testing)

---

## Phase 2: Foundational (Blocking Prerequisites) ✅

**Purpose**: Core infrastructure for structured data support that MUST be complete before ANY user story

**✅ COMPLETE**: Foundation ready - user story work can now begin

### Tests for Foundational (Write FIRST - must FAIL)

- [x] T006 [P] Unit test for VectorstoreTool structured data field validation in tests/unit/models/test_tool_models.py
- [x] T007 [P] Unit test for RecordPathError exception in tests/unit/test_errors.py
- [x] T008 [P] Unit test for StructuredQueryResult dataclass in tests/unit/test_structured_record.py
- [x] T009 [P] Unit test for create_structured_record_class() factory in tests/unit/test_structured_record.py

### Implementation for Foundational (Make tests PASS)

- [x] T010 Extend VectorstoreTool model with structured data fields (id_field, field_separator, delimiter) in src/holodeck/models/tool.py
- [x] T011 Add Pydantic validators for structured data configuration (id_field required with vector_field) in src/holodeck/models/tool.py
- [x] T012 [P] Create RecordPathError exception class in src/holodeck/lib/errors.py
- [x] T013 [P] Create StructuredQueryResult dataclass in src/holodeck/lib/vector_store.py
- [x] T014 Create create_structured_record_class() factory function in src/holodeck/lib/vector_store.py

**Checkpoint**: Foundation ready - all foundational tests pass - user story implementation can now begin

---

## Phase 3: User Story 1 - Semantic Search Over Structured Data with Field Mapping (Priority: P1) ✅ MVP

**Goal**: Enable semantic search over CSV/JSON data sources with explicit field mapping (vector_field, metadata_fields, id_field)

**Independent Test**: Configure a vectorstore tool pointing to products.csv with `vector_field: "description"` and `metadata_fields: ["id", "title", "category"]`, run queries, verify semantically relevant records are returned with their metadata.

### Tests for User Story 1 (Write FIRST - must FAIL)

- [x] T015 [P] [US1] Unit test for detect_file_type() function in tests/unit/test_structured_loader.py
- [x] T016 [P] [US1] Unit test for detect_csv_delimiter() function in tests/unit/test_structured_loader.py
- [x] T017 [P] [US1] Unit test for validate_schema() method in tests/unit/test_structured_loader.py
- [x] T018 [P] [US1] Unit test for load_csv_records() generator in tests/unit/test_structured_loader.py
- [x] T019 [P] [US1] Unit test for load_json_records() function in tests/unit/test_structured_loader.py
- [x] T020 [P] [US1] Unit test for load_jsonl_records() generator in tests/unit/test_structured_loader.py
- [x] T021 [P] [US1] Unit test for iter_records() method in tests/unit/test_structured_loader.py
- [x] T022 [P] [US1] Unit test for iter_batches() method in tests/unit/test_structured_loader.py
- [x] T023 [P] [US1] Integration test for structured vectorstore CSV ingestion and search in tests/integration/test_structured_vectorstore.py
- [x] T024 [P] [US1] Integration test for structured vectorstore JSON ingestion and search in tests/integration/test_structured_vectorstore.py

### Implementation for User Story 1 (Make tests PASS)

- [x] T025 [US1] Create StructuredDataLoader class skeleton in src/holodeck/lib/structured_loader.py
- [x] T026 [US1] Implement detect_file_type() function (CSV, JSON, JSONL detection) in src/holodeck/lib/structured_loader.py
- [x] T027 [US1] Implement detect_csv_delimiter() function with csv.Sniffer in src/holodeck/lib/structured_loader.py
- [x] T028 [US1] Implement validate_schema() method to validate fields against source schema in src/holodeck/lib/structured_loader.py
- [x] T029 [US1] Implement load_csv_records() generator for streaming CSV records in src/holodeck/lib/structured_loader.py
- [x] T030 [US1] Implement load_json_records() function for JSON array files in src/holodeck/lib/structured_loader.py
- [x] T031 [US1] Implement load_jsonl_records() generator for JSONL files in src/holodeck/lib/structured_loader.py
- [x] T032 [US1] Implement iter_records() method combining loaders with field extraction in src/holodeck/lib/structured_loader.py
- [x] T033 [US1] Implement iter_batches() method for batch processing in src/holodeck/lib/structured_loader.py
- [x] T034 [US1] Add structured data mode detection (check vector_field presence) in src/holodeck/tools/vectorstore_tool.py
- [x] T035 [US1] Integrate StructuredDataLoader into vectorstore tool execution for ingestion in src/holodeck/tools/vectorstore_tool.py
- [x] T036 [US1] Wire create_structured_record_class() into vectorstore tool for record generation in src/holodeck/tools/vectorstore_tool.py
- [x] T037 [US1] Add embedding generation for structured records using existing embedding infrastructure in src/holodeck/tools/vectorstore_tool.py
- [x] T038 [US1] Implement search result conversion to StructuredQueryResult in src/holodeck/tools/vectorstore_tool.py

**Checkpoint**: User Story 1 complete - all US1 tests pass - semantic search over CSV/JSON with single vector_field works

---

## Phase 4: User Story 2 - Multiple Vector Fields for Richer Embeddings (Priority: P2)

**Goal**: Support combining multiple fields into a single embedding for richer semantic representations

**Independent Test**: Configure `vector_fields: [title, description, tags]` with `field_separator: "\n\n"`, verify all fields are concatenated and embedded together.

### Tests for User Story 2 (Write FIRST - must FAIL)

- [ ] T039 [P] [US2] Unit test for vector_field normalization (singular vs plural) in tests/unit/test_structured_loader.py
- [ ] T040 [P] [US2] Unit test for field concatenation with separator in tests/unit/test_structured_loader.py
- [ ] T041 [P] [US2] Unit test for null/empty field handling in tests/unit/test_structured_loader.py
- [ ] T042 [P] [US2] Unit test for get_nested_value() dot notation access in tests/unit/test_structured_loader.py
- [ ] T043 [P] [US2] Unit test for parse_path_segments() path parsing in tests/unit/test_structured_loader.py
- [ ] T044 [P] [US2] Unit test for resolve_record_path() navigation in tests/unit/test_structured_loader.py
- [ ] T045 [P] [US2] Unit test for record_path error messages with available keys in tests/unit/test_structured_loader.py
- [ ] T046 [P] [US2] Integration test for multi-field embedding in tests/integration/test_structured_vectorstore.py
- [ ] T047 [P] [US2] Integration test for nested JSON with record_path in tests/integration/test_structured_vectorstore.py

### Implementation for User Story 2 (Make tests PASS)

- [ ] T048 [US2] Update StructuredDataLoader to normalize vector_field vs vector_fields to list in src/holodeck/lib/structured_loader.py
- [ ] T049 [US2] Implement field concatenation with configurable separator in iter_records() in src/holodeck/lib/structured_loader.py
- [ ] T050 [US2] Handle null/empty values in vector fields (skip with warning, continue if partial) in src/holodeck/lib/structured_loader.py
- [ ] T051 [US2] Implement get_nested_value() for dot notation field access (e.g., "details.description") in src/holodeck/lib/structured_loader.py
- [ ] T052 [US2] Implement parse_path_segments() for record_path parsing (dot notation + array indexing) in src/holodeck/lib/structured_loader.py
- [ ] T053 [US2] Implement resolve_record_path() for deep nested JSON navigation in src/holodeck/lib/structured_loader.py
- [ ] T054 [US2] Add helpful error messages with available keys when record_path fails in src/holodeck/lib/structured_loader.py
- [ ] T055 [US2] Integrate record_path support into load_json_records() in src/holodeck/lib/structured_loader.py

**Checkpoint**: User Story 2 complete - all US2 tests pass - multi-field embeddings and nested JSON navigation work

---

## Phase 5: User Story 3 - Database Source Integration (Priority: P3)

**Goal**: Connect to databases (SQLite, PostgreSQL) via Semantic Kernel's existing vector store providers

**Independent Test**: Configure database connection string and SQL query, verify data is extracted and field mappings work correctly.

### Tests for User Story 3 (Write FIRST - must FAIL)

- [ ] T056 [P] [US3] Unit test for DatabaseConfig model validation in tests/unit/test_tool_model.py
- [ ] T057 [P] [US3] Unit test for database source detection in tests/unit/test_structured_loader.py
- [ ] T058 [P] [US3] Unit test for database connection validation in tests/unit/test_structured_loader.py
- [ ] T059 [P] [US3] Unit test for load_database_records() with SQLite in tests/unit/test_structured_loader.py
- [ ] T060 [P] [US3] Unit test for database connection error handling in tests/unit/test_structured_loader.py
- [ ] T061 [P] [US3] Integration test for SQLite database ingestion and search in tests/integration/test_structured_vectorstore.py

### Implementation for User Story 3 (Make tests PASS)

- [ ] T062 [US3] Add DatabaseConfig model for database source configuration in src/holodeck/models/tool.py
- [ ] T063 [US3] Extend StructuredDataLoader to detect database sources in src/holodeck/lib/structured_loader.py
- [ ] T064 [US3] Implement database connection validation using Semantic Kernel connectors in src/holodeck/lib/structured_loader.py
- [ ] T065 [US3] Implement load_database_records() for SQL query execution in src/holodeck/lib/structured_loader.py
- [ ] T066 [US3] Add connection error handling with troubleshooting guidance in src/holodeck/lib/structured_loader.py
- [ ] T067 [US3] Implement incremental re-ingestion based on modification tracking (id_field comparison) in src/holodeck/lib/structured_loader.py

**Checkpoint**: User Story 3 complete - all US3 tests pass - database sources work with field mapping

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

### Tests for Polish (Write FIRST - must FAIL)

- [ ] T068 [P] Unit test for type conversion (int, float, bool → str) in tests/unit/test_structured_loader.py
- [ ] T069 [P] Unit test for "did you mean?" field suggestions in tests/unit/test_structured_loader.py

### Implementation for Polish (Make tests PASS)

- [ ] T070 [P] Add type conversion for non-string values in vector fields (int, float, bool → str) in src/holodeck/lib/structured_loader.py
- [ ] T071 [P] Add progress logging for large file batch processing in src/holodeck/lib/structured_loader.py
- [ ] T072 [P] Add "did you mean?" suggestions for field name typos in error messages in src/holodeck/lib/structured_loader.py
- [ ] T073 Validate quickstart.md examples work end-to-end
- [ ] T074 Run make format, make lint, make type-check, and fix any issues

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can proceed in parallel (if staffed) or sequentially in priority order
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Builds on US1 loader infrastructure
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Independent from US1/US2, adds database support

### TDD Cycle Within Each Phase

1. **RED**: Write all tests for the phase/story first - they MUST fail
2. **GREEN**: Implement code to make tests pass (minimum viable implementation)
3. **REFACTOR**: Clean up code while keeping tests green

### Within Each User Story

- Write ALL tests first (can be parallel - different test files)
- Verify tests FAIL (red phase)
- Implement code to make tests pass (sequential within implementation)
- Verify ALL tests pass (green phase)
- Refactor if needed (keep tests green)

### Parallel Opportunities

**Phase 1 (All parallel)**:
```bash
Task: "Create test fixture directory"
Task: "Create products.csv fixture"
Task: "Create faqs.json fixture"
Task: "Create nested_data.json fixture"
Task: "Create products.jsonl fixture"
```

**Phase 2 Tests (All parallel)**:
```bash
Task: "Unit test for VectorstoreTool structured data field validation"
Task: "Unit test for RecordPathError exception"
Task: "Unit test for StructuredQueryResult dataclass"
Task: "Unit test for create_structured_record_class() factory"
```

**Phase 3 Tests (All parallel)**:
```bash
Task: "Unit test for detect_file_type()"
Task: "Unit test for detect_csv_delimiter()"
Task: "Unit test for validate_schema()"
# ... all US1 tests can run in parallel
```

**Phase 4 Tests (All parallel)**:
```bash
Task: "Unit test for vector_field normalization"
Task: "Unit test for field concatenation with separator"
# ... all US2 tests can run in parallel
```

---

## Implementation Strategy

### TDD MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (5 tasks)
2. Complete Phase 2: Foundational
   - Write tests (4 tasks) → verify FAIL
   - Implement (5 tasks) → verify PASS
3. Complete Phase 3: User Story 1
   - Write tests (10 tasks) → verify FAIL
   - Implement (14 tasks) → verify PASS
4. **STOP and VALIDATE**: All tests green, CSV/JSON semantic search works
5. Deploy/demo if ready

### Incremental TDD Delivery

1. Complete Setup + Foundational (tests → impl) → Foundation ready
2. US1 tests → impl → All green → Deploy/Demo (MVP!)
3. US2 tests → impl → All green → Deploy/Demo
4. US3 tests → impl → All green → Deploy/Demo
5. Each story adds value, all tests remain green

---

## Summary

| Phase | Tests | Implementation | Total | Parallel |
|-------|-------|----------------|-------|----------|
| Phase 1: Setup | 0 | 5 | 5 | 5 |
| Phase 2: Foundational | 4 | 5 | 9 | 6 |
| Phase 3: US1 - MVP | 10 | 14 | 24 | 10 |
| Phase 4: US2 | 9 | 8 | 17 | 9 |
| Phase 5: US3 | 6 | 6 | 12 | 6 |
| Phase 6: Polish | 2 | 5 | 7 | 4 |
| **Total** | **31** | **43** | **74** | **40** |

**MVP Scope**: Phases 1-3 (38 tasks: 5 setup + 9 foundational + 24 US1) - Delivers semantic search over CSV/JSON with field mapping

**Test Coverage**: 31 test tasks covering unit tests and integration tests for all functionality
