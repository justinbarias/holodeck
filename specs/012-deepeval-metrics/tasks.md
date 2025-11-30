# Tasks: DeepEval Metrics Integration

**Input**: Design documents from `/specs/012-deepeval-metrics/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Unit tests included as this is a library feature requiring thorough validation.

**Organization**: Tasks grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure) ✅

**Purpose**: Add DeepEval dependency and create module structure

- [x] T001 Add `deepeval>=0.21.0,<1.0.0` to dependencies in pyproject.toml
- [x] T002 [P] Create deepeval evaluators module directory at src/holodeck/lib/evaluators/deepeval/
- [x] T003 [P] Create src/holodeck/lib/evaluators/deepeval/__init__.py with module exports
- [x] T004 [P] Create tests/unit/lib/evaluators/deepeval/ directory structure

---

## Phase 2: Foundational (Blocking Prerequisites) ✅

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T005 Implement ProviderNotSupportedError in src/holodeck/lib/evaluators/deepeval/errors.py
- [x] T006 Implement DeepEvalError in src/holodeck/lib/evaluators/deepeval/errors.py
- [x] T007 Implement DeepEvalModelConfig with provider validation in src/holodeck/lib/evaluators/deepeval/config.py
- [x] T008 Implement to_deepeval_model() method using native DeepEval classes (GPTModel, AzureOpenAIModel, AnthropicModel, OllamaModel) in src/holodeck/lib/evaluators/deepeval/config.py
- [x] T009 Implement DeepEvalBaseEvaluator with _build_test_case and _evaluate_impl in src/holodeck/lib/evaluators/deepeval/base.py
- [x] T010 [P] Create unit tests for DeepEvalModelConfig in tests/unit/lib/evaluators/deepeval/test_config.py
- [x] T011 [P] Create unit tests for DeepEvalBaseEvaluator in tests/unit/lib/evaluators/deepeval/test_base.py

**Checkpoint**: Foundation ready - user story implementation can now begin ✅

---

## Phase 3: User Story 1 - Multi-Provider Support (Priority: P1) 🎯 MVP ✅

**Goal**: Enable evaluation using OpenAI, Anthropic, or Ollama as judge models

**Independent Test**: Configure OpenAI provider and run G-Eval evaluation on sample response

### Tests for User Story 1

- [x] T012 [P] [US1] Create unit tests for GEvalEvaluator in tests/unit/lib/evaluators/deepeval/test_geval.py

### Implementation for User Story 1

- [x] T013 [US1] Implement GEvalEvaluator with criteria and evaluation_steps support in src/holodeck/lib/evaluators/deepeval/geval.py
- [x] T014 [US1] Add _create_metric() method wrapping DeepEval's GEval in src/holodeck/lib/evaluators/deepeval/geval.py
- [x] T015 [US1] Add _extract_result() method normalizing score to 0-1 in src/holodeck/lib/evaluators/deepeval/geval.py (Note: handled by base class)
- [x] T016 [US1] Add logging for evaluation scores and reasoning using HoloDeck logger in src/holodeck/lib/evaluators/deepeval/base.py (Note: already in base class)
- [x] T017 [US1] Update src/holodeck/lib/evaluators/deepeval/__init__.py to export GEvalEvaluator
- [x] T018 [US1] Update src/holodeck/lib/evaluators/__init__.py to include deepeval module exports

### TestExecutor Integration for User Story 1

- [x] T049 [US1] Create GEvalMetric model (discriminated union) with criteria, evaluation_steps, strict_mode, evaluation_params fields in src/holodeck/models/evaluation.py
- [x] T050 [US1] Update EvaluationConfig.metrics to accept Union[EvaluationMetric, GEvalMetric] with discriminator on type field in src/holodeck/models/evaluation.py
- [x] T051 [P] [US1] Create unit tests for GEvalMetric model validation in tests/unit/models/test_evaluation_deepeval.py
- [x] T052 [US1] Add DeepEval imports (GEvalEvaluator, DeepEvalModelConfig) to src/holodeck/lib/test_runner/executor.py
- [x] T053 [US1] Add _build_deepeval_config() helper method to convert LLMProvider to DeepEvalModelConfig in src/holodeck/lib/test_runner/executor.py
- [x] T054 [US1] Extend _create_evaluators() to handle GEvalMetric instances in src/holodeck/lib/test_runner/executor.py
- [x] T055 [P] [US1] Create unit tests for TestExecutor GEval integration in tests/unit/lib/test_runner/test_executor_deepeval.py

**Checkpoint**: User Story 1 complete - can evaluate with any LLM provider using G-Eval ✅

---

## Phase 4: User Story 2 - Custom Evaluation Criteria (Priority: P1) ✅

**Goal**: Define custom evaluation criteria in natural language

**Independent Test**: Create custom G-Eval metric with "professionalism" criteria and evaluate informal response

### Tests for User Story 2

- [x] T019 [P] [US2] Create unit tests for custom criteria with threshold validation in tests/unit/lib/evaluators/deepeval/test_geval.py

### Implementation for User Story 2

- [x] T020 [US2] Add evaluation_params support (input, actual_output, expected_output, context, retrieval_context) in src/holodeck/lib/evaluators/deepeval/geval.py
- [x] T021 [US2] Add strict_mode support for binary scoring in src/holodeck/lib/evaluators/deepeval/geval.py
- [x] T022 [US2] Add threshold configuration with pass/fail logic in src/holodeck/lib/evaluators/deepeval/geval.py (Note: inherited from base)
- [x] T023 [US2] Add auto-generation of evaluation_steps when not provided in src/holodeck/lib/evaluators/deepeval/geval.py (Note: delegated to DeepEval)

**Checkpoint**: User Story 2 complete - custom criteria evaluation working ✅

---

## Phase 5: User Story 3 - RAG Pipeline Evaluation (Priority: P2)

**Goal**: Evaluate retrieval quality and response faithfulness separately

**Independent Test**: Provide retrieval_context and run Faithfulness + ContextualRelevancy metrics

### Tests for User Story 3

- [ ] T024 [P] [US3] Create unit tests for FaithfulnessEvaluator in tests/unit/lib/evaluators/deepeval/test_faithfulness.py
- [ ] T025 [P] [US3] Create unit tests for ContextualRelevancyEvaluator in tests/unit/lib/evaluators/deepeval/test_contextual_relevancy.py
- [ ] T026 [P] [US3] Create unit tests for ContextualPrecisionEvaluator in tests/unit/lib/evaluators/deepeval/test_contextual_precision.py
- [ ] T027 [P] [US3] Create unit tests for ContextualRecallEvaluator in tests/unit/lib/evaluators/deepeval/test_contextual_recall.py

### Implementation for User Story 3

- [ ] T028 [P] [US3] Implement FaithfulnessEvaluator wrapping DeepEval's FaithfulnessMetric in src/holodeck/lib/evaluators/deepeval/faithfulness.py
- [ ] T029 [P] [US3] Implement ContextualRelevancyEvaluator in src/holodeck/lib/evaluators/deepeval/contextual_relevancy.py
- [ ] T030 [P] [US3] Implement ContextualPrecisionEvaluator in src/holodeck/lib/evaluators/deepeval/contextual_precision.py
- [ ] T031 [P] [US3] Implement ContextualRecallEvaluator in src/holodeck/lib/evaluators/deepeval/contextual_recall.py
- [ ] T032 [US3] Add retrieval_context validation (required for RAG metrics) in src/holodeck/lib/evaluators/deepeval/base.py
- [ ] T033 [US3] Update src/holodeck/lib/evaluators/deepeval/__init__.py to export all RAG evaluators

### TestExecutor Integration for User Story 3

- [ ] T056 [US3] Add retrieval_context field to TestCaseModel for RAG metrics in src/holodeck/models/test_case.py
- [ ] T057 [US3] Create RAGMetric model (discriminated union) for faithfulness, contextual_relevancy, contextual_precision, contextual_recall in src/holodeck/models/evaluation.py
- [ ] T058 [US3] Update EvaluationConfig.metrics Union to include RAGMetric in src/holodeck/models/evaluation.py
- [ ] T059 [P] [US3] Create unit tests for RAGMetric model validation in tests/unit/models/test_evaluation_deepeval.py
- [ ] T060 [US3] Add RAG evaluator imports (FaithfulnessEvaluator, ContextualRelevancyEvaluator, etc.) to src/holodeck/lib/test_runner/executor.py
- [ ] T061 [US3] Extend _create_evaluators() to handle RAGMetric instances in src/holodeck/lib/test_runner/executor.py
- [ ] T062 [US3] Update _run_evaluations() to pass retrieval_context for RAG metrics when available in src/holodeck/lib/test_runner/executor.py
- [ ] T063 [P] [US3] Create unit tests for TestExecutor RAG metrics integration in tests/unit/lib/test_runner/test_executor_rag_metrics.py

**Checkpoint**: User Story 3 complete - RAG evaluation suite working

---

## Phase 6: User Story 4 - Azure AI Provider Validation (Priority: P2)

**Goal**: Fail early with clear error when non-Azure provider configured for Azure AI evaluator

**Independent Test**: Configure AzureAIEvaluator with OpenAI provider, verify ProviderNotSupportedError raised

### Tests for User Story 4

- [ ] T034 [P] [US4] Create unit tests for Azure AI provider validation in tests/unit/lib/evaluators/test_azure_ai_validation.py

### Implementation for User Story 4

- [ ] T035 [US4] Add _validate_provider() method to AzureAIEvaluator in src/holodeck/lib/evaluators/azure_ai.py
- [ ] T036 [US4] Call _validate_provider() in AzureAIEvaluator.__init__() in src/holodeck/lib/evaluators/azure_ai.py
- [ ] T037 [US4] Import ProviderNotSupportedError from deepeval.errors in src/holodeck/lib/evaluators/azure_ai.py
- [ ] T038 [US4] Update src/holodeck/lib/evaluators/__init__.py to export ProviderNotSupportedError

**Checkpoint**: User Story 4 complete - Azure AI evaluator fails early on misconfiguration

---

## Phase 7: User Story 5 - Answer Relevancy Evaluation (Priority: P2)

**Goal**: Evaluate response relevance to user queries

**Independent Test**: Provide query-response pair, verify relevancy score returned

### Tests for User Story 5

- [ ] T039 [P] [US5] Create unit tests for AnswerRelevancyEvaluator in tests/unit/lib/evaluators/deepeval/test_answer_relevancy.py

### Implementation for User Story 5

- [ ] T040 [US5] Implement AnswerRelevancyEvaluator wrapping DeepEval's AnswerRelevancyMetric in src/holodeck/lib/evaluators/deepeval/answer_relevancy.py
- [ ] T041 [US5] Add include_reason parameter support in src/holodeck/lib/evaluators/deepeval/answer_relevancy.py
- [ ] T042 [US5] Update src/holodeck/lib/evaluators/deepeval/__init__.py to export AnswerRelevancyEvaluator

### TestExecutor Integration for User Story 5

- [ ] T064 [US5] Create AnswerRelevancyMetric model (discriminated union) with include_reason field in src/holodeck/models/evaluation.py
- [ ] T065 [US5] Update EvaluationConfig.metrics Union to include AnswerRelevancyMetric in src/holodeck/models/evaluation.py
- [ ] T066 [P] [US5] Create unit tests for AnswerRelevancyMetric model validation in tests/unit/models/test_evaluation_deepeval.py
- [ ] T067 [US5] Add AnswerRelevancyEvaluator import to src/holodeck/lib/test_runner/executor.py
- [ ] T068 [US5] Extend _create_evaluators() to handle AnswerRelevancyMetric instances in src/holodeck/lib/test_runner/executor.py
- [ ] T069 [P] [US5] Create unit tests for TestExecutor AnswerRelevancy integration in tests/unit/lib/test_runner/test_executor_answer_relevancy.py

**Checkpoint**: User Story 5 complete - answer relevancy evaluation working

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T043 [P] Run make format && make lint-fix to ensure code quality
- [ ] T044 [P] Run make type-check to verify MyPy strict mode compliance
- [ ] T045 [P] Run make test to verify all unit tests pass
- [ ] T046 Create integration test with real Ollama in tests/integration/lib/evaluators/test_deepeval_integration.py
- [ ] T047 [P] Run make security to verify Bandit/Safety compliance
- [ ] T048 Validate quickstart.md examples work with implemented code

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user stories
- **User Stories (Phases 3-7)**: All depend on Foundational (Phase 2) completion
  - US1 + US2 can proceed in parallel (P1 priority, both use GEval)
  - US3, US4, US5 can proceed in parallel after US1 (different evaluators)
- **Polish (Phase 8)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational - No dependencies on other stories
- **User Story 2 (P1)**: Can start after US1 (extends GEvalEvaluator)
- **User Story 3 (P2)**: Can start after Foundational - Independent RAG evaluators
- **User Story 4 (P2)**: Can start after Foundational - Modifies existing Azure AI code
- **User Story 5 (P2)**: Can start after Foundational - Independent evaluator

### Within Each User Story

- Tests SHOULD be written first (TDD approach)
- Base evaluator methods before specific evaluator classes
- Core implementation before exports/integration
- Story complete before Polish phase

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- T010, T011 (Foundational tests) can run in parallel
- All RAG evaluator tests (T024-T027) can run in parallel
- All RAG evaluator implementations (T028-T031) can run in parallel
- All Polish tasks marked [P] can run in parallel

---

## Parallel Example: User Story 3 (RAG Evaluators)

```bash
# Launch all RAG metric tests in parallel:
T024: "Create unit tests for FaithfulnessEvaluator in tests/unit/lib/evaluators/deepeval/test_faithfulness.py"
T025: "Create unit tests for ContextualRelevancyEvaluator in tests/unit/lib/evaluators/deepeval/test_contextual_relevancy.py"
T026: "Create unit tests for ContextualPrecisionEvaluator in tests/unit/lib/evaluators/deepeval/test_contextual_precision.py"
T027: "Create unit tests for ContextualRecallEvaluator in tests/unit/lib/evaluators/deepeval/test_contextual_recall.py"

# Launch all RAG metric implementations in parallel:
T028: "Implement FaithfulnessEvaluator in src/holodeck/lib/evaluators/deepeval/faithfulness.py"
T029: "Implement ContextualRelevancyEvaluator in src/holodeck/lib/evaluators/deepeval/contextual_relevancy.py"
T030: "Implement ContextualPrecisionEvaluator in src/holodeck/lib/evaluators/deepeval/contextual_precision.py"
T031: "Implement ContextualRecallEvaluator in src/holodeck/lib/evaluators/deepeval/contextual_recall.py"
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 2 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Multi-Provider + G-Eval)
4. Complete Phase 4: User Story 2 (Custom Criteria)
5. **STOP and VALIDATE**: Test G-Eval with custom criteria independently
6. Deploy/demo if ready - this is a usable MVP

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 + 2 → Test G-Eval independently → **MVP!**
3. Add User Story 3 → Test RAG metrics independently
4. Add User Story 4 → Test Azure AI validation independently
5. Add User Story 5 → Test AnswerRelevancy independently
6. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 + 2 (G-Eval)
   - Developer B: User Story 3 (RAG metrics)
   - Developer C: User Story 4 (Azure validation) + User Story 5 (Answer Relevancy)
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Uses DeepEval's native model classes (GPTModel, AzureOpenAIModel, AnthropicModel, OllamaModel)
- Default model is Ollama with gpt-oss:20b per clarification session
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
