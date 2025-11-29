# Implementation Plan: DeepEval Metrics Integration

**Branch**: `012-deepeval-metrics` | **Date**: 2025-01-30 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/012-deepeval-metrics/spec.md`

## Summary

Add DeepEval as a new evaluation framework supporting multiple LLM providers (OpenAI, Azure OpenAI, Anthropic, Ollama) as judge models. Implements G-Eval for custom criteria evaluation and RAG metrics (AnswerRelevancy, Faithfulness, ContextualRelevancy, ContextualPrecision, ContextualRecall). Also adds provider validation to existing Azure AI evaluators to fail early when non-Azure providers are configured.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: deepeval>=0.21.0, pydantic>=2.0.0, ollama>=0.4
**Storage**: N/A (evaluation metrics are stateless)
**Testing**: pytest with pytest-asyncio for async evaluator tests
**Target Platform**: Linux/macOS/Windows (CLI and library)
**Project Type**: Single Python package (existing HoloDeck structure)
**Performance Goals**: <30 seconds per evaluation (per SC-001)
**Constraints**: Must integrate with existing BaseEvaluator retry/timeout infrastructure
**Scale/Scope**: 6 new evaluator classes + 1 config adapter + 1 error type

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. No-Code-First Agent Definition | ✅ PASS | Evaluators configured via YAML agent config, not code |
| II. MCP for API Integrations | ✅ N/A | DeepEval is an evaluation library, not an API integration |
| III. Test-First with Multimodal Support | ✅ PASS | Feature adds evaluation metrics for test validation |
| IV. OpenTelemetry-Native Observability | ✅ PASS | FR-013 requires logging via HoloDeck logger |
| V. Evaluation Flexibility with Model Overrides | ✅ PASS | FR-011 supports per-metric model config; default Ollama gpt-oss:20b |

**Architecture Constraints**:
- ✅ Feature adds to Evaluation Framework engine only
- ✅ No cross-engine coupling introduced

**Code Quality**:
- ✅ Python 3.10+ target
- ✅ Will use Google Style Guide, Black, Ruff
- ✅ MyPy strict mode required
- ✅ pytest with unit/integration markers
- ✅ 80%+ coverage target

## Project Structure

### Documentation (this feature)

```
specs/012-deepeval-metrics/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (internal Python interfaces)
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```
src/holodeck/
├── lib/
│   └── evaluators/
│       ├── __init__.py          # Update exports
│       ├── base.py              # Existing BaseEvaluator (unchanged)
│       ├── azure_ai.py          # Add ProviderNotSupportedError + validation
│       ├── nlp_metrics.py       # Existing NLP metrics (unchanged)
│       └── deepeval/            # NEW: DeepEval evaluators module
│           ├── __init__.py      # Module exports
│           ├── config.py        # DeepEvalModelConfig adapter
│           ├── base.py          # DeepEvalBaseEvaluator
│           ├── geval.py         # GEvalEvaluator
│           ├── answer_relevancy.py
│           ├── faithfulness.py
│           ├── contextual_relevancy.py
│           ├── contextual_precision.py
│           └── contextual_recall.py
└── models/
    └── llm.py                   # Existing LLMProvider (may need extension)

tests/
├── unit/
│   └── lib/
│       └── evaluators/
│           ├── test_azure_ai_validation.py  # NEW: Provider validation tests
│           └── deepeval/                     # NEW: DeepEval unit tests
│               ├── test_config.py
│               ├── test_geval.py
│               ├── test_answer_relevancy.py
│               ├── test_faithfulness.py
│               ├── test_contextual_relevancy.py
│               ├── test_contextual_precision.py
│               └── test_contextual_recall.py
└── integration/
    └── lib/
        └── evaluators/
            └── test_deepeval_integration.py  # NEW: End-to-end with real LLM
```

**Structure Decision**: Extends existing single-project structure. DeepEval evaluators placed in new submodule `lib/evaluators/deepeval/` to maintain separation from Azure AI evaluators while sharing the BaseEvaluator infrastructure.

## Complexity Tracking

*No Constitution violations - table empty*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| (none)    | -          | -                                   |

---

## Post-Design Constitution Re-Check

*Verified after Phase 1 design artifacts complete.*

| Principle | Status | Design Verification |
|-----------|--------|---------------------|
| I. No-Code-First | ✅ PASS | YAML config in quickstart.md; no Python required for users |
| II. MCP for APIs | ✅ N/A | DeepEval is evaluation library, not API integration |
| III. Test-First | ✅ PASS | Evaluators enable test case validation; contracts define test interfaces |
| IV. OTel Observability | ✅ PASS | FR-013 logging via HoloDeck logger; integrates with existing infrastructure |
| V. Eval Flexibility | ✅ PASS | Per-metric model override in DeepEvalModelConfig; default Ollama gpt-oss:20b |

**Architecture**: Feature adds to Evaluation Framework only. No cross-engine coupling.
