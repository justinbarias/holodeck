# Data Model: DeepEval Metrics Integration

**Feature**: 012-deepeval-metrics
**Date**: 2025-01-30

## Overview

This document defines the data entities, configuration models, and result structures for the DeepEval metrics integration. All entities use Pydantic for validation and follow existing HoloDeck patterns.

---

## 1. Configuration Entities

### 1.1 DeepEvalModelConfig

Adapter that bridges HoloDeck's `LLMProvider` model to DeepEval's native model classes.

**DeepEval provides native model classes** - no custom implementation required:
- `GPTModel` for OpenAI
- `AzureOpenAIModel` for Azure OpenAI
- `AnthropicModel` for Anthropic
- `OllamaModel` for Ollama

```
DeepEvalModelConfig
├── provider: ProviderEnum           # openai | azure_openai | anthropic | ollama
├── model_name: str                  # e.g., "gpt-4o", "claude-3-5-sonnet-latest", "gpt-oss:20b"
├── api_key: str | None              # Required for cloud providers, None for Ollama
├── endpoint: str | None             # Required for Azure OpenAI; optional for Ollama (base_url)
├── api_version: str | None          # Azure OpenAI only (default: "2024-02-15-preview")
├── deployment_name: str | None      # Azure OpenAI only
└── temperature: float               # Default: 0.0 (deterministic for evaluation)
```

**Maps to DeepEval Native Classes**:

| Provider | DeepEval Class | Import |
|----------|---------------|--------|
| openai | `GPTModel` | `from deepeval.models import GPTModel` |
| azure_openai | `AzureOpenAIModel` | `from deepeval.models import AzureOpenAIModel` |
| anthropic | `AnthropicModel` | `from deepeval.models import AnthropicModel` |
| ollama | `OllamaModel` | `from deepeval.models import OllamaModel` |

**Validation Rules**:
- If `provider == azure_openai`: `endpoint`, `deployment_name`, `api_key` required
- If `provider == openai`: `api_key` required (or `OPENAI_API_KEY` env var)
- If `provider == anthropic`: `api_key` required (or `ANTHROPIC_API_KEY` env var)
- If `provider == ollama`: `endpoint` optional (defaults to `http://localhost:11434`)

**Default Instance**:
```python
DeepEvalModelConfig(
    provider=ProviderEnum.OLLAMA,
    model_name="gpt-oss:20b",
    endpoint="http://localhost:11434",
    temperature=0.0
)
# Creates: OllamaModel(model="gpt-oss:20b", base_url="http://localhost:11434", temperature=0)
```

**Conversion to DeepEval Model**:
```python
def to_deepeval_model(config: DeepEvalModelConfig):
    if config.provider == ProviderEnum.OPENAI:
        return GPTModel(model=config.model_name, temperature=config.temperature)
    elif config.provider == ProviderEnum.AZURE_OPENAI:
        return AzureOpenAIModel(
            model_name=config.model_name,
            deployment_name=config.deployment_name,
            azure_endpoint=config.endpoint,
            azure_openai_api_key=config.api_key,
            openai_api_version=config.api_version,
            temperature=config.temperature
        )
    elif config.provider == ProviderEnum.ANTHROPIC:
        return AnthropicModel(model=config.model_name, temperature=config.temperature)
    elif config.provider == ProviderEnum.OLLAMA:
        return OllamaModel(
            model=config.model_name,
            base_url=config.endpoint or "http://localhost:11434",
            temperature=config.temperature
        )
```

### 1.2 GEvalConfig

Configuration specific to G-Eval custom criteria evaluator.

```
GEvalConfig
├── name: str                        # Metric identifier (e.g., "Professionalism")
├── criteria: str                    # Natural language evaluation criteria
├── evaluation_steps: list[str] | None  # Optional explicit steps (auto-generated if None)
├── evaluation_params: list[str]     # Test case fields to include
│   └── Valid values: "input", "actual_output", "expected_output",
│                     "context", "retrieval_context"
├── threshold: float                 # Pass/fail cutoff (default: 0.5)
└── strict_mode: bool                # Binary scoring (default: False)
```

**Validation Rules**:
- `criteria` must be non-empty string
- `evaluation_params` must contain at least `["actual_output"]`
- `threshold` must be in range [0.0, 1.0]

### 1.3 RAGMetricConfig

Configuration for RAG evaluation metrics (shared base).

```
RAGMetricConfig
├── threshold: float                 # Pass/fail cutoff (default: 0.5)
├── include_reason: bool             # Include reasoning in output (default: True)
└── model_config: DeepEvalModelConfig | None  # Override default model
```

---

## 2. Input Entities

### 2.1 EvaluationInput

Unified input structure for all DeepEval evaluators. Maps to DeepEval's `LLMTestCase`.

```
EvaluationInput
├── input: str                       # User query/prompt (alias: "query")
├── actual_output: str               # Agent's response (alias: "response")
├── expected_output: str | None      # Ground truth (alias: "ground_truth")
├── context: str | None              # General context
├── retrieval_context: list[str] | None  # Retrieved chunks for RAG
└── metadata: dict[str, Any] | None  # Additional context (ignored by metrics)
```

**Field Aliases** (for backward compatibility with existing evaluators):
- `query` → `input`
- `response` → `actual_output`
- `ground_truth` → `expected_output`

**Validation Rules**:
- `input` required for all metrics except FaithfulnessMetric
- `actual_output` always required
- `retrieval_context` required for: Faithfulness, ContextualRelevancy, ContextualPrecision, ContextualRecall
- `expected_output` required for: ContextualPrecision, ContextualRecall

---

## 3. Output Entities

### 3.1 EvaluationResult

Standard output structure returned by all evaluators.

```
EvaluationResult
├── score: float                     # Normalized score (0.0 - 1.0)
├── passed: bool                     # score >= threshold
├── reasoning: str                   # LLM-generated explanation
├── metric_name: str                 # e.g., "GEval", "AnswerRelevancy"
├── threshold: float                 # Configured threshold
├── raw_score: float | None          # Original score before normalization (if applicable)
├── evaluation_steps: list[str] | None  # Steps used (G-Eval only)
└── metadata: dict[str, Any]         # Metric-specific additional data
```

**Invariants**:
- `score` always in range [0.0, 1.0]
- `passed == (score >= threshold)`
- `reasoning` never empty (LLM always provides explanation)

### 3.2 MetricSpecificMetadata

Additional data included in `EvaluationResult.metadata` by metric type.

**G-Eval**:
```
{
    "criteria": str,
    "auto_generated_steps": bool,
    "strict_mode": bool
}
```

**Faithfulness**:
```
{
    "claims_count": int,
    "supported_claims": int,
    "unsupported_claims": list[str]
}
```

**ContextualRelevancy**:
```
{
    "relevant_chunks": int,
    "total_chunks": int,
    "irrelevant_chunk_indices": list[int]
}
```

**ContextualPrecision**:
```
{
    "precision_at_k": dict[int, float]  # e.g., {1: 1.0, 3: 0.67, 5: 0.6}
}
```

**ContextualRecall**:
```
{
    "expected_facts": int,
    "retrieved_facts": int,
    "missing_facts": list[str]
}
```

---

## 4. Error Entities

### 4.1 ProviderNotSupportedError

Raised when an evaluator is used with an incompatible LLM provider.

```
ProviderNotSupportedError(EvaluationError)
├── message: str                     # Human-readable error
├── evaluator_type: str              # e.g., "AzureAIEvaluator"
├── configured_provider: str         # e.g., "openai"
└── supported_providers: list[str]   # e.g., ["azure_openai"]
```

### 4.2 DeepEvalError

Wraps errors from DeepEval library with additional context.

```
DeepEvalError(EvaluationError)
├── message: str
├── original_error: Exception | None
├── metric_name: str
└── test_case_summary: dict[str, str]  # Truncated input/output for debugging
```

---

## 5. Entity Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                        Configuration                             │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │DeepEvalModelConfig│◄──┤   GEvalConfig   │                    │
│  └────────┬────────┘    └─────────────────┘                     │
│           │                                                      │
│           │             ┌─────────────────┐                     │
│           └─────────────┤  RAGMetricConfig │                    │
│                         └─────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Evaluators                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              DeepEvalBaseEvaluator                       │   │
│  │  (inherits from BaseEvaluator)                           │   │
│  └───────────────────────┬─────────────────────────────────┘   │
│                          │                                       │
│    ┌────────────┬────────┼────────┬────────────┬────────────┐  │
│    ▼            ▼        ▼        ▼            ▼            ▼   │
│ GEval    AnswerRel  Faithful  CtxRel    CtxPrec    CtxRecall   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Input / Output                              │
│  ┌─────────────────┐              ┌─────────────────┐           │
│  │ EvaluationInput │──evaluate()──►│ EvaluationResult│          │
│  └─────────────────┘              └─────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. State Transitions

Evaluators are stateless. Each `evaluate()` call is independent.

**LLM Model Lifecycle** (per evaluator instance):
```
UNINITIALIZED ──__init__()──► CONFIGURED ──evaluate()──► CONFIGURED
                                  │                          │
                                  └──────────────────────────┘
```

**Evaluation Call Lifecycle**:
```
START ──build_test_case()──► TEST_CASE_READY ──measure()──► SCORING
  │                                                            │
  │                                                            ▼
  │                                                    RESULT_AVAILABLE
  │                                                            │
  └────────────────on_error()──────────────────────────────────┘
                         │
                         ▼
                 RETRY (up to max_retries) or FAILED
```

---

## 7. Validation Summary

| Entity | Field | Rule |
|--------|-------|------|
| DeepEvalModelConfig | provider | Must be valid ProviderEnum |
| DeepEvalModelConfig | api_key | Required for openai, azure_openai, anthropic |
| DeepEvalModelConfig | endpoint | Required for azure_openai |
| GEvalConfig | criteria | Non-empty string |
| GEvalConfig | threshold | 0.0 ≤ value ≤ 1.0 |
| GEvalConfig | evaluation_params | Contains at least "actual_output" |
| EvaluationInput | actual_output | Always required |
| EvaluationInput | retrieval_context | Required for RAG metrics |
| EvaluationResult | score | 0.0 ≤ value ≤ 1.0 |
| EvaluationResult | passed | Equals (score >= threshold) |
