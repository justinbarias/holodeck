# Research: DeepEval Metrics Integration

**Feature**: 012-deepeval-metrics
**Date**: 2025-01-30
**Status**: Complete

## Executive Summary

This research validates the technical approach for integrating DeepEval as an alternative evaluation framework to Azure AI Evaluation. DeepEval provides multi-provider LLM-as-judge capabilities with G-Eval for custom criteria and comprehensive RAG metrics.

---

## 1. DeepEval Library Architecture

### Decision: Use DeepEval v0.21+ with native provider integrations

**Rationale**:
- DeepEval is the leading open-source LLM evaluation framework (50+ metrics)
- Native support for OpenAI, Azure OpenAI, Anthropic, Ollama, and custom LLMs
- G-Eval implementation follows the original research paper with chain-of-thought scoring
- Active maintenance and community support

**Alternatives Considered**:
| Alternative | Why Rejected |
|-------------|--------------|
| RAGAS | Limited to RAG metrics only; no custom criteria support |
| LangSmith Evals | Requires LangChain dependency; less flexible provider support |
| Custom Implementation | High development cost; reinventing well-tested algorithms |
| TruLens | More complex setup; heavier dependency footprint |

**Sources**:
- [DeepEval GitHub](https://github.com/confident-ai/deepeval)
- [DeepEval Documentation](https://deepeval.com/docs/getting-started)

---

## 2. G-Eval Algorithm

### Decision: Implement GEvalEvaluator wrapping DeepEval's GEval metric

**Rationale**:
G-Eval is a two-step algorithm from "NLG Evaluation using GPT-4 with Better Human Alignment":

1. **Step Generation Phase**: Auto-generates evaluation steps from criteria using chain-of-thought
2. **Scoring Phase**: Uses generated steps + test case parameters to produce 1-5 score, normalized to 0-1

Key parameters:
- `name`: Metric identifier
- `criteria`: Natural language evaluation criteria
- `evaluation_steps`: Optional override for auto-generated steps
- `evaluation_params`: Which LLMTestCase fields to include (INPUT, ACTUAL_OUTPUT, EXPECTED_OUTPUT, CONTEXT, RETRIEVAL_CONTEXT)
- `threshold`: Pass/fail cutoff (default 0.5)
- `model`: LLM judge model

**Implementation Pattern**:
```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

metric = GEval(
    name="Correctness",
    criteria="Evaluate if actual_output matches expected_output factually",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=0.7
)
```

**Alternatives Considered**:
| Alternative | Why Rejected |
|-------------|--------------|
| DAGMetric | Deterministic but requires decision tree definition |
| Direct LLM prompting | Lacks token probability weighting for accurate scoring |

**Sources**:
- [G-Eval Documentation](https://deepeval.com/docs/metrics-llm-evals)
- [G-Eval Research Blog](https://www.confident-ai.com/blog/g-eval-the-definitive-guide)

---

## 3. RAG Metrics Implementation

### Decision: Implement 5 RAG evaluators using DeepEval's built-in metrics

**Rationale**:
DeepEval provides research-backed implementations of standard RAG metrics:

| Metric | Purpose | Required Inputs |
|--------|---------|-----------------|
| AnswerRelevancyMetric | Response relevance to query | input, actual_output |
| FaithfulnessMetric | Hallucination detection | input, actual_output, retrieval_context |
| ContextualRelevancyMetric | Retrieval relevance | input, actual_output, retrieval_context |
| ContextualPrecisionMetric | Retrieval ranking quality | input, actual_output, expected_output, retrieval_context |
| ContextualRecallMetric | Retrieval completeness | input, actual_output, expected_output, retrieval_context |

**Implementation Pattern**:
```python
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

metric = AnswerRelevancyMetric(threshold=0.7, model=custom_model)
test_case = LLMTestCase(
    input="What is the return policy?",
    actual_output="We offer 30-day returns."
)
metric.measure(test_case)
# Access: metric.score, metric.reason, metric.is_successful()
```

**Sources**:
- [RAG Evaluation Metrics](https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more)
- [DeepEval Metrics Introduction](https://deepeval.com/docs/metrics-introduction)

---

## 4. Multi-Provider LLM Integration

### Decision: Use DeepEval's native model classes for all 4 providers

**Rationale**:
DeepEval provides **native model classes** for all supported providers - no custom `DeepEvalBaseLLM` implementation required:

1. **OpenAI**: `GPTModel` class from `deepeval.models`
2. **Azure OpenAI**: `AzureOpenAIModel` class from `deepeval.models`
3. **Anthropic**: `AnthropicModel` class from `deepeval.models`
4. **Ollama**: `OllamaModel` class from `deepeval.models`

**Provider Configuration Mapping**:

| HoloDeck Provider | DeepEval Native Class | Key Parameters |
|-------------------|----------------------|----------------|
| `openai` | `GPTModel` | `model`, `temperature`, `cost_per_input_token`, `cost_per_output_token` |
| `azure_openai` | `AzureOpenAIModel` | `model_name`, `deployment_name`, `azure_endpoint`, `azure_openai_api_key`, `openai_api_version` |
| `anthropic` | `AnthropicModel` | `model`, `temperature` |
| `ollama` | `OllamaModel` | `model`, `base_url`, `temperature` |

**Implementation Patterns**:

```python
# OpenAI
from deepeval.models import GPTModel
model = GPTModel(model="gpt-4o", temperature=0)

# Azure OpenAI
from deepeval.models import AzureOpenAIModel
model = AzureOpenAIModel(
    model_name="gpt-4o",
    deployment_name="my-deployment",
    azure_endpoint="https://my-resource.openai.azure.com/",
    azure_openai_api_key="...",
    openai_api_version="2024-02-15-preview",
    temperature=0
)

# Anthropic
from deepeval.models import AnthropicModel
model = AnthropicModel(model="claude-3-5-sonnet-latest", temperature=0)

# Ollama (Default)
from deepeval.models import OllamaModel
model = OllamaModel(
    model="gpt-oss:20b",
    base_url="http://localhost:11434",
    temperature=0
)
```

**Supported Models by Provider**:

| Provider | Example Models |
|----------|---------------|
| OpenAI | gpt-4o, gpt-4.1, gpt-3.5-turbo, o1 |
| Azure OpenAI | Same as OpenAI (deployment-dependent) |
| Anthropic | claude-3-5-sonnet-latest, claude-3-opus-latest, claude-3-haiku-20240307 |
| Ollama | deepseek-r1, llama3.1, mistral, qwen, phi3, gpt-oss:20b |

**Alternatives Considered**:
| Alternative | Why Rejected |
|-------------|--------------|
| Custom DeepEvalBaseLLM subclasses | Unnecessary - native classes available for all providers |
| LiteLLM wrapper | Additional dependency; native integration is simpler |
| Environment-only config | Doesn't support per-metric model selection |

**Sources**:
- [DeepEval OpenAI Integration](https://deepeval.com/integrations/models/openai)
- [DeepEval Azure OpenAI Integration](https://deepeval.com/integrations/models/azure-openai)
- [DeepEval Anthropic Integration](https://deepeval.com/integrations/models/anthropic)
- [DeepEval Ollama Integration](https://deepeval.com/integrations/models/ollama)

---

## 5. Structured Output Handling

### Decision: Rely on DeepEval's native model classes for structured output

**Rationale**:
DeepEval's native model classes (`GPTModel`, `AzureOpenAIModel`, `AnthropicModel`, `OllamaModel`) handle structured JSON output internally. No custom implementation required.

- **OpenAI/Azure**: Native `response_format={"type": "json_object"}` handled by DeepEval
- **Anthropic**: DeepEval's `AnthropicModel` handles JSON formatting internally
- **Ollama**: DeepEval's `OllamaModel` handles JSON formatting internally

**Error Handling**:
If structured output fails, DeepEval raises: "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."

Our implementation will:
1. Retry with configured retry policy (via BaseEvaluator)
2. Log the raw response for debugging
3. Raise `EvaluationError` with context

**Sources**:
- [DeepEval Native Model Classes](https://deepeval.com/integrations/models/openai)

---

## 6. BaseEvaluator Integration

### Decision: DeepEval evaluators inherit from existing BaseEvaluator

**Rationale**:
The existing `BaseEvaluator` in `holodeck/lib/evaluators/base.py` provides:
- Retry logic with exponential backoff (`RetryConfig`)
- Timeout handling via `asyncio.wait_for`
- Common `evaluate()` interface
- Logging infrastructure

DeepEval metrics are synchronous, so the wrapper will:
1. Call DeepEval's synchronous `metric.measure(test_case)` in `_evaluate_impl`
2. Let BaseEvaluator handle retry/timeout
3. Normalize output to standard format: `{score, passed, reasoning, ...metric_specific}`

**Implementation Pattern**:
```python
from deepeval.models import OllamaModel, GPTModel, AzureOpenAIModel, AnthropicModel

class DeepEvalBaseEvaluator(BaseEvaluator):
    def __init__(self, model_config: DeepEvalModelConfig | None = None, ...):
        super().__init__(timeout=timeout, retry_config=retry_config)
        self.model = self._create_model(model_config)

    def _create_model(self, config: DeepEvalModelConfig | None):
        if config is None:
            # Default to Ollama with gpt-oss:20b
            return OllamaModel(model="gpt-oss:20b", temperature=0)

        # Use DeepEval's native model classes
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

    async def _evaluate_impl(self, **kwargs) -> dict[str, Any]:
        test_case = self._build_test_case(**kwargs)
        metric = self._create_metric()
        metric.measure(test_case)  # Synchronous DeepEval call
        return {
            "score": metric.score,
            "passed": metric.is_successful(),
            "reasoning": metric.reason,
        }
```

---

## 7. Azure AI Provider Validation

### Decision: Add `ProviderNotSupportedError` with early validation in `__init__`

**Rationale**:
Current Azure AI evaluators accept any `ModelConfig` but fail at runtime with cryptic Azure SDK errors when non-Azure providers are used.

**Implementation**:
```python
class ProviderNotSupportedError(EvaluationError):
    """Raised when an evaluator is used with an incompatible provider."""
    pass

class AzureAIEvaluator(BaseEvaluator):
    def __init__(self, model_config: ModelConfig, ...):
        self._validate_provider(model_config)
        super().__init__(...)

    def _validate_provider(self, config: ModelConfig) -> None:
        # ModelConfig is Azure-specific, so check at initialization
        # The current ModelConfig assumes Azure - this validation
        # catches cases where users might try to use it incorrectly
        if not config.azure_endpoint or not config.azure_deployment:
            raise ProviderNotSupportedError(
                "Azure AI Evaluator requires Azure OpenAI provider. "
                "Missing azure_endpoint or azure_deployment."
            )
```

---

## 8. Default Model Configuration

### Decision: Default to Ollama with gpt-oss:20b (per clarification session)

**Rationale**:
- User explicitly requested Ollama as default
- `gpt-oss:20b` provides good balance of quality and local execution
- No API key required for local evaluation
- Reduces cost for iterative testing

**Fallback Behavior**:
If Ollama is not running, the evaluator will:
1. Raise `ConnectionError` on first call
2. Retry per `RetryConfig`
3. Fail with clear message: "Ollama not available. Start Ollama or configure alternative provider."

---

## 9. Dependency Addition

### Decision: Add `deepeval>=0.21.0` to pyproject.toml dependencies

**Rationale**:
- v0.21+ includes latest G-Eval and RAG metric implementations
- Ollama integration stabilized in recent versions
- `instructor` is a transitive dependency for Anthropic support

**pyproject.toml Addition**:
```toml
dependencies = [
    # ... existing
    "deepeval>=0.21.0,<1.0.0",
]
```

---

## Open Questions (Resolved)

| Question | Resolution |
|----------|------------|
| Default model when none specified | Ollama gpt-oss:20b (per clarification) |
| Logging verbosity | Standard: scores, reasoning, retries (per clarification) |
| Async support | Sync DeepEval calls wrapped in async BaseEvaluator |

---

## Next Steps

1. **Phase 1**: Create data-model.md with entity definitions
2. **Phase 1**: Generate Python interface contracts
3. **Phase 1**: Write quickstart.md for users
4. **Phase 2**: Generate tasks.md from spec requirements
