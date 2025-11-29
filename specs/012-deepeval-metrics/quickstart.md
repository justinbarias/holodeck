# Quickstart: DeepEval Metrics

**Feature**: 012-deepeval-metrics
**Date**: 2025-01-30

## Overview

DeepEval metrics provide LLM-as-a-judge evaluation capabilities supporting multiple providers (OpenAI, Azure OpenAI, Anthropic, Ollama). This guide covers basic setup and common use cases.

---

## Prerequisites

1. **Ollama installed** (for default model):
   ```bash
   # macOS
   brew install ollama

   # Linux
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Pull the default model**:
   ```bash
   ollama pull gpt-oss:20b
   ```

3. **Start Ollama** (if not running):
   ```bash
   ollama serve
   ```

---

## Basic Usage

### 1. G-Eval: Custom Criteria Evaluation

Evaluate agent responses against any criteria you define:

```python
from holodeck.lib.evaluators.deepeval import GEvalEvaluator

# Create evaluator with custom criteria
evaluator = GEvalEvaluator(
    name="Professionalism",
    criteria="Evaluate if the response uses professional, business-appropriate language",
    threshold=0.7
)

# Run evaluation
result = await evaluator.evaluate(
    input="Write me an email to my boss",
    actual_output="Hey boss, gonna be late tmrw, peace out"
)

print(f"Score: {result['score']:.2f}")
print(f"Passed: {result['passed']}")
print(f"Reasoning: {result['reasoning']}")
```

### 2. Answer Relevancy

Check if responses directly address the user's question:

```python
from holodeck.lib.evaluators.deepeval import AnswerRelevancyEvaluator

evaluator = AnswerRelevancyEvaluator(threshold=0.7)

result = await evaluator.evaluate(
    input="What is your return policy?",
    actual_output="We offer 30-day returns with full refund for unused items."
)
# High score - response directly addresses the question
```

### 3. Faithfulness (Hallucination Detection)

Verify responses are grounded in retrieved context:

```python
from holodeck.lib.evaluators.deepeval import FaithfulnessEvaluator

evaluator = FaithfulnessEvaluator(threshold=0.8)

result = await evaluator.evaluate(
    input="What are the store hours?",
    actual_output="The store is open Monday through Friday, 9am to 5pm.",
    retrieval_context=[
        "Store hours: Monday-Friday 9:00 AM - 5:00 PM",
        "Weekend hours: Closed"
    ]
)
# High score - response matches context
```

---

## YAML Configuration

Configure evaluations in your agent YAML:

```yaml
# agent.yaml
evaluations:
  # Default model for all metrics (optional - defaults to Ollama gpt-oss:20b)
  model:
    provider: ollama
    name: gpt-oss:20b

  metrics:
    # Custom G-Eval metric
    - type: geval
      name: Helpfulness
      criteria: "Evaluate if the response provides actionable, useful information"
      threshold: 0.7

    # Built-in RAG metrics
    - type: answer_relevancy
      threshold: 0.6

    - type: faithfulness
      threshold: 0.8

    - type: contextual_relevancy
      threshold: 0.6
```

---

## Using Different Providers

DeepEval provides **native model classes** for all supported providers - no custom implementation required:

- `GPTModel` for OpenAI
- `AzureOpenAIModel` for Azure OpenAI
- `AnthropicModel` for Anthropic
- `OllamaModel` for Ollama

### OpenAI

```python
from holodeck.lib.evaluators.deepeval import (
    DeepEvalModelConfig,
    GEvalEvaluator,
    ProviderEnum
)

config = DeepEvalModelConfig(
    provider=ProviderEnum.OPENAI,
    model_name="gpt-4o",
    api_key="sk-..."  # Or set OPENAI_API_KEY env var
)
# Internally creates: GPTModel(model="gpt-4o", temperature=0)

evaluator = GEvalEvaluator(
    name="Quality",
    criteria="Evaluate response quality",
    model_config=config
)
```

### Azure OpenAI

```python
config = DeepEvalModelConfig(
    provider=ProviderEnum.AZURE_OPENAI,
    model_name="gpt-4o",
    deployment_name="my-gpt4-deployment",
    endpoint="https://my-resource.openai.azure.com/",
    api_key="...",
    api_version="2024-02-15-preview"
)
# Internally creates: AzureOpenAIModel(model_name="gpt-4o", deployment_name="my-gpt4-deployment", ...)
```

### Anthropic Claude

```python
config = DeepEvalModelConfig(
    provider=ProviderEnum.ANTHROPIC,
    model_name="claude-3-5-sonnet-latest",  # or claude-3-opus-latest
    api_key="sk-ant-..."  # Or set ANTHROPIC_API_KEY env var
)
# Internally creates: AnthropicModel(model="claude-3-5-sonnet-latest", temperature=0)
```

### Ollama (Default)

```python
config = DeepEvalModelConfig(
    provider=ProviderEnum.OLLAMA,
    model_name="gpt-oss:20b",
    endpoint="http://localhost:11434"  # Optional, this is the default
)
# Internally creates: OllamaModel(model="gpt-oss:20b", base_url="http://localhost:11434", temperature=0)
```

**Note**: All providers use DeepEval's native model classes internally. The `DeepEvalModelConfig` is just an adapter that simplifies configuration.

---

## RAG Evaluation Suite

For RAG pipelines, use the full suite of retrieval metrics:

```python
from holodeck.lib.evaluators.deepeval import (
    AnswerRelevancyEvaluator,
    FaithfulnessEvaluator,
    ContextualRelevancyEvaluator,
    ContextualPrecisionEvaluator,
    ContextualRecallEvaluator,
)

# Common test data
test_data = {
    "input": "What features does the Pro plan include?",
    "actual_output": "The Pro plan includes unlimited users and priority support.",
    "expected_output": "Pro plan: unlimited users, priority support, API access",
    "retrieval_context": [
        "Pro Plan Features: Unlimited users, Priority support",
        "Enterprise Plan: Custom pricing",
        "API access available on Pro and Enterprise plans"
    ]
}

# Run all RAG metrics
evaluators = [
    AnswerRelevancyEvaluator(threshold=0.7),
    FaithfulnessEvaluator(threshold=0.8),
    ContextualRelevancyEvaluator(threshold=0.6),
    ContextualPrecisionEvaluator(threshold=0.6),
    ContextualRecallEvaluator(threshold=0.7),
]

for evaluator in evaluators:
    result = await evaluator.evaluate(**test_data)
    print(f"{evaluator.name}: {result['score']:.2f} ({'PASS' if result['passed'] else 'FAIL'})")
```

---

## Error Handling

```python
from holodeck.lib.evaluators.base import EvaluationError
from holodeck.lib.evaluators.deepeval import DeepEvalError

try:
    result = await evaluator.evaluate(
        input="test query",
        actual_output="test response"
    )
except DeepEvalError as e:
    print(f"DeepEval error: {e}")
    print(f"Metric: {e.metric_name}")
except EvaluationError as e:
    print(f"Evaluation failed: {e}")
```

---

## Integration with Test Runner

DeepEval metrics work with the HoloDeck test runner:

```bash
# Run tests with DeepEval metrics
holodeck test --agent my-agent.yaml --metrics geval,faithfulness
```

Test case configuration:

```yaml
# tests/test_cases.yaml
test_cases:
  - name: "Return policy question"
    input: "What is the return policy?"
    ground_truth: "30-day full refund for unused items"
    retrieval_context:
      - "Return Policy: 30 days for full refund on unused items"
      - "Shipping: Free returns within US"
    expected_metrics:
      answer_relevancy: 0.8
      faithfulness: 0.9
```

---

## Performance Tips

1. **Use Ollama for development**: Fast, free, local evaluation
2. **Batch evaluations**: Run multiple test cases in parallel
3. **Cache model loading**: Reuse evaluator instances across tests
4. **Adjust timeouts**: Increase for complex criteria or slow models

```python
# Reuse evaluator for multiple evaluations
evaluator = GEvalEvaluator(
    name="Quality",
    criteria="...",
    timeout=120.0  # Increase for complex evaluations
)

results = []
for test_case in test_cases:
    result = await evaluator.evaluate(**test_case)
    results.append(result)
```

---

## Next Steps

- See [Data Model](./data-model.md) for detailed entity definitions
- See [Contracts](./contracts/evaluator_interfaces.md) for full API documentation
- See [Research](./research.md) for implementation rationale
