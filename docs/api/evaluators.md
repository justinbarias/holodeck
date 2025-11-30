# Evaluation Framework API

HoloDeck provides a flexible evaluation framework for measuring agent response quality. The framework supports three tiers of metrics:

1. **DeepEval Metrics (Recommended)** - LLM-as-a-judge with GEval and RAG metrics
2. **NLP Metrics (Standard)** - Algorithmic text comparison
3. **Legacy AI Metrics (Deprecated)** - Azure AI-based metrics

## Evaluation Configuration Models

::: holodeck.models.evaluation.EvaluationConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.evaluation.MetricType
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.evaluation.GEvalMetric
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.evaluation.RAGMetric
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.evaluation.EvaluationMetric
    options:
      docstring_style: google
      show_source: true

---

## DeepEval Evaluators (Recommended)

DeepEval provides powerful LLM-as-a-judge evaluation using the DeepEval library.

### Base Classes

::: holodeck.lib.evaluators.deepeval.DeepEvalBaseEvaluator
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.evaluators.deepeval.DeepEvalModelConfig
    options:
      docstring_style: google
      show_source: true

### GEval Evaluator

The GEval evaluator uses the G-Eval algorithm with chain-of-thought prompting for custom criteria evaluation.

::: holodeck.lib.evaluators.deepeval.GEvalEvaluator
    options:
      docstring_style: google
      show_source: true

### RAG Evaluators

RAG evaluators measure retrieval-augmented generation pipeline quality.

::: holodeck.lib.evaluators.deepeval.FaithfulnessEvaluator
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.evaluators.deepeval.AnswerRelevancyEvaluator
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.evaluators.deepeval.ContextualRelevancyEvaluator
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.evaluators.deepeval.ContextualPrecisionEvaluator
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.evaluators.deepeval.ContextualRecallEvaluator
    options:
      docstring_style: google
      show_source: true

---

## Usage Examples

### DeepEval GEval Metrics

```python
from holodeck.lib.evaluators.deepeval import GEvalEvaluator, DeepEvalModelConfig
from holodeck.models.llm import LLMProvider

# Configure model
model_config = DeepEvalModelConfig(
    provider=LLMProvider.OLLAMA,
    name="llama3.2:latest",
    temperature=0.0
)

# Create evaluator with custom criteria
evaluator = GEvalEvaluator(
    name="Coherence",
    criteria="Evaluate whether the response is clear and well-structured.",
    evaluation_steps=[
        "Check if the response uses clear language.",
        "Assess if the explanation is easy to follow."
    ],
    evaluation_params=["actual_output"],
    model_config=model_config,
    threshold=0.7
)

# Evaluate
result = await evaluator.evaluate(
    actual_output="The password can be reset by clicking 'Forgot Password' on the login page.",
    input="How do I reset my password?"
)

print(f"Score: {result.score}")
print(f"Passed: {result.passed}")
print(f"Reason: {result.reason}")
```

### DeepEval RAG Metrics

```python
from holodeck.lib.evaluators.deepeval import (
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
    DeepEvalModelConfig
)
from holodeck.models.llm import LLMProvider

# Configure model
model_config = DeepEvalModelConfig(
    provider=LLMProvider.OLLAMA,
    name="llama3.2:latest",
    temperature=0.0
)

# Faithfulness - detect hallucinations
faithfulness = FaithfulnessEvaluator(
    model_config=model_config,
    threshold=0.8,
    include_reason=True
)

result = await faithfulness.evaluate(
    input="What is our return policy?",
    actual_output="You can return items within 30 days for a full refund.",
    retrieval_context=[
        "Our return policy allows returns within 30 days of purchase.",
        "Full refunds are provided for items in original condition."
    ]
)

# Answer Relevancy - check response addresses query
relevancy = AnswerRelevancyEvaluator(
    model_config=model_config,
    threshold=0.7
)

result = await relevancy.evaluate(
    input="How do I reset my password?",
    actual_output="Click 'Forgot Password' on the login page and follow the email instructions."
)
```

### NLP Metrics

```python
from holodeck.lib.evaluators.nlp_metrics import compute_f1_score, compute_rouge

# Compute F1 score
prediction = "the cat is on the mat"
reference = "a cat is on the mat"
f1 = compute_f1_score(prediction, reference)
print(f"F1 Score: {f1}")

# Compute ROUGE scores
scores = compute_rouge(prediction, reference)
print(f"ROUGE-1: {scores['rouge1']}")
print(f"ROUGE-2: {scores['rouge2']}")
print(f"ROUGE-L: {scores['rougeL']}")
```

---

## Metric Configuration in YAML

### DeepEval GEval Metric

```yaml
evaluations:
  model:
    provider: ollama
    name: llama3.2:latest
    temperature: 0.0

  metrics:
    - type: geval
      name: "Coherence"
      criteria: "Evaluate whether the response is clear and well-structured."
      evaluation_steps:
        - "Check if the response uses clear language."
        - "Assess if the explanation is easy to follow."
      evaluation_params:
        - actual_output
        - input
      threshold: 0.7
      enabled: true
      fail_on_error: false
```

### DeepEval RAG Metrics

```yaml
evaluations:
  model:
    provider: ollama
    name: llama3.2:latest
    temperature: 0.0

  metrics:
    # Faithfulness - hallucination detection
    - type: rag
      metric_type: faithfulness
      threshold: 0.8
      include_reason: true

    # Answer Relevancy
    - type: rag
      metric_type: answer_relevancy
      threshold: 0.7

    # Contextual Relevancy
    - type: rag
      metric_type: contextual_relevancy
      threshold: 0.75

    # Contextual Precision
    - type: rag
      metric_type: contextual_precision
      threshold: 0.8

    # Contextual Recall
    - type: rag
      metric_type: contextual_recall
      threshold: 0.7
```

### NLP Metrics

```yaml
evaluations:
  metrics:
    - type: standard
      metric: f1_score
      threshold: 0.8

    - type: standard
      metric: bleu
      threshold: 0.6

    - type: standard
      metric: rouge
      threshold: 0.7

    - type: standard
      metric: meteor
      threshold: 0.65
```

### Per-Metric Model Override

```yaml
evaluations:
  model:
    provider: ollama
    name: llama3.2:latest  # Default: free, local

  metrics:
    - type: rag
      metric_type: faithfulness
      threshold: 0.9
      model:  # Override for critical metric
        provider: openai
        name: gpt-4
```

---

## Legacy AI Metrics (Deprecated)

> **DEPRECATED**: Azure AI-based metrics are deprecated and will be removed in a future version.
> Migrate to DeepEval metrics for better flexibility and local model support.

### Migration Guide

| Legacy Metric | Recommended Replacement |
|---------------|------------------------|
| `groundedness` | `type: rag`, `metric_type: faithfulness` |
| `relevance` | `type: rag`, `metric_type: answer_relevancy` |
| `coherence` | `type: geval` with custom criteria |
| `safety` | `type: geval` with custom criteria |

### Legacy Usage (Not Recommended)

```python
# DEPRECATED - Use DeepEval evaluators instead
from holodeck.lib.evaluators.azure_ai import AzureAIEvaluator

evaluator = AzureAIEvaluator(model="gpt-4", api_key="your-key")

result = await evaluator.evaluate_groundedness(
    response="Paris is the capital of France",
    context="France's capital city is known for the Eiffel Tower",
)
```

### Legacy YAML Configuration (Not Recommended)

```yaml
# DEPRECATED - Use type: geval or type: rag instead
evaluations:
  metrics:
    - type: standard
      metric: groundedness  # Deprecated
      threshold: 0.8

    - type: standard
      metric: relevance     # Deprecated
      threshold: 0.75

    - type: standard
      metric: coherence     # Deprecated
      threshold: 0.7

    - type: standard
      metric: safety        # Deprecated
      threshold: 0.9
```

---

## Integration with Test Runner

The test runner automatically:

1. Loads evaluation configuration from agent YAML
2. Creates appropriate evaluators based on metric type
3. Invokes evaluators on test outputs
4. Extracts retrieval_context from tool results (for RAG metrics)
5. Collects metric scores
6. Compares against thresholds
7. Includes results in test report

### Test Runner Evaluator Creation

```python
# Internal test runner logic (simplified)
def _create_evaluators(self, metrics: list[MetricType]) -> dict:
    evaluators = {}

    for metric in metrics:
        if metric.type == "geval":
            evaluators[metric.name] = GEvalEvaluator(
                name=metric.name,
                criteria=metric.criteria,
                evaluation_steps=metric.evaluation_steps,
                evaluation_params=metric.evaluation_params,
                model_config=self._get_model_config(metric),
                threshold=metric.threshold,
                strict_mode=metric.strict_mode
            )
        elif metric.type == "rag":
            evaluator_class = RAG_EVALUATOR_MAP[metric.metric_type]
            evaluators[metric.metric_type] = evaluator_class(
                model_config=self._get_model_config(metric),
                threshold=metric.threshold,
                include_reason=metric.include_reason
            )
        elif metric.type == "standard":
            # NLP or legacy metrics
            evaluators[metric.metric] = self._create_standard_evaluator(metric)

    return evaluators
```

---

## Error Handling

### DeepEval Errors

```python
from holodeck.lib.evaluators.deepeval.errors import (
    DeepEvalError,
    ProviderNotSupportedError
)

try:
    result = await evaluator.evaluate(actual_output="...")
except DeepEvalError as e:
    print(f"Evaluation failed: {e.message}")
    print(f"Metric: {e.metric_name}")
    print(f"Test case: {e.test_case_summary}")
except ProviderNotSupportedError as e:
    print(f"Provider not supported: {e}")
```

### Soft vs Hard Failures

```yaml
metrics:
  # Soft failure - continues on error
  - type: geval
    name: "Quality"
    criteria: "..."
    fail_on_error: false  # Default

  # Hard failure - stops test on error
  - type: rag
    metric_type: faithfulness
    fail_on_error: true
```

---

## Related Documentation

- [Evaluations Guide](../guides/evaluations.md): Configuration and usage guide
- [Test Runner](test-runner.md): Test execution framework
- [Data Models](models.md): EvaluationConfig and MetricConfig models
- [Configuration Loading](config-loader.md): Loading evaluation configs
