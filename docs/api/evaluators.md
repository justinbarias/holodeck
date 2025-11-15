# Evaluation Framework API

HoloDeck provides a flexible evaluation framework configuration. Evaluations can be configured at three levels:

- **Global default**: Apply same model to all metrics
- **Run-level override**: Change model for an evaluation run
- **Per-metric override**: Use different models for specific metrics

## Evaluation Configuration Models

::: holodeck.models.evaluation.EvaluationConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.evaluation.EvaluationMetric
    options:
      docstring_style: google
      show_source: true

## Usage Examples

### NLP Metrics

```python
from holodeck.lib.evaluators.nlp_metrics import compute_f1_score, compute_rouge

# Compute F1 score
prediction = "the cat is on the mat"
reference = "a cat is on the mat"
f1 = compute_f1_score(prediction, reference)

# Compute ROUGE scores
scores = compute_rouge(prediction, reference)
print(f"ROUGE-1: {scores['rouge1']}")
print(f"ROUGE-2: {scores['rouge2']}")
print(f"ROUGE-L: {scores['rougeL']}")
```

### Azure AI Metrics

```python
from holodeck.lib.evaluators.azure_ai import AzureAIEvaluator

evaluator = AzureAIEvaluator(model="gpt-4", api_key="your-key")

# Evaluate groundedness
result = await evaluator.evaluate_groundedness(
    response="Paris is the capital of France",
    context="France's capital city is known for the Eiffel Tower",
)
```

### Per-Test Evaluation

Tests are automatically evaluated if configured in agent YAML:

```yaml
evaluations:
  model:
    provider: openai
    model: gpt-4
  metrics:
    - type: groundedness
    - type: relevance
    - type: f1  # NLP metric
    - type: bleu
  thresholds:
    groundedness: 0.8
    relevance: 0.7
```

## Metric Configuration

In agent YAML, specify evaluation metrics:

```yaml
evaluations:
  # Global model for all metrics
  model:
    provider: openai
    model: gpt-4-turbo

  # Per-metric configuration
  metrics:
    # AI-powered metrics
    - name: groundedness
      model:
        provider: openai
        model: gpt-4  # Override for this metric

    # NLP metrics (no model needed)
    - name: f1
    - name: bleu
    - name: rouge
    - name: meteor

  # Minimum thresholds for pass/fail
  thresholds:
    groundedness: 0.8
    relevance: 0.7
    f1: 0.6
```

## Integration with Test Runner

The test runner automatically:

1. Loads evaluation configuration from agent YAML
2. Invokes evaluators on test outputs
3. Collects metric scores
4. Compares against thresholds
5. Includes results in test report

## Related Documentation

- [Test Runner](test-runner.md): Test execution framework
- [Data Models](models.md): EvaluationConfig and MetricConfig models
- [Configuration Loading](config-loader.md): Loading evaluation configs
