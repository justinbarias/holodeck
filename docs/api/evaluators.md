# Evaluation Framework API

HoloDeck provides a flexible evaluation framework for measuring agent response quality. The framework supports three tiers of metrics:

1. **Standard NLP Metrics** -- Traditional text-comparison metrics (BLEU, ROUGE, METEOR) that require no LLM
2. **Azure AI Metrics** -- AI-assisted quality metrics via Azure AI Evaluation SDK (groundedness, relevance, coherence, fluency, similarity)
3. **DeepEval Metrics** -- LLM-as-a-judge evaluation with multi-provider support (G-Eval custom criteria, RAG pipeline metrics)

All evaluators share a common base class with retry logic, timeout handling, and a unified parameter specification system.

---

## Architecture Overview

```
BaseEvaluator (base.py)
├── BLEUEvaluator (nlp_metrics.py)
├── ROUGEEvaluator (nlp_metrics.py)
├── METEOREvaluator (nlp_metrics.py)
├── AzureAIEvaluator (azure_ai.py)
│   ├── GroundednessEvaluator
│   ├── RelevanceEvaluator
│   ├── CoherenceEvaluator
│   ├── FluencyEvaluator
│   └── SimilarityEvaluator
└── DeepEvalBaseEvaluator (deepeval/base.py)
    ├── GEvalEvaluator (deepeval/geval.py)
    ├── FaithfulnessEvaluator (deepeval/faithfulness.py)
    ├── AnswerRelevancyEvaluator (deepeval/answer_relevancy.py)
    ├── ContextualRelevancyEvaluator (deepeval/contextual_relevancy.py)
    ├── ContextualPrecisionEvaluator (deepeval/contextual_precision.py)
    └── ContextualRecallEvaluator (deepeval/contextual_recall.py)
```

---

## Configuration Models

Evaluation metrics are configured in `agent.yaml` using Pydantic models from `holodeck.models.evaluation`. The `metrics` list uses a discriminated union on the `type` field (`standard`, `geval`, or `rag`).

### YAML Configuration Example

```yaml
evaluations:
  model:                          # Default LLM for all LLM-based metrics
    provider: openai
    name: gpt-4o
    temperature: 0.0
  metrics:
    # Standard NLP metric (no LLM required)
    - type: standard
      metric: bleu
      threshold: 0.4

    # G-Eval custom criteria (LLM-as-judge)
    - type: geval
      name: Helpfulness
      criteria: "Evaluate if the response provides actionable information"
      evaluation_params: [actual_output, input]
      threshold: 0.7

    # RAG pipeline metric
    - type: rag
      metric_type: faithfulness
      threshold: 0.8
      include_reason: true
```

### EvaluationConfig

::: holodeck.models.evaluation.EvaluationConfig
    options:
      docstring_style: google
      show_source: true

### MetricType

The discriminated union that routes to the correct metric model based on the `type` field:

```python
MetricType = Annotated[
    EvaluationMetric | GEvalMetric | RAGMetric,
    Field(discriminator="type"),
]
```

### EvaluationMetric

Standard metric configuration (`type: standard`).

::: holodeck.models.evaluation.EvaluationMetric
    options:
      docstring_style: google
      show_source: true

### GEvalMetric

G-Eval custom criteria configuration (`type: geval`).

::: holodeck.models.evaluation.GEvalMetric
    options:
      docstring_style: google
      show_source: true

### RAGMetric

RAG pipeline metric configuration (`type: rag`).

::: holodeck.models.evaluation.RAGMetric
    options:
      docstring_style: google
      show_source: true

### RAGMetricType

::: holodeck.models.evaluation.RAGMetricType
    options:
      docstring_style: google
      show_source: true

---

## Base Framework

All evaluators inherit from `BaseEvaluator`, which provides retry logic with exponential backoff, timeout handling, and a parameter specification system.

### BaseEvaluator

::: holodeck.lib.evaluators.base.BaseEvaluator
    options:
      docstring_style: google
      show_source: true

### RetryConfig

::: holodeck.lib.evaluators.base.RetryConfig
    options:
      docstring_style: google
      show_source: true

### EvaluationError

::: holodeck.lib.evaluators.base.EvaluationError
    options:
      docstring_style: google
      show_source: true

---

## Parameter Specification

The `param_spec` module defines a standard way for evaluators to declare their required and optional inputs. This enables the test runner to validate inputs before calling the evaluator.

### EvalParam

::: holodeck.lib.evaluators.param_spec.EvalParam
    options:
      docstring_style: google
      show_source: true

### ParamSpec

::: holodeck.lib.evaluators.param_spec.ParamSpec
    options:
      docstring_style: google
      show_source: true

### DEEPEVAL_PARAMS

A `frozenset` of DeepEval-specific parameter names (`INPUT`, `ACTUAL_OUTPUT`, `EXPECTED_OUTPUT`) used by `ParamSpec.uses_deepeval_params()` to detect the DeepEval naming convention.

```python
DEEPEVAL_PARAMS = frozenset(
    {EvalParam.INPUT, EvalParam.ACTUAL_OUTPUT, EvalParam.EXPECTED_OUTPUT}
)
```

Two naming conventions exist side-by-side:

| Convention | Query | Response | Reference |
|---|---|---|---|
| Azure AI / NLP | `query` | `response` | `ground_truth` |
| DeepEval | `input` | `actual_output` | `expected_output` |

Both conventions share `context` and `retrieval_context`.

---

## Standard NLP Metrics

Traditional text-comparison metrics that do not require an LLM. All NLP evaluators require `response` and `ground_truth` parameters.

### BLEUEvaluator

Uses SacreBLEU with exponential smoothing. Scores are normalized from SacreBLEU's 0--100 scale to 0.0--1.0.

::: holodeck.lib.evaluators.nlp_metrics.BLEUEvaluator
    options:
      docstring_style: google
      show_source: true

### ROUGEEvaluator

Returns all three ROUGE variants (`rouge1`, `rouge2`, `rougeL`). The `variant` parameter controls which variant is used for the threshold check.

::: holodeck.lib.evaluators.nlp_metrics.ROUGEEvaluator
    options:
      docstring_style: google
      show_source: true

### METEOREvaluator

Synonym-aware matching with stemming for better correlation with human judgment.

::: holodeck.lib.evaluators.nlp_metrics.METEOREvaluator
    options:
      docstring_style: google
      show_source: true

### NLPMetricsError

::: holodeck.lib.evaluators.nlp_metrics.NLPMetricsError
    options:
      docstring_style: google
      show_source: true

### NLP Metrics Usage

```python
from holodeck.lib.evaluators.nlp_metrics import BLEUEvaluator, ROUGEEvaluator

bleu = BLEUEvaluator(threshold=0.5)
result = await bleu.evaluate(
    response="The cat sat on the mat",
    ground_truth="The cat is on the mat",
)
print(result["bleu"])    # 0.0-1.0
print(result["passed"])  # True if >= 0.5

rouge = ROUGEEvaluator(threshold=0.6, variant="rougeL")
result = await rouge.evaluate(
    response="The cat sat on the mat",
    ground_truth="The cat is on the mat",
)
print(result["rouge1"], result["rouge2"], result["rougeL"])
```

### NLP Metrics Summary

| Metric | Score Key | Score Range | Use Case |
|---|---|---|---|
| `BLEUEvaluator` | `bleu` | 0.0--1.0 | Precision-focused n-gram matching |
| `ROUGEEvaluator` | `rouge1`, `rouge2`, `rougeL` | 0.0--1.0 | Recall-focused overlap (summarization) |
| `METEOREvaluator` | `meteor` | 0.0--1.0 | Synonym-aware semantic similarity |

---

## Azure AI Metrics

AI-assisted quality metrics powered by the Azure AI Evaluation SDK. All Azure evaluators normalize scores from a 1--5 scale to 0.0--1.0.

### ModelConfig

::: holodeck.lib.evaluators.azure_ai.ModelConfig
    options:
      docstring_style: google
      show_source: true

### AzureAIEvaluator

::: holodeck.lib.evaluators.azure_ai.AzureAIEvaluator
    options:
      docstring_style: google
      show_source: true

### GroundednessEvaluator

Assesses whether all claims in the response are supported by the provided context. Use an expensive model (e.g., `gpt-4o`) for this critical metric.

::: holodeck.lib.evaluators.azure_ai.GroundednessEvaluator
    options:
      docstring_style: google
      show_source: true

### RelevanceEvaluator

Measures whether the response directly addresses the user's question.

::: holodeck.lib.evaluators.azure_ai.RelevanceEvaluator
    options:
      docstring_style: google
      show_source: true

### CoherenceEvaluator

Evaluates logical flow and readability of the response.

::: holodeck.lib.evaluators.azure_ai.CoherenceEvaluator
    options:
      docstring_style: google
      show_source: true

### FluencyEvaluator

Assesses grammar, spelling, punctuation, word choice, and sentence structure.

::: holodeck.lib.evaluators.azure_ai.FluencyEvaluator
    options:
      docstring_style: google
      show_source: true

### SimilarityEvaluator

Compares semantic similarity between response and ground truth.

::: holodeck.lib.evaluators.azure_ai.SimilarityEvaluator
    options:
      docstring_style: google
      show_source: true

### Azure AI Usage

```python
from holodeck.lib.evaluators.azure_ai import (
    ModelConfig,
    GroundednessEvaluator,
    RelevanceEvaluator,
)

config = ModelConfig(
    azure_endpoint="https://my-resource.openai.azure.com/",
    api_key="my-api-key",
    azure_deployment="gpt-4o",
)

groundedness = GroundednessEvaluator(model_config=config)
result = await groundedness.evaluate(
    query="What is the capital of France?",
    response="The capital of France is Paris.",
    context="France is a country in Europe. Its capital is Paris.",
)
print(result["score"])          # 0.0-1.0 (normalized from 1-5)
print(result["groundedness"])   # Raw 1-5 score
print(result["reasoning"])      # LLM explanation
```

### Azure AI Metrics Summary

| Evaluator | Required Params | Optional Params | Score Key |
|---|---|---|---|
| `GroundednessEvaluator` | `response`, `context` | `query` | `groundedness` |
| `RelevanceEvaluator` | `response`, `query` | `context` | `relevance` |
| `CoherenceEvaluator` | `response`, `query` | -- | `coherence` |
| `FluencyEvaluator` | `response`, `query` | -- | `fluency` |
| `SimilarityEvaluator` | `response`, `query`, `ground_truth` | -- | `similarity` |

---

## DeepEval Metrics

LLM-as-a-judge evaluation with multi-provider support (OpenAI, Azure OpenAI, Anthropic, Ollama). DeepEval metrics use a different parameter naming convention (`input`, `actual_output`, `expected_output`) but HoloDeck's `DeepEvalBaseEvaluator` also accepts Azure/NLP aliases (`query`, `response`, `ground_truth`).

### DeepEvalModelConfig

::: holodeck.lib.evaluators.deepeval.config.DeepEvalModelConfig
    options:
      docstring_style: google
      show_source: true

### DeepEvalBaseEvaluator

::: holodeck.lib.evaluators.deepeval.base.DeepEvalBaseEvaluator
    options:
      docstring_style: google
      show_source: true

### DeepEvalError

::: holodeck.lib.evaluators.deepeval.errors.DeepEvalError
    options:
      docstring_style: google
      show_source: true

### ProviderNotSupportedError

::: holodeck.lib.evaluators.deepeval.errors.ProviderNotSupportedError
    options:
      docstring_style: google
      show_source: true

---

### G-Eval: Custom Criteria

#### GEvalEvaluator

::: holodeck.lib.evaluators.deepeval.geval.GEvalEvaluator
    options:
      docstring_style: google
      show_source: true

#### G-Eval Usage

```python
from holodeck.lib.evaluators.deepeval import GEvalEvaluator, DeepEvalModelConfig
from holodeck.models.llm import ProviderEnum

config = DeepEvalModelConfig(
    provider=ProviderEnum.OPENAI,
    model_name="gpt-4o",
    api_key="sk-...",
)

evaluator = GEvalEvaluator(
    name="Professionalism",
    criteria="Evaluate if the response uses professional language and avoids slang.",
    evaluation_params=["actual_output", "input"],
    evaluation_steps=[
        "Check if the language is formal and professional",
        "Verify no slang or casual expressions are used",
    ],
    model_config=config,
    threshold=0.7,
    strict_mode=False,
)

result = await evaluator.evaluate(
    input="Write a business email",
    actual_output="Dear Sir/Madam, I am writing to inquire about...",
)
print(result["score"])      # 0.0-1.0
print(result["passed"])     # True if >= 0.7
print(result["reasoning"])  # LLM-generated explanation
```

#### G-Eval YAML Configuration

```yaml
evaluations:
  model:
    provider: openai
    name: gpt-4o
    temperature: 0.0
  metrics:
    - type: geval
      name: Professionalism
      criteria: |
        Evaluate if the response uses professional language,
        avoids slang, and maintains a respectful tone.
      evaluation_steps:
        - "Check if the language is formal and professional"
        - "Verify no slang or casual expressions are used"
        - "Assess the overall respectful tone"
      evaluation_params:
        - actual_output
        - input
      threshold: 0.7
      strict_mode: false
```

Valid `evaluation_params` values: `input`, `actual_output`, `expected_output`, `context`, `retrieval_context`.

---

### RAG Pipeline Metrics

RAG evaluators measure retrieval-augmented generation quality. All RAG evaluators (except `AnswerRelevancyEvaluator`) require `retrieval_context`.

#### FaithfulnessEvaluator

Detects hallucinations by checking whether the response is supported by the retrieval context.

::: holodeck.lib.evaluators.deepeval.faithfulness.FaithfulnessEvaluator
    options:
      docstring_style: google
      show_source: true

#### AnswerRelevancyEvaluator

Measures whether response statements are relevant to the input query. Does **not** require `retrieval_context`.

::: holodeck.lib.evaluators.deepeval.answer_relevancy.AnswerRelevancyEvaluator
    options:
      docstring_style: google
      show_source: true

#### ContextualRelevancyEvaluator

Measures the proportion of retrieved chunks that are relevant to the query.

::: holodeck.lib.evaluators.deepeval.contextual_relevancy.ContextualRelevancyEvaluator
    options:
      docstring_style: google
      show_source: true

#### ContextualPrecisionEvaluator

Evaluates ranking quality -- whether relevant chunks appear before irrelevant ones.

::: holodeck.lib.evaluators.deepeval.contextual_precision.ContextualPrecisionEvaluator
    options:
      docstring_style: google
      show_source: true

#### ContextualRecallEvaluator

Measures retrieval completeness -- whether the context contains all facts needed to produce the expected output.

::: holodeck.lib.evaluators.deepeval.contextual_recall.ContextualRecallEvaluator
    options:
      docstring_style: google
      show_source: true

#### RAG Metrics Usage

```python
from holodeck.lib.evaluators.deepeval import (
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
    ContextualRelevancyEvaluator,
    ContextualPrecisionEvaluator,
    ContextualRecallEvaluator,
    DeepEvalModelConfig,
)

config = DeepEvalModelConfig()  # Default: Ollama with gpt-oss:20b

# Faithfulness (hallucination detection)
faithfulness = FaithfulnessEvaluator(model_config=config, threshold=0.8)
result = await faithfulness.evaluate(
    input="What are the store hours?",
    actual_output="Store is open 24/7.",
    retrieval_context=["Store hours: Mon-Fri 9am-5pm"],
)
print(result["score"])  # Low score -- hallucination detected

# Answer Relevancy (no retrieval_context needed)
relevancy = AnswerRelevancyEvaluator(model_config=config, threshold=0.7)
result = await relevancy.evaluate(
    input="What is the return policy?",
    actual_output="We offer 30-day returns at no extra cost.",
)

# Contextual Precision (ranking quality)
precision = ContextualPrecisionEvaluator(model_config=config, threshold=0.7)
result = await precision.evaluate(
    input="What is X?",
    actual_output="X is a programming concept.",
    expected_output="X is a well-known programming paradigm.",
    retrieval_context=["X is a programming paradigm.", "Unrelated info"],
)
```

#### RAG YAML Configuration

```yaml
evaluations:
  model:
    provider: openai
    name: gpt-4o
  metrics:
    - type: rag
      metric_type: faithfulness
      threshold: 0.8
      include_reason: true

    - type: rag
      metric_type: answer_relevancy
      threshold: 0.7

    - type: rag
      metric_type: contextual_relevancy
      threshold: 0.6

    - type: rag
      metric_type: contextual_precision
      threshold: 0.7

    - type: rag
      metric_type: contextual_recall
      threshold: 0.6
```

#### RAG Metrics Summary

| Evaluator | Required Params | Requires `retrieval_context` | Measures |
|---|---|---|---|
| `FaithfulnessEvaluator` | `input`, `actual_output`, `retrieval_context` | Yes | Hallucination detection |
| `AnswerRelevancyEvaluator` | `input`, `actual_output` | No | Response relevance to query |
| `ContextualRelevancyEvaluator` | `input`, `actual_output`, `retrieval_context` | Yes | Chunk relevance to query |
| `ContextualPrecisionEvaluator` | `input`, `actual_output`, `expected_output`, `retrieval_context` | Yes | Ranking quality of chunks |
| `ContextualRecallEvaluator` | `input`, `actual_output`, `expected_output`, `retrieval_context` | Yes | Retrieval completeness |

---

## Complete Agent Configuration Example

```yaml
name: customer-support-agent
model:
  provider: openai
  name: gpt-4o

evaluations:
  model:
    provider: openai
    name: gpt-4o
    temperature: 0.0
  metrics:
    # Standard NLP metrics (no LLM required)
    - type: standard
      metric: bleu
      threshold: 0.4
    - type: standard
      metric: rouge
      threshold: 0.5

    # Custom G-Eval criteria
    - type: geval
      name: Helpfulness
      criteria: "Evaluate if the response provides actionable, helpful information"
      evaluation_params: [actual_output, input]
      threshold: 0.7

    # RAG evaluation
    - type: rag
      metric_type: faithfulness
      threshold: 0.8
      include_reason: true

test_cases:
  - name: "Refund policy question"
    input: "What is your refund policy?"
    ground_truth: "We offer a 30-day money-back guarantee on all products."
    retrieval_context:
      - "Refund Policy: All products come with a 30-day money-back guarantee."
      - "Returns must be initiated within 30 days of purchase."

  - name: "Product recommendation"
    input: "I need a laptop for video editing"
    expected_tools: [search_products, get_specifications]
    evaluations:
      - type: geval
        name: TechnicalAccuracy
        criteria: "Verify the response contains accurate technical specifications"
        threshold: 0.8
```

Run tests with:

```bash
holodeck test agent.yaml --verbose --output report.md --format markdown
```
