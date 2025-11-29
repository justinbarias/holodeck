"""Evaluation framework for HoloDeck test results.

This package provides evaluators for assessing agent performance using
multiple evaluation approaches:

- **NLP Metrics**: Traditional natural language processing metrics
  - F1 Score: Harmonic mean of precision and recall
  - BLEU: Bilingual Evaluation Understudy score for translation quality
  - ROUGE: Recall-Oriented Understudy for Gisting Evaluation
  - METEOR: Metric for Evaluation of Translation with Explicit ORdering

- **Azure AI Metrics**: AI-powered semantic evaluation metrics
  - Groundedness: How factually grounded responses are
  - Relevance: How well responses address the query
  - Coherence: How logically coherent responses are
  - (Customizable) Additional metrics via Azure AI Evaluation

- **DeepEval Metrics**: LLM-as-a-judge evaluation with multi-provider support
  - GEvalEvaluator: Custom criteria evaluation using G-Eval algorithm
  - Support for OpenAI, Azure OpenAI, Anthropic, and Ollama providers

- **Base Evaluator**: Abstract base class with retry logic

Example:
    from holodeck.lib.evaluators.nlp_metrics import compute_f1_score
    from holodeck.lib.evaluators.azure_ai import AzureAIEvaluator
    from holodeck.lib.evaluators.deepeval import GEvalEvaluator

    # NLP metric
    score = compute_f1_score(prediction, reference)

    # AI-powered metric (Azure)
    evaluator = AzureAIEvaluator(model="gpt-4")
    results = await evaluator.evaluate(response, ground_truth)

    # AI-powered metric (DeepEval)
    evaluator = GEvalEvaluator(
        name="Helpfulness",
        criteria="Evaluate if the response is helpful"
    )
    results = await evaluator.evaluate(input=query, actual_output=response)

Classes:
    BaseEvaluator: Abstract evaluator base class
    NLPEvaluator: NLP metrics implementation
    AzureAIEvaluator: Azure AI evaluation metrics
    GEvalEvaluator: DeepEval G-Eval custom criteria evaluator

Functions:
    compute_f1_score: Compute F1 score between prediction and reference
    compute_bleu: Compute BLEU score for translation quality
    compute_rouge: Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
    compute_meteor: Compute METEOR score with stemmer and synonyms
"""

# DeepEval evaluators and configuration
from holodeck.lib.evaluators.deepeval import (
    DEFAULT_MODEL_CONFIG,
    DeepEvalBaseEvaluator,
    DeepEvalError,
    DeepEvalModelConfig,
    GEvalEvaluator,
    ProviderNotSupportedError,
)

__all__ = [
    # DeepEval exports
    "DeepEvalBaseEvaluator",
    "DeepEvalModelConfig",
    "DEFAULT_MODEL_CONFIG",
    "DeepEvalError",
    "GEvalEvaluator",
    "ProviderNotSupportedError",
]
