# Feature Specification: DeepEval Metrics Integration

**Feature Branch**: `012-deepeval-metrics`
**Created**: 2025-01-30
**Status**: Draft
**Input**: User description: "Add DeepEval as a new set of evaluation metrics. The current azure_ai.py eval metrics only support Azure OpenAI, and nothing else. Also add error handling in azure_ai.py to fail early if another LLM provider aside from Azure OpenAI is used."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Evaluate Agent Responses with Multi-Provider Support (Priority: P1)

As a HoloDeck user with OpenAI, Anthropic, or other LLM providers, I want to evaluate my agent's responses using industry-standard LLM-as-a-judge metrics so that I can assess response quality without being locked into Azure OpenAI.

**Why this priority**: The current evaluation system only supports Azure OpenAI, preventing users with other LLM providers from using AI-powered evaluation metrics. This is the core value proposition of adding DeepEval support.

**Independent Test**: Can be fully tested by configuring a non-Azure LLM provider (e.g., OpenAI) and running G-Eval metric evaluations on sample agent responses.

**Acceptance Scenarios**:

1. **Given** a user has configured OpenAI as their LLM provider, **When** they run a G-Eval evaluation on an agent response, **Then** the evaluation completes successfully and returns a normalized score (0-1), reasoning, and pass/fail status.

2. **Given** a user has configured Anthropic Claude as their LLM provider, **When** they run an AnswerRelevancy evaluation, **Then** the evaluation uses Claude as the judge model and returns valid results.

3. **Given** a user has no API key configured for their selected provider, **When** they attempt to run an evaluation, **Then** the system returns a clear error message indicating the missing credentials.

---

### User Story 2 - Define Custom Evaluation Criteria with G-Eval (Priority: P1)

As a HoloDeck user, I want to define custom evaluation criteria in natural language so that I can assess agent responses against my specific quality standards without writing code.

**Why this priority**: G-Eval's ability to evaluate any custom criteria using natural language is the most versatile feature of DeepEval and enables domain-specific evaluation.

**Independent Test**: Can be fully tested by creating a custom G-Eval metric with criteria like "professionalism" or "technical accuracy" and running it against sample responses.

**Acceptance Scenarios**:

1. **Given** a user defines custom criteria "Evaluate whether the response is professional and avoids slang", **When** they evaluate an agent response containing informal language, **Then** the G-Eval metric returns a low score with reasoning explaining why.

2. **Given** a user provides both criteria and explicit evaluation_steps, **When** they run the evaluation, **Then** the metric uses the provided steps instead of auto-generating them.

3. **Given** a user configures a threshold of 0.7, **When** the evaluation returns a score of 0.65, **Then** the result indicates passed=False.

---

### User Story 3 - Evaluate RAG Pipeline Components (Priority: P2)

As a HoloDeck user with a RAG-based agent, I want to evaluate retrieval quality and response faithfulness separately so that I can identify whether issues stem from retrieval or generation.

**Why this priority**: RAG pipelines are common in production agents, and separate retriever/generator metrics help identify specific areas for improvement.

**Independent Test**: Can be fully tested by providing retrieval_context alongside query and response, then running Faithfulness and ContextualRelevancy metrics.

**Acceptance Scenarios**:

1. **Given** a RAG agent response with retrieval_context, **When** I run the FaithfulnessMetric, **Then** it evaluates whether the response is grounded in the retrieved context and returns a score with reasoning.

2. **Given** a retrieval_context with relevant and irrelevant chunks, **When** I run ContextualRelevancyMetric, **Then** it returns a score indicating what proportion of chunks were relevant.

3. **Given** I provide expected_output alongside retrieval_context, **When** I run ContextualRecallMetric, **Then** it evaluates whether the retrieval contains all information needed to produce the ideal response.

---

### User Story 4 - Azure AI Evaluator Provider Validation (Priority: P2)

As a HoloDeck developer, I want the Azure AI evaluator to fail early with a clear error when a non-Azure LLM provider is configured so that users understand why their evaluation failed.

**Why this priority**: Prevents confusing runtime errors when users accidentally misconfigure the Azure AI evaluator with incompatible providers.

**Independent Test**: Can be fully tested by configuring the Azure AI evaluator with an OpenAI or Anthropic provider and verifying it raises a descriptive error before attempting evaluation.

**Acceptance Scenarios**:

1. **Given** a user configures AzureAIEvaluator with an OpenAI provider, **When** they attempt to create the evaluator, **Then** a ProviderNotSupportedError is raised with message "Azure AI Evaluator requires Azure OpenAI provider. Configured provider: openai".

2. **Given** a user configures AzureAIEvaluator with Anthropic provider, **When** they call evaluate(), **Then** the same clear error is raised before any Azure SDK calls are attempted.

3. **Given** a user provides a valid Azure OpenAI configuration, **When** they create the evaluator, **Then** no error is raised and evaluation proceeds normally.

---

### User Story 5 - Answer Relevancy Evaluation (Priority: P2)

As a HoloDeck user, I want to evaluate how relevant agent responses are to user queries so that I can ensure agents stay on topic.

**Why this priority**: Answer relevancy is a fundamental quality metric applicable to all types of agents.

**Independent Test**: Can be fully tested by providing various query-response pairs with clear relevancy differences.

**Acceptance Scenarios**:

1. **Given** a query "What is the return policy?" and response "We offer 30-day returns", **When** I run AnswerRelevancyMetric, **Then** it returns a high score (>0.8).

2. **Given** a query "What is the return policy?" and response "Our store hours are 9am-5pm", **When** I run AnswerRelevancyMetric, **Then** it returns a low score (<0.5).

---

### Edge Cases

- What happens when the LLM judge returns an invalid/unparseable response? The evaluator should retry with the configured retry policy, then raise an EvaluationError with details.
- What happens when evaluation_steps for G-Eval are empty or invalid? The system should auto-generate steps from criteria.
- What happens when retrieval_context is required but not provided? The evaluator should raise a ValueError with a clear message about required parameters.
- What happens when the configured LLM model doesn't support structured output? The evaluator should attempt to parse unstructured responses and provide degraded functionality warnings.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a DeepEval-based evaluator module that supports multiple LLM providers (OpenAI, Azure OpenAI, Anthropic, Ollama) as evaluation judges.
- **FR-002**: System MUST implement GEvalEvaluator that accepts custom criteria in natural language and optional evaluation_steps.
- **FR-003**: System MUST implement AnswerRelevancyEvaluator that measures response relevance to queries.
- **FR-004**: System MUST implement FaithfulnessEvaluator that detects hallucinations by comparing responses to retrieval context.
- **FR-005**: System MUST implement ContextualRelevancyEvaluator that measures retrieval quality.
- **FR-006**: System MUST implement ContextualPrecisionEvaluator that evaluates ranking quality of retrieved chunks.
- **FR-007**: System MUST implement ContextualRecallEvaluator that measures retrieval completeness.
- **FR-008**: All DeepEval evaluators MUST inherit from BaseEvaluator and support retry logic, timeout, and threshold configuration.
- **FR-009**: All evaluators MUST return normalized scores in 0.0-1.0 range with reasoning explanations.
- **FR-010**: Azure AI evaluators MUST validate that the configured provider is Azure OpenAI and raise ProviderNotSupportedError otherwise.
- **FR-011**: DeepEval evaluators MUST support per-metric model configuration allowing different models for different metrics. When no model is specified, evaluators MUST default to Ollama with gpt-oss:20b model.
- **FR-012**: System MUST provide a DeepEvalModelConfig class that wraps provider configuration for DeepEval compatibility.
- **FR-013**: All DeepEval evaluators MUST log evaluation scores, reasoning, and retry attempts using the existing HoloDeck logging infrastructure.

### Key Entities

- **DeepEvalModelConfig**: Configuration wrapper that adapts HoloDeck's LLMProvider model to DeepEval's model interface. Contains provider type, model name, API credentials, and optional parameters.
- **GEvalEvaluator**: Custom criteria evaluator supporting natural language evaluation definitions. Key attributes: criteria (str), evaluation_steps (list[str], optional), evaluation_params (list[str]).
- **RAGEvaluator Suite**: Collection of evaluators for RAG pipelines including AnswerRelevancy, Faithfulness, ContextualRelevancy, ContextualPrecision, ContextualRecall.
- **ProviderNotSupportedError**: Exception raised when an evaluator is used with an incompatible LLM provider.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can evaluate agent responses using OpenAI, Anthropic, or Ollama as judge models within 30 seconds per evaluation.
- **SC-002**: Custom G-Eval criteria defined in natural language produce consistent scores (variance < 0.1 across 10 identical evaluations).
- **SC-003**: Azure AI evaluator provider validation prevents 100% of misconfiguration errors by failing early with clear error messages.
- **SC-004**: RAG evaluation metrics (Faithfulness, ContextualRelevancy) correctly identify hallucinations and irrelevant retrievals with at least 80% agreement with human evaluation.
- **SC-005**: All DeepEval metrics integrate seamlessly with existing test runner and report results in the same format as NLP metrics.
- **SC-006**: Documentation and examples enable users to configure and run DeepEval metrics within 15 minutes of reading the guide.

## Clarifications

### Session 2025-01-30

- Q: What should happen when a user doesn't specify a model for a DeepEval metric? → A: Default to Ollama with gpt-oss:20b model
- Q: What level of observability should DeepEval metrics provide? → A: Standard logging (scores, reasoning, retry attempts) using existing HoloDeck logger

## Assumptions

- DeepEval library version 0.21+ is used (supports latest GEval and RAG metrics)
- Users have valid API credentials for their chosen LLM provider (Ollama runs locally and requires no API key)
- Ollama is installed and running locally for default model usage (gpt-oss:20b)
- The instructor library is available for Anthropic structured output support
- Network connectivity to LLM provider APIs is reliable (retry logic handles transient failures)
- All LLM providers support the reasoning capabilities required for LLM-as-a-judge evaluation
