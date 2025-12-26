# Feature Specification: Vectorstore Reranking Extension

**Feature Branch**: `015-vectorstore-reranking`
**Created**: 2025-12-23
**Status**: Draft
**Input**: User description: "Create a spec for a new extension for vectorstore-type tools to support reranking in vectorsearch. Ensure support for cohere API and vllm-based reranking models. Reranking must be opt-in and should be a flag for vectorstore-type tools."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Enable Reranking for Improved Search Quality (Priority: P1)

A HoloDeck user has configured a vectorstore tool for semantic search over their documentation. While the initial vector search returns relevant results, they want to improve the quality and relevance ordering of search results by adding a reranking step that uses a more sophisticated model to re-score and reorder the top candidates.

**Why this priority**: This is the core functionality of the feature. Without the ability to enable and configure reranking, the entire feature has no value. This establishes the foundational opt-in pattern and basic reranker configuration.

**Independent Test**: Can be fully tested by configuring a vectorstore tool with `rerank: true` and verifying that search results are reordered based on reranker scores.

**Acceptance Scenarios**:

1. **Given** a vectorstore tool without reranking configured, **When** a search is performed, **Then** results are returned based solely on vector similarity scores (existing behavior preserved)
2. **Given** a vectorstore tool with `rerank: true` configured, **When** a search is performed, **Then** initial vector search results are passed to a reranker and returned in reranked order
3. **Given** a vectorstore tool with reranking enabled, **When** reranker processing fails, **Then** the system falls back to original vector search results and logs a warning

---

### User Story 2 - Configure Cohere Reranking (Priority: P2)

A user wants to leverage Cohere's reranking API for high-quality search result reranking. They need to configure their API credentials and select an appropriate Cohere reranker model.

**Why this priority**: Cohere is a widely-used commercial reranking provider with excellent out-of-the-box performance. Supporting Cohere enables users to quickly achieve high-quality reranking without infrastructure setup.

**Independent Test**: Can be fully tested by configuring a vectorstore tool with Cohere reranker provider, API key, and model, then verifying that Cohere API is called and results are reranked.

**Acceptance Scenarios**:

1. **Given** a vectorstore tool with Cohere reranker configured, **When** a search is performed, **Then** the Cohere Rerank API is called with the query and candidate documents
2. **Given** a Cohere reranker configuration with a valid API key, **When** the reranker is initialized, **Then** the connection is validated successfully
3. **Given** a Cohere reranker configuration with an invalid API key, **When** a search is performed, **Then** an appropriate error is raised with a clear message about the authentication failure

---

### User Story 3 - Configure vLLM-based Reranking (Priority: P2)

A user wants to run reranking locally using a vLLM-hosted reranking model for data privacy, cost control, or customization. They need to configure the endpoint URL and model name for their vLLM deployment.

**Why this priority**: vLLM support enables self-hosted reranking for users with privacy requirements or those wanting to use open-source reranking models. This is equally important to Cohere for different use cases.

**Independent Test**: Can be fully tested by configuring a vectorstore tool with vLLM reranker provider, endpoint URL, and model, then verifying that the vLLM endpoint is called and results are reranked.

**Acceptance Scenarios**:

1. **Given** a vectorstore tool with vLLM reranker configured, **When** a search is performed, **Then** the vLLM reranker endpoint is called with the query and candidate documents
2. **Given** a vLLM reranker configuration with a valid endpoint, **When** the reranker is initialized, **Then** the connection is validated successfully
3. **Given** a vLLM reranker configuration pointing to an unreachable endpoint, **When** a search is performed, **Then** an appropriate error is raised or graceful fallback occurs

---

### User Story 4 - Fine-tune Reranking Behavior (Priority: P3)

A user wants to control the reranking behavior to balance between quality and performance. They need options to specify how many candidates to fetch before reranking and how many final results to return.

**Why this priority**: Fine-tuning parameters allow users to optimize the trade-off between reranking quality (more candidates) and latency/cost (fewer candidates). This is important for production deployments but not required for basic functionality.

**Independent Test**: Can be fully tested by configuring custom `rerank_top_n` and observing that the specified number of candidates are reranked.

**Acceptance Scenarios**:

1. **Given** a vectorstore tool with `rerank_top_n: 20` configured, **When** a search is performed, **Then** the top 20 vector search results are sent to the reranker before returning the final `top_k` results
2. **Given** a vectorstore tool with reranking but no `rerank_top_n` specified, **When** a search is performed, **Then** a reasonable default number of candidates are reranked

---

### Edge Cases

- What happens when the number of vector search results is less than `rerank_top_n`? (All available results should be reranked)
- What happens when the reranker API rate limit is exceeded? (Graceful fallback to original results with warning)
- What happens when reranker returns fewer results than requested? (Return all available reranked results)
- What happens when using reranking with `min_similarity_score` filter? (Apply similarity filter before reranking)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support an opt-in `rerank` boolean flag on vectorstore tools (default: false)
- **FR-002**: System MUST support a `reranker` configuration object that specifies reranker provider and settings
- **FR-003**: System MUST support Cohere as a reranker provider with configurable API key and model name
- **FR-004**: System MUST support vLLM as a reranker provider with configurable endpoint URL and model name
- **FR-005**: System MUST provide a `rerank_top_n` parameter to control how many candidates are sent to the reranker
- **FR-006**: System MUST fall back to original vector search results when reranking fails (graceful degradation)

**Error Handling Decision Matrix** (for FR-006):

| Error Type | Error Class | Behavior | Rationale |
|------------|-------------|----------|-----------|
| Network errors | `RerankerConnectionError` | Fallback to vector results | Transient, service may recover |
| Timeouts | `RerankerConnectionError` | Fallback to vector results | Transient, may succeed on retry |
| Rate limits | `RerankerRateLimitError` | Fallback to vector results | Transient, will recover after backoff |
| Authentication errors | `RerankerAuthError` | **Fail fast** (raise error) | Configuration error, won't self-heal |
| Invalid model name | `RerankerError` | **Fail fast** (raise error) | Configuration error, won't self-heal |
| Invalid API response | `RerankerError` | Fallback to vector results | May be transient server issue |

**Implementation Note**: Errors with `recoverable=True` trigger fallback behavior. Errors with `recoverable=False` (authentication, invalid configuration) fail fast to surface configuration problems early rather than silently degrading.

- **FR-007**: System MUST validate reranker configuration at startup/configuration load time
- **FR-008**: System MUST apply `min_similarity_score` filtering before sending candidates to the reranker
- **FR-009**: System MUST maintain backward compatibility - existing vectorstore configurations without reranking must work unchanged
- **FR-010**: System MUST support environment variable substitution for sensitive reranker configuration values (API keys, URLs)
- **FR-011**: System MUST log reranker calls at debug level, including latency and result counts for operational visibility

### Key Entities

- **RerankerConfig**: Configuration for the reranking provider, including provider type, credentials, model selection, and provider-specific settings
- **RerankerProvider**: Enumeration of supported reranker providers (cohere, vllm)
- **CohereRerankerConfig**: Cohere-specific configuration (API key, model name)
- **VLLMRerankerConfig**: vLLM-specific configuration (endpoint URL, model name)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can enable reranking by adding a single flag (`rerank: true`) to their existing vectorstore configuration
- **SC-002**: Reranked search results demonstrate improved relevance ordering compared to pure vector similarity (validated through user testing and evaluation metrics)
- **SC-003**: Search latency with reranking enabled remains acceptable for interactive use cases (search completes within user-defined timeout)
- **SC-004**: System handles reranker failures gracefully without disrupting the overall search functionality
- **SC-005**: Configuration errors related to reranking are reported clearly at startup, not at runtime during searches
- **SC-006**: All existing vectorstore configurations continue to work without modification (100% backward compatibility)

## Clarifications

### Session 2025-12-23

- Q: What should the default value for `rerank_top_n` be? → A: Default to `top_k * 3` (e.g., 30 candidates if top_k=10)
- Q: Which API protocol should vLLM reranker use? → A: OpenAI-compatible `/v1/rerank` endpoint (JSON with query + documents)
- Q: What observability should reranking operations have? → A: Log reranker calls with latency and result counts (debug level)

## Assumptions

- Users will provide their own API keys for Cohere or host their own vLLM instances
- Cohere Rerank API follows standard REST patterns and is available at their documented endpoints
- vLLM reranking endpoints use the OpenAI-compatible `/v1/rerank` endpoint (JSON with query + documents)
- The reranking step adds acceptable latency for the improved relevance (users opt-in knowing this trade-off)
- `rerank_top_n` should be greater than or equal to `top_k` for meaningful reranking (system should warn if not)
- Default `rerank_top_n` is `top_k * 3` when not explicitly specified
