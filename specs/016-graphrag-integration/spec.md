# Feature Specification: GraphRAG Engine Integration

**Feature Branch**: `016-graphrag-integration`
**Created**: 2025-12-27
**Status**: Draft
**Input**: User description: "Integrate GraphRAG as an engine option for HoloDeck vectorstore tools, enabling knowledge graph-based retrieval with entity extraction, community detection, and hierarchical summarization."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Configure GraphRAG Engine for Entity-Centric Search (Priority: P1)

As a HoloDeck user, I want to configure a vectorstore tool to use GraphRAG as its engine so that I can perform knowledge graph-based searches that understand entities and their relationships within my document corpus.

**Why this priority**: This is the foundational capability that enables all GraphRAG features. Without the ability to configure and use GraphRAG as an engine, no other functionality can work.

**Independent Test**: Can be fully tested by creating a YAML configuration with `engine: graphrag` and `search_mode: local`, indexing a sample document corpus, and querying for entities and their relationships.

**Acceptance Scenarios**:

1. **Given** a vectorstore tool configuration with `engine: graphrag` and `search_mode: local`, **When** the tool is initialized with a document source, **Then** the system builds a knowledge graph index containing extracted entities and relationships
2. **Given** an initialized GraphRAG tool with local search mode, **When** a user queries "Who is the CEO of Acme Corp?", **Then** the system returns entity-focused results with relationship context
3. **Given** an incomplete GraphRAG configuration, **When** the user attempts to initialize the tool, **Then** the system creates sensible defaults and logs them for transparency

---

### User Story 2 - Perform Global Analytical Queries (Priority: P2)

As a data analyst using HoloDeck, I want to use global search mode to analyze themes and patterns across my entire document corpus so that I can answer high-level analytical questions that require synthesizing information from multiple sources.

**Why this priority**: Global search provides unique analytical capabilities that differentiate GraphRAG from traditional RAG. It's essential for users who need dataset-wide insights but builds on the core engine integration.

**Independent Test**: Can be fully tested by configuring a tool with `search_mode: global`, indexing documents, and querying for cross-document themes or patterns.

**Acceptance Scenarios**:

1. **Given** a vectorstore tool configured with `search_mode: global`, **When** a user queries "What are the main themes in this dataset?", **Then** the system synthesizes insights from community reports across the knowledge graph
2. **Given** a global search configuration with `community_level: 1`, **When** performing analysis, **Then** the system uses more detailed community granularity for nuanced insights
3. **Given** a global search configuration with `community_level: 3`, **When** performing analysis, **Then** the system uses broader community summaries for high-level patterns

---

### User Story 3 - Incremental Index Updates (Priority: P3)

As a HoloDeck user with evolving document collections, I want the system to detect when my source documents have changed and automatically update the index so that my searches always reflect current content without manual intervention.

**Why this priority**: Incremental updates improve operational efficiency but are not required for basic GraphRAG functionality. Users can manually force reindexing if needed.

**Independent Test**: Can be fully tested by modifying source documents after initial indexing and verifying the system detects changes and updates the index on next initialization.

**Acceptance Scenarios**:

1. **Given** an existing GraphRAG index and unchanged source documents, **When** the tool initializes, **Then** the system uses the cached index without rebuilding
2. **Given** an existing GraphRAG index and modified source documents, **When** the tool initializes, **Then** the system detects the change and rebuilds the index
3. **Given** an existing GraphRAG index, **When** the user explicitly requests forced reindexing, **Then** the system rebuilds the index regardless of source file changes

---

### User Story 4 - Configure Separate Models for Indexing and Search (Priority: P4)

As a cost-conscious HoloDeck user, I want to configure different language models for indexing versus search operations so that I can optimize costs by using cheaper models for the computationally intensive indexing phase while using higher-quality models for user-facing search.

**Why this priority**: Model configuration flexibility is an optimization feature. The system works with default models, making this a nice-to-have enhancement.

**Independent Test**: Can be fully tested by configuring `indexing_model` and `search_model` separately, running indexing and search operations, and verifying different models are used for each.

**Acceptance Scenarios**:

1. **Given** a GraphRAG configuration with `indexing_model: gpt-4o-mini` and `search_model: gpt-4o`, **When** the tool indexes documents, **Then** the system uses the cheaper model for entity extraction and summarization
2. **Given** a GraphRAG configuration with `indexing_model: gpt-4o-mini` and `search_model: gpt-4o`, **When** a user performs a search, **Then** the system uses the higher-quality model for generating responses
3. **Given** a GraphRAG configuration with no model specifications, **When** the tool operates, **Then** the system uses sensible default models

---

### Edge Cases

- What happens when the GraphRAG package is not installed?
  - System provides a clear error message with installation instructions
- What happens when source documents are empty or contain no extractable entities?
  - System completes indexing with zero entities and returns appropriate empty results
- What happens when LLM API keys are missing or invalid?
  - System fails fast with a clear error indicating which credentials are needed
- What happens when indexing is interrupted mid-process?
  - System detects incomplete index and rebuilds on next initialization
- What happens when disk space is insufficient for index storage?
  - System fails with a clear error message about storage requirements

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support `engine: graphrag` as a configuration option for vectorstore tools
- **FR-002**: System MUST support `local` and `global` search modes for GraphRAG tools
- **FR-003**: System MUST automatically build a knowledge graph index when a GraphRAG tool is initialized for the first time
- **FR-004**: System MUST extract entities, relationships, and communities from source documents during indexing
- **FR-005**: System MUST cache index artifacts for reuse across sessions
- **FR-006**: System MUST detect source document changes and rebuild the index when needed
- **FR-007**: System MUST support configurable entity types for extraction
- **FR-008**: System MUST support separate model configuration for indexing and search operations
- **FR-009**: System MUST provide clear error messages when the GraphRAG package is not installed
- **FR-010**: System MUST integrate seamlessly with existing vectorstore tool patterns (initialize, search interface)
- **FR-011**: System MUST support configurable community hierarchy levels for global search
- **FR-012**: System MUST persist index metadata including entity counts, relationship counts, and indexing timestamps
- **FR-013**: System MUST support OpenAI and Azure OpenAI as LLM providers for GraphRAG operations
- **FR-014**: System MUST NOT require the GraphRAG package unless `engine: graphrag` is configured (optional dependency)
- **FR-015**: System MUST support chunk size and overlap configuration for text unit processing

### Key Entities

- **GraphRAG Configuration**: Settings that control GraphRAG behavior including search mode, model configurations, entity types, and storage paths
- **Index Metadata**: Information about the built index including source hash, timestamps, entity/relationship/community counts, and version information
- **Entity**: A person, organization, location, event, or concept extracted from documents with attributes and descriptions
- **Relationship**: A connection between two entities with type and description
- **Community**: A cluster of related entities detected through graph analysis
- **Community Report**: A summarized description of a community's key themes and relationships

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can configure a GraphRAG vectorstore tool using 5 or fewer YAML configuration lines (minimal configuration)
- **SC-002**: Entity-centric queries return relationship context that traditional RAG cannot provide
- **SC-003**: Global search mode can synthesize themes across 100+ documents in a single query
- **SC-004**: Subsequent tool initializations use cached indexes when source documents are unchanged, completing in under 5 seconds
- **SC-005**: Users receive clear, actionable error messages within 3 seconds when configuration is invalid
- **SC-006**: GraphRAG tools integrate with existing HoloDeck CLI commands (test, chat) without modification
- **SC-007**: Documentation and examples enable users to configure GraphRAG tools without external references

## Assumptions

- Users have access to OpenAI or Azure OpenAI API keys for GraphRAG operations
- Source documents are in text, markdown, or PDF format
- The GraphRAG library (Microsoft) is the underlying implementation
- Index storage uses local file system (parquet files for artifacts)
- LLM caching is enabled by default to reduce indexing costs
- Users understand the cost implications of GraphRAG indexing (LLM-intensive)
- Default entity types are: organization, person, location, event, concept

## Out of Scope

- DRIFT search mode (advanced GraphRAG feature)
- Basic search mode (uses traditional RAG approach)
- External vector store backends (e.g., Pinecone, Weaviate) for GraphRAG
- Real-time streaming during indexing progress
- Multi-tenant index isolation
- Distributed indexing across multiple machines
- Integration with reranking extension (kept separate per user requirement)
