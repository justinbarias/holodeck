# Feature Specification: HierarchicalDocumentTool

**Feature Branch**: `020-structured-document-tool`
**Created**: 2026-01-29
**Status**: Draft
**Input**: User description: "Enhancement to HoloDeck's vectorstore tool to implement advanced hybrid vector/keyword/exact match search with structure-aware document parsing, contextual embeddings, and domain-agnostic hierarchical chunking."

## Clarifications

### Session 2026-01-29

- Q: What is the default and maximum number of search results? → A: Default 10, configurable up to 100 results
- Q: What is the index persistence strategy? → A: In-memory by default, configurable to any supported vector store (postgres, azure-ai-search, qdrant, weaviate, chromadb, faiss, pinecone, etc.) for persistence
- Q: What is the maximum chunk size for embeddings? → A: 800 tokens default (per Anthropic contextual retrieval baseline), configurable
- Q: How should definition sections be detected? → A: Heading keywords ("Definitions", "Glossary", "Terms") + pattern matching (e.g., `"Term" means...`)
- Q: How should cross-references be detected? → A: Section number patterns (e.g., "Section 203", "§4.2") + natural language patterns ("see X", "as defined in", "pursuant to")

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Semantic Search Across Documents (Priority: P1)

As an agent user, I want to search my indexed documents using natural language queries so that I can find relevant information even when I don't know the exact terminology.

**Why this priority**: Semantic search is the foundational use case that provides the core value proposition of understanding user intent and finding conceptually related content.

**Independent Test**: Can be fully tested by ingesting a sample document, querying with natural language, and verifying relevant sections are returned with accurate source citations.

**Acceptance Scenarios**:

1. **Given** a document corpus has been ingested, **When** a user queries "What are the reporting requirements?", **Then** the system returns sections discussing reporting, compliance obligations, or submission deadlines regardless of exact wording.
2. **Given** multiple documents are indexed, **When** a user queries conceptually, **Then** results include relevance scores and source locations (document name, section path).
3. **Given** a query has no relevant matches, **When** results are returned, **Then** the system indicates low confidence or no matches found.

---

### User Story 2 - Exact Match and Keyword Search (Priority: P1)

As an agent user, I want to search for exact phrases, section numbers, or specific keywords so that I can locate precise references without semantic interpretation.

**Why this priority**: Users frequently need to find exact citations, legal references, or specific terminology that semantic search may dilute or miss entirely.

**Independent Test**: Can be fully tested by ingesting a document with known section numbers, querying for "Section 203(a)(1)", and verifying the exact section is returned as the top result.

**Acceptance Scenarios**:

1. **Given** a document with hierarchical sections, **When** a user queries "Section 403(b)(2)", **Then** the exact section is returned as the primary result.
2. **Given** a document containing specific terminology, **When** a user queries the exact phrase "reasonable best efforts", **Then** all occurrences of that exact phrase are returned with surrounding context.
3. **Given** a mixed query with keywords and concepts, **When** the user searches, **Then** both exact matches and semantically similar content are returned with clear distinction.

---

### User Story 3 - Structure-Aware Document Ingestion (Priority: P1)

As a HoloDeck user, I want to ingest documents while preserving their hierarchical structure so that search results maintain context about where information appears in the document.

**Why this priority**: Preserving document structure is essential for meaningful retrieval - users need to know not just what was found, but where it sits within the document's hierarchy.

**Independent Test**: Can be fully tested by ingesting a structured document (with headings, sections, subsections), then verifying that retrieved chunks include their full parent chain (e.g., "Title I > Chapter 2 > Section 203").

**Acceptance Scenarios**:

1. **Given** a document with nested structure (headings, sections, subsections), **When** ingested, **Then** each chunk retains metadata about its position in the hierarchy.
2. **Given** a markdown document converted from PDF/Word, **When** ingested, **Then** heading levels map to hierarchical depth.
3. **Given** a document with definitions section, **When** ingested, **Then** definitions are extracted and made available as global context for all queries.

---

### User Story 4 - Hybrid Search with Result Fusion (Priority: P2)

As an agent user, I want search results that combine semantic understanding with keyword precision so that I get the best of both approaches in a single query.

**Why this priority**: Combining search modalities dramatically improves retrieval quality, but requires the foundational P1 capabilities to be in place first.

**Independent Test**: Can be fully tested by querying with a mixed intent (concept + specific term), verifying results come from both semantic and keyword indices, and checking the fused ranking is sensible.

**Acceptance Scenarios**:

1. **Given** both semantic and keyword indices exist, **When** a user queries, **Then** results are merged using a fusion algorithm that considers both relevance types.
2. **Given** configurable weights for search modalities, **When** the agent configuration specifies weights, **Then** the fusion algorithm respects those weights.
3. **Given** a query that matches strongly in one modality but weakly in another, **When** results are returned, **Then** the strong matches are not diluted by weak cross-modality matches.

---

### User Story 5 - Contextual Embeddings for Improved Retrieval (Priority: P2)

As a HoloDeck user, I want chunks to be embedded with their structural context prepended so that semantic search understands where content appears in the document.

**Why this priority**: Contextual embeddings significantly improve retrieval quality but build on top of the basic structure-aware ingestion capability.

**Independent Test**: Can be fully tested by comparing retrieval accuracy between context-prepended embeddings and raw chunk embeddings on the same query set.

**Acceptance Scenarios**:

1. **Given** a chunk like "The Administrator shall...", **When** embedded, **Then** the embedding includes document summary and location context.
2. **Given** identical text appearing in different document locations, **When** queried in context, **Then** the contextually appropriate occurrence ranks higher.
3. **Given** the contextual embedding option is configurable, **When** disabled, **Then** chunks are embedded without context prepending.

---

### User Story 6 - YAML Configuration for HierarchicalDocumentTool (Priority: P2)

As a HoloDeck user, I want to configure the HierarchicalDocumentTool in my agent YAML so that I can customize ingestion, chunking, and search behavior without writing code.

**Why this priority**: YAML-based configuration is core to HoloDeck's no-code philosophy and must be available for this tool to integrate with the platform.

**Independent Test**: Can be fully tested by creating an agent.yaml with HierarchicalDocumentTool configuration, running ingestion, and verifying the configuration options are respected.

**Acceptance Scenarios**:

1. **Given** an agent.yaml with HierarchicalDocumentTool configuration, **When** the agent is loaded, **Then** the tool initializes with specified settings.
2. **Given** configuration for search weights, **When** the tool executes a search, **Then** the configured weights are applied.
3. **Given** configuration for chunking strategy, **When** documents are ingested, **Then** the specified strategy is used.

---

### User Story 7 - Definition and Cross-Reference Extraction (Priority: P3)

As an agent user working with formal documents, I want definitions and cross-references automatically extracted so that I can understand terminology and navigate related sections.

**Why this priority**: Definition extraction enhances usability for formal documents but is an advanced feature that builds on core functionality.

**Independent Test**: Can be fully tested by ingesting a document with a definitions section, then verifying definitions are available as lookup and included as context when relevant terms appear in queries.

**Acceptance Scenarios**:

1. **Given** a document with a definitions section, **When** ingested, **Then** definitions are extracted into a separate, always-available reference.
2. **Given** a query containing a defined term, **When** results are returned, **Then** the definition is included as supplementary context.
3. **Given** a section with cross-references, **When** retrieved, **Then** the cross-referenced sections are identified and can be navigated.

---

### User Story 8 - Optional Reranking (Priority: P3)

As an advanced user, I want to optionally enable a reranker to improve result quality so that the most relevant documents appear at the top.

**Why this priority**: Reranking is an optimization that improves result quality but adds latency and complexity; it's optional for users who prioritize precision.

**Independent Test**: Can be fully tested by running the same query with and without reranking enabled, comparing result ordering and relevance.

**Acceptance Scenarios**:

1. **Given** reranking is enabled in configuration, **When** search results are returned, **Then** results pass through a reranker before being presented.
2. **Given** reranking is disabled, **When** search results are returned, **Then** results are returned directly from fusion without additional processing.
3. **Given** reranking configuration specifies a model, **When** reranking executes, **Then** the specified model is used.

---

### Edge Cases

- What happens when a document has no discernible structure (flat text)?
  - System falls back to token-based chunking with configurable overlap
- How does the system handle extremely long sections that exceed embedding limits?
  - Sections are split at natural boundaries (sentences, paragraphs) while preserving parent context
- What happens when keyword search returns many results but semantic search returns few?
  - Fusion algorithm handles imbalanced result sets; configurable fallback behavior
- How does the system handle documents in non-English languages?
  - Embedding model must support the language; system warns if language detection indicates potential incompatibility
- What happens when definitions conflict across multiple documents?
  - Each document maintains its own definition namespace; conflicts are surfaced to the user

## Requirements *(mandatory)*

### Functional Requirements

**Document Processing Pipeline**

- **FR-001**: System MUST convert any supported document type to markdown using existing markitdown integration before processing.
- **FR-002**: System MUST parse markdown to identify hierarchical structure (headings, sections, subsections).
- **FR-003**: System MUST chunk documents based on structural boundaries rather than token count alone, with a default maximum of 800 tokens per chunk (configurable).
- **FR-004**: System MUST preserve parent chain metadata for each chunk (e.g., ["Title I", "Chapter 2", "Section 203"]).
- **FR-005**: System MUST extract definitions into a separate, queryable reference when detected via heading keywords ("Definitions", "Glossary", "Terms") or definition patterns (e.g., `"Term" means...`).
- **FR-006**: System MUST identify and store cross-references between sections via section number patterns (e.g., "Section 203", "§4.2") and natural language patterns ("see X", "as defined in", "pursuant to").

**Indexing**

- **FR-007**: System MUST create a dense (embedding) index for semantic search.
- **FR-008**: System MUST create a sparse (BM25 or equivalent) index for keyword search.
- **FR-009**: System MUST create an exact match index for precise phrase and identifier lookup.
- **FR-010**: System MUST support contextual embedding by prepending structural context before embedding.
- **FR-011**: System MUST allow users to configure which index types are enabled.

**Search & Retrieval**

- **FR-012**: System MUST support pure semantic queries.
- **FR-013**: System MUST support pure keyword/exact match queries.
- **FR-014**: System MUST support hybrid queries that combine both modalities.
- **FR-015**: System MUST merge results using Reciprocal Rank Fusion (RRF) or configurable alternative.
- **FR-016**: System MUST return source attribution with each result (document, section path, chunk ID).
- **FR-017**: System MUST support configurable weights for each search modality in hybrid mode.
- **FR-018**: System MUST optionally support reranking of fused results.
- **FR-018a**: System MUST return 10 results by default, configurable up to a maximum of 100 results per query.

**Configuration**

- **FR-019**: System MUST be configurable via YAML in the agent configuration file.
- **FR-020**: System MUST support configuration of chunking strategy (structure-aware vs. token-based fallback).
- **FR-021**: System MUST support configuration of search weights per modality.
- **FR-022**: System MUST support configuration of embedding model.
- **FR-023**: System MUST support configuration of reranker enablement and model selection.

**Integration**

- **FR-024**: System MUST integrate as a tool type in HoloDeck's existing tool framework.
- **FR-025**: System MUST be invocable by agents during conversation.
- **FR-026**: System MUST support incremental document updates (add/remove documents without full reindex).

### Key Entities

- **HierarchicalDocument**: Represents an ingested document with metadata (source path, document summary, language).
- **DocumentChunk**: A parsed section of a document with text, parent chain, chunk ID, type classification (definition, requirement, etc.), and cross-references.
- **DefinitionEntry**: An extracted definition with term, definition text, source location, and any exceptions.
- **HybridIndex**: Container for the three index types (dense, sparse, exact) for a document corpus.
- **SearchResult**: A retrieved chunk with relevance scores per modality, fused score, and source attribution.
- **SearchConfiguration**: User-specified settings for weights, enabled indices, reranking, and contextual embedding.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users find relevant information in 90% of queries as measured by user feedback or automated relevance testing.
- **SC-002**: Exact match queries (section numbers, specific phrases) return the correct result as the top result in 99% of cases.
- **SC-003**: Hybrid search improves retrieval relevance by 20% compared to semantic-only search as measured on a standard test set.
- **SC-004**: Document ingestion completes in under 30 seconds per 100 pages for typical documents.
- **SC-005**: Search queries return results in under 2 seconds for corpora up to 10,000 chunks.
- **SC-006**: Users can configure the tool entirely through YAML without writing any code.
- **SC-007**: Structure is preserved for 95% of documents with identifiable hierarchical formatting (headings, numbered sections).
- **SC-008**: Retrieved results include accurate source attribution (document name, section path) in 100% of cases.

## Assumptions

- **A-001**: The existing markitdown integration handles document-to-markdown conversion; this feature focuses on post-conversion processing.
- **A-002**: Users have access to embedding models (via configured LLM providers) for dense indexing.
- **A-003**: The vector store backend follows the existing vectorstore_tool pattern: in-memory by default (indices rebuilt on restart), with optional persistence via any supported provider (postgres, qdrant, weaviate, chromadb, faiss, pinecone, azure-ai-search, etc.) when database is configured.
- **A-004**: Documents primarily contain hierarchical structure identifiable through markdown headings; unstructured documents fall back to token-based chunking.
- **A-005**: Performance targets assume standard hardware; large-scale deployments may require tuning.
- **A-006**: Reranking models are optional and user-provided; the system works without them.

## Dependencies

- **D-001**: Existing HoloDeck tool framework for integration.
- **D-002**: Existing markitdown integration for document conversion.
- **D-003**: Embedding model access through HoloDeck's LLM provider configuration.
- **D-004**: Vector store backend (to be determined during implementation planning).

## Out of Scope

- Real-time document synchronization (documents are ingested on demand or via explicit refresh).
- Multi-tenant isolation (handled at the HoloDeck deployment level, not within this tool).
- OCR for scanned documents (relies on markitdown's existing capabilities).
- Specific domain parsers (e.g., legal-specific, medical-specific) - the tool is domain-agnostic.
