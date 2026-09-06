# Feature Specification: Unstructured Vector Ingestion and Search

**Feature Branch**: `008-unstructured-vector-ingestion-search`
**Created**: 2025-11-23
**Status**: Draft
**Input**: User description: "Create a spec for US 6.1.1 - which is part of the parent feature 6.1 - Vector Search Tool operations, and is a parent of US6 - Agent Tools. The vectorstore tool supports unstructured text data in multiple formats and creates searchable embeddings from full content."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Semantic Search Over Unstructured Data (Priority: P1)

An agent developer configures a vectorstore tool to enable semantic search over unstructured text data by pointing to either a single file or a directory. The tool automatically ingests the content, creates embeddings, and enables query-based search that returns relevant results ranked by semantic similarity.

**Why this priority**: This is the core MVP functionality - enabling semantic search over knowledge sources (single file or multiple files). It delivers immediate value by allowing agents to answer questions based on documented information without requiring manual keyword matching or separate configurations for files vs. directories.

**Independent Test**: Can be fully tested by configuring a vectorstore tool pointing to either a single markdown file or a directory of mixed file types, running queries, and verifying semantically relevant results are returned with source references. Delivers value by enabling basic question-answering capabilities.

**Acceptance Scenarios**:

1. **Given** a tool defined as `type: vectorstore` with `source: data/faqs.md` in agent.yaml, **When** the agent calls this tool with a query, **Then** the markdown file content is embedded and semantically similar results are returned
2. **Given** a vectorstore tool with `source: data/docs/` pointing to a directory, **When** the agent executes, **Then** all supported text files in the directory are loaded and embedded (supports: .txt, .md, .pdf, .csv, .json)
3. **Given** a vectorstore tool configured with any valid source (file or directory), **When** the agent executes a query, **Then** results are ranked by relevance score and returned in order with source file reference
4. **Given** a vectorstore tool with a source containing multiple documents, **When** querying for specific information, **Then** the most relevant content across all documents is returned with highest relevance scores

---

### User Story 2 - Custom Embedding Model Configuration (Priority: P2)

An agent developer specifies a custom embedding model to optimize for their specific use case (e.g., domain-specific embeddings, cost optimization, or performance requirements).

**Why this priority**: Provides flexibility and optimization but is not required for basic functionality. Users can start with default embeddings and customize later based on performance needs.

**Independent Test**: Can be tested by configuring `embedding_model: text-embedding-3-small` in the vectorstore tool definition, verifying embeddings are generated using the specified model, and comparing results against default model behavior.

**Acceptance Scenarios**:

1. **Given** a vectorstore tool with unstructured data and embedding_model specified (e.g., `text-embedding-3-small`), **When** the agent executes, **Then** embeddings are generated using the specified model
2. **Given** a vectorstore tool without an explicit embedding_model parameter, **When** the agent executes, **Then** embeddings are generated using the default model for the configured LLM provider
3. **Given** different embedding models configured across multiple vectorstore tools, **When** comparing search quality, **Then** each tool uses its specified embedding model independently

---

### User Story 3 - Vector Database Persistence (Priority: P3)

An agent developer configures persistent vector storage using Redis (via Semantic Kernel's vector store connector) to enable efficient search at scale and persistence across agent sessions.

**Why this priority**: Required for production deployments with large knowledge bases or multiple agent instances, but not needed for development/testing or small-scale deployments. Can be added later without changing the tool interface.

**Independent Test**: Can be tested by configuring a Redis vector database connection (using VectorstoreConfig with Semantic Kernel connector), ingesting test data, verifying data persists across agent restarts, and measuring search performance with large datasets.

**Acceptance Scenarios**:

1. **Given** a vectorstore tool with a configured vector database (Redis via Semantic Kernel connector), **When** the agent executes, **Then** the data source is ingested using the file_processor and stored in the vector database for search
2. **Given** a vectorstore tool with persisted embeddings in the database, **When** the agent restarts and source files are unchanged, **Then** search is available immediately without re-ingesting source files
3. **Given** a vectorstore tool with persisted embeddings, **When** a source file is modified, **Then** embeddings for that file are automatically regenerated on next agent execution
4. **Given** a vectorstore tool, **When** executed with `--force-ingest` flag, **Then** all source files are re-ingested and re-embedded regardless of modification timestamps
5. **Given** a vectorstore tool with no database configured, **When** the agent executes, **Then** embeddings are stored in-memory for the current session only

---

### Edge Cases

- What happens when a source file is empty or contains only whitespace? → **Skip with warning log**
- How does the system handle unsupported file types in a directory source? → **Skip with warning log, continue processing**
- What happens when a specified source file or directory doesn't exist? → **Fail with clear error message (FR-020)**
- How does the system handle file encoding issues (non-UTF8 files)? → **Skip with warning log, continue processing (FR-016)**
- What happens when an embedding model is specified but not available/supported? → **Fail with clear error message**
- How does the system handle very large files (>100MB)? → **Process with chunking (semantic_kernel.text.text_chunker)**
- What happens when the vector database connection fails or is unavailable? → **Fall back to in-memory storage (FR-011)**
- How does the system handle duplicate files or content across multiple files? → **Embed all instances (no deduplication)**
- What happens when query results have identical relevance scores? → **Maintain insertion order or filename sort**
- How does the system handle special characters or formatting in PDF/CSV files? → **Extract as plain text, skip if extraction fails with warning log**
- What happens when a directory source contains nested subdirectories? → **Process recursively (FR-019)**
- How does the system handle file permission errors when reading source files? → **Skip with warning log, continue processing**

## Clarifications

### Session 2025-11-23

- Q: When processing large documents for embedding, how should the system chunk the content? → A: Use semantic_kernel.text.text_chunker
- Q: How should the system handle errors when processing files (e.g., corrupted PDFs, encoding issues, unsupported files in directories)? → A: Skip individual failed files with warning logs and continue processing remaining files
- Q: What vector database backends should be supported, and how should connection configuration be specified in the YAML? → A: Use Semantic Kernel's built-in vector stores (Redis, in-memory only) with connection config objects (VectorstoreConfig from src/holodeck/models/config.py)
- Q: How many search results should be returned by default when querying the vector store, and should this be configurable? → A: Default top 5 results with optional top_k parameter and similarity threshold filter
- Q: When should embeddings be regenerated for existing documents (e.g., when switching embedding models or when source files are modified)? → A: Auto-regenerate when source file modified; manual regeneration when model changes; add --force-ingest flag to chat/test commands

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support vectorstore tools defined with `type: vectorstore` in agent configuration
- **FR-002**: System MUST accept both file and directory paths via the `source` parameter
- **FR-003**: System MUST support the following file formats for ingestion: .txt, .md, .pdf, .csv, .json
- **FR-004**: System MUST extract full text content from all supported file formats and chunk using semantic_kernel.text.text_chunker for embedding generation
- **FR-005**: System MUST generate embeddings for all ingested content using the configured embedding model
- **FR-006**: System MUST accept queries and return top 5 semantically similar results by default, with optional `top_k` parameter and `min_similarity_score` threshold filter
- **FR-007**: System MUST include source file references in all search results
- **FR-008**: System MUST support custom embedding model specification via `embedding_model` parameter
- **FR-009**: System MUST use provider-default embedding model when no explicit model is specified
- **FR-010**: System MUST support vector database storage via Semantic Kernel's Redis connector and in-memory store, configured using VectorstoreConfig objects
- **FR-011**: System MUST fall back to in-memory storage when no vector database is configured
- **FR-012**: System MUST process all supported files in a directory when source is a directory path
- **FR-013**: System MUST return search results ordered by descending relevance score, limited to top_k count (default 5) and filtered by min_similarity_score threshold if specified
- **FR-014**: System MUST validate source paths exist before attempting ingestion
- **FR-015**: System MUST handle file encoding detection and conversion to UTF-8
- **FR-016**: System MUST skip unsupported file types and failed file processing (corrupted files, encoding errors) in directory sources with warning-level logging, continuing to process remaining files
- **FR-017**: System MUST persist embeddings in configured vector database for reuse across sessions
- **FR-021**: System MUST track source file modification timestamps and automatically regenerate embeddings when files are modified
- **FR-022**: System MUST support `--force-ingest` flag in `holodeck chat` and `holodeck test` commands to force re-ingestion and re-embedding regardless of timestamps
- **FR-023**: System MUST require manual re-ingestion (via `--force-ingest`) when the embedding model is changed in configuration
- **FR-018**: System MUST detect and skip empty or whitespace-only files during ingestion
- **FR-019**: System MUST handle nested subdirectories when source is a directory path
- **FR-020**: System MUST provide clear error messages when source paths don't exist or are inaccessible

### Key Entities

- **VectorStore Tool**: Represents the configured vectorstore tool instance with source path (file or directory), optional embedding model, optional database connection, and optional search parameters (top_k, min_similarity_score)
- **Document**: Represents a single ingested file with extracted text content, metadata (file path, format, size, modification timestamp), and generated embeddings
- **Embedding**: Represents the vector representation of document content chunks (generated via semantic_kernel.text.text_chunker), created by the specified embedding model
- **Query Result**: Represents a search result containing matched document content, relevance score, and source file reference
- **File Processor**: Handles file format detection, content extraction, and encoding normalization across supported file types

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Agent developers can configure semantic search over single files or directories in under 5 lines of YAML configuration
- **SC-002**: Search queries return relevant results within 2 seconds for knowledge bases up to 1000 documents
- **SC-003**: 95% of semantically relevant content appears in top 5 search results
- **SC-004**: System successfully ingests and indexes all supported file formats without manual intervention
- **SC-005**: Vector database integrations maintain search performance with knowledge bases exceeding 10,000 documents
- **SC-006**: Search results include accurate source file attribution for 100% of returned content
- **SC-007**: 90% of users successfully implement vectorstore tools without consulting documentation beyond configuration examples
- **SC-008**: File processing handles common encoding issues (UTF-8, Latin-1, ASCII) without errors in 95% of cases
- **SC-009**: Directory sources with nested subdirectories are fully indexed with all supported files discovered automatically
