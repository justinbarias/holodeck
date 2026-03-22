# Utilities and Support API

HoloDeck provides a rich set of utility modules for template rendering, file processing, search, chunking, context generation, tool initialization, logging, validation, and more.

---

## Template Engine

Jinja2-based template rendering with restricted filters and YAML validation against the `AgentConfig` schema. Used by `holodeck init` to scaffold projects from built-in templates.

::: holodeck.lib.template_engine.TemplateRenderer
    options:
      docstring_style: google
      show_source: true

---

## File Processing

Multimodal file processor that converts Office documents, PDFs, images (OCR), CSV, and JSON into markdown for LLM consumption. Supports local and remote files with caching, page/sheet/range extraction, and configurable timeouts.

::: holodeck.lib.file_processor.SourceFile
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.file_processor.FileProcessor
    options:
      docstring_style: google
      show_source: true

---

## Search

Hybrid search combining semantic (vector) similarity with keyword (full-text) matching via Reciprocal Rank Fusion (RRF). Includes tiered strategy selection based on vector store provider capabilities.

### Hybrid Search

::: holodeck.lib.hybrid_search.SearchResult
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.hybrid_search.reciprocal_rank_fusion
    options:
      docstring_style: google
      show_source: true

### Keyword Search

::: holodeck.lib.keyword_search.KeywordSearchStrategy
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.keyword_search.KeywordDocument
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.keyword_search.KeywordSearchProvider
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.keyword_search.InMemoryBM25KeywordProvider
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.keyword_search.OpenSearchKeywordProvider
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.keyword_search.HybridSearchExecutor
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.keyword_search.get_keyword_search_strategy
    options:
      docstring_style: google
      show_source: true

---

## Chunking

Text splitting and structure-aware chunking for preparing documents for embedding generation and hierarchical retrieval.

### Text Chunker

Token-based text splitting using Semantic Kernel's paragraph splitter.

::: holodeck.lib.text_chunker.TextChunker
    options:
      docstring_style: google
      show_source: true

### Structured Chunker

Structure-aware markdown parsing with heading hierarchy extraction, chunk type classification, and parent chain building.

::: holodeck.lib.structured_chunker.SubsectionPattern
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.structured_chunker.ChunkType
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.structured_chunker.DocumentChunk
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.structured_chunker.StructuredChunker
    options:
      docstring_style: google
      show_source: true

### Structured Data Loader

Loads and iterates over structured data from CSV, JSON, and JSONL files with field mapping, schema validation, and batch processing for vector store ingestion.

::: holodeck.lib.structured_loader.StructuredDataLoader
    options:
      docstring_style: google
      show_source: true

---

## Context Generation

Implements the Anthropic contextual retrieval approach -- generating short context snippets for document chunks to improve semantic search retrieval by 35-49%.

### Claude SDK Context Generator

Uses the Claude Agent SDK `query()` function for contextual embeddings. Supports batched prompts with JSON array parsing and automatic single-chunk fallback.

::: holodeck.lib.claude_context_generator.ClaudeContextConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.claude_context_generator.ClaudeSDKContextGenerator
    options:
      docstring_style: google
      show_source: true

### LLM Context Generator

Uses Semantic Kernel chat completion services for contextual embeddings. Supports adaptive concurrency on rate limiting and exponential backoff retry logic.

::: holodeck.lib.llm_context_generator.RetryConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.llm_context_generator.LLMContextGenerator
    options:
      docstring_style: google
      show_source: true

---

## Tool Initialization

Shared tool initialization for VectorStoreTool and HierarchicalDocumentTool. Provider-agnostic: works for both SK and Claude backend paths.

::: holodeck.lib.tool_initializer.resolve_embedding_model
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.tool_initializer.create_embedding_service
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.tool_initializer.initialize_tools
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.tool_initializer.initialize_hierarchical_doc_tools
    options:
      docstring_style: google
      show_source: true

---

## Instruction Resolution

Resolves agent instructions from `Instructions` config objects, supporting both inline text and file-based instructions with base directory resolution.

::: holodeck.lib.instruction_resolver.resolve_instructions
    options:
      docstring_style: google
      show_source: true

---

## Vector Store

Unified interface for working with various vector storage backends through Semantic Kernel's VectorStoreCollection abstractions. Supports PostgreSQL (pgvector), Azure AI Search, Qdrant, Weaviate, ChromaDB, FAISS, Pinecone, and more.

::: holodeck.lib.vector_store.ChromaConnectionParams
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.vector_store.QdrantConnectionParams
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.vector_store.PineconeConnectionParams
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.vector_store.PostgresConnectionParams
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.vector_store.QueryResult
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.vector_store.StructuredQueryResult
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.vector_store.create_document_record_class
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.vector_store.create_structured_record_class
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.vector_store.create_hierarchical_document_record_class
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.vector_store.get_collection_class
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.vector_store.get_collection_factory
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.vector_store.parse_chromadb_connection_string
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.vector_store.parse_qdrant_connection_string
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.vector_store.parse_pinecone_connection_string
    options:
      docstring_style: google
      show_source: true

---

## Logging

Centralized logging configuration with support for console and file handlers, environment variable configuration, log rotation, and structured logging patterns.

### Logging Configuration

::: holodeck.lib.logging_config.setup_logging
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.logging_config.get_logger
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.logging_config.set_log_level
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.logging_config.configure_third_party_loggers
    options:
      docstring_style: google
      show_source: true

### Logging Utilities

::: holodeck.lib.logging_utils.LogTimer
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.logging_utils.log_operation
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.logging_utils.log_context
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.logging_utils.log_with_context
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.logging_utils.log_exception
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.logging_utils.log_retry
    options:
      docstring_style: google
      show_source: true

---

## Validation

Shared validation functions and constants for agent name validation, chat input validation, tool name sanitization, and tool output sanitization.

::: holodeck.lib.validation.validate_agent_name
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.validation.ValidationPipeline
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.validation.sanitize_tool_output
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.validation.sanitize_tool_name
    options:
      docstring_style: google
      show_source: true

---

## Chat History

Utilities for extracting tool information from agent execution results.

::: holodeck.lib.chat_history_utils.extract_tool_names
    options:
      docstring_style: google
      show_source: true

---

## Definition Extraction

Data structures for representing extracted definitions from documents. Definitions are key terms and their explanations used to enhance search results with contextual information.

::: holodeck.lib.definition_extractor.DefinitionEntry
    options:
      docstring_style: google
      show_source: true

---

## UI Utilities

Terminal detection, color output, and spinner animation utilities for the CLI layer.

### Colors

::: holodeck.lib.ui.colors.ANSIColors
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.ui.colors.colorize
    options:
      docstring_style: google
      show_source: true

### Spinner

::: holodeck.lib.ui.spinner.SpinnerMixin
    options:
      docstring_style: google
      show_source: true

### Terminal Detection

::: holodeck.lib.ui.terminal.is_tty
    options:
      docstring_style: google
      show_source: true

---

## Error Hierarchy

Exception classes for HoloDeck library operations. All exceptions inherit from `HoloDeckError`, enabling generic error handling with `except HoloDeckError`.

::: holodeck.lib.exceptions.HoloDeckError
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.exceptions.ValidationError
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.exceptions.InitError
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.exceptions.TemplateError
    options:
      docstring_style: google
      show_source: true
