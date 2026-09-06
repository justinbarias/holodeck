# Feature Specification: Structured Data Field Mapping and Ingestion

**Feature Branch**: `014-structured-data-ingestion`
**Created**: 2025-12-18
**Status**: Draft
**Input**: User description: "Create a spec for US 6.1.2 (Structured Data Field Mapping) and ingestion - enabling vectorstore tools to handle structured data sources with explicit field mapping"
**Parent Feature**: US 6.1 - Vector Search Tool Operations
**Sibling Feature**: US 6.1.1 - Unstructured Vector Ingestion and Search (008)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Semantic Search Over Structured Data with Field Mapping (Priority: P1)

An agent developer configures a vectorstore tool to enable semantic search over structured data sources (CSV, JSON) by defining which fields to embed and which fields to include as searchable metadata. The tool extracts data according to the field mapping, creates embeddings from the specified content fields, and enables query-based search that returns relevant records with their metadata.

**Why this priority**: This is the core MVP functionality - enabling semantic search over structured knowledge bases where specific fields contain the searchable content. It delivers immediate value by allowing agents to answer questions based on structured data (product catalogs, FAQs databases, knowledge bases) without requiring manual data transformation.

**Independent Test**: Can be fully tested by configuring a vectorstore tool pointing to a CSV file with `vector_field: "description"` and `metadata_fields: ["id", "title", "category"]`, running queries, and verifying semantically relevant records are returned with their metadata. Delivers value by enabling question-answering over product catalogs, FAQ databases, or any structured knowledge source.

**Acceptance Scenarios**:

1. **Given** a vectorstore tool with `source: data/products.csv` and `vector_field: description` in agent.yaml, **When** the agent calls this tool with a query, **Then** the description field content from each row is embedded and semantically similar records are returned with their metadata
2. **Given** a vectorstore tool with `source: data/faqs.json` and `vector_fields: [question, answer]` configured, **When** the agent executes a query, **Then** both question and answer fields are concatenated and embedded for each record
3. **Given** a vectorstore tool with `metadata_fields: [id, title, category]` configured, **When** search results are returned, **Then** each result includes the specified metadata fields alongside the matched content
4. **Given** a vectorstore tool configured with structured data source, **When** the agent executes a query, **Then** results are ranked by relevance score and returned in order with full record metadata

---

### User Story 2 - Multiple Vector Fields for Richer Embeddings (Priority: P2)

An agent developer configures a vectorstore tool with multiple fields to embed (e.g., title, description, tags), allowing the system to create richer semantic representations by combining content from multiple fields into a single embedding.

**Why this priority**: Provides enhanced search quality for complex structured data but is not required for basic functionality. Users can start with single-field embeddings and combine fields later based on search quality needs.

**Independent Test**: Can be tested by configuring `vector_fields: [title, description, tags]`, verifying that all fields are concatenated and embedded together for each record.

**Acceptance Scenarios**:

1. **Given** a vectorstore tool with `vector_fields: [title, description]` configured, **When** the agent executes, **Then** both fields are concatenated and embedded as a single representation for each record
2. **Given** a vectorstore tool with vector_fields configured, **When** a record has missing/null values for some fields, **Then** available fields are still concatenated and embedded, and the record is searchable
3. **Given** a vectorstore tool with vector_fields and a custom `field_separator` configured, **When** fields are concatenated, **Then** the specified separator is used between field values

---

### User Story 3 - Database Source Integration (Priority: P3)

An agent developer configures a vectorstore tool to connect to a database (SQLite, PostgreSQL) and extract records based on a query, enabling semantic search over large-scale structured data that doesn't fit in files.

**Why this priority**: Required for enterprise deployments with large-scale data in databases, but not needed for development/testing or file-based data sources. Can be added later without changing the core field mapping interface.

**Independent Test**: Can be tested by configuring a database connection string and SQL query, verifying data is extracted, field mappings are applied, and semantic search works over the extracted records.

**Acceptance Scenarios**:

1. **Given** a vectorstore tool with `source: database` and `connection_string: sqlite:///data/products.db` configured, **When** the agent executes, **Then** records are extracted using the specified query and embedded according to field mapping
2. **Given** a vectorstore tool with database source and `query: "SELECT id, title, description FROM products WHERE active = 1"`, **When** the agent executes, **Then** only matching records are ingested and embedded
3. **Given** a vectorstore tool with database source, **When** the underlying database data changes, **Then** the system detects changes on re-ingestion and updates embeddings for modified records
4. **Given** a vectorstore tool with database source, **When** connection fails, **Then** a clear error message is provided with connection troubleshooting guidance

---

### Edge Cases

- What happens when a specified vector_field doesn't exist in the data source? **Fail with clear error message listing available fields**
- How does the system handle records with null/empty values in vector_field? **Skip record with warning log, continue processing**
- What happens when metadata_fields reference non-existent columns? **Fail with clear error message listing available fields**
- How does the system handle CSV files with different delimiters? **Auto-detect common delimiters (comma, semicolon, tab) or accept delimiter parameter**
- What happens when JSON data is nested and field path is required? **Support dot notation for nested fields (e.g., "details.description")**
- How does the system handle very large structured files (>1M rows)? **Process in batches with progress logging**
- What happens when both vector_field (singular) and vector_fields (plural) are specified? **Fail validation with clear error - mutually exclusive**
- How does the system handle records with mixed data types in vector fields? **Convert all values to string for embedding**
- What happens when database query returns no results? **Complete successfully with warning log, no embeddings created**
- How does the system handle special characters in field names? **Support escaped field names or bracket notation**

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support vectorstore tools with structured data sources defined by `type: vectorstore` with `source_type: structured` in agent configuration
- **FR-002**: System MUST accept CSV, JSON, and JSONL file formats as structured data sources
- **FR-003**: System MUST support `vector_field` (singular) parameter to specify a single field for embedding
- **FR-004**: System MUST support `vector_fields` (plural) parameter to specify multiple fields for embedding
- **FR-005**: System MUST validate that vector_field and vector_fields are mutually exclusive
- **FR-006**: System MUST support optional `metadata_fields` parameter to specify fields included in search results; when omitted, all fields are included as metadata
- **FR-006a**: System MUST require `id_field` parameter to specify the unique record identifier field for incremental re-ingestion and deduplication
- **FR-007**: System MUST validate that specified fields exist in the data source schema
- **FR-008**: System MUST concatenate multiple vector_fields with a configurable separator (default: newline) before embedding
- **FR-009**: System MUST return all metadata_fields values in search results alongside matched content
- **FR-012**: System MUST skip records with null/empty values in all vector fields with warning-level logging
- **FR-013**: System MUST support dot notation for nested JSON field paths (e.g., "details.description")
- **FR-013a**: System MUST support `record_path` parameter for extracting records from deeply nested JSON structures (e.g., "data.results.items", "batches[0].records")
- **FR-013b**: System MUST support array indexing in record_path (e.g., "[0]", "[2]") for accessing specific array elements
- **FR-013c**: System MUST provide clear error messages when record_path navigation fails, including available keys at each traversal level
- **FR-014**: System MUST auto-detect CSV delimiters or accept optional `delimiter` parameter
- **FR-015**: System MUST convert non-string values in vector fields to string representations
- **FR-016**: System MUST process large files in batches to manage memory usage
- **FR-017**: System MUST support database sources via Semantic Kernel's existing vector store providers (postgres, sql-server, etc.) without requiring additional ORM libraries
- **FR-018**: System MUST validate database connection using Semantic Kernel's connector validation before attempting data extraction
- **FR-019**: System MUST leverage Semantic Kernel's 11+ supported vector store providers for database integration (ChromaDB, Qdrant, PostgreSQL, Pinecone, etc.)
- **FR-020**: System MUST provide clear error messages when fields are not found, including list of available fields
- **FR-021**: System MUST track source file/database modification state and support incremental re-ingestion
- **FR-022**: System MUST support the same embedding model configuration as unstructured vectorstore tools (embedding_model parameter)
- **FR-023**: System MUST integrate with existing vector database persistence (Redis, in-memory) from US 6.1.1

### Key Entities

- **Structured VectorStore Tool**: Extends VectorStore Tool with field mapping configuration (vector_field/vector_fields, metadata_fields, id_field, source_type)
- **Field Mapping**: Defines which data fields to embed (vector_field/vector_fields), which to include as metadata (metadata_fields), and the unique record identifier (id_field)
- **Structured Record**: Represents a single row/document from the structured source with all fields extracted according to the schema
- **Embedding**: Represents the vector representation of combined vector field content, created by the specified embedding model
- **Query Result**: Represents a search result containing matched content, relevance score, and all specified metadata fields

## Clarifications

### Session 2025-12-18

- Q: How should individual records be uniquely identified for incremental re-ingestion and deduplication? → A: Require explicit `id_field` parameter in configuration
- Q: Should field_weights be supported for weighted multi-field search? → A: No, remove weights functionality to simplify scope; multi-field embeddings use simple concatenation
- Q: What is the default behavior when metadata_fields is not specified? → A: Include all fields as metadata when not specified
- Q: Should we use SQLAlchemy or another ORM for database source integration? → A: No additional ORMs required. Use Semantic Kernel's existing vector store abstractions which already support postgres, sql-server, and 11+ other providers
- Q: How should deeply nested JSON files be handled? → A: Support robust `record_path` parameter with dot notation and array indexing (e.g., "data.results.items", "batches[0].records", "response.data[2].entries"). Provide clear error messages with available keys at each traversal level when navigation fails

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Agent developers can configure semantic search over structured data in under 8 lines of YAML configuration
- **SC-002**: Search queries return relevant results within 2 seconds for structured sources up to 100,000 records
- **SC-003**: 95% of semantically relevant records appear in top 5 search results when appropriate vector_field is configured
- **SC-004**: System successfully ingests CSV, JSON, and JSONL files without manual format configuration
- **SC-005**: Field mapping validation catches 100% of misconfigured field names before ingestion begins
- **SC-006**: Search results include all specified metadata fields for 100% of returned records
- **SC-007**: 90% of users successfully implement structured vectorstore tools without consulting documentation beyond configuration examples
- **SC-008**: Multi-field embedding improves search recall by combining relevant content from multiple fields into richer semantic representations
- **SC-009**: Database source ingestion handles tables with 1M+ records without memory errors
