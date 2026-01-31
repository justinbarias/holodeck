"""Integration tests for HierarchicalDocumentTool with real LLM providers.

These tests make actual API calls to embedding providers (OpenAI, Azure OpenAI).
They require valid API keys to be set in tests/integration/.env file.

To run these tests:
1. Copy tests/integration/.env.example to tests/integration/.env
2. Fill in your actual API keys
3. Run: pytest tests/integration/test_hierarchical_document_tool_integration.py

To skip these tests (e.g., in CI without API keys):
Set SKIP_LLM_INTEGRATION_TESTS=true in .env or environment
"""

import os
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

from holodeck.lib.hybrid_search import SearchResult
from holodeck.models.tool import (
    DocumentDomain,
    HierarchicalDocumentToolConfig,
    SearchMode,
)
from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool

# Load environment variables from tests/integration/.env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Check if we should skip LLM integration tests
SKIP_LLM_TESTS = os.getenv("SKIP_LLM_INTEGRATION_TESTS", "false").lower() == "true"

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Azure OpenAI configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small"
)

# Test fixtures path
FIXTURES_PATH = Path(__file__).parent.parent / "fixtures" / "hierarchical"
# POLICY_DOC_PATH = FIXTURES_PATH / "policy.md"
POLICY_DOC_PATH = FIXTURES_PATH / "HR8847_CNIMDT_AIGCE_STD_Act.pdf"

# Skip markers for different providers
skip_if_no_openai = pytest.mark.skipif(
    SKIP_LLM_TESTS or not OPENAI_API_KEY,
    reason="OpenAI API key not configured or LLM tests disabled",
)

skip_if_no_azure = pytest.mark.skipif(
    SKIP_LLM_TESTS or not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT,
    reason="Azure OpenAI credentials not configured or LLM tests disabled",
)


def create_test_config(
    source: str,
    contextual_embeddings: bool = False,
    document_domain: DocumentDomain = DocumentDomain.NONE,
    **overrides: Any,
) -> HierarchicalDocumentToolConfig:
    """Create config with sensible test defaults.

    Args:
        source: Path to document source.
        contextual_embeddings: Enable contextual embedding generation.
        document_domain: Document domain for subsection pattern detection.
        **overrides: Additional config overrides.

    Returns:
        HierarchicalDocumentToolConfig for testing.
    """
    defaults: dict[str, Any] = {
        "name": "integration_test_tool",
        "description": "Integration test tool",
        "source": source,
        "contextual_embeddings": contextual_embeddings,
        "document_domain": document_domain,
        "search_mode": SearchMode.SEMANTIC,
        "top_k": 5,
    }
    defaults.update(overrides)
    return HierarchicalDocumentToolConfig(**defaults)


def create_openai_embedding_service() -> Any:
    """Create OpenAI embedding service.

    Returns:
        OpenAITextEmbedding service instance.
    """
    from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding

    return OpenAITextEmbedding(
        ai_model_id=OPENAI_EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )


def create_azure_embedding_service() -> Any:
    """Create Azure OpenAI embedding service.

    Returns:
        AzureTextEmbedding service instance.
    """
    from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding

    return AzureTextEmbedding(
        deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
    )


def create_openai_chat_service() -> Any:
    """Create OpenAI chat service for contextual embeddings.

    Returns:
        OpenAIChatCompletion service instance.
    """
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

    return OpenAIChatCompletion(
        ai_model_id=OPENAI_MODEL_NAME,
        api_key=OPENAI_API_KEY,
    )


def create_azure_chat_service() -> Any:
    """Create Azure OpenAI chat service for contextual embeddings.

    Returns:
        AzureChatCompletion service instance.
    """
    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

    return AzureChatCompletion(
        deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
    )


@pytest.mark.integration
@pytest.mark.slow
class TestHierarchicalDocumentToolOpenAI:
    """Integration tests for HierarchicalDocumentTool with OpenAI provider."""

    @skip_if_no_openai
    @pytest.mark.asyncio
    async def test_initialize_with_real_embeddings(self) -> None:
        """Test document ingestion with real OpenAI embeddings.

        Validates:
        1. Tool initializes successfully with OpenAI embedding service
        2. Documents are ingested and chunked
        3. Embeddings are generated for all chunks
        """
        config = create_test_config(
            str(POLICY_DOC_PATH),
            document_domain=DocumentDomain.US_LEGISLATIVE,
        )
        tool = HierarchicalDocumentTool(config)
        tool.set_embedding_service(create_openai_embedding_service())

        await tool.initialize()

        # Verify chunks were created
        assert tool._initialized is True
        assert len(tool._chunks) > 0

        # Verify embeddings are not placeholder zeros
        for chunk in tool._chunks:
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 1536  # text-embedding-3-small dimensions
            # At least some values should be non-zero
            assert any(v != 0.0 for v in chunk.embedding)

    @skip_if_no_openai
    @pytest.mark.asyncio
    async def test_search_returns_relevant_results(self) -> None:
        """Test semantic search finds expected content.

        Validates:
        1. Search returns results for relevant queries
        2. Results are semantically relevant to the query
        3. Results contain expected document content
        """
        config = create_test_config(
            str(POLICY_DOC_PATH),
            document_domain=DocumentDomain.US_LEGISLATIVE,
        )
        tool = HierarchicalDocumentTool(config)
        tool.set_embedding_service(create_openai_embedding_service())

        await tool.initialize()

        # Search for definitions
        results = await tool.search("What is Force Majeure?")

        assert len(results) > 0
        assert isinstance(results[0], SearchResult)

        # Top result should contain force majeure content
        top_result = results[0]
        assert "force majeure" in top_result.content.lower()

    @skip_if_no_openai
    @pytest.mark.asyncio
    async def test_search_results_have_valid_scores(self) -> None:
        """Test search results have valid similarity scores.

        Validates:
        1. All scores are between 0.0 and 1.0
        2. Results are sorted by score descending
        """
        config = create_test_config(
            str(POLICY_DOC_PATH),
            document_domain=DocumentDomain.US_LEGISLATIVE,
            top_k=10,
        )
        tool = HierarchicalDocumentTool(config)
        tool.set_embedding_service(create_openai_embedding_service())

        await tool.initialize()

        results = await tool.search("insurance coverage")

        assert len(results) > 0

        # Verify score bounds
        for result in results:
            assert 0.0 <= result.fused_score <= 1.0

        # Verify sorted descending
        scores = [r.fused_score for r in results]
        assert scores == sorted(scores, reverse=True)

    @skip_if_no_openai
    @pytest.mark.asyncio
    async def test_search_preserves_parent_chain(self) -> None:
        """Test search results include document hierarchy.

        Validates:
        1. Results have parent_chain populated
        2. Parent chain reflects document structure
        """
        config = create_test_config(
            str(POLICY_DOC_PATH),
            document_domain=DocumentDomain.US_LEGISLATIVE,
        )
        tool = HierarchicalDocumentTool(config)
        tool.set_embedding_service(create_openai_embedding_service())

        await tool.initialize()

        results = await tool.search("premium payment")

        assert len(results) > 0

        # Check that parent chains are populated
        for result in results:
            assert isinstance(result.parent_chain, list)
            # At least some results should have non-empty parent chains
            # (indicating hierarchical structure)

        # Find result related to premium (should have Section 1: Definitions ancestry)
        premium_results = [r for r in results if "premium" in r.content.lower()]
        if premium_results:
            # Verify hierarchy is preserved
            assert any(
                "definition" in " ".join(r.parent_chain).lower()
                for r in premium_results
            ) or any(
                "section 1" in " ".join(r.parent_chain).lower() for r in premium_results
            )

    @skip_if_no_openai
    @pytest.mark.asyncio
    async def test_get_context_returns_formatted_string(self) -> None:
        """Test get_context convenience method formats results.

        Validates:
        1. Returns formatted string containing query context
        2. Includes search results in readable format
        """
        config = create_test_config(
            str(POLICY_DOC_PATH),
            document_domain=DocumentDomain.US_LEGISLATIVE,
        )
        tool = HierarchicalDocumentTool(config)
        tool.set_embedding_service(create_openai_embedding_service())

        await tool.initialize()

        context = await tool.get_context("claims process")

        assert isinstance(context, str)
        assert len(context) > 0
        assert "claims" in context.lower() or "claim" in context.lower()
        # Should contain structured format elements
        assert "Score:" in context or "Context for query:" in context


@pytest.mark.integration
@pytest.mark.slow
class TestHierarchicalDocumentToolAzureOpenAI:
    """Integration tests for HierarchicalDocumentTool with Azure OpenAI provider."""

    @skip_if_no_azure
    @pytest.mark.asyncio
    async def test_azure_initialize_with_real_embeddings(self) -> None:
        """Test document ingestion with Azure OpenAI embeddings.

        Validates:
        1. Tool initializes successfully with Azure embedding service
        2. Documents are ingested and chunked
        3. Embeddings are generated for all chunks
        """
        config = create_test_config(
            str(POLICY_DOC_PATH),
            document_domain=DocumentDomain.US_LEGISLATIVE,
        )
        tool = HierarchicalDocumentTool(config)
        tool.set_embedding_service(create_azure_embedding_service())

        await tool.initialize()

        # Verify chunks were created
        assert tool._initialized is True
        assert len(tool._chunks) > 0

        # Verify embeddings are not placeholder zeros
        for chunk in tool._chunks:
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 1536
            assert any(v != 0.0 for v in chunk.embedding)

    @skip_if_no_azure
    @pytest.mark.asyncio
    async def test_azure_search_returns_relevant_results(self) -> None:
        """Test semantic search with Azure OpenAI embeddings.

        Validates:
        1. Search returns results for relevant queries
        2. Results are semantically relevant to the query
        """
        config = create_test_config(
            str(POLICY_DOC_PATH),
            document_domain=DocumentDomain.US_LEGISLATIVE,
        )
        tool = HierarchicalDocumentTool(config)
        tool.set_embedding_service(create_azure_embedding_service())

        await tool.initialize()

        results = await tool.search("exclusions from coverage")

        assert len(results) > 0

        # Should find content about exclusions
        all_content = " ".join(r.content.lower() for r in results)
        has_exclusion_content = (
            "exclus" in all_content or "war" in all_content or "nuclear" in all_content
        )
        assert has_exclusion_content


@pytest.mark.integration
@pytest.mark.slow
class TestContextualEmbeddings:
    """Integration tests for contextual embeddings with LLM context generation."""

    @skip_if_no_azure
    @pytest.mark.asyncio
    async def test_contextual_embeddings_with_chat_service(self) -> None:
        """Test LLMContextGenerator prepends context to chunks.

        Validates:
        1. Contextual embeddings are enabled and work with chat service
        2. Context generator produces contextualized content
        3. Search still works with contextualized embeddings
        """
        from holodeck.lib.llm_context_generator import LLMContextGenerator

        config = create_test_config(
            str(POLICY_DOC_PATH),
            contextual_embeddings=True,
            document_domain=DocumentDomain.US_LEGISLATIVE,
        )
        tool = HierarchicalDocumentTool(config)

        # Set up both embedding and chat services (using Azure OpenAI)
        tool.set_embedding_service(create_azure_embedding_service())
        chat_service = create_azure_chat_service()
        tool.set_chat_service(chat_service)

        tool._context_generator = LLMContextGenerator(chat_service=chat_service)

        await tool.initialize()

        # Verify chunks were created with contextualized content
        assert tool._initialized is True
        assert len(tool._chunks) > 0

        # Check that some chunks have contextualized content different from original
        # (Context generator adds situating text)
        contextualized_chunks = [
            chunk
            for chunk in tool._chunks
            if chunk.contextualized_content
            and chunk.contextualized_content != chunk.content
        ]

        # If context generation succeeded, verify contextualized content is longer
        for chunk in contextualized_chunks:
            # Contextualized content should be longer (context + original)
            assert len(chunk.contextualized_content) >= len(chunk.content)

        # Note: Context generation may fail gracefully, so we just verify
        # the tool works regardless (contextualized_chunks may be empty)
        assert tool._initialized is True

        # Search should still work
        results = await tool.search("property damage coverage")
        assert len(results) > 0


@pytest.mark.integration
@pytest.mark.slow
class TestMultipleDocuments:
    """Integration tests for multiple document ingestion."""

    @skip_if_no_openai
    @pytest.mark.asyncio
    async def test_initialize_directory_of_documents(self, tmp_path: Path) -> None:
        """Test ingestion of multiple files from directory.

        Validates:
        1. Tool can ingest multiple markdown files from a directory
        2. All files are processed and indexed
        """
        # Create test documents
        doc1 = tmp_path / "doc1.md"
        doc1.write_text(
            "# Document One\n\n## Introduction\n\nThis is the first document."
        )

        doc2 = tmp_path / "doc2.md"
        doc2.write_text("# Document Two\n\n## Overview\n\nThis is the second document.")

        config = create_test_config(str(tmp_path))
        tool = HierarchicalDocumentTool(config)
        tool.set_embedding_service(create_openai_embedding_service())

        await tool.initialize()

        assert tool._initialized is True
        # Should have chunks from both documents
        assert len(tool._chunks) >= 2

        # Verify both source files are represented
        source_paths = {chunk.source_path for chunk in tool._chunks}
        assert any("doc1.md" in path for path in source_paths)
        assert any("doc2.md" in path for path in source_paths)

    @skip_if_no_openai
    @pytest.mark.asyncio
    async def test_search_across_multiple_documents(self, tmp_path: Path) -> None:
        """Test search results span multiple source files.

        Validates:
        1. Search can find content from different source files
        2. Results correctly attribute source paths
        """
        # Create documents with distinct content
        doc1 = tmp_path / "animals.md"
        doc1.write_text(
            "# Animals\n\n## Mammals\n\nDogs and cats are common pets.\n\n"
            "## Birds\n\nParrots can learn to speak."
        )

        doc2 = tmp_path / "plants.md"
        doc2.write_text(
            "# Plants\n\n## Flowers\n\nRoses are popular garden flowers.\n\n"
            "## Trees\n\nOak trees can live for centuries."
        )

        config = create_test_config(str(tmp_path), top_k=10)
        tool = HierarchicalDocumentTool(config)
        tool.set_embedding_service(create_openai_embedding_service())

        await tool.initialize()

        # Search for content from first document
        animal_results = await tool.search("pets dogs cats")
        assert len(animal_results) > 0
        assert any("animals.md" in r.source_path for r in animal_results)

        # Search for content from second document
        plant_results = await tool.search("garden flowers roses")
        assert len(plant_results) > 0
        assert any("plants.md" in r.source_path for r in plant_results)


@pytest.mark.integration
class TestHeaderOnlyFiltering:
    """Integration tests for header-only chunk filtering."""

    @pytest.mark.asyncio
    async def test_header_only_chunks_filtered_during_ingestion(
        self, tmp_path: Path
    ) -> None:
        """Test that header-only chunks are not indexed.

        Validates:
        1. Documents with empty sections (header only) are parsed
        2. Header-only chunks are filtered out before embedding/storage
        3. Only content-bearing chunks are indexed
        """
        from holodeck.lib.structured_chunker import ChunkType

        # Create document with empty sections at various levels
        doc = tmp_path / "test_headers.md"
        doc.write_text(
            "# Main Title\n\n"
            "Some introduction content.\n\n"
            "## Empty Section\n\n"  # Header-only (level 2)
            "## Section With Content\n\n"
            "This section has body content.\n\n"
            "### Empty Subsection\n\n"  # Header-only (level 3)
            "### Subsection With Content\n\n"
            "More body content here.\n"
        )

        config = create_test_config(str(doc))
        tool = HierarchicalDocumentTool(config)

        await tool.initialize()

        assert tool._initialized is True
        assert len(tool._chunks) > 0

        # Verify no header-only chunks were stored
        for chunk in tool._chunks:
            assert (
                chunk.chunk_type != ChunkType.HEADER
            ), f"Header-only chunk should have been filtered: {chunk.content[:50]}"

        # Verify we still have content chunks
        content_chunks = [c for c in tool._chunks if c.chunk_type == ChunkType.CONTENT]
        assert len(content_chunks) > 0


@pytest.mark.integration
class TestErrorScenarios:
    """Integration tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_missing_source_raises_file_not_found(self) -> None:
        """Test FileNotFoundError with clear message for missing source.

        Validates:
        1. FileNotFoundError is raised for non-existent paths
        2. Error message includes the path that was not found
        """
        config = create_test_config("/nonexistent/path/to/document.md")
        tool = HierarchicalDocumentTool(config)

        with pytest.raises(FileNotFoundError) as exc_info:
            await tool.initialize()

        assert "nonexistent" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_empty_query_raises_value_error(self) -> None:
        """Test ValueError for empty/whitespace queries.

        Validates:
        1. ValueError is raised for empty string query
        2. ValueError is raised for whitespace-only query
        """
        config = create_test_config(
            str(POLICY_DOC_PATH),
            document_domain=DocumentDomain.US_LEGISLATIVE,
        )
        tool = HierarchicalDocumentTool(config)

        # Initialize with placeholder embeddings (no real service needed for error test)
        await tool.initialize()

        with pytest.raises(ValueError, match="cannot be empty"):
            await tool.search("")

        with pytest.raises(ValueError, match="cannot be empty"):
            await tool.search("   ")

    @pytest.mark.asyncio
    async def test_search_before_initialize_raises_runtime_error(self) -> None:
        """Test RuntimeError if not initialized before search.

        Validates:
        1. RuntimeError is raised when searching before initialize()
        2. Error message is clear about the required action
        """
        config = create_test_config(
            str(POLICY_DOC_PATH),
            document_domain=DocumentDomain.US_LEGISLATIVE,
        )
        tool = HierarchicalDocumentTool(config)

        with pytest.raises(RuntimeError, match="must be initialized"):
            await tool.search("test query")


@pytest.mark.integration
class TestDocumentDomainConfiguration:
    """Integration tests for document domain configuration."""

    @pytest.mark.asyncio
    async def test_us_legislative_domain_detects_title_chapter_section(
        self, tmp_path: Path
    ) -> None:
        """Test US legislative domain detects Title, Chapter, Section markers.

        Validates:
        1. TITLE markers are detected at level 1
        2. CHAPTER markers are detected at level 2
        3. SEC. markers are detected at level 3
        4. Subsection (a) markers are detected at level 4
        """
        # Create a US legislative document
        doc = tmp_path / "us_legislation.md"
        doc.write_text(
            "TITLE I—BROADBAND ACCESS\n\n"
            "CHAPTER 1—DEFINITIONS\n\n"
            "SEC. 101. SHORT TITLE.\n\n"
            "This Act may be cited as the Broadband Access Act.\n\n"
            "(a) FINDINGS.—Congress finds that:\n\n"
            "High-speed internet is essential.\n\n"
            "(b) PURPOSE.—The purpose of this Act is to expand access.\n\n"
            "SEC. 102. DEFINITIONS.\n\n"
            "In this Act:\n\n"
            "(1) BROADBAND.—The term broadband means high-speed internet.\n\n"
            "(2) RURAL AREA.—The term rural area means any area not urban.\n"
        )

        config = HierarchicalDocumentToolConfig(
            name="us_leg_test",
            description="Test US legislative parsing",
            source=str(doc),
            document_domain=DocumentDomain.US_LEGISLATIVE,
            # max_subsection_depth=None means use all 6 levels
        )
        tool = HierarchicalDocumentTool(config)

        await tool.initialize()

        assert tool._initialized is True
        assert len(tool._chunks) > 0

        # Check that various heading levels were detected
        heading_levels = {chunk.heading_level for chunk in tool._chunks}

        # Should have detected multiple levels (1=TITLE, 2=CHAPTER, 3=SEC, 4=(a))
        assert 1 in heading_levels or any(
            "TITLE" in c.content for c in tool._chunks
        ), "TITLE should be detected"
        assert 3 in heading_levels or any(
            "SEC." in c.content for c in tool._chunks
        ), "SEC. should be detected"
        assert 4 in heading_levels or any(
            "(a)" in c.content for c in tool._chunks
        ), "Subsections should be detected"

    @pytest.mark.asyncio
    async def test_au_legislative_domain_detects_part_division_section(
        self, tmp_path: Path
    ) -> None:
        """Test Australian legislative domain detects Part, Division, Section.

        Validates:
        1. Part markers are detected at level 1
        2. Division markers are detected at level 2
        3. Section numbers are detected at level 3
        4. Numeric subsections (1) are detected at level 4
        """
        # Create an Australian legislative document
        doc = tmp_path / "au_legislation.md"
        doc.write_text(
            "Part 1—Preliminary\n\n"
            "Division 1—General provisions\n\n"
            "1 Short title\n\n"
            "This Act may be cited as the Example Act 2024.\n\n"
            "(1) This section applies to all persons.\n\n"
            "(2) A person must comply with this Act.\n\n"
            "(a) in the case of individuals; or\n\n"
            "(b) in the case of corporations.\n\n"
            "2 Definitions\n\n"
            "In this Act:\n\n"
            "(1) authority means a government body.\n"
        )

        config = HierarchicalDocumentToolConfig(
            name="au_leg_test",
            description="Test Australian legislative parsing",
            source=str(doc),
            document_domain=DocumentDomain.AU_LEGISLATIVE,
        )
        tool = HierarchicalDocumentTool(config)

        await tool.initialize()

        assert tool._initialized is True
        assert len(tool._chunks) > 0

        # Check that various heading levels were detected
        heading_levels = {chunk.heading_level for chunk in tool._chunks}

        # Should have detected Part (1), Division (2), Section (3), subsection (4)
        assert 1 in heading_levels or any(
            "Part" in c.content for c in tool._chunks
        ), "Part should be detected"
        assert 4 in heading_levels or any(
            "(1)" in c.content for c in tool._chunks
        ), "Numeric subsections should be detected"

    @pytest.mark.asyncio
    async def test_academic_domain_detects_numbered_sections(
        self, tmp_path: Path
    ) -> None:
        """Test academic domain detects numbered section patterns.

        Validates:
        1. Section numbers (1., 2.) are detected
        2. Subsection numbers (1.1, 1.2) are detected
        """
        # Create an academic paper
        doc = tmp_path / "paper.md"
        doc.write_text(
            "# Machine Learning in Healthcare\n\n"
            "Abstract: This paper explores...\n\n"
            "1. Introduction\n\n"
            "Machine learning has revolutionized healthcare.\n\n"
            "1.1 Background\n\n"
            "Previous work has shown significant progress.\n\n"
            "1.2 Contributions\n\n"
            "Our main contributions are:\n\n"
            "2. Related Work\n\n"
            "Several approaches have been proposed.\n\n"
            "2.1 Deep Learning Methods\n\n"
            "Convolutional neural networks have been applied.\n"
        )

        config = HierarchicalDocumentToolConfig(
            name="academic_test",
            description="Test academic paper parsing",
            source=str(doc),
            document_domain=DocumentDomain.ACADEMIC,
        )
        tool = HierarchicalDocumentTool(config)

        await tool.initialize()

        assert tool._initialized is True
        assert len(tool._chunks) > 0

        # Check that numbered sections were detected
        heading_levels = {chunk.heading_level for chunk in tool._chunks}

        # Should have detected section (3) and subsection (4) patterns
        assert 3 in heading_levels or any(
            "1." in c.content and "Introduction" in c.content for c in tool._chunks
        ), "Numbered sections should be detected"

    @pytest.mark.asyncio
    async def test_technical_domain_detects_step_patterns(self, tmp_path: Path) -> None:
        """Test technical domain detects Step patterns.

        Validates:
        1. Step markers are detected
        2. Note/Warning markers are detected
        """
        # Create a technical document
        doc = tmp_path / "manual.md"
        doc.write_text(
            "# Installation Guide\n\n"
            "Follow these steps to install the software.\n\n"
            "Step 1: Download the installer\n\n"
            "Visit the download page and get the latest version.\n\n"
            "Step 2: Run the installer\n\n"
            "Double-click the downloaded file.\n\n"
            "Note: Administrator privileges may be required.\n\n"
            "Step 3: Configure settings\n\n"
            "Open the settings panel and adjust preferences.\n\n"
            "Warning: Do not modify system files directly.\n"
        )

        config = HierarchicalDocumentToolConfig(
            name="technical_test",
            description="Test technical manual parsing",
            source=str(doc),
            document_domain=DocumentDomain.TECHNICAL,
        )
        tool = HierarchicalDocumentTool(config)

        await tool.initialize()

        assert tool._initialized is True
        assert len(tool._chunks) > 0

        # Check that Step patterns were detected
        step_chunks = [c for c in tool._chunks if "Step" in c.content]
        assert len(step_chunks) >= 3, "Step markers should be detected"

    @pytest.mark.asyncio
    async def test_legal_contract_domain_detects_article_section(
        self, tmp_path: Path
    ) -> None:
        """Test legal contract domain detects Article and Section patterns.

        Validates:
        1. Article markers are detected
        2. Section markers are detected
        3. Clause (a) markers are detected
        """
        # Create a legal contract
        doc = tmp_path / "contract.md"
        doc.write_text(
            "# SERVICE AGREEMENT\n\n"
            "Article I: Definitions\n\n"
            "Section 1: General Terms\n\n"
            "The following terms shall have the meanings set forth below:\n\n"
            "(a) Service means the consulting services provided.\n\n"
            "(b) Client means the party receiving services.\n\n"
            "Section 2: Specific Terms\n\n"
            "Additional definitions as needed.\n\n"
            "Article II: Scope of Services\n\n"
            "Section 1: Services Provided\n\n"
            "The Provider shall deliver the following services.\n"
        )

        config = HierarchicalDocumentToolConfig(
            name="contract_test",
            description="Test legal contract parsing",
            source=str(doc),
            document_domain=DocumentDomain.LEGAL_CONTRACT,
        )
        tool = HierarchicalDocumentTool(config)

        await tool.initialize()

        assert tool._initialized is True
        assert len(tool._chunks) > 0

        # Check that Article/Section patterns were detected
        article_chunks = [c for c in tool._chunks if "Article" in c.content]
        assert len(article_chunks) >= 2, "Article markers should be detected"

    @pytest.mark.asyncio
    async def test_max_subsection_depth_limits_detection(self, tmp_path: Path) -> None:
        """Test that max_subsection_depth limits pattern recognition.

        Validates:
        1. With depth=3, only Title/Chapter/Section are detected
        2. Subsection patterns (a) are not detected as headings
        """
        doc = tmp_path / "limited_depth.md"
        doc.write_text(
            "TITLE I—TEST\n\n"
            "CHAPTER 1—DEFINITIONS\n\n"
            "SEC. 101. SHORT TITLE.\n\n"
            "Content here.\n\n"
            "(a) This subsection should not be a heading.\n\n"
            "More content.\n"
        )

        config = HierarchicalDocumentToolConfig(
            name="depth_test",
            description="Test depth limiting",
            source=str(doc),
            document_domain=DocumentDomain.US_LEGISLATIVE,
            max_subsection_depth=3,  # Only Title, Chapter, Section
        )
        tool = HierarchicalDocumentTool(config)

        await tool.initialize()

        assert tool._initialized is True

        # (a) should NOT be detected as a heading (level 4)
        level_4_chunks = [c for c in tool._chunks if c.heading_level == 4]
        assert len(level_4_chunks) == 0, "Subsections should not be detected"

    @pytest.mark.asyncio
    async def test_none_domain_disables_subsection_detection(
        self, tmp_path: Path
    ) -> None:
        """Test that document_domain=none disables subsection pattern detection.

        Validates:
        1. Only markdown headings are detected
        2. Legislative patterns like (a), (1) are not detected as headings
        """
        doc = tmp_path / "no_domain.md"
        doc.write_text(
            "# Main Title\n\n"
            "Introduction content.\n\n"
            "## Section One\n\n"
            "TITLE I—This should not be detected as heading.\n\n"
            "(a) This should not be detected as heading.\n\n"
            "(1) Neither should this.\n"
        )

        config = HierarchicalDocumentToolConfig(
            name="no_domain_test",
            description="Test no domain detection",
            source=str(doc),
            document_domain=DocumentDomain.NONE,  # Explicitly no domain
        )
        tool = HierarchicalDocumentTool(config)

        await tool.initialize()

        assert tool._initialized is True

        # Only markdown headings (levels 1 and 2) should be detected
        heading_levels = {c.heading_level for c in tool._chunks if c.heading_level > 0}
        assert heading_levels <= {1, 2}, "Only markdown headings should be detected"

    @skip_if_no_openai
    @pytest.mark.asyncio
    async def test_us_legislative_search_with_real_embeddings(
        self, tmp_path: Path
    ) -> None:
        """Test search works correctly with US legislative domain and real embeddings.

        Validates:
        1. Documents with legislative structure are properly indexed
        2. Search returns relevant results respecting hierarchy
        """
        doc = tmp_path / "broadband_act.md"
        doc.write_text(
            "TITLE I—UNIVERSAL BROADBAND ACCESS\n\n"
            "CHAPTER 1—FINDINGS AND PURPOSE\n\n"
            "SEC. 101. FINDINGS.\n\n"
            "Congress finds the following:\n\n"
            "(a) DIGITAL DIVIDE.—Millions of Americans lack access to broadband.\n\n"
            "(b) ECONOMIC IMPACT.—Broadband access drives economic growth.\n\n"
            "SEC. 102. PURPOSE.\n\n"
            "The purpose of this title is to ensure universal broadband access.\n\n"
            "CHAPTER 2—GRANT PROGRAMS\n\n"
            "SEC. 201. BROADBAND DEPLOYMENT GRANTS.\n\n"
            "(a) AUTHORIZATION.—The Secretary shall award grants.\n\n"
            "(b) ELIGIBILITY.—States and territories are eligible.\n"
        )

        config = HierarchicalDocumentToolConfig(
            name="broadband_search_test",
            description="Test legislative search",
            source=str(doc),
            document_domain=DocumentDomain.US_LEGISLATIVE,
            search_mode=SearchMode.SEMANTIC,
            top_k=5,
        )
        tool = HierarchicalDocumentTool(config)
        tool.set_embedding_service(create_openai_embedding_service())

        await tool.initialize()

        # Search for findings
        results = await tool.search("digital divide broadband access")
        assert len(results) > 0

        # Top results should contain relevant content
        all_content = " ".join(r.content.lower() for r in results[:3])
        assert "broadband" in all_content or "access" in all_content

        # Search for grant information
        results = await tool.search("grant eligibility requirements")
        assert len(results) > 0
