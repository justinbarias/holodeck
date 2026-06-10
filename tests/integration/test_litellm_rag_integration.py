"""Integration tests for the LiteLLM-backed RAG layer.

Covers the two seams that moved from Semantic Kernel to LiteLLM:

1. Embedding generation — ``LiteLLMEmbeddingService.generate_embeddings``
   (consumed by vectorstore / hierarchical-document tools).
2. Context prefix generation (contextual retrieval) —
   ``LLMContextGenerator`` backed by ``litellm.acompletion``.

These tests make real API calls to OpenAI / Azure OpenAI. They require valid
API keys in tests/integration/.env (see .env.example). To skip them (e.g. in
CI without keys), set SKIP_LLM_INTEGRATION_TESTS=true.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from holodeck.lib.litellm_support import (
    LiteLLMEmbeddingService,
    LiteLLMModelSpec,
    resolve_litellm_model,
)
from holodeck.lib.llm_context_generator import LLMContextGenerator
from holodeck.lib.structured_chunker import ChunkType, DocumentChunk
from holodeck.models.llm import LLMProvider, ProviderEnum

# Load environment variables from tests/integration/.env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Check if we should skip LLM integration tests
SKIP_LLM_TESTS = os.getenv("SKIP_LLM_INTEGRATION_TESTS", "false").lower() == "true"


def _real_credential(value: str | None) -> str | None:
    """Treat unset or .env.example placeholder values ("your-...") as missing."""
    if not value or value.startswith("your-"):
        return None
    return value


# OpenAI configuration
OPENAI_API_KEY = _real_credential(os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Azure OpenAI configuration
AZURE_OPENAI_API_KEY = _real_credential(os.getenv("AZURE_OPENAI_API_KEY"))
AZURE_OPENAI_ENDPOINT = _real_credential(os.getenv("AZURE_OPENAI_ENDPOINT"))
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small"
)

# Skip markers for different providers
skip_if_no_openai = pytest.mark.skipif(
    SKIP_LLM_TESTS or not OPENAI_API_KEY,
    reason="OpenAI API key not configured or LLM tests disabled",
)

skip_if_no_azure = pytest.mark.skipif(
    SKIP_LLM_TESTS or not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT,
    reason="Azure OpenAI credentials not configured or LLM tests disabled",
)

SAMPLE_DOCUMENT = """# Insurance Policy

## Section 1: Definitions

Premium means the amount payable by the policyholder for coverage.
Force Majeure means any event beyond the reasonable control of either party.

## Section 2: Claims

Claims must be lodged within 30 days of the insured event. The insurer
will assess each claim within 10 business days of receipt.
"""


def make_chunk(content: str, index: int = 0) -> DocumentChunk:
    """Create a DocumentChunk for context-generation tests."""
    return DocumentChunk(
        id=f"chunk_{index}",
        source_path="/test/policy.md",
        chunk_index=index,
        content=content,
        parent_chain=["Insurance Policy"],
        section_id=f"sec_{index}",
        chunk_type=ChunkType.CONTENT,
    )


def openai_embedding_service() -> LiteLLMEmbeddingService:
    """Build the OpenAI embedding service under test."""
    return LiteLLMEmbeddingService(
        LiteLLMModelSpec(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    )


def azure_provider() -> LLMProvider:
    """Azure OpenAI LLMProvider built from the integration environment."""
    return LLMProvider(
        provider=ProviderEnum.AZURE_OPENAI,
        name=AZURE_OPENAI_DEPLOYMENT_NAME,
        api_key=AZURE_OPENAI_API_KEY,
        endpoint=AZURE_OPENAI_ENDPOINT,
    )


def azure_embedding_service() -> LiteLLMEmbeddingService:
    """Build the Azure embedding service via the production resolver."""
    spec = resolve_litellm_model(
        azure_provider(),
        kind="embedding",
        model_name=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    )
    return LiteLLMEmbeddingService(spec)


def openai_context_generator() -> LLMContextGenerator:
    """Build an OpenAI-backed context generator under test."""
    return LLMContextGenerator(
        model_spec=LiteLLMModelSpec(model=OPENAI_MODEL_NAME, api_key=OPENAI_API_KEY)
    )


def azure_context_generator() -> LLMContextGenerator:
    """Build an Azure-backed context generator via the production resolver."""
    return LLMContextGenerator(
        model_spec=resolve_litellm_model(azure_provider(), kind="chat")
    )


@pytest.mark.integration
class TestLiteLLMEmbeddings:
    """Embedding generation through the LiteLLM shim with real providers."""

    @skip_if_no_openai
    @pytest.mark.asyncio
    async def test_openai_embeddings_shape_and_values(self) -> None:
        """OpenAI embeddings: one vector per input, expected dims, non-zero."""
        service = openai_embedding_service()

        result = await service.generate_embeddings(
            ["What is Force Majeure?", "Premium payment terms"]
        )

        assert len(result) == 2
        for vector in result:
            assert len(vector) == 1536  # text-embedding-3-small dimensions
            assert any(v != 0.0 for v in vector)
            assert all(isinstance(v, float) for v in vector)
        # Distinct inputs must produce distinct vectors
        assert result[0] != result[1]

    @skip_if_no_azure
    @pytest.mark.asyncio
    async def test_azure_embeddings_shape_and_values(self) -> None:
        """Azure OpenAI embeddings: one vector per input, expected dims."""
        service = azure_embedding_service()

        result = await service.generate_embeddings(["claims assessment window"])

        assert len(result) == 1
        assert len(result[0]) == 1536
        assert any(v != 0.0 for v in result[0])

    @skip_if_no_azure
    @pytest.mark.asyncio
    async def test_create_embedding_service_wiring(self) -> None:
        """The production factory builds a working service from agent config."""
        from holodeck.lib.tool_initializer import create_embedding_service
        from holodeck.models.agent import Agent, Instructions
        from holodeck.models.tool import VectorstoreTool

        agent = Agent(
            name="litellm_rag_test",
            model=azure_provider(),
            instructions=Instructions(inline="test"),
            tools=[
                VectorstoreTool(
                    name="kb",
                    description="kb",
                    source="./docs",
                    embedding_model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                )
            ],
        )

        service = create_embedding_service(agent)
        result = await service.generate_embeddings(["wiring smoke test"])

        assert len(result) == 1
        assert len(result[0]) == 1536


@pytest.mark.integration
class TestLiteLLMContextPrefixGeneration:
    """Contextual-retrieval prefix generation with real chat models."""

    @skip_if_no_openai
    @pytest.mark.asyncio
    async def test_openai_generate_context_returns_situating_text(self) -> None:
        """generate_context returns a short, non-empty situating snippet."""
        generator = openai_context_generator()

        context = await generator.generate_context(
            chunk_text="Claims must be lodged within 30 days of the insured event.",
            document_text=SAMPLE_DOCUMENT,
        )

        assert isinstance(context, str)
        assert len(context) > 0
        # max_tokens is bounded to max_context_tokens (100) — the snippet
        # must be much shorter than the source document.
        assert len(context) < len(SAMPLE_DOCUMENT)

    @skip_if_no_openai
    @pytest.mark.asyncio
    async def test_openai_contextualize_chunk_prepends_prefix(self) -> None:
        """contextualize_chunk output is '{context}\\n\\n{original chunk}'."""
        generator = openai_context_generator()
        chunk = make_chunk(
            "Premium means the amount payable by the policyholder for coverage."
        )

        result = await generator.contextualize_chunk(chunk, SAMPLE_DOCUMENT)

        # Original chunk content is preserved verbatim at the end
        assert result.endswith(chunk.content)
        # A non-empty context prefix was prepended with the \n\n separator
        assert result != chunk.content
        prefix = result[: -len(chunk.content)]
        assert prefix.endswith("\n\n")
        assert len(prefix.strip()) > 0

    @skip_if_no_openai
    @pytest.mark.asyncio
    async def test_openai_contextualize_batch_preserves_order(self) -> None:
        """contextualize_batch returns one result per chunk, in input order."""
        generator = openai_context_generator()
        chunks = [
            make_chunk("Force Majeure means any event beyond reasonable control.", 0),
            make_chunk("Claims must be lodged within 30 days.", 1),
        ]

        results = await generator.contextualize_batch(chunks, SAMPLE_DOCUMENT)

        assert len(results) == 2
        # Each result carries its own chunk's content (order preserved);
        # context generation may degrade gracefully to the bare content.
        assert chunks[0].content in results[0]
        assert chunks[1].content in results[1]

    @skip_if_no_azure
    @pytest.mark.asyncio
    async def test_azure_generate_context_returns_situating_text(self) -> None:
        """Azure OpenAI chat deployment also produces context prefixes."""
        generator = azure_context_generator()

        context = await generator.generate_context(
            chunk_text="The insurer will assess each claim within 10 business days.",
            document_text=SAMPLE_DOCUMENT,
        )

        assert isinstance(context, str)
        assert len(context) > 0

    @skip_if_no_azure
    @pytest.mark.asyncio
    async def test_resolve_context_generator_wiring(self) -> None:
        """The production resolver builds a working LiteLLM-backed generator
        from a hierarchical-document tool's context_model config."""
        from holodeck.lib.tool_initializer import _resolve_context_generator
        from holodeck.models.agent import Agent, Instructions
        from holodeck.models.tool import HierarchicalDocumentToolConfig

        context_model = azure_provider()
        tool_config = HierarchicalDocumentToolConfig(
            name="policy_docs",
            description="Policy documents",
            source="./docs",
            context_model=context_model,
        )
        agent = Agent(
            name="litellm_ctx_test",
            model=context_model,
            instructions=Instructions(inline="test"),
            tools=[tool_config],
        )

        generator = _resolve_context_generator(agent=agent, tool_config=tool_config)

        assert isinstance(generator, LLMContextGenerator)
        context = await generator.generate_context(
            chunk_text="Premium means the amount payable by the policyholder.",
            document_text=SAMPLE_DOCUMENT,
        )
        assert isinstance(context, str)
        assert len(context) > 0
