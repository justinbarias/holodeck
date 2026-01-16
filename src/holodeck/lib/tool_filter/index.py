"""Tool index for fast semantic search over tools.

This module provides the ToolIndex class for building and searching
an in-memory index of tool metadata. Supports semantic search using
embeddings, BM25 keyword search, and hybrid combinations.

Key features:
- Build index from Semantic Kernel plugins
- Semantic search using cosine similarity
- BM25 keyword-based search
- Hybrid search combining both methods
- Usage tracking for adaptive optimization
"""

import math
import re

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.embedding_generator_base import (
    EmbeddingGeneratorBase,
)
from semantic_kernel.functions.kernel_function import KernelFunction
from semantic_kernel.functions.kernel_parameter_metadata import KernelParameterMetadata
from semantic_kernel.functions.kernel_plugin import KernelPlugin

from holodeck.lib.logging_config import get_logger
from holodeck.lib.tool_filter.models import ToolMetadata

logger = get_logger(__name__)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Cosine similarity score between -1.0 and 1.0.
    """
    if len(vec_a) != len(vec_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25 search.

    Splits text on non-alphanumeric characters INCLUDING underscores,
    so that tool names like "brave_web_search" become ["brave", "web", "search"].
    This enables matching individual terms like "web" against tool names.

    Args:
        text: Input text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    # Split on non-alphanumeric characters (excluding underscores from word chars)
    # This ensures "brave_web_search" -> ["brave", "web", "search"]
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return tokens


class ToolIndex:
    """In-memory index for fast tool searching.

    Maintains a collection of ToolMetadata objects and supports
    multiple search methods for finding relevant tools based on
    user queries.

    Attributes:
        tools: Dictionary mapping full_name to ToolMetadata.
        _idf_cache: Cached IDF values for BM25 search.
        _doc_lengths: Document lengths for BM25 normalization.
        _avg_doc_length: Average document length for BM25.
    """

    # BM25 parameters
    _BM25_K1 = 1.5
    _BM25_B = 0.75

    def __init__(self) -> None:
        """Initialize an empty tool index."""
        self.tools: dict[str, ToolMetadata] = {}
        self._idf_cache: dict[str, float] = {}
        self._doc_lengths: dict[str, int] = {}
        self._avg_doc_length: float = 0.0

    async def build_from_kernel(
        self,
        kernel: Kernel,
        embedding_service: EmbeddingGeneratorBase | None = None,
        defer_loading_map: dict[str, bool] | None = None,
    ) -> None:
        """Build index from Semantic Kernel plugins.

        Extracts all registered functions from the kernel's plugins
        and creates ToolMetadata entries with optional embeddings.

        Args:
            kernel: Semantic Kernel with registered plugins.
            embedding_service: Optional TextEmbedding service for generating embeddings.
            defer_loading_map: Optional mapping of tool names to defer_loading flags.
                               Defaults to True for all tools if not provided.
        """
        defer_loading_map = defer_loading_map or {}
        documents_for_bm25: list[tuple[str, str]] = []

        # Get all plugins and their functions
        plugins: dict[str, KernelPlugin] = getattr(kernel, "plugins", {})
        if not plugins:
            logger.debug("No plugins found in kernel")
            return

        for plugin_name, plugin in plugins.items():
            functions: dict[str, KernelFunction] = getattr(plugin, "functions", {})
            for func_name, func in functions.items():
                # Build full name
                full_name = f"{plugin_name}-{func_name}" if plugin_name else func_name

                # Extract description
                description = getattr(func, "description", "") or ""
                if not description:
                    # Try to get from metadata
                    metadata = getattr(func, "metadata", None)
                    if metadata:
                        description = getattr(metadata, "description", "") or ""

                if not description:
                    description = f"Function {func_name} from plugin {plugin_name}"

                # Extract parameter descriptions
                parameters: list[str] = []
                try:
                    func_params: list[KernelParameterMetadata] | None = getattr(
                        func, "parameters", None
                    )
                    if func_params:
                        for param in func_params:
                            if param.description:
                                parameters.append(f"{param.name}: {param.description}")
                            elif param.name:
                                parameters.append(param.name)
                except Exception as e:
                    logger.debug(f"Could not extract parameters for {full_name}: {e}")

                # Determine defer_loading
                defer_loading = defer_loading_map.get(full_name, True)

                # Create metadata
                tool_metadata = ToolMetadata(
                    name=func_name,
                    plugin_name=plugin_name,
                    full_name=full_name,
                    description=description,
                    parameters=parameters,
                    defer_loading=defer_loading,
                )

                self.tools[full_name] = tool_metadata

                # Collect for BM25
                doc_text = self._create_searchable_text(tool_metadata)
                documents_for_bm25.append((full_name, doc_text))

                logger.debug(
                    f"Indexed tool: {full_name} | "
                    f"searchable_text: {doc_text[:200]}..."
                )

        # Build BM25 index
        self._build_bm25_index(documents_for_bm25)

        # Generate embeddings if service provided
        if embedding_service and self.tools:
            await self._generate_embeddings(embedding_service)

        logger.info(f"Built tool index with {len(self.tools)} tools")

    def _create_searchable_text(self, tool: ToolMetadata) -> str:
        """Create combined searchable text from tool metadata.

        Args:
            tool: ToolMetadata to create text from.

        Returns:
            Combined text for search indexing.
        """
        parts = [
            tool.name,
            tool.plugin_name,
            tool.description,
        ]
        parts.extend(tool.parameters)
        return " ".join(parts)

    def _build_bm25_index(self, documents: list[tuple[str, str]]) -> None:
        """Build BM25 index from documents.

        Args:
            documents: List of (full_name, text) tuples.
        """
        if not documents:
            return

        # Tokenize all documents
        tokenized_docs: dict[str, list[str]] = {}
        total_length = 0

        for full_name, text in documents:
            tokens = _tokenize(text)
            tokenized_docs[full_name] = tokens
            self._doc_lengths[full_name] = len(tokens)
            total_length += len(tokens)

        self._avg_doc_length = total_length / len(documents) if documents else 0.0

        # Calculate IDF for all terms
        term_doc_freq: dict[str, int] = {}
        for tokens in tokenized_docs.values():
            seen_terms: set[str] = set()
            for token in tokens:
                if token not in seen_terms:
                    term_doc_freq[token] = term_doc_freq.get(token, 0) + 1
                    seen_terms.add(token)

        # IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
        n = len(documents)
        for term, df in term_doc_freq.items():
            self._idf_cache[term] = math.log((n - df + 0.5) / (df + 0.5) + 1)

    async def _generate_embeddings(
        self, embedding_service: EmbeddingGeneratorBase
    ) -> None:
        """Generate embeddings for all tools using the embedding service.

        Args:
            embedding_service: TextEmbedding service for generating vectors.
        """
        texts: list[str] = []
        full_names: list[str] = []

        for full_name, tool in self.tools.items():
            text = self._create_searchable_text(tool)
            texts.append(text)
            full_names.append(full_name)

        if not texts:
            return

        try:
            # Generate embeddings in batch
            embeddings = await embedding_service.generate_embeddings(texts)

            # Assign embeddings to tools
            for full_name, embedding in zip(full_names, embeddings, strict=False):
                self.tools[full_name].embedding = list(embedding)

            logger.debug(f"Generated embeddings for {len(texts)} tools")

        except Exception as e:
            logger.warning(f"Failed to generate tool embeddings: {e}")

    async def search(
        self,
        query: str,
        top_k: int,
        method: str = "semantic",
        threshold: float = 0.0,
        embedding_service: EmbeddingGeneratorBase | None = None,
    ) -> list[tuple[ToolMetadata, float]]:
        """Search for relevant tools based on query.

        Args:
            query: User query to match against tools.
            top_k: Maximum number of results to return.
            method: Search method (semantic, bm25, or hybrid).
            threshold: Minimum score threshold for inclusion.
            embedding_service: TextEmbedding service (required for semantic search).

        Returns:
            List of (ToolMetadata, score) tuples sorted by relevance.
        """
        if not self.tools:
            return []

        if method == "semantic":
            results = await self._semantic_search(query, embedding_service)
        elif method == "bm25":
            results = self._bm25_search(query)
        elif method == "hybrid":
            results = await self._hybrid_search(query, embedding_service)
        else:
            logger.warning(f"Unknown search method: {method}, falling back to semantic")
            results = await self._semantic_search(query, embedding_service)

        # Sort all results by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Log ALL tool scores for debugging (helps diagnose ranking issues)
        logger.debug(
            f"Tool search ({method}) all scores: "
            f"{[(t.full_name, f'{s:.4f}') for t, s in results]}"
        )

        # Filter by threshold
        filtered = [(tool, score) for tool, score in results if score >= threshold]

        top_results = filtered[:top_k]

        # Log top matches for visibility
        if top_results:
            top_matches = [(t.full_name, f"{s:.3f}") for t, s in top_results[:3]]
            logger.info(f"Tool search ({method}): top matches {top_matches}")

        return top_results

    async def _semantic_search(
        self, query: str, embedding_service: EmbeddingGeneratorBase | None
    ) -> list[tuple[ToolMetadata, float]]:
        """Perform semantic search using embeddings.

        Args:
            query: User query.
            embedding_service: TextEmbedding service for query embedding.

        Returns:
            List of (ToolMetadata, similarity_score) tuples.
        """
        if not embedding_service:
            logger.warning("No embedding service provided for semantic search")
            return self._bm25_search(query)

        try:
            # Generate query embedding
            query_embeddings = await embedding_service.generate_embeddings([query])
            query_embedding = list(query_embeddings[0])

            results: list[tuple[ToolMetadata, float]] = []
            for tool in self.tools.values():
                if tool.embedding:
                    score = _cosine_similarity(query_embedding, tool.embedding)
                    results.append((tool, score))
                else:
                    # No embedding, use BM25 fallback for this tool
                    bm25_score = self._bm25_score_single(query, tool)
                    results.append((tool, bm25_score))

            return results

        except Exception as e:
            logger.warning(f"Semantic search failed, falling back to BM25: {e}")
            return self._bm25_search(query)

    def _bm25_search(self, query: str) -> list[tuple[ToolMetadata, float]]:
        """Perform BM25 keyword search.

        Args:
            query: User query.

        Returns:
            List of (ToolMetadata, normalized_score) tuples with scores in 0-1 range.
        """
        raw_results: list[tuple[ToolMetadata, float]] = []

        for tool in self.tools.values():
            score = self._bm25_score_single(query, tool)
            raw_results.append((tool, score))

        # Normalize BM25 scores to 0-1 range
        # Raw BM25 scores can be arbitrarily high (typically 0-10+)
        max_score = max((score for _, score in raw_results), default=1.0)
        if max_score > 0:
            return [(tool, score / max_score) for tool, score in raw_results]
        return raw_results

    def _bm25_score_single(self, query: str, tool: ToolMetadata) -> float:
        """Calculate BM25 score for a single tool.

        Args:
            query: User query.
            tool: ToolMetadata to score.

        Returns:
            BM25 score (higher is more relevant).
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return 0.0

        doc_text = self._create_searchable_text(tool)
        doc_tokens = _tokenize(doc_text)
        doc_length = len(doc_tokens)

        if doc_length == 0:
            return 0.0

        # Count term frequencies in document
        term_freq: dict[str, int] = {}
        for token in doc_tokens:
            term_freq[token] = term_freq.get(token, 0) + 1

        score = 0.0
        for term in query_tokens:
            if term not in term_freq:
                continue

            tf = term_freq[term]
            idf = self._idf_cache.get(term, 0.0)

            # BM25 formula
            numerator = tf * (self._BM25_K1 + 1)
            denominator = tf + self._BM25_K1 * (
                1 - self._BM25_B + self._BM25_B * doc_length / self._avg_doc_length
            )
            score += idf * (numerator / denominator)

        return score

    async def _hybrid_search(
        self, query: str, embedding_service: EmbeddingGeneratorBase | None
    ) -> list[tuple[ToolMetadata, float]]:
        """Perform hybrid search combining semantic and BM25.

        Uses reciprocal rank fusion to combine results.

        Args:
            query: User query.
            embedding_service: TextEmbedding service for semantic component.

        Returns:
            List of (ToolMetadata, combined_score) tuples.
        """
        # Get semantic results
        semantic_results = await self._semantic_search(query, embedding_service)

        # Get BM25 results
        bm25_results = self._bm25_search(query)

        # Log intermediate results for debugging
        semantic_sorted_debug = sorted(
            semantic_results, key=lambda x: x[1], reverse=True
        )
        bm25_sorted_debug = sorted(bm25_results, key=lambda x: x[1], reverse=True)

        logger.debug(
            f"Hybrid search - Semantic top 5: "
            f"{[(t.full_name, f'{s:.4f}') for t, s in semantic_sorted_debug[:5]]}"
        )
        logger.debug(
            f"Hybrid search - BM25 top 5: "
            f"{[(t.full_name, f'{s:.4f}') for t, s in bm25_sorted_debug[:5]]}"
        )

        # Combine using reciprocal rank fusion (RRF)
        k = 60  # RRF constant
        rrf_scores: dict[str, float] = {}

        # Process semantic results
        semantic_sorted = sorted(semantic_results, key=lambda x: x[1], reverse=True)
        for rank, (tool, _) in enumerate(semantic_sorted):
            rrf_scores[tool.full_name] = rrf_scores.get(tool.full_name, 0.0) + 1 / (
                k + rank + 1
            )

        # Process BM25 results
        bm25_sorted = sorted(bm25_results, key=lambda x: x[1], reverse=True)
        for rank, (tool, _) in enumerate(bm25_sorted):
            rrf_scores[tool.full_name] = rrf_scores.get(tool.full_name, 0.0) + 1 / (
                k + rank + 1
            )

        # Normalize RRF scores to 0-1 range
        # Raw RRF scores are typically 0.01-0.03, which breaks threshold filtering
        # Normalizing ensures consistency with semantic search scores
        max_score = max(rrf_scores.values()) if rrf_scores else 1.0
        if max_score > 0:
            normalized_scores = {
                name: score / max_score for name, score in rrf_scores.items()
            }
        else:
            normalized_scores = rrf_scores

        # Build final results with normalized scores
        results: list[tuple[ToolMetadata, float]] = []
        for full_name, score in normalized_scores.items():
            found_tool = self.tools.get(full_name)
            if found_tool:
                results.append((found_tool, score))

        return results

    def update_usage(self, tool_name: str) -> None:
        """Increment usage count for a tool.

        Args:
            tool_name: Full name of the tool that was used.
        """
        if tool_name in self.tools:
            self.tools[tool_name].usage_count += 1
            logger.debug(
                f"Updated usage for {tool_name}: {self.tools[tool_name].usage_count}"
            )

    def get_top_n_used(self, n: int) -> list[ToolMetadata]:
        """Get the N most frequently used tools.

        Args:
            n: Number of top tools to return.

        Returns:
            List of ToolMetadata sorted by usage_count descending.
        """
        if n <= 0:
            return []

        sorted_tools = sorted(
            self.tools.values(), key=lambda t: t.usage_count, reverse=True
        )
        return sorted_tools[:n]

    def get_tool(self, full_name: str) -> ToolMetadata | None:
        """Get a tool by its full name.

        Args:
            full_name: Tool's full name (plugin_name-function_name).

        Returns:
            ToolMetadata if found, None otherwise.
        """
        return self.tools.get(full_name)

    def get_all_tool_names(self) -> list[str]:
        """Get all tool full names in the index.

        Returns:
            List of all tool full names.
        """
        return list(self.tools.keys())
