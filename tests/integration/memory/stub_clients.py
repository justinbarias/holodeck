"""Stub model clients for the Graphiti integration tests (spec 043, T10).

The LLM client returns canned, schema-valid replies and counts its calls. The
embedder returns fixed vectors. Nothing here calls a model or the network.

Importing this module skips the importing test module when the ``graphiti``
extra is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "graphiti_core", reason="the graphiti extra is not installed", exc_type=ImportError
)

from typing import Any  # noqa: E402

from graphiti_core.llm_client.client import LLMClient  # noqa: E402
from graphiti_core.llm_client.config import LLMConfig, ModelSize  # noqa: E402
from pydantic import BaseModel  # noqa: E402

ENTITY_SOURCE = "Rate A"
ENTITY_TARGET = "Agency B"
EDGE_FACT = "Rate A is administered by Agency B"


def _minimal(schema: dict[str, Any], defs: dict[str, Any]) -> Any:
    """The smallest value that satisfies a JSON schema's required parts."""
    if "$ref" in schema:
        return _minimal(defs[schema["$ref"].split("/")[-1]], defs)
    if "anyOf" in schema:
        return None
    kind = schema.get("type")
    if kind == "object":
        required = schema.get("required", [])
        return {
            key: _minimal(value, defs)
            for key, value in schema.get("properties", {}).items()
            if key in required
        }
    return {
        "array": [],
        "string": "",
        "integer": 0,
        "number": 0.0,
        "boolean": False,
    }.get(kind)


class StubLLMClient(LLMClient):
    """A Graphiti LLM client with canned replies.

    Attributes:
        calls: Number of ``generate_response`` calls, including failed ones.
        prompts: The ``prompt_name`` of every call, in order.
        fail_with: When set, every call raises it.
    """

    def __init__(self) -> None:
        super().__init__(LLMConfig(model="stub", small_model="stub"), cache=False)
        self.calls = 0
        self.prompts: list[str] = []
        self.fail_with: Exception | None = None

    async def _generate_response(  # type: ignore[override]
        self,
        messages: Any,
        response_model: type[BaseModel] | None = None,
        max_tokens: int = 0,
        model_size: ModelSize = ModelSize.medium,
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
        raise AssertionError("StubLLMClient answers in generate_response")

    async def generate_response(  # type: ignore[override]
        self,
        messages: Any,
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
        *,
        attribute_extraction: bool = False,
        model: str | None = None,
    ) -> dict[str, Any]:
        self.calls += 1
        self.prompts.append(str(prompt_name))
        if self.fail_with is not None:
            raise self.fail_with
        self.token_tracker.record(prompt_name, 10, 5)
        if prompt_name == "extract_nodes.extract_text":
            return {
                "extracted_entities": [
                    {"name": ENTITY_SOURCE, "entity_type_id": 0},
                    {"name": ENTITY_TARGET, "entity_type_id": 0},
                ]
            }
        if prompt_name == "extract_edges.edge":
            return {
                "edges": [
                    {
                        "source_entity_name": ENTITY_SOURCE,
                        "target_entity_name": ENTITY_TARGET,
                        "relation_type": "ADMINISTERED_BY",
                        "fact": EDGE_FACT,
                    }
                ]
            }
        assert response_model is not None, f"no canned reply for {prompt_name}"
        schema = response_model.model_json_schema()
        return dict(_minimal(schema, schema.get("$defs", {})))


class StubEmbeddingService:
    """Fixed, deterministic vectors; one per text, in order."""

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.5, 0.25, float(len(text) % 7)] for text in texts]
