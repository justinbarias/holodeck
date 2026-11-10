"""Unit tests for the native Pinecone adapter, on a fake asynchronous client.

The fake returns the SDK's own response types, so a renamed field in the client
shows up here. ``tests/contract/retrieval`` runs the same adapter against
Pinecone Local.
"""

from __future__ import annotations

import hashlib
import logging
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import pytest
from pinecone import (
    AsyncPinecone,
    DescribeIndexStatsResponse,
    FetchByMetadataResponse,
    FetchResponse,
    ListResponse,
    NotFoundError,
    PineconeError,
    QueryResponse,
    ScoredVector,
    Vector,
)
from pinecone.errors import PineconeValueError
from pinecone.models.indexes.schema import DenseVectorField, IndexSchema
from pinecone.models.vectors.responses import ListItem, NamespaceSummary, Pagination

from holodeck.lib.retrieval import (
    CollectionSchema,
    FieldSpec,
    FieldType,
    IncompatibleCollectionError,
    LexicalQuery,
    Operation,
    Record,
    RecordFilter,
    SchemaError,
    StoreConnectionError,
    UnsupportedOperationError,
    VectorQuery,
    create_adapter,
)
from holodeck.lib.retrieval.adapters import pinecone as module
from holodeck.lib.retrieval.adapters.pinecone import (
    PineconeAdapter,
    PineconeSettings,
    normalize_index_name,
    parse_pinecone_connection_string,
    settings_from_connection,
)

API_KEY = "pcsk_SECRET_key_123"
DIMENSIONS = 4


def schema(name: str = "contract-docs") -> CollectionSchema:
    return CollectionSchema(
        name,
        DIMENSIONS,
        (
            FieldSpec("source_path", FieldType.STRING, indexed=True),
            FieldSpec("chunk_index", FieldType.INTEGER, indexed=True),
            FieldSpec("content", FieldType.STRING, full_text=True),
            FieldSpec("mtime", FieldType.FLOAT),
            FieldSpec("pinned", FieldType.BOOLEAN),
        ),
    )


def record(
    record_id: str,
    *,
    source: str = "a.md",
    index: int = 0,
    content: str = "text",
    vector: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
) -> Record:
    return Record(
        id=record_id,
        fields={
            "source_path": source,
            "chunk_index": index,
            "content": content,
            "mtime": 1.5,
            "pinned": False,
        },
        vector=list(vector),
    )


# -- fakes -----------------------------------------------------------------------


def _matches(metadata: Mapping[str, object], expression: Mapping[str, object]) -> bool:
    """Evaluate the ``$eq`` / ``$and`` subset the adapter builds."""
    if "$and" in expression:
        clauses = cast(list[Mapping[str, object]], expression["$and"])
        return all(_matches(metadata, clause) for clause in clauses)
    ((name, condition),) = expression.items()
    assert isinstance(condition, Mapping)
    return name in metadata and metadata[name] == condition["$eq"]


@dataclass
class FakeIndex:
    """An in-memory ``AsyncIndex`` that logs its calls."""

    metadata_fetch: bool = True
    lag: int = 0
    stored: dict[str, Vector] = field(default_factory=dict)
    calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    closed: int = 0
    failure: Exception | None = None
    namespaces: dict[str, int] | None = None
    float_digits: int | None = None

    def _log(self, name: str, **arguments: object) -> None:
        self.calls.append((name, arguments))
        if self.failure is not None:
            raise self.failure

    def names(self, name: str) -> list[dict[str, object]]:
        return [args for called, args in self.calls if called == name]

    async def upsert(self, *, vectors: Sequence[Vector], namespace: str = "") -> object:
        self._log("upsert", vectors=list(vectors), namespace=namespace)
        for vector in vectors:
            self.stored[vector.id] = vector
        return None

    async def fetch(self, *, ids: Sequence[str], namespace: str = "") -> FetchResponse:
        self._log("fetch", ids=list(ids), namespace=namespace)
        if self.lag > 0:
            self.lag -= 1
            return FetchResponse(vectors={}, namespace=namespace)
        found = {
            key: self._served(self.stored[key]) for key in ids if key in self.stored
        }
        return FetchResponse(vectors=found, namespace=namespace)

    def _served(self, vector: Vector) -> Vector:
        if self.float_digits is None or not vector.metadata:
            return vector
        digits = self.float_digits
        metadata = {
            key: round(value, digits) if isinstance(value, float) else value
            for key, value in vector.metadata.items()
        }
        return Vector(id=vector.id, values=vector.values, metadata=metadata)

    async def fetch_by_metadata(
        self,
        *,
        filter: Mapping[str, object],
        namespace: str = "",
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> FetchByMetadataResponse:
        self._log(
            "fetch_by_metadata",
            filter=filter,
            limit=limit,
            pagination_token=pagination_token,
        )
        if not self.metadata_fetch:
            raise NotFoundError()
        hits = [
            vector
            for vector in self.stored.values()
            if _matches(vector.metadata or {}, filter)
        ]
        start = int(pagination_token or 0)
        size = limit or len(hits)
        page = hits[start : start + size]
        more = start + size < len(hits)
        return FetchByMetadataResponse(
            vectors={vector.id: vector for vector in page},
            namespace=namespace,
            pagination=Pagination(next=str(start + size)) if more else None,
        )

    async def delete(
        self,
        *,
        ids: Sequence[str] | None = None,
        delete_all: bool = False,
        namespace: str = "",
    ) -> None:
        self._log(
            "delete", ids=list(ids or ()), delete_all=delete_all, namespace=namespace
        )
        if delete_all:
            self.stored.clear()
            if self.namespaces is not None:
                self.namespaces.pop(namespace, None)
            return
        for key in ids or ():
            self.stored.pop(key, None)

    async def list_paginated(
        self,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
        namespace: str = "",
    ) -> ListResponse:
        self._log("list_paginated", limit=limit, pagination_token=pagination_token)
        keys = list(self.stored)
        start = int(pagination_token or 0)
        size = limit or len(keys)
        page = keys[start : start + size]
        more = start + size < len(keys)
        return ListResponse(
            vectors=[ListItem(id=key) for key in page],
            pagination=Pagination(next=str(start + size)) if more else None,
            namespace=namespace,
        )

    async def query(
        self,
        *,
        top_k: int,
        vector: Sequence[float],
        namespace: str = "",
        filter: Mapping[str, object] | None = None,
        include_values: bool = False,
        include_metadata: bool = False,
    ) -> QueryResponse:
        self._log(
            "query",
            top_k=top_k,
            filter=filter,
            include_metadata=include_metadata,
            include_values=include_values,
            namespace=namespace,
        )
        hits = [
            ScoredVector(id=v.id, score=2.0 - i, metadata=v.metadata)
            for i, v in enumerate(self.stored.values())
            if filter is None or _matches(v.metadata or {}, filter)
        ]
        return QueryResponse(matches=hits[:top_k], namespace=namespace)

    async def describe_index_stats(self) -> DescribeIndexStatsResponse:
        self._log("describe_index_stats")
        counts = self.namespaces
        if counts is None:
            counts = {"__default__": len(self.stored)}
        return DescribeIndexStatsResponse(
            namespaces={k: NamespaceSummary(vector_count=v) for k, v in counts.items()},
            dimension=DIMENSIONS,
        )

    async def close(self) -> None:
        self.closed += 1


@dataclass
class FakeModel:
    host: str | None
    schema: IndexSchema


@dataclass
class FakeIndexes:
    """The control plane: one optional index."""

    exists: bool = False
    dimension: int = DIMENSIONS
    metric: str = "cosine"
    created: list[dict[str, object]] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    failure: Exception | None = None

    async def describe(self, name: str) -> FakeModel:
        if self.failure is not None:
            raise self.failure
        if not self.exists:
            raise NotFoundError()
        field_ = DenseVectorField(dimension=self.dimension, metric=self.metric)
        return FakeModel("index.example", IndexSchema(fields={"_values": field_}))

    async def create(self, **arguments: object) -> None:
        self.created.append(arguments)
        self.exists = True

    async def delete(self, name: str, *, timeout: int | None = None) -> None:
        self.deleted.append(name)
        self.exists = False


@dataclass
class FakeClient:
    indexes: FakeIndexes
    data: FakeIndex
    hosts: list[str] = field(default_factory=list)
    closed: int = 0

    async def index(self, *, host: str) -> FakeIndex:
        self.hosts.append(host)
        return self.data

    async def close(self) -> None:
        self.closed += 1


@dataclass
class Harness:
    adapter: PineconeAdapter
    client: FakeClient
    index: FakeIndex
    indexes: FakeIndexes


def harness(
    *,
    exists: bool = True,
    namespace: str = "",
    collection: str = "contract-docs",
    **index_options: object,
) -> Harness:
    index = FakeIndex(**index_options)  # type: ignore[arg-type]
    indexes = FakeIndexes(exists=exists)
    client = FakeClient(indexes, index)
    adapter = PineconeAdapter(
        schema(collection), PineconeSettings(api_key=API_KEY, namespace=namespace)
    )
    adapter._client = cast(AsyncPinecone, client)
    return Harness(adapter, client, index, indexes)


@pytest.fixture(autouse=True)
def no_poll_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_POLL_START", 0.0)
    monkeypatch.setattr(module, "_POLL_MAX", 0.0)


# -- names and settings ------------------------------------------------------------


def _digest(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:8]


@pytest.mark.parametrize("name", ["documents-dim1536", "a", "a-b", "9lives", "a" * 45])
def test_a_valid_index_name_is_used_unchanged(name: str) -> None:
    assert normalize_index_name(name) == name


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("documents_dim1536", "documents-dim1536"),
        ("contract_0123456789ab", "contract-0123456789ab"),
        ("Product Catalog.v2", "product-catalog-v2"),
        ("--edge--", "edge"),
        ("a__b", "a-b"),
    ],
)
def test_a_changed_index_name_gets_a_hash_of_the_original(name: str, slug: str) -> None:
    assert normalize_index_name(name) == f"{slug}-{_digest(name)}"


def test_an_unusable_name_becomes_holodeck_and_a_hash() -> None:
    assert normalize_index_name("___") == f"holodeck-{_digest('___')}"


def test_names_that_differ_only_by_underscore_and_hyphen_never_share_an_index() -> None:
    names = ["a_b", "a-b", "a__b", "A_b", "a b"]

    index_names = [normalize_index_name(name) for name in names]

    assert len(set(index_names)) == len(names)
    assert normalize_index_name("a-b") == "a-b"


def test_long_names_keep_their_head_and_a_hash() -> None:
    first = "hierarchical_docs_" + "x" * 40 + "_dim1536"
    second = "hierarchical_docs_" + "x" * 40 + "_dim768"

    one, two = normalize_index_name(first), normalize_index_name(second)

    assert len(one) <= 45 and len(two) <= 45
    assert one != two
    assert one == normalize_index_name(first)
    assert one.startswith("hierarchical-docs-")
    assert one.endswith(_digest(first))
    assert not one.endswith("-") and "_" not in one


def test_a_valid_long_name_over_the_limit_is_shortened() -> None:
    name = "a" * 60

    assert len(normalize_index_name(name)) == 45
    assert normalize_index_name(name).endswith(_digest(name))


class TestConnectionString:
    def test_bare_key(self) -> None:
        assert parse_pinecone_connection_string("pc-abc123def456") == {
            "api_key": "pc-abc123def456"
        }
        assert parse_pinecone_connection_string("my-custom-api-key-12345") == {
            "api_key": "my-custom-api-key-12345"
        }

    def test_url_with_key(self) -> None:
        assert parse_pinecone_connection_string("pinecone://pc-abc123") == {
            "api_key": "pc-abc123"
        }

    def test_url_with_namespace(self) -> None:
        assert parse_pinecone_connection_string("pinecone://pc-abc123@my-ns") == {
            "api_key": "pc-abc123",
            "namespace": "my-ns",
        }

    def test_url_with_namespace_in_path(self) -> None:
        assert parse_pinecone_connection_string("pinecone://pc-abc123/production") == {
            "api_key": "pc-abc123",
            "namespace": "production",
        }

    def test_empty_path_namespace_is_ignored(self) -> None:
        assert parse_pinecone_connection_string("pinecone://pc-abc123/") == {
            "api_key": "pc-abc123"
        }

    def test_the_key_keeps_its_case_and_an_at_namespace_is_lower_cased(self) -> None:
        """The old parser read ``@namespace`` as a URL host, which lower-cases it."""
        parsed = parse_pinecone_connection_string("pinecone://pcsk_AbC_dEf@Prod")

        assert parsed == {"api_key": "pcsk_AbC_dEf", "namespace": "prod"}

    def test_a_path_namespace_keeps_its_case(self) -> None:
        parsed = parse_pinecone_connection_string("pinecone://pcsk_AbC_dEf/Prod")

        assert parsed == {"api_key": "pcsk_AbC_dEf", "namespace": "Prod"}

    def test_an_explicit_namespace_setting_keeps_its_case(self) -> None:
        settings = settings_from_connection(
            {"connection_string": "pinecone://key@Prod", "namespace": "Prod"}
        )

        assert settings.namespace == "Prod"
        assert (
            settings_from_connection(
                {"connection_string": "pinecone://key@Prod"}
            ).namespace
            == "prod"
        )

    def test_empty_string_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_pinecone_connection_string("")


def test_number_filters_follow_the_field_type() -> None:
    adapter = harness().adapter

    expression = adapter._filter_expression(
        RecordFilter.where(mtime=2, chunk_index=3.0)
    )

    assert expression == {
        "$and": [{"mtime": {"$eq": 2.0}}, {"chunk_index": {"$eq": 3}}]
    }
    assert isinstance(expression["$and"][0]["mtime"]["$eq"], float)  # type: ignore[index]
    with pytest.raises(SchemaError):
        adapter._filter_expression(RecordFilter.where(chunk_index=True))


class TestSettings:
    def test_explicit_settings_win_over_the_connection_string(self) -> None:
        settings = settings_from_connection(
            {
                "connection_string": "pinecone://from-string@string-ns",
                "api_key": "explicit",
                "namespace": "explicit-ns",
                "cloud": "gcp",
                "region": "europe-west4",
                "host": "http://localhost:5080",
                "index_host": "http://localhost:5081",
                "index_name": "my-index",
            }
        )

        assert settings.api_key == "explicit"
        assert settings.namespace == "explicit-ns"
        assert (settings.cloud, settings.region) == ("gcp", "europe-west4")
        assert settings.host == "http://localhost:5080"
        assert settings.index_host == "http://localhost:5081"
        assert settings.index_name == "my-index"

    def test_connection_string_supplies_key_and_namespace(self) -> None:
        settings = settings_from_connection(
            {"connection_string": "pinecone://key-1@ns-1"}
        )

        assert (settings.api_key, settings.namespace) == ("key-1", "ns-1")
        assert (settings.cloud, settings.region) == ("aws", "us-east-1")

    def test_serverless_spec_mapping(self) -> None:
        settings = settings_from_connection(
            {
                "api_key": "k",
                "spec": {"serverless": {"cloud": "azure", "region": "eastus2"}},
            }
        )

        assert (settings.cloud, settings.region) == ("azure", "eastus2")

    def test_environment_key_is_the_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PINECONE_API_KEY", "from-env")

        assert settings_from_connection({}).api_key == "from-env"

    def test_missing_key_is_a_connection_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)

        with pytest.raises(StoreConnectionError, match="No API key"):
            settings_from_connection({})

    def test_unknown_keys_are_rejected_by_name(self) -> None:
        with pytest.raises(StoreConnectionError, match=r"\['regoin'\]"):
            settings_from_connection({"api_key": "k", "regoin": "x"})

    def test_mistyped_values_are_rejected(self) -> None:
        with pytest.raises(StoreConnectionError, match="'namespace' must be"):
            settings_from_connection({"api_key": "k", "namespace": 5})
        with pytest.raises(StoreConnectionError, match="'spec' must look like"):
            settings_from_connection({"api_key": "k", "spec": "serverless"})

    def test_bad_namespace_is_rejected(self) -> None:
        with pytest.raises(StoreConnectionError, match="Namespace"):
            settings_from_connection({"api_key": "k", "namespace": "café"})

    def test_a_connection_string_without_a_key_is_a_connection_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)

        with pytest.raises(StoreConnectionError, match="No API key"):
            settings_from_connection({"connection_string": "pinecone://"})

    def test_use_grpc_is_ignored_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            settings_from_connection({"api_key": "k", "use_grpc": True})

        assert "use_grpc is ignored" in caplog.text

    def test_the_key_is_not_in_the_repr(self) -> None:
        assert API_KEY not in repr(PineconeSettings(api_key=API_KEY))


def test_registry_builds_the_native_adapter() -> None:
    adapter = create_adapter("pinecone", schema(), {"api_key": API_KEY})

    assert isinstance(adapter, PineconeAdapter)
    assert adapter.provider == "pinecone"


def test_capabilities() -> None:
    capabilities = PineconeAdapter(
        schema(), PineconeSettings(api_key=API_KEY)
    ).capabilities

    assert set(capabilities.supported) == {
        Operation.UPSERT,
        Operation.GET,
        Operation.DELETE,
        Operation.FIND,
        Operation.LIST,
        Operation.DELETE_WHERE,
        Operation.COUNT,
        Operation.VECTOR_SEARCH,
    }
    assert not capabilities.supports(Operation.LEXICAL_SEARCH)
    assert not capabilities.supports(Operation.HYBRID_SEARCH)


@pytest.mark.asyncio
async def test_lexical_and_hybrid_raise_before_any_request() -> None:
    test = harness()

    with pytest.raises(UnsupportedOperationError):
        await test.adapter.lexical_search(LexicalQuery("a", 1, "content"))
    assert test.index.calls == []


def test_field_names_that_look_like_operators_are_rejected() -> None:
    bad = CollectionSchema("docs", 4, (FieldSpec("$or"),))

    with pytest.raises(SchemaError, match=r"\$"):
        PineconeAdapter(bad, PineconeSettings(api_key=API_KEY))


# -- control plane -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_collection_creates_the_layout_semantic_kernel_wrote() -> None:
    test = harness(exists=False, collection="documents_dim4")

    await test.adapter.ensure_collection()

    (created,) = test.indexes.created
    assert created == {
        "name": f"documents-dim4-{_digest('documents_dim4')}",
        "schema": {
            "fields": {
                "_values": {"type": "dense_vector", "dimension": 4, "metric": "cosine"}
            }
        },
        "deployment": {
            "deployment_type": "managed",
            "cloud": "aws",
            "region": "us-east-1",
        },
        "timeout": module.READY_TIMEOUT,
    }


@pytest.mark.asyncio
async def test_ensure_collection_leaves_an_existing_index_alone() -> None:
    test = harness(exists=True)

    await test.adapter.ensure_collection()
    await test.adapter.ensure_collection()

    assert test.indexes.created == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dimension", "metric", "message"),
    [(8, "cosine", "8 dimensions"), (4, "euclidean", "euclidean metric")],
)
async def test_an_incompatible_existing_index_is_refused(
    dimension: int, metric: str, message: str
) -> None:
    test = harness(exists=True)
    test.indexes.dimension, test.indexes.metric = dimension, metric

    with pytest.raises(IncompatibleCollectionError, match=message):
        await test.adapter.ensure_collection()


@pytest.mark.asyncio
async def test_collection_exists_and_delete() -> None:
    test = harness(exists=False)

    assert await test.adapter.collection_exists() is False
    await test.adapter.delete_collection()
    assert test.indexes.deleted == []

    await test.adapter.ensure_collection()
    assert await test.adapter.collection_exists() is True
    await test.adapter.upsert([record("a")])
    await test.adapter.delete_collection()

    assert test.indexes.deleted == ["contract-docs"]
    assert test.index.closed == 1
    assert await test.adapter.collection_exists() is False


@pytest.mark.asyncio
async def test_a_named_namespace_is_the_collection() -> None:
    """With a named namespace the collection is (index, namespace) (042 review).

    Deleting it deletes the namespace's records only; the index, which other
    namespaces share, stays. Existence and count read the namespace.
    """
    test = harness(exists=True, namespace="team-a", namespaces={"team-b": 4})

    assert await test.adapter.collection_exists() is False
    assert await test.adapter.count() == 0
    await test.adapter.delete_collection()
    assert test.index.names("delete") == []  # nothing in the namespace

    test.index.namespaces = {"team-a": 2, "team-b": 4}
    assert await test.adapter.collection_exists() is True
    assert await test.adapter.count() == 2
    await test.adapter.delete_collection()

    assert test.indexes.deleted == []
    assert test.indexes.exists is True
    (call,) = test.index.names("delete")
    assert call["delete_all"] is True and call["namespace"] == "team-a"
    assert test.index.namespaces == {"team-b": 4}
    assert await test.adapter.collection_exists() is False


@pytest.mark.asyncio
async def test_the_default_namespace_collection_is_the_index() -> None:
    test = harness(exists=True, namespaces={"__default__": 0})
    assert await test.adapter.collection_exists() is True
    await test.adapter.delete_collection()
    assert test.indexes.deleted == ["contract-docs"]
    assert test.index.names("delete") == []


@pytest.mark.asyncio
async def test_reads_on_a_missing_index_are_empty_and_send_nothing() -> None:
    test = harness(exists=False)
    adapter = test.adapter

    assert await adapter.count() == 0
    assert await adapter.get(["a"]) == []
    assert await adapter.find(RecordFilter.where(source_path="a.md")) == []
    assert (await adapter.list_page()).records == ()
    assert await adapter.vector_search(VectorQuery([1, 0, 0, 0], 3)) == []
    await adapter.delete(["a"])
    assert await adapter.delete_where(RecordFilter.where(source_path="a.md")) == 0
    assert test.index.calls == []
    assert test.indexes.created == []


@pytest.mark.asyncio
async def test_upsert_creates_a_missing_index() -> None:
    test = harness(exists=False)

    await test.adapter.upsert([record("a")])

    assert len(test.indexes.created) == 1
    assert "a" in test.index.stored


# -- lifecycle -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_data_handle_is_reused_and_close_releases_both() -> None:
    test = harness()

    await test.adapter.upsert([record("a")])
    await test.adapter.get(["a"])
    await test.adapter.count()
    assert test.client.hosts == ["index.example"]

    await test.adapter.close()
    await test.adapter.close()

    assert test.index.closed == 1
    assert test.client.closed == 1


@pytest.mark.asyncio
async def test_the_adapter_is_usable_after_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test = harness()
    await test.adapter.upsert([record("a")])
    await test.adapter.close()
    rebuilt = harness()
    monkeypatch.setattr(test.adapter, "_pinecone", lambda: rebuilt.client)

    await test.adapter.upsert([record("b")])

    assert "b" in rebuilt.index.stored
    assert rebuilt.client.hosts == ["index.example"]


@pytest.mark.asyncio
async def test_an_async_context_closes_the_adapter() -> None:
    test = harness()

    async with test.adapter as adapter:
        await adapter.count()

    assert test.client.closed == 1


# -- writes -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_records_map_to_vectors_with_ids_and_metadata_unchanged() -> None:
    test = harness(namespace="team-a")

    await test.adapter.upsert([record("docs/a.md_chunk_0", content="hello")])

    (call,) = test.index.names("upsert")
    (vector,) = cast(list[Vector], call["vectors"])
    assert call["namespace"] == "team-a"
    assert vector.id == "docs/a.md_chunk_0"
    assert vector.values == [1.0, 0.0, 0.0, 0.0]
    assert vector.metadata == {
        "source_path": "a.md",
        "chunk_index": 0,
        "content": "hello",
        "mtime": 1.5,
        "pinned": False,
    }


@pytest.mark.asyncio
async def test_none_fields_are_omitted_and_restored_with_type_defaults() -> None:
    test = harness()
    sparse = Record(
        "a",
        {"source_path": "a.md", "content": None, "chunk_index": None},
        [1.0, 0.0, 0.0, 0.0],
    )

    await test.adapter.upsert([sparse])

    assert test.index.stored["a"].metadata == {"source_path": "a.md"}
    (read,) = await test.adapter.get(["a"])
    assert dict(read.fields) == {
        "source_path": "a.md",
        "chunk_index": 0,
        "content": "",
        "mtime": 0.0,
        "pinned": False,
    }
    assert isinstance(read.fields["chunk_index"], int)


@pytest.mark.asyncio
async def test_numbers_come_back_in_the_schema_type() -> None:
    test = harness()
    test.index.stored["a"] = Vector(
        id="a",
        values=[1.0, 0.0, 0.0, 0.0],
        metadata={"chunk_index": 3.0, "mtime": 2, "pinned": True, "content": 7},
    )

    (read,) = await test.adapter.get(["a"])

    assert read.fields["chunk_index"] == 3
    assert isinstance(read.fields["chunk_index"], int)
    assert isinstance(read.fields["mtime"], float)
    assert read.fields["pinned"] is True
    assert read.fields["content"] == "7"


@pytest.mark.asyncio
async def test_upsert_splits_requests_by_record_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "UPSERT_MAX_RECORDS", 2)
    test = harness()

    await test.adapter.upsert([record(f"r{i}") for i in range(5)])

    sizes = [len(cast(list[Vector], c["vectors"])) for c in test.index.names("upsert")]
    assert sizes == [2, 2, 1]


@pytest.mark.asyncio
async def test_upsert_splits_requests_by_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "UPSERT_MAX_BYTES", 700)
    test = harness()
    big = "x" * 300

    await test.adapter.upsert([record(f"r{i}", content=big) for i in range(4)])

    sizes = [len(cast(list[Vector], c["vectors"])) for c in test.index.names("upsert")]
    assert sum(sizes) == 4
    assert len(sizes) > 1
    assert all(size >= 1 for size in sizes)


@pytest.mark.asyncio
async def test_an_empty_upsert_sends_nothing() -> None:
    test = harness()

    await test.adapter.upsert([])

    assert test.index.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id", ["café", "a\0b", "x" * 513])
async def test_ids_pinecone_cannot_store_are_rejected_before_any_write(
    bad_id: str,
) -> None:
    test = harness()

    with pytest.raises(SchemaError, match="not a valid Pinecone vector ID"):
        await test.adapter.upsert([record("ok"), record(bad_id)])

    assert test.index.calls == []


@pytest.mark.asyncio
async def test_a_record_outside_the_schema_is_rejected_before_any_write() -> None:
    test = harness()
    bad = Record("x", {"nope": 1}, [1.0, 0.0, 0.0, 0.0])

    with pytest.raises(SchemaError):
        await test.adapter.upsert([record("ok"), bad])

    assert test.index.calls == []


@pytest.mark.asyncio
async def test_upsert_waits_until_the_last_record_is_readable() -> None:
    test = harness(lag=3)

    await test.adapter.upsert([record("a"), record("b")])

    probes = test.index.names("fetch")
    assert len(probes) == 4
    assert all(probe["ids"] == ["b"] for probe in probes)


@pytest.mark.asyncio
async def test_the_wait_tolerates_floats_that_lose_digits(
    caplog: pytest.LogCaptureFixture,
) -> None:
    test = harness(float_digits=4)
    stamped = Record(
        "a",
        {"source_path": "a.md", "mtime": 1794294901.5036001, "chunk_index": 1},
        [1.0, 0.0, 0.0, 0.0],
    )

    with caplog.at_level(logging.WARNING):
        await test.adapter.upsert([stamped])

    assert "did not show a write" not in caplog.text
    assert len(test.index.names("fetch")) == 1


@pytest.mark.asyncio
async def test_a_write_that_never_shows_logs_and_returns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(module, "VISIBILITY_TIMEOUT", 0.0)
    test = harness(lag=10**6)

    with caplog.at_level(logging.WARNING):
        await test.adapter.upsert([record("a")])

    assert "did not show a write" in caplog.text


# -- point reads -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_keeps_request_order_and_skips_missing_ids() -> None:
    test = harness()
    await test.adapter.upsert([record("a"), record("b"), record("c")])

    found = await test.adapter.get(["c", "missing", "a"])

    assert [r.id for r in found] == ["c", "a"]
    assert all(r.vector is None for r in found)


@pytest.mark.asyncio
async def test_get_returns_vectors_on_request() -> None:
    test = harness()
    await test.adapter.upsert([record("a", vector=[0.5, 0.5, 0.0, 0.0])])

    (found,) = await test.adapter.get(["a"], include_vectors=True)

    assert found.vector == [0.5, 0.5, 0.0, 0.0]


@pytest.mark.asyncio
async def test_get_batches_ids_and_skips_unusable_ones() -> None:
    test = harness()
    ids = [f"id-{i}" for i in range(250)]

    await test.adapter.get([*ids, "café", ""])

    sizes = [len(cast(list[str], c["ids"])) for c in test.index.names("fetch")]
    assert sizes == [100, 100, 50]


@pytest.mark.asyncio
async def test_get_keeps_each_request_under_the_url_budget() -> None:
    test = harness()
    ids = [f"{i:03d}-" + "x" * 400 for i in range(25)]

    await test.adapter.get(ids)

    for call in test.index.names("fetch"):
        assert sum(len(key) for key in cast(list[str], call["ids"])) <= 4000
    assert sum(len(cast(list[str], c["ids"])) for c in test.index.names("fetch")) == 25


@pytest.mark.asyncio
async def test_get_without_usable_ids_sends_nothing() -> None:
    test = harness()

    assert await test.adapter.get([]) == []
    assert await test.adapter.get(["café"]) == []
    assert test.index.calls == []


@pytest.mark.asyncio
async def test_delete_batches_ignores_unusable_ids_and_waits() -> None:
    test = harness()
    await test.adapter.upsert([record(f"r{i}") for i in range(3)])
    test.index.calls.clear()

    await test.adapter.delete(["r0", "r0", "café", "missing"])

    (call,) = test.index.names("delete")
    assert call["ids"] == ["r0", "missing"]
    assert "r0" not in test.index.stored
    assert test.index.names("fetch")[-1]["ids"] == ["missing"]


@pytest.mark.asyncio
async def test_delete_splits_requests_at_one_thousand_ids() -> None:
    test = harness()

    await test.adapter.delete([f"r{i}" for i in range(2500)])

    sizes = [len(cast(list[str], c["ids"])) for c in test.index.names("delete")]
    assert sizes == [1000, 1000, 500]


# -- filters, find and delete_where ----------------------------------------------------


@pytest.mark.asyncio
async def test_find_pushes_an_exact_filter_to_fetch_by_metadata() -> None:
    test = harness()
    await test.adapter.upsert(
        [record("a0"), record("a1", index=1), record("b0", source="b.md")]
    )

    found = await test.adapter.find(
        RecordFilter.where(source_path="a.md", chunk_index=1)
    )

    assert [r.id for r in found] == ["a1"]
    assert found[0].vector is None
    (call,) = test.index.names("fetch_by_metadata")
    assert call["filter"] == {
        "$and": [
            {"source_path": {"$eq": "a.md"}},
            {"chunk_index": {"$eq": 1}},
        ]
    }
    assert test.index.names("list_paginated") == []


@pytest.mark.asyncio
async def test_a_single_clause_filter_has_no_and() -> None:
    test = harness()

    await test.adapter.find(RecordFilter.where(source_path="a.md"))

    (call,) = test.index.names("fetch_by_metadata")
    assert call["filter"] == {"source_path": {"$eq": "a.md"}}


@pytest.mark.asyncio
async def test_find_follows_pagination_to_the_end() -> None:
    test = harness()
    await test.adapter.upsert([record(f"a{i}", index=i) for i in range(5)])

    found = await test.adapter.find(RecordFilter.where(source_path="a.md"))

    assert sorted(r.id for r in found) == [f"a{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_find_limit_caps_the_result_and_the_page_size() -> None:
    test = harness()
    await test.adapter.upsert([record(f"a{i}", index=i) for i in range(5)])

    found = await test.adapter.find(RecordFilter.where(source_path="a.md"), limit=2)

    assert len(found) == 2
    assert test.index.names("fetch_by_metadata")[0]["limit"] == 2
    assert (
        await test.adapter.find(RecordFilter.where(source_path="a.md"), limit=0) == []
    )


@pytest.mark.asyncio
async def test_find_falls_back_to_a_scan_when_the_route_is_missing() -> None:
    test = harness(metadata_fetch=False)
    await test.adapter.upsert(
        [record("a0"), record("a1", index=1), record("b0", source="b.md")]
    )
    test.index.calls.clear()

    first = await test.adapter.find(RecordFilter.where(source_path="a.md"))
    second = await test.adapter.find(RecordFilter.where(source_path="b.md"))

    assert sorted(r.id for r in first) == ["a0", "a1"]
    assert [r.id for r in second] == ["b0"]
    assert len(test.index.names("fetch_by_metadata")) == 1  # probed once
    assert len(test.index.names("list_paginated")) == 2


@pytest.mark.asyncio
async def test_the_scan_is_exact_for_types_and_limit() -> None:
    test = harness(metadata_fetch=False)
    await test.adapter.upsert(
        [
            record("a0"),
            record("a1", index=1),
            Record(
                "p",
                {"source_path": "p.md", "pinned": True},
                [1.0, 0.0, 0.0, 0.0],
            ),
        ]
    )

    assert [r.id for r in await test.adapter.find(RecordFilter.where(pinned=True))] == [
        "p"
    ]
    assert [
        r.id for r in await test.adapter.find(RecordFilter.where(chunk_index=1))
    ] == ["a1"]
    assert (
        len(await test.adapter.find(RecordFilter.where(source_path="a.md"), limit=1))
        == 1
    )


@pytest.mark.asyncio
async def test_an_empty_filter_scans_every_record() -> None:
    test = harness()
    await test.adapter.upsert([record(f"r{i}") for i in range(3)])

    found = await test.adapter.find(RecordFilter())

    assert sorted(r.id for r in found) == ["r0", "r1", "r2"]
    assert test.index.names("fetch_by_metadata") == []


@pytest.mark.asyncio
async def test_delete_where_deletes_the_matching_ids_and_counts_them() -> None:
    test = harness()
    await test.adapter.upsert(
        [record("a0"), record("a1", index=1), record("b0", source="b.md")]
    )

    removed = await test.adapter.delete_where(RecordFilter.where(source_path="a.md"))

    assert removed == 2
    assert list(test.index.stored) == ["b0"]
    assert await test.adapter.delete_where(RecordFilter.where(source_path="a.md")) == 0


@pytest.mark.asyncio
async def test_replace_source_removes_stale_chunks() -> None:
    test = harness()
    await test.adapter.upsert(
        [record("a0"), record("a1", index=1), record("a2", index=2)]
    )

    await test.adapter.replace_source(
        RecordFilter.where(source_path="a.md"), [record("a0", content="new")]
    )

    assert list(test.index.stored) == ["a0"]
    assert test.index.stored["a0"].metadata is not None
    assert test.index.stored["a0"].metadata["content"] == "new"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("clause", "message"),
    [
        (RecordFilter.where(source_path=None), "cannot be filtered on None"),
        (RecordFilter.where(chunk_index="1"), "does not fit integer"),
        (RecordFilter.where(chunk_index=True), "does not fit integer"),
        (RecordFilter.where(pinned=1), "does not fit boolean"),
        (RecordFilter.where(source_path=1), "does not fit string"),
        (RecordFilter.where(unknown="x"), "no field 'unknown'"),
    ],
)
async def test_filters_that_cannot_be_exact_raise(
    clause: RecordFilter, message: str
) -> None:
    test = harness()

    with pytest.raises(SchemaError, match=message):
        await test.adapter.find(clause)
    with pytest.raises(SchemaError, match=message):
        await test.adapter.vector_search(VectorQuery([1, 0, 0, 0], 1, clause))

    assert test.index.calls == []


@pytest.mark.asyncio
async def test_a_float_filter_accepts_an_integer_value() -> None:
    test = harness()

    await test.adapter.find(RecordFilter.where(mtime=2))

    (call,) = test.index.names("fetch_by_metadata")
    assert call["filter"] == {"mtime": {"$eq": 2}}


# -- enumeration ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_page_walks_every_record_once() -> None:
    test = harness()
    await test.adapter.upsert([record(f"r{i}") for i in range(5)])

    seen: list[str] = []
    cursor: str | None = None
    while True:
        page = await test.adapter.list_page(cursor=cursor, limit=2)
        assert len(page.records) <= 2
        assert all(r.vector is None for r in page.records)
        seen.extend(r.id for r in page.records)
        cursor = page.next_cursor
        if cursor is None:
            break

    assert seen == [f"r{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_list_page_asks_for_at_most_one_hundred_ids_per_call() -> None:
    test = harness()
    test.index.stored.update(
        {f"r{i}": Vector(id=f"r{i}", values=[0.0] * 4) for i in range(250)}
    )

    page = await test.adapter.list_page(limit=230)

    assert len(page.records) == 230
    limits = [c["limit"] for c in test.index.names("list_paginated")]
    assert limits == [100, 100, 30]
    assert page.next_cursor is not None
    rest = await test.adapter.list_page(cursor=page.next_cursor, limit=230)
    assert len(rest.records) == 20
    assert rest.next_cursor is None


@pytest.mark.asyncio
async def test_list_page_rejects_a_bad_limit() -> None:
    test = harness()

    with pytest.raises(SchemaError):
        await test.adapter.list_page(limit=0)


# -- search ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vector_search_pushes_the_filter_down_and_clamps_scores() -> None:
    test = harness(namespace="ns")
    await test.adapter.upsert(
        [record("a0"), record("a1", index=1), record("b0", source="b.md")]
    )

    hits = await test.adapter.vector_search(
        VectorQuery([1, 0, 0, 0], 5, RecordFilter.where(source_path="a.md"))
    )

    assert [h.record.id for h in hits] == ["a0", "a1"]
    assert [h.score for h in hits] == [1.0, 1.0]  # 2.0 and 1.0 clamp to 1.0
    assert all(h.record.vector is None for h in hits)
    (call,) = test.index.names("query")
    assert call["filter"] == {"source_path": {"$eq": "a.md"}}
    assert call["include_metadata"] is True
    assert call["include_values"] is False
    assert call["namespace"] == "ns"
    assert hits[0].record.fields["content"] == "text"


@pytest.mark.asyncio
async def test_scores_below_zero_clamp_to_zero() -> None:
    test = harness()
    test.index.stored["a"] = Vector(id="a", values=[1, 0, 0, 0], metadata={})

    async def negative(**_: object) -> QueryResponse:
        return QueryResponse(matches=[ScoredVector(id="a", score=-0.4)], namespace="")

    test.index.query = negative  # type: ignore[method-assign, assignment]

    (hit,) = await test.adapter.vector_search(VectorQuery([1, 0, 0, 0], 1))

    assert hit.score == 0.0


@pytest.mark.asyncio
async def test_vector_search_caps_top_k_and_validates_the_vector() -> None:
    test = harness()

    await test.adapter.vector_search(VectorQuery([1, 0, 0, 0], 50_000))

    assert test.index.names("query")[0]["top_k"] == 10_000
    with pytest.raises(SchemaError, match="5 dimensions"):
        await test.adapter.vector_search(VectorQuery([1.0] * 5, 1))
    assert await test.adapter.vector_search(VectorQuery([1, 0, 0, 0], 0)) == []


# -- count -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_count_reads_the_namespace_of_the_adapter() -> None:
    default = harness(namespaces={"__default__": 7, "other": 3})
    named = harness(namespace="other", namespaces={"__default__": 7, "other": 3})
    empty = harness(namespace="absent", namespaces={"__default__": 7})

    assert await default.adapter.count() == 7
    assert await named.adapter.count() == 3
    assert await empty.adapter.count() == 0


# -- errors ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sdk_errors_become_connection_errors() -> None:
    test = harness()
    test.index.failure = PineconeError("the index is on fire")

    with pytest.raises(StoreConnectionError, match="on fire") as raised:
        await test.adapter.get(["a"])

    assert raised.value.__cause__ is not None
    assert "pinecone" in str(raised.value)


@pytest.mark.asyncio
async def test_rejected_input_becomes_a_schema_error() -> None:
    test = harness()
    test.index.failure = PineconeValueError("top_k exceeds the maximum")

    with pytest.raises(SchemaError, match="top_k"):
        await test.adapter.vector_search(VectorQuery([1, 0, 0, 0], 1))


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["get", "upsert", "count", "find", "control"])
async def test_the_api_key_never_appears_in_an_error(operation: str) -> None:
    test = harness()
    await test.adapter.get(["a"])  # open the data handle
    leaky = PineconeError(f"401 for key {API_KEY} on host x")
    test.index.failure = leaky
    test.indexes.failure = leaky

    with pytest.raises(StoreConnectionError) as raised:
        if operation == "get":
            await test.adapter.get(["a"])
        elif operation == "upsert":
            await test.adapter.upsert([record("a")])
        elif operation == "count":
            await test.adapter.count()
        elif operation == "find":
            await test.adapter.find(RecordFilter.where(source_path="a"))
        else:
            await test.adapter.collection_exists()

    assert API_KEY not in str(raised.value)
    assert "***" in str(raised.value)
    assert raised.value.__cause__ is None  # the original message held the key


@pytest.mark.parametrize("bad", [f"{API_KEY}\n", f"{API_KEY}\r\n", f"a\tb{API_KEY}"])
def test_a_key_with_a_control_character_is_rejected_without_echoing_it(
    bad: str,
) -> None:
    for connection in ({"api_key": bad}, {"connection_string": bad}):
        with pytest.raises(StoreConnectionError, match="control character") as raised:
            settings_from_connection(connection)
        assert API_KEY not in "".join(traceback.format_exception(raised.value))


@pytest.mark.asyncio
@pytest.mark.parametrize("echoed", [f"b'{API_KEY}\\n'", f"key={API_KEY}"])
async def test_an_echoed_key_form_is_masked_and_the_chain_dropped(
    echoed: str,
) -> None:
    test = harness()
    await test.adapter.get(["a"])
    test.index.failure = PineconeError(f"Illegal header value {echoed}")

    with pytest.raises(StoreConnectionError) as raised:
        await test.adapter.get(["a"])

    text = "".join(traceback.format_exception(raised.value))
    assert API_KEY not in text
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__


@pytest.mark.asyncio
async def test_a_connection_string_is_masked_as_a_whole() -> None:
    settings = settings_from_connection(
        {"connection_string": f"pinecone://{API_KEY}@ns"}
    )
    adapter = PineconeAdapter(schema("docs"), settings)

    assert f"pinecone://{API_KEY}@ns" in adapter._secrets
    assert adapter._scrub(f"bad pinecone://{API_KEY}@ns") == "bad ***"


@pytest.mark.asyncio
async def test_a_missing_index_host_is_a_connection_error() -> None:
    test = harness()

    class NoHost(FakeIndexes):
        async def describe(self, name: str) -> FakeModel:
            return FakeModel(None, IndexSchema(fields={}))

    test.client.indexes = NoHost()

    with pytest.raises(StoreConnectionError, match="no host yet"):
        await test.adapter.collection_exists()
