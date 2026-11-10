"""Unit tests for the native Azure AI Search adapter, on a fake of its clients.

The fake (``tests/contract/retrieval/azure_search_fake.py``) returns the SDK's
own model types and enforces the service's rules for keys, filters and index
definitions. Nothing here talks to Azure AI Search; the live run is open in the
042 exec plan. ``test_azure_ai_search_requests.py`` checks the REST requests the
real SDK builds from the adapter's calls.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections.abc import Mapping, Sequence

import pytest
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceExistsError,
)
from azure.search.documents.indexes.models import SearchIndex
from azure.search.documents.models import IndexingResult, VectorizedQuery

from holodeck.lib.retrieval import (
    CollectionSchema,
    FieldSpec,
    FieldType,
    HybridQuery,
    IncompatibleCollectionError,
    LexicalQuery,
    Operation,
    Record,
    RecordFilter,
    SchemaError,
    ScoreKind,
    StoreConnectionError,
    VectorQuery,
    create_adapter,
)
from holodeck.lib.retrieval.adapters import azure_ai_search as module
from holodeck.lib.retrieval.adapters.azure_ai_search import (
    AzureAISearchAdapter,
    AzureSearchSettings,
    build_index,
    decode_key,
    encode_key,
    normalize_index_name,
    odata_string,
    settings_from_connection,
    similarity_from_score,
)
from tests.contract.retrieval.azure_search_fake import (
    FakeSearchClient,
    FakeSearchService,
)

ENDPOINT = "https://svc.search.windows.net"
API_KEY = "AzureKey-SECRET-123"
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


def adapter_on(
    service: FakeSearchService, name: str = "contract-docs", **settings: object
) -> AzureAISearchAdapter:
    return AzureAISearchAdapter(
        schema(name),
        AzureSearchSettings(endpoint=ENDPOINT, api_key=API_KEY, **settings),  # type: ignore[arg-type]
        client_factory=service.pair,
    )


@pytest.fixture
def service() -> FakeSearchService:
    return FakeSearchService()


@pytest.fixture
def adapter(service: FakeSearchService) -> AzureAISearchAdapter:
    return adapter_on(service)


@pytest.fixture(autouse=True)
def no_poll_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_POLL_START", 0.0)
    monkeypatch.setattr(module, "_POLL_MAX", 0.0)


# -- index names -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("documents_dim1536", "documents_dim1536"),
        ("contract_0123456789ab", "contract_0123456789ab"),
        ("hierarchical_docs_policy_dim768", "hierarchical_docs_policy_dim768"),
        ("my-index", "my-index"),
        ("Product Catalog.v2", "product-catalog-v2-{}"),
        ("a__b", "a-b-{}"),
        ("a--b", "a-b-{}"),
        ("-edge-", "edge-{}"),
        ("_lead", "lead-{}"),
        ("a" + "-" + "_", "a-{}"),
        ("x", "x-{}"),
    ],
)
def test_index_names_follow_the_service_rules(name: str, expected: str) -> None:
    """A valid name is unchanged; any other change appends a hash of the original."""
    assert normalize_index_name(name) == expected.format(_digest(name))


def _digest(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:8]


def test_names_that_clean_to_the_same_text_get_different_indexes() -> None:
    names = ["a_b", "a-b", "A-B", "a.b", "a b", "A_b", "a__b", "a--b"]
    indexes = [normalize_index_name(name) for name in names]
    assert len(set(indexes)) == len(names)
    assert normalize_index_name("a-b") == "a-b"
    assert normalize_index_name("a_b") == "a_b"


@pytest.mark.parametrize("name", ["x", "___", "日本語", "a" * 200, "UPPER_and spaces"])
def test_every_normalized_name_is_a_valid_index_name(name: str) -> None:
    result = normalize_index_name(name)

    assert 2 <= len(result) <= 128
    assert re.fullmatch(r"[a-z0-9]+([-_][a-z0-9]+)*", result)
    assert result == normalize_index_name(name)


def test_long_names_keep_their_head_and_a_hash() -> None:
    first = "hierarchical_docs_" + "x" * 150 + "_dim1536"
    second = "hierarchical_docs_" + "x" * 150 + "_dim768"

    one, two = normalize_index_name(first), normalize_index_name(second)

    assert len(one) <= 128 and one != two
    assert one.startswith("hierarchical-docs-xxx")


def test_index_name_setting_replaces_the_collection_name(
    service: FakeSearchService,
) -> None:
    assert adapter_on(service, index_name="Shared Index").index_name == (
        f"shared-index-{_digest('Shared Index')}"
    )
    assert adapter_on(service, name="my_docs").index_name == "my_docs"


# -- record IDs as document keys -------------------------------------------------------

IDS = [
    "docs/policy.md_chunk_0",
    "C:\\data\\file name.txt_chunk_12",
    "résumé/日本語.md_chunk_3",
    "a" * 700,
    "_starts_with_underscore",
    "b64-looks-encoded",
    "b64-Zm9v",
    "with space",
    "plain_chunk_0",
    "dash-and=equals_",
    "3f2c9d2e-6c1b-5f0e-9a43-0c1d2e3f4a5b",
]


@pytest.mark.parametrize("record_id", IDS)
def test_keys_round_trip_and_are_valid_for_the_service(record_id: str) -> None:
    key = encode_key(record_id)

    assert re.fullmatch(r"[A-Za-z0-9=-][A-Za-z0-9_=-]*", key)
    assert len(key) <= 1024
    assert decode_key(key) == record_id


def test_a_valid_id_is_its_own_key_so_sk_collections_stay_readable() -> None:
    assert encode_key("plain_chunk_0") == "plain_chunk_0"
    assert decode_key("plain_chunk_0") == "plain_chunk_0"
    assert encode_key("3f2c9d2e-6c1b-5f0e-9a43-0c1d2e3f4a5b").count("b64-") == 0


def test_encoding_is_one_to_one() -> None:
    keys = {encode_key(record_id) for record_id in IDS}

    assert len(keys) == len(IDS)


def test_a_raw_key_that_looks_encoded_reads_as_itself() -> None:
    # "b64-Zm9v" would decode to "foo", but "foo" encodes to "foo", not to this key.
    assert decode_key("b64-Zm9v") == "b64-Zm9v"
    assert decode_key(encode_key("b64-Zm9v")) == "b64-Zm9v"
    assert decode_key("b64-%%%") == "b64-%%%"


def test_ids_that_do_not_fit_raise_schema_error() -> None:
    with pytest.raises(SchemaError, match="too long"):
        encode_key("é" * 600)
    with pytest.raises(SchemaError):
        encode_key("")


# -- OData -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "literal"),
    [
        ("plain", "'plain'"),
        ("o'brien", "'o''brien'"),
        ("''", "''''''"),
        ("a' or id ne '", "'a'' or id ne '''"),
        ("日本語 \u00e9", "'日本語 \u00e9'"),
        ("back\\slash", "'back\\slash'"),
        ("", "''"),
    ],
)
def test_string_literals_double_single_quotes(value: str, literal: str) -> None:
    assert odata_string(value) == literal


def test_filters_use_each_type_literal(adapter: AzureAISearchAdapter) -> None:
    expression = adapter._filter_expression(
        RecordFilter.where(source_path="a.md", chunk_index=2)
    )

    assert expression == "source_path eq 'a.md' and chunk_index eq 2"
    assert adapter._filter_expression(RecordFilter()) is None


def test_filter_on_a_field_that_is_not_indexed_fails_before_a_request(
    adapter: AzureAISearchAdapter,
) -> None:
    with pytest.raises(SchemaError, match="not indexed"):
        adapter._filter_expression(RecordFilter.where(mtime=1.5))


@pytest.mark.parametrize(
    "clause",
    [
        {"chunk_index": "1"},
        {"chunk_index": True},
        {"source_path": 5},
        {"chunk_index": float("nan")},
    ],
)
def test_filter_values_must_fit_the_field_type(
    adapter: AzureAISearchAdapter, clause: Mapping[str, object]
) -> None:
    with pytest.raises(SchemaError, match="does not fit|another type"):
        adapter._filter_expression(RecordFilter.where(**clause))  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_hostile_values_match_only_themselves(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    hostile = [
        "o'brien.md",
        "a' or source_path ne 'x",
        "x' and chunk_index eq 0 and source_path eq '",
        "日本語/résumé.md",
        "back\\slash.md",
        "''",
    ]
    await adapter.ensure_collection()
    await adapter.upsert(
        [record(f"id/{n}.md_chunk_0", source=value) for n, value in enumerate(hostile)]
        + [record("other_chunk_0", source="innocent.md")]
    )

    for n, value in enumerate(hostile):
        found = await adapter.find(RecordFilter.where(source_path=value))
        assert [hit.id for hit in found] == [f"id/{n}.md_chunk_0"], value
        assert found[0].fields["source_path"] == value
    assert await adapter.find(RecordFilter.where(source_path="o'brien")) == []
    sent = [call["filter"] for call in service.names("search")]
    assert "source_path eq 'o''brien.md'" in sent


# -- index definition ------------------------------------------------------------------


def test_index_definition_is_the_expected_layout() -> None:
    index = build_index(schema(), "contract-docs")

    assert index.as_dict() == {
        "name": "contract-docs",
        "fields": [
            {
                "name": "id",
                "type": "Edm.String",
                "key": True,
                "searchable": False,
                "filterable": True,
                "sortable": True,
                "facetable": False,
                "retrievable": True,
            },
            _field("source_path", "Edm.String", filterable=True),
            _field("chunk_index", "Edm.Int64", filterable=True),
            _field("content", "Edm.String", searchable=True),
            _field("mtime", "Edm.Double"),
            _field("pinned", "Edm.Boolean"),
            {
                "name": "embedding",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "retrievable": True,
                "dimensions": DIMENSIONS,
                "vectorSearchProfile": "embedding_profile",
            },
        ],
        "vectorSearch": {
            "profiles": [
                {"name": "embedding_profile", "algorithm": "embedding_algorithm"}
            ],
            "algorithms": [
                {
                    "name": "embedding_algorithm",
                    "hnswParameters": {"metric": "cosine"},
                    "kind": "hnsw",
                }
            ],
        },
    }


def _field(
    name: str, edm: str, *, searchable: bool = False, filterable: bool = False
) -> dict[str, object]:
    return {
        "name": name,
        "type": edm,
        "searchable": searchable,
        "filterable": filterable,
        "sortable": False,
        "facetable": False,
        "retrievable": True,
    }


def test_only_string_fields_can_be_full_text() -> None:
    bad = CollectionSchema(
        "docs", 4, (FieldSpec("n", FieldType.INTEGER, full_text=True),)
    )

    with pytest.raises(SchemaError, match="only string fields"):
        AzureAISearchAdapter(bad, AzureSearchSettings(endpoint=ENDPOINT))


# -- connection settings ---------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(module.ENDPOINT_ENV, raising=False)
    monkeypatch.delenv(module.API_KEY_ENV, raising=False)


@pytest.mark.usefixtures("clean_env")
class TestSettings:
    def test_connection_string_is_the_endpoint(self) -> None:
        settings = settings_from_connection(
            {"connection_string": f"{ENDPOINT}/", "api_key": API_KEY}
        )

        assert settings == AzureSearchSettings(ENDPOINT, API_KEY, None)

    @pytest.mark.parametrize("key", ["endpoint", "search_endpoint"])
    def test_endpoint_aliases(self, key: str) -> None:
        assert settings_from_connection({key: ENDPOINT}).endpoint == ENDPOINT

    def test_explicit_endpoint_wins_over_connection_string(self) -> None:
        settings = settings_from_connection(
            {"endpoint": ENDPOINT, "connection_string": "https://other.example"}
        )

        assert settings.endpoint == ENDPOINT

    def test_environment_fallbacks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(module.ENDPOINT_ENV, ENDPOINT)
        monkeypatch.setenv(module.API_KEY_ENV, API_KEY)

        assert settings_from_connection({}) == AzureSearchSettings(ENDPOINT, API_KEY)

    def test_no_key_means_entra_id(self) -> None:
        assert settings_from_connection({"endpoint": ENDPOINT}).api_key is None

    def test_index_and_collection_names(self) -> None:
        settings = settings_from_connection(
            {"endpoint": ENDPOINT, "index_name": "idx", "collection_name": "c"}
        )

        assert settings.index_name == "idx"

    def test_unknown_keys_are_an_error(self) -> None:
        with pytest.raises(StoreConnectionError, match=r"Unknown connection settings"):
            settings_from_connection({"endpoint": ENDPOINT, "api_version": "x"})

    def test_an_endpoint_is_required(self) -> None:
        with pytest.raises(StoreConnectionError, match="No endpoint"):
            settings_from_connection({"api_key": API_KEY})

    @pytest.mark.parametrize(
        "endpoint",
        ["http://svc.search.windows.net", "svc.search.windows.net", "https://"],
    )
    def test_the_endpoint_must_be_https(self, endpoint: str) -> None:
        with pytest.raises(StoreConnectionError, match="https"):
            settings_from_connection({"endpoint": endpoint})

    def test_settings_must_be_strings(self) -> None:
        with pytest.raises(StoreConnectionError, match="must be a string"):
            settings_from_connection({"endpoint": ENDPOINT, "api_key": 12})

    def test_the_key_is_not_in_the_repr(self) -> None:
        assert API_KEY not in repr(AzureSearchSettings(ENDPOINT, API_KEY))

    def test_registry_builds_the_adapter_without_a_request(self) -> None:
        built = create_adapter("azure-ai-search", schema(), {"endpoint": ENDPOINT})

        assert isinstance(built, AzureAISearchAdapter)
        assert built.provider == "azure-ai-search"
        assert built.capabilities.supported == frozenset(Operation)


# -- capabilities ----------------------------------------------------------------------


def test_every_operation_is_supported(adapter: AzureAISearchAdapter) -> None:
    for operation in Operation:
        assert adapter.capabilities.supports(operation), operation


# -- score conversion ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "similarity"),
    [
        (1.0, 1.0),  # cosine 1: distance 0
        (1 / 1.5, 0.5),  # distance 0.5
        (0.5, 0.0),  # orthogonal: distance 1
        (1 / 3, 0.0),  # opposite: distance 2, cosine -1, clamped
        (0.4, 0.0),  # cosine -0.5, clamped
        (0.0, 0.0),
        (-1.0, 0.0),
        (float("nan"), 0.0),
        (1.2, 1.0),  # above the documented range, clamped
    ],
)
def test_score_conversion(score: float, similarity: float) -> None:
    assert similarity_from_score(score) == pytest.approx(similarity)


def test_conversion_matches_the_documented_formula_and_keeps_the_order() -> None:
    def documented(score: float) -> float:
        cosine_distance = (1 - score) / score  # ScoreToSimilarity in the docs
        return -cosine_distance + 1

    scores = [1 / (1 + distance / 10) for distance in range(0, 21)]
    converted = [similarity_from_score(score) for score in scores]

    for score, value in zip(scores, converted, strict=True):
        assert value == pytest.approx(max(0.0, min(1.0, documented(score))))
    assert converted == sorted(converted, reverse=True)
    assert all(0.0 <= value <= 1.0 for value in converted)
    assert not math.isnan(sum(converted))


# -- lifecycle -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clients_are_built_on_first_use_and_reused(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    assert service.built == []

    await adapter.collection_exists()
    await adapter.ensure_collection()
    await adapter.count()

    assert service.built == ["contract-docs"]


@pytest.mark.asyncio
async def test_close_releases_both_clients_and_the_adapter_is_reusable(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    await adapter.ensure_collection()

    await adapter.close()
    await adapter.close()

    assert service.closed == 2
    assert await adapter.collection_exists() is True
    assert service.built == ["contract-docs", "contract-docs"]


@pytest.mark.asyncio
async def test_close_before_any_use_does_nothing(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    await adapter.close()

    assert service.built == [] and service.closed == 0


@pytest.mark.asyncio
async def test_a_closing_failure_does_not_stop_the_other_clients() -> None:
    service = FakeSearchService()
    adapter = adapter_on(service)
    await adapter.ensure_collection()
    pair = adapter._clients
    assert pair is not None

    async def explode() -> None:
        raise HttpResponseError(message="close failed")

    pair.documents.close = explode  # type: ignore[method-assign]
    await adapter.close()

    assert service.closed == 1  # the index client still closed


@pytest.mark.asyncio
async def test_the_index_is_created_with_the_expected_definition(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    await adapter.ensure_collection()

    stored = service.indexes["contract-docs"].definition
    assert stored.as_dict() == build_index(schema(), "contract-docs").as_dict()
    assert [call for call, _ in service.calls].count("create_index") == 1
    await adapter.ensure_collection()
    assert [call for call, _ in service.calls].count("create_index") == 1


@pytest.mark.asyncio
async def test_delete_collection_drops_the_index_and_ignores_a_missing_one(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    await adapter.ensure_collection()

    await adapter.delete_collection()
    await adapter.delete_collection()

    assert service.indexes == {}
    assert await adapter.count() == 0
    assert await adapter.get(["x"]) == []
    assert await adapter.find(RecordFilter.where(source_path="a.md")) == []
    assert (await adapter.list_page()).records == ()
    assert await adapter.vector_search(VectorQuery([1.0, 0.0, 0.0, 0.0], 3)) == []


@pytest.mark.asyncio
async def test_upsert_creates_a_missing_index_and_recreates_a_deleted_one(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    await adapter.upsert([record("a_chunk_0")])
    assert "contract-docs" in service.indexes

    del service.indexes["contract-docs"]  # deleted behind the adapter's back
    await adapter.upsert([record("a_chunk_0")])

    assert [r.id for r in await adapter.get(["a_chunk_0"])] == ["a_chunk_0"]


# -- compatibility with an existing index ----------------------------------------------


async def _existing(service: FakeSearchService, mutate: object) -> AzureAISearchAdapter:
    built = adapter_on(service)
    await built.ensure_collection()
    index = service.indexes["contract-docs"].definition
    assert callable(mutate)
    mutate(index)
    return adapter_on(service)


@pytest.mark.asyncio
async def test_an_index_with_other_dimensions_is_incompatible(
    service: FakeSearchService,
) -> None:
    def mutate(index: SearchIndex) -> None:
        index.fields[-1].vector_search_dimensions = 8

    other = await _existing(service, mutate)

    with pytest.raises(IncompatibleCollectionError, match="8 dimensions"):
        await other.ensure_collection()


@pytest.mark.asyncio
async def test_an_index_whose_key_cannot_be_sorted_is_incompatible(
    service: FakeSearchService,
) -> None:
    def mutate(index: SearchIndex) -> None:
        index.fields[0].sortable = False  # what Semantic Kernel creates

    other = await _existing(service, mutate)

    with pytest.raises(IncompatibleCollectionError, match="sortable"):
        await other.ensure_collection()


@pytest.mark.asyncio
async def test_an_index_missing_a_field_is_incompatible(
    service: FakeSearchService,
) -> None:
    def mutate(index: SearchIndex) -> None:
        index.fields = [f for f in index.fields if f.name != "content"]

    other = await _existing(service, mutate)

    with pytest.raises(IncompatibleCollectionError, match="no field 'content'"):
        await other.ensure_collection()


@pytest.mark.asyncio
async def test_an_indexed_field_that_is_not_filterable_is_incompatible(
    service: FakeSearchService,
) -> None:
    def mutate(index: SearchIndex) -> None:
        next(f for f in index.fields if f.name == "source_path").filterable = False

    other = await _existing(service, mutate)

    with pytest.raises(IncompatibleCollectionError, match="not filterable"):
        await other.ensure_collection()


# -- upsert, get, delete ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_ids_with_paths_are_stored_under_valid_keys_and_read_back_whole(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    ids = ["docs/policy.md_chunk_0", "résumé/日本語.md_chunk_1", "plain_chunk_2"]
    await adapter.upsert([record(record_id) for record_id in ids])

    stored = set(service.indexes["contract-docs"].docs)
    assert stored == {encode_key(record_id) for record_id in ids}
    assert all(re.fullmatch(r"[A-Za-z0-9=-][A-Za-z0-9_=-]*", key) for key in stored)
    assert [r.id for r in await adapter.get(ids)] == ids
    assert sorted(r.id for r in await adapter.find(RecordFilter())) == sorted(ids)
    await adapter.delete(["docs/policy.md_chunk_0"])
    assert [r.id for r in await adapter.get(ids)] == ids[1:]


@pytest.mark.asyncio
async def test_a_record_written_by_the_sk_path_under_a_valid_key_is_readable(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    await adapter.ensure_collection()
    legacy = service.indexes["contract-docs"]
    legacy.docs["legacy_chunk_0"] = {
        "id": "legacy_chunk_0",
        "source_path": "old.md",
        "chunk_index": 0,
        "content": "written before the move",
        "mtime": 3.0,
        "pinned": None,
        "embedding": [1.0, 0.0, 0.0, 0.0],
    }

    (found,) = await adapter.get(["legacy_chunk_0"])
    listed = await adapter.find(RecordFilter.where(source_path="old.md"))

    assert found.fields["content"] == "written before the move"
    assert [r.id for r in listed] == ["legacy_chunk_0"]
    assert found.fields["pinned"] is None


@pytest.mark.asyncio
async def test_upsert_replaces_the_whole_record_including_fields_left_out(
    adapter: AzureAISearchAdapter,
) -> None:
    await adapter.upsert([record("a_chunk_0", content="first")])

    await adapter.upsert(
        [
            Record(
                id="a_chunk_0",
                fields={"source_path": "a.md"},
                vector=[0.0, 1.0, 0.0, 0.0],
            )
        ]
    )

    (stored,) = await adapter.get(["a_chunk_0"])
    assert stored.fields["source_path"] == "a.md"
    assert stored.fields["content"] is None  # merge would have kept "first"
    assert stored.fields["chunk_index"] is None


@pytest.mark.asyncio
async def test_upsert_sends_every_schema_field_as_json_null_when_absent(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    await adapter.upsert(
        [Record(id="a", fields={"source_path": "a.md"}, vector=[1.0, 0, 0, 0])]
    )

    (document,) = service.names("merge_or_upload")[0]["documents"]  # type: ignore[misc]
    assert document == {
        "id": "a",
        "source_path": "a.md",
        "chunk_index": None,
        "content": None,
        "mtime": None,
        "pinned": None,
        "embedding": [1.0, 0.0, 0.0, 0.0],
    }


@pytest.mark.asyncio
async def test_values_are_coerced_to_the_declared_types(
    adapter: AzureAISearchAdapter,
) -> None:
    await adapter.upsert(
        [
            Record(
                id="a",
                fields={"chunk_index": 3.0, "mtime": 2, "pinned": 1, "content": 12},
                vector=[1.0, 0, 0, 0],
            )
        ]
    )

    (stored,) = await adapter.get(["a"])
    assert stored.fields["chunk_index"] == 3 and isinstance(
        stored.fields["chunk_index"], int
    )
    assert stored.fields["mtime"] == 2.0 and isinstance(stored.fields["mtime"], float)
    assert stored.fields["pinned"] is True
    assert stored.fields["content"] == "12"


@pytest.mark.parametrize(
    "fields",
    [{"chunk_index": "x"}, {"mtime": float("inf")}, {"mtime": float("nan")}],
)
@pytest.mark.asyncio
async def test_values_that_do_not_fit_the_type_raise_schema_error(
    adapter: AzureAISearchAdapter, fields: Mapping[str, object]
) -> None:
    with pytest.raises(SchemaError):
        await adapter.upsert(
            [Record(id="a", fields=fields, vector=[1.0, 0, 0, 0])]  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_non_finite_vectors_are_rejected(adapter: AzureAISearchAdapter) -> None:
    with pytest.raises(SchemaError, match="finite"):
        await adapter.upsert([record("a", vector=[float("nan"), 0, 0, 0])])


@pytest.mark.asyncio
async def test_an_empty_upsert_sends_nothing(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    await adapter.upsert([])

    assert service.calls == []


@pytest.mark.asyncio
async def test_get_keeps_request_order_skips_missing_and_repeats_once(
    adapter: AzureAISearchAdapter,
) -> None:
    await adapter.upsert([record("a_chunk_0"), record("b_chunk_0"), record("c")])

    found = await adapter.get(["c", "nope", "a_chunk_0", "c", "é" * 600])

    assert [r.id for r in found] == ["c", "a_chunk_0"]
    assert await adapter.get([]) == []


@pytest.mark.asyncio
async def test_get_selects_only_stored_fields_and_the_vector_on_request(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    await adapter.upsert([record("a")])

    await adapter.get(["a"])
    (plain,) = service.names("get_document")
    (vectors,) = [await adapter.get(["a"], include_vectors=True)]

    assert "embedding" not in plain["selected_fields"]  # type: ignore[operator]
    assert vectors[0].vector == [1.0, 0.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_delete_ignores_missing_and_unusable_ids(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    await adapter.upsert([record("a"), record("b")])

    await adapter.delete(["a", "missing", "é" * 600, "a"])
    await adapter.delete([])

    assert [r.id for r in await adapter.get(["a", "b"])] == ["b"]
    sent = service.names("delete")[-1]["documents"]
    assert sent == [{"id": "a"}, {"id": "missing"}]


# -- batching --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_batches_stay_under_the_document_limit(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    await adapter.upsert([record(f"id_{n}") for n in range(2500)])

    sizes = [len(call["documents"]) for call in service.names("merge_or_upload")]  # type: ignore[arg-type]
    assert sizes == [1000, 1000, 500]
    assert await adapter.count() == 2500


def test_upload_batches_also_stay_under_the_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "UPLOAD_MAX_BYTES", 1_000)
    documents: list[dict[str, object]] = [
        {"id": str(n), "text": "x" * 300} for n in range(7)
    ]

    batches = list(module._upload_batches(documents))

    assert [len(batch) for batch in batches] == [2, 2, 2, 1]
    assert sum(len(batch) for batch in batches) == 7


def test_one_oversized_document_still_goes_out_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "UPLOAD_MAX_BYTES", 100)

    batches = list(module._upload_batches([{"id": "a", "text": "x" * 500}]))

    assert batches == [[{"id": "a", "text": "x" * 500}]]


@pytest.mark.asyncio
async def test_delete_batches_stay_under_the_document_limit(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    ids = [f"id_{n}" for n in range(1500)]
    await adapter.upsert([record(record_id) for record_id in ids])

    await adapter.delete(ids)

    assert [len(c["documents"]) for c in service.names("delete")] == [1000, 500]  # type: ignore[arg-type]
    assert await adapter.count() == 0


# -- enumeration -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_pages_by_key_range_not_by_skip(
    adapter: AzureAISearchAdapter,
    service: FakeSearchService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "SCAN_PAGE_SIZE", 3)
    ids = [f"id_{n:02d}" for n in range(8)]
    await adapter.upsert([record(record_id, source="a.md") for record_id in ids])
    service.calls.clear()

    found = await adapter.find(RecordFilter.where(source_path="a.md"))

    assert [r.id for r in found] == ids
    searches = service.names("search")
    # 3 + 3 + 2 results: the third page is short, so the scan stops.
    assert [s["filter"] for s in searches] == [
        "source_path eq 'a.md'",
        "source_path eq 'a.md' and id gt 'id_02'",
        "source_path eq 'a.md' and id gt 'id_05'",
    ]
    assert all(s["order_by"] == ["id asc"] and s["top"] == 3 for s in searches)
    assert all(s["skip"] is None for s in searches)


@pytest.mark.asyncio
async def test_find_with_a_limit_asks_for_no_more_than_the_limit(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    await adapter.upsert([record(f"id_{n}", source="a.md") for n in range(5)])
    service.calls.clear()

    assert len(await adapter.find(RecordFilter.where(source_path="a.md"), limit=2)) == 2
    assert await adapter.find(RecordFilter(), limit=0) == []
    assert service.names("search")[0]["top"] == 2


@pytest.mark.asyncio
async def test_find_and_list_do_not_return_vectors(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    await adapter.upsert([record("a")])
    service.calls.clear()

    (hit,) = await adapter.find(RecordFilter())
    page = await adapter.list_page()

    assert hit.vector is None and page.records[0].vector is None
    assert all("embedding" not in s["select"] for s in service.names("search"))  # type: ignore[operator]


@pytest.mark.asyncio
async def test_list_page_cursor_is_the_last_key(
    adapter: AzureAISearchAdapter,
) -> None:
    ids = [f"id_{n}" for n in range(5)]
    await adapter.upsert([record(record_id) for record_id in ids])

    first = await adapter.list_page(limit=2)
    second = await adapter.list_page(cursor=first.next_cursor, limit=2)
    last = await adapter.list_page(cursor=second.next_cursor, limit=2)

    assert first.next_cursor == "id_1" and second.next_cursor == "id_3"
    assert [r.id for r in first.records + second.records + last.records] == ids
    assert last.next_cursor is None


@pytest.mark.asyncio
async def test_list_page_of_an_exact_fit_has_no_extra_page(
    adapter: AzureAISearchAdapter,
) -> None:
    await adapter.upsert([record("a"), record("b")])

    page = await adapter.list_page(limit=2)

    assert len(page.records) == 2 and page.next_cursor is None


@pytest.mark.parametrize("cursor", ["x' or '1'='1", "has space", "a/b", ""])
@pytest.mark.asyncio
async def test_a_cursor_that_is_not_a_key_is_rejected(
    adapter: AzureAISearchAdapter, cursor: str
) -> None:
    with pytest.raises(SchemaError, match="Invalid cursor"):
        await adapter.list_page(cursor=cursor)


@pytest.mark.asyncio
async def test_list_page_limit_must_be_positive(adapter: AzureAISearchAdapter) -> None:
    with pytest.raises(SchemaError, match="positive"):
        await adapter.list_page(limit=0)


@pytest.mark.asyncio
async def test_a_scan_that_does_not_advance_fails_instead_of_looping(
    adapter: AzureAISearchAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    await adapter.upsert([record(f"id_{n}") for n in range(4)])
    monkeypatch.setattr(module, "SCAN_PAGE_SIZE", 2)

    async def stuck(
        self: FakeSearchClient, search_text: str | None = None, **kwargs: object
    ) -> object:
        return await original(self, search_text, **{**kwargs, "filter": None})  # type: ignore[arg-type]

    original = FakeSearchClient.search
    monkeypatch.setattr(FakeSearchClient, "search", stuck)

    with pytest.raises(StoreConnectionError, match="ascending order"):
        await adapter.find(RecordFilter())


@pytest.mark.asyncio
async def test_delete_where_deletes_what_find_returns(
    adapter: AzureAISearchAdapter,
) -> None:
    await adapter.upsert(
        [
            record("a/1.md_chunk_0", source="a.md"),
            record("a/1.md_chunk_1", source="a.md"),
            record("b_chunk_0", source="b.md"),
        ]
    )

    assert await adapter.delete_where(RecordFilter.where(source_path="a.md")) == 2

    assert [r.id for r in await adapter.find(RecordFilter())] == ["b_chunk_0"]


@pytest.mark.asyncio
async def test_replace_source_removes_stale_chunks_with_path_ids(
    adapter: AzureAISearchAdapter,
) -> None:
    await adapter.upsert(
        [record(f"docs/a.md_chunk_{n}", index=n, source="docs/a.md") for n in range(4)]
    )

    await adapter.replace_source(
        RecordFilter.where(source_path="docs/a.md"),
        [record("docs/a.md_chunk_0", source="docs/a.md", content="shorter")],
    )

    (left,) = await adapter.find(RecordFilter.where(source_path="docs/a.md"))
    assert left.id == "docs/a.md_chunk_0" and left.fields["content"] == "shorter"


# -- search request shapes -------------------------------------------------------------


@pytest.mark.asyncio
async def test_vector_search_sends_a_prefiltered_vector_query(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    await adapter.upsert([record("a_chunk_0"), record("b_chunk_0", source="b.md")])
    service.calls.clear()

    hits = await adapter.vector_search(
        VectorQuery([1.0, 0.0, 0.0, 0.0], 3, RecordFilter.where(source_path="b.md"))
    )

    (call,) = service.names("search")
    (vector_query,) = call["vector_queries"]  # type: ignore[misc]
    assert isinstance(vector_query, VectorizedQuery)
    assert vector_query.fields == "embedding" and vector_query.k_nearest_neighbors == 3
    assert list(vector_query.vector) == [1.0, 0.0, 0.0, 0.0]
    assert call["search_text"] is None and call["top"] == 3
    assert call["filter"] == "source_path eq 'b.md'"
    assert call["vector_filter_mode"] == "preFilter"
    assert "embedding" not in call["select"]  # type: ignore[operator]
    assert [h.record.id for h in hits] == ["b_chunk_0"]
    assert hits[0].kind is ScoreKind.SIMILARITY


@pytest.mark.asyncio
async def test_vector_scores_are_cosine_similarities(
    adapter: AzureAISearchAdapter,
) -> None:
    await adapter.upsert(
        [
            record("same", vector=[1.0, 0.0, 0.0, 0.0]),
            record("near", vector=[0.9, 0.1, 0.0, 0.0]),
            record("orthogonal", vector=[0.0, 1.0, 0.0, 0.0]),
            record("opposite", vector=[-1.0, 0.0, 0.0, 0.0]),
        ]
    )

    hits = await adapter.vector_search(VectorQuery([1.0, 0.0, 0.0, 0.0], 4))

    assert [h.record.id for h in hits] == ["same", "near", "orthogonal", "opposite"]
    assert [h.score for h in hits] == pytest.approx([1.0, 0.993884, 0.0, 0.0], abs=1e-5)


@pytest.mark.asyncio
async def test_top_k_is_capped_at_the_service_limit(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    await adapter.ensure_collection()
    service.calls.clear()

    await adapter.vector_search(VectorQuery([1.0, 0, 0, 0], 5000))
    assert await adapter.vector_search(VectorQuery([1.0, 0, 0, 0], 0)) == []

    (call,) = service.names("search")
    assert call["top"] == 1000
    assert call["vector_queries"][0].k_nearest_neighbors == 1000  # type: ignore[index]


@pytest.mark.asyncio
async def test_lexical_search_scopes_the_text_to_the_field_and_strips_operators(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    await adapter.upsert(
        [record("a", content="alpha apple"), record("b", content="beta zeppelin")]
    )
    service.calls.clear()

    hits = await adapter.lexical_search(
        LexicalQuery('+zeppelin -"nothing" (x*) | ~2', 4, "content")
    )

    (call,) = service.names("search")
    assert call["search_text"] == "zeppelin nothing x 2"
    assert call["search_fields"] == ["content"]
    assert call["query_type"] == "simple" and call["search_mode"] == "any"
    assert call["vector_queries"] is None
    assert [h.record.id for h in hits] == ["b"]
    assert hits[0].kind is ScoreKind.LEXICAL and hits[0].score > 0


@pytest.mark.asyncio
async def test_lexical_search_with_no_words_returns_nothing_without_a_request(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    await adapter.ensure_collection()
    service.calls.clear()

    assert await adapter.lexical_search(LexicalQuery("+-*() ~", 3, "content")) == []
    assert await adapter.lexical_search(LexicalQuery("word", 0, "content")) == []
    assert service.names("search") == []


@pytest.mark.parametrize("field", ["source_path", "nope"])
@pytest.mark.asyncio
async def test_text_search_needs_a_full_text_field(
    adapter: AzureAISearchAdapter, field: str
) -> None:
    with pytest.raises(SchemaError):
        await adapter.lexical_search(LexicalQuery("x", 3, field))
    with pytest.raises(SchemaError):
        await adapter.hybrid_search(HybridQuery("x", [1.0, 0, 0, 0], 3, field))


@pytest.mark.asyncio
async def test_hybrid_search_is_one_request_with_text_and_a_vector_query(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    await adapter.upsert(
        [
            record("a", vector=[1.0, 0.0, 0.0, 0.0], content="alpha"),
            record("b", vector=[0.0, 0.0, 1.0, 0.0], content="zeppelin hangar"),
            record("c", vector=[0.0, 0.0, 0.0, 1.0], content="unrelated"),
        ]
    )
    service.calls.clear()

    hits = await adapter.hybrid_search(
        HybridQuery(
            "zeppelin",
            [1.0, 0.0, 0.0, 0.0],
            2,
            "content",
            RecordFilter.where(chunk_index=0),
        )
    )

    (call,) = service.names("search")
    assert call["search_text"] == "zeppelin" and call["search_fields"] == ["content"]
    assert call["query_type"] == "simple" and call["search_mode"] == "any"
    assert call["filter"] == "chunk_index eq 0" and call["top"] == 2
    (vector_query,) = call["vector_queries"]  # type: ignore[misc]
    assert vector_query.k_nearest_neighbors == 20  # max(4 * 2, 20)
    assert vector_query.fields == "embedding"
    assert {h.record.id for h in hits} == {"a", "b"}
    assert all(h.kind is ScoreKind.FUSED for h in hits)
    assert all(h.score > 0 for h in hits)


@pytest.mark.asyncio
async def test_hybrid_pool_grows_with_top_k(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    await adapter.ensure_collection()
    service.calls.clear()

    await adapter.hybrid_search(HybridQuery("word", [1.0, 0, 0, 0], 30, "content"))
    await adapter.hybrid_search(HybridQuery("word", [1.0, 0, 0, 0], 400, "content"))

    searches = service.names("search")
    pools = [c["vector_queries"][0].k_nearest_neighbors for c in searches]  # type: ignore[index]
    assert pools == [120, 1000]


@pytest.mark.asyncio
async def test_hybrid_with_no_usable_words_is_the_vector_side_alone(
    adapter: AzureAISearchAdapter, service: FakeSearchService
) -> None:
    await adapter.upsert([record("a", content="alpha")])
    service.calls.clear()

    hits = await adapter.hybrid_search(HybridQuery("?!", [1.0, 0, 0, 0], 3, "content"))

    (call,) = service.names("search")
    assert call["search_text"] is None
    assert [h.record.id for h in hits] == ["a"]
    assert hits[0].kind is ScoreKind.FUSED


@pytest.mark.asyncio
async def test_search_rejects_a_wrong_sized_vector(
    adapter: AzureAISearchAdapter,
) -> None:
    with pytest.raises(SchemaError, match="dimensions"):
        await adapter.vector_search(VectorQuery([1.0], 3))
    with pytest.raises(SchemaError, match="dimensions"):
        await adapter.hybrid_search(HybridQuery("x", [1.0], 3, "content"))


# -- consistency -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_waits_until_search_shows_the_last_document(
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = FakeSearchService(search_lag=3)
    adapter = adapter_on(service)
    await adapter.ensure_collection()

    await adapter.upsert([record("a", content="new")])

    found = await adapter.find(RecordFilter.where(source_path="a.md"))
    assert [r.fields["content"] for r in found] == ["new"]
    assert not caplog.records


@pytest.mark.asyncio
async def test_upsert_waits_for_the_new_values_of_an_existing_document() -> None:
    service = FakeSearchService()
    adapter = adapter_on(service)
    await adapter.upsert([record("a", content="old")])
    service.search_lag = 2

    await adapter.upsert([record("a", content="new")])

    (found,) = await adapter.find(RecordFilter.where(source_path="a.md"))
    assert found.fields["content"] == "new"


@pytest.mark.asyncio
async def test_delete_waits_until_search_no_longer_shows_the_document() -> None:
    service = FakeSearchService()
    adapter = adapter_on(service)
    await adapter.upsert([record("a"), record("b")])
    service.search_lag = 2

    await adapter.delete(["b"])

    assert [r.id for r in await adapter.find(RecordFilter())] == ["a"]


@pytest.mark.asyncio
async def test_a_write_that_never_shows_up_logs_a_warning_and_returns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(module, "VISIBILITY_TIMEOUT", 0.0)
    service = FakeSearchService(search_lag=1000)
    adapter = adapter_on(service)
    await adapter.ensure_collection()

    with caplog.at_level(logging.WARNING, logger=module.logger.name):
        await adapter.upsert([record("a")])

    assert "did not show a write" in caplog.text


# -- errors ----------------------------------------------------------------------------


class _Failing:
    """Wraps a fake client and fails the named methods with ``error``."""

    def __init__(self, inner: object, error: Exception, *names: str) -> None:
        self._inner, self._error, self._names = inner, error, names

    def __getattr__(self, name: str) -> object:
        if name in self._names:

            async def fail(*args: object, **kwargs: object) -> None:
                raise self._error

            return fail
        return getattr(self._inner, name)


async def _failing_adapter(error: Exception, *names: str) -> AzureAISearchAdapter:
    service = FakeSearchService()
    built = adapter_on(service)
    await built.ensure_collection()
    pair = await built._pair()
    built._clients = module.ClientPair(
        documents=_Failing(pair.documents, error, *names),  # type: ignore[arg-type]
        indexes=_Failing(pair.indexes, error, *names),  # type: ignore[arg-type]
    )
    return built


@pytest.mark.parametrize(
    "error",
    [
        HttpResponseError(message=f"503 service unavailable for key {API_KEY}"),
        ClientAuthenticationError(message=f"401 bad key {API_KEY}"),
    ],
    ids=["http", "auth"],
)
@pytest.mark.asyncio
async def test_the_api_key_never_reaches_an_error_message(
    error: HttpResponseError,
) -> None:
    built = await _failing_adapter(error, "get_document_count", "search")

    for call in (
        built.count(),
        built.find(RecordFilter()),
        built.vector_search(VectorQuery([1.0, 0, 0, 0], 1)),
    ):
        with pytest.raises(StoreConnectionError) as raised:
            await call
        assert API_KEY not in str(raised.value)
        assert "***" in str(raised.value)
        assert raised.value.__cause__ is None  # the SDK error holds the key
        assert raised.value.provider == "azure-ai-search"


@pytest.mark.asyncio
async def test_authentication_errors_say_what_to_check() -> None:
    built = await _failing_adapter(
        ClientAuthenticationError(message="denied"), "get_document_count"
    )

    with pytest.raises(StoreConnectionError, match="API key or the Microsoft Entra"):
        await built.count()


@pytest.mark.asyncio
async def test_http_400_is_a_schema_error_and_other_errors_keep_their_cause() -> None:
    bad = HttpResponseError(message="bad filter")
    bad.status_code = 400
    built = await _failing_adapter(bad, "search")
    with pytest.raises(SchemaError, match="bad filter"):
        await built.find(RecordFilter())

    down = HttpResponseError(message="down")
    down.status_code = 503
    built = await _failing_adapter(down, "search")
    with pytest.raises(StoreConnectionError) as raised:
        await built.find(RecordFilter())
    assert raised.value.__cause__ is down


@pytest.mark.asyncio
async def test_per_document_failures_are_raised_not_swallowed() -> None:
    service = FakeSearchService()
    built = adapter_on(service)
    await built.ensure_collection()
    pair = await built._pair()
    real = pair.documents.merge_or_upload_documents

    async def refuse(documents: list[dict[str, object]]) -> list[IndexingResult]:
        results = await real(documents)
        return [
            IndexingResult(
                key=r.key,
                succeeded=False,
                status_code=503,
                error_message=f"busy {API_KEY}",
            )
            for r in results
        ]

    pair.documents.merge_or_upload_documents = refuse  # type: ignore[method-assign]

    with pytest.raises(
        StoreConnectionError, match="upload failed for 2 of 2"
    ) as raised:
        await built.upsert([record("a/b.md_chunk_0"), record("c")])

    assert "'a/b.md_chunk_0'" in str(raised.value)  # the original ID, not the key
    assert API_KEY not in str(raised.value)


@pytest.mark.asyncio
async def test_rejected_documents_with_status_400_are_a_schema_error() -> None:
    service = FakeSearchService()
    built = adapter_on(service)
    await built.ensure_collection()
    pair = await built._pair()

    async def reject(documents: list[dict[str, object]]) -> list[IndexingResult]:
        return [
            IndexingResult(
                key=d["id"], succeeded=False, status_code=400, error_message="no"
            )  # type: ignore[arg-type]
            for d in documents
        ]

    pair.documents.merge_or_upload_documents = reject  # type: ignore[method-assign]

    with pytest.raises(SchemaError, match="HTTP 400"):
        await built.upsert([record("a")])


@pytest.mark.asyncio
async def test_a_create_conflict_is_not_an_error() -> None:
    service = FakeSearchService()
    built = adapter_on(service)
    pair = await built._pair()
    real_create = pair.indexes.create_index

    async def conflict(index: SearchIndex) -> SearchIndex:
        await real_create(index)  # another writer got there first
        raise ResourceExistsError(message="exists")

    pair.indexes.create_index = conflict  # type: ignore[method-assign]

    await built.ensure_collection()

    assert await built.collection_exists() is True


# -- review fixes (042 phase 3) ----------------------------------------------------


def test_every_adapter_written_key_decodes_to_its_id() -> None:
    """An ID that starts with the prefix is always encoded, so decoding is exact."""
    tricky = [
        "b64-",
        "b64-b64-",
        "b64-Zm9v",
        "b64-%%%",
        encode_key("b64-Zm9v"),
        encode_key("a/b"),
        "B64-x",
        "b64",
    ]
    keys = [encode_key(record_id) for record_id in tricky]
    assert len(set(keys)) == len(tricky)
    assert [decode_key(key) for key in keys] == tricky
    assert all(key.startswith("b64-") for key in keys[:6])


@pytest.mark.parametrize(
    "bad",
    [
        {"chunk_index": 1.5},
        {"chunk_index": "1"},
        {"source_path": 5},
        {"chunk_index": True},
    ],
)
def test_a_filter_value_that_cannot_equal_a_stored_value_is_rejected(
    adapter: AzureAISearchAdapter, bad: Mapping[str, object]
) -> None:
    with pytest.raises(SchemaError, match="another type|boolean is not a number"):
        adapter._filter_expression(RecordFilter.where(**bad))  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        record("a_chunk_0", vector=(float("nan"), 0.0, 0.0, 0.0)),
        record("é" * 600),
        record("a" * 2000),
        Record("a_chunk_0", {"chunk_index": "abc"}, [1.0, 0.0, 0.0, 0.0]),
    ],
    ids=["nan-vector", "long-encoded-id", "long-raw-id", "string-in-integer-field"],
)
async def test_replace_source_converts_every_record_before_deleting(
    service: FakeSearchService, adapter: AzureAISearchAdapter, bad: Record
) -> None:
    await adapter.upsert([record("a_chunk_0"), record("a_chunk_1", index=1)])
    deletes = len(service.names("delete"))
    with pytest.raises(SchemaError):
        await adapter.replace_source(RecordFilter.where(source_path="a.md"), [bad])
    assert len(service.names("delete")) == deletes
    kept = await adapter.find(RecordFilter.where(source_path="a.md"))
    assert sorted(r.id for r in kept) == ["a_chunk_0", "a_chunk_1"]


@pytest.mark.asyncio
async def test_delete_waits_until_none_of_the_deleted_documents_shows() -> None:
    # The last key leaves search at once; the first one lags behind.
    service = FakeSearchService()
    adapter = adapter_on(service)
    await adapter.upsert([record("a"), record("b"), record("c")])
    service.key_lag = {"a": 3}

    await adapter.delete(["a", "b", "c"])

    assert await adapter.find(RecordFilter()) == []


@pytest.mark.asyncio
async def test_upsert_waits_until_every_written_document_shows() -> None:
    # The last document is searchable at once; the first one lags behind.
    service = FakeSearchService(key_lag={"a": 3})
    adapter = adapter_on(service)
    await adapter.ensure_collection()

    await adapter.upsert([record("a"), record("b"), record("c")])

    assert sorted(r.id for r in await adapter.find(RecordFilter())) == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_upsert_waits_for_the_new_embedding_of_the_last_document() -> None:
    service = FakeSearchService()
    adapter = adapter_on(service)
    await adapter.upsert([record("a", vector=(1.0, 0.0, 0.0, 0.0))])
    service.search_lag = 2

    await adapter.upsert([record("a", vector=(0.0, 1.0, 0.0, 0.0))])

    (found,) = await adapter.get(["a"], include_vectors=True)
    assert found.vector == pytest.approx([0.0, 1.0, 0.0, 0.0])


@pytest.mark.asyncio
async def test_the_wait_checks_keys_in_one_batched_filter() -> None:
    service = FakeSearchService()
    adapter = adapter_on(service)
    await adapter.upsert([record("a"), record("b")])
    before = len(service.names("search"))

    await adapter.delete(["a", "b"])

    waits = service.names("search")[before:]
    assert len(waits) == 1
    assert waits[0]["filter"] == "search.in(id, 'a,b', ',')"


# -- the index metric (review fix 9) -----------------------------------------------


@pytest.mark.asyncio
async def test_an_index_with_another_metric_is_incompatible(
    service: FakeSearchService,
) -> None:
    def mutate(index: SearchIndex) -> None:
        index.vector_search.algorithms[0].parameters.metric = "euclidean"

    other = await _existing(service, mutate)

    with pytest.raises(IncompatibleCollectionError, match="cosine"):
        await other.ensure_collection()


@pytest.mark.asyncio
async def test_an_index_whose_vector_profile_is_missing_is_incompatible(
    service: FakeSearchService,
) -> None:
    def mutate(index: SearchIndex) -> None:
        index.vector_search.profiles = []

    other = await _existing(service, mutate)

    with pytest.raises(IncompatibleCollectionError, match="profile"):
        await other.ensure_collection()
