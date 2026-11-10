"""The REST requests the real Azure SDK builds from the adapter's calls.

The fake clients used elsewhere accept whatever objects the adapter builds, so
they cannot show that the real ``azure-search-documents`` client accepts them.
Here the adapter runs on the real ``SearchClient`` and ``SearchIndexClient``
over a stub ``AsyncHttpTransport``: no network, but the full SDK pipeline
(model serialization, URL building, ``api-version``, authentication header).

This checks request shape against the SDK, not against the live service. A
field the service would reject (a bad analyzer, an unsupported combination of
attributes) is not visible here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.transport import AsyncHttpTransport
from azure.core.rest import HttpRequest
from azure.core.rest._http_response_impl_async import AsyncHttpResponseImpl
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient

from holodeck.lib.retrieval import (
    CollectionSchema,
    FieldSpec,
    FieldType,
    HybridQuery,
    LexicalQuery,
    Record,
    RecordFilter,
    VectorQuery,
)
from holodeck.lib.retrieval.adapters import azure_ai_search as module
from holodeck.lib.retrieval.adapters.azure_ai_search import (
    AzureAISearchAdapter,
    AzureSearchSettings,
    ClientPair,
    build_index,
)

ENDPOINT = "https://svc.search.windows.net"
API_KEY = "AzureKey-SECRET-123"
API_VERSION = "2026-04-01"  # azure-search-documents 12.0.0 default

SCHEMA = CollectionSchema(
    "docs_dim4",
    4,
    (
        FieldSpec("source_path", FieldType.STRING, indexed=True),
        FieldSpec("chunk_index", FieldType.INTEGER, indexed=True),
        FieldSpec("content", FieldType.STRING, full_text=True),
        FieldSpec("mtime", FieldType.FLOAT),
        FieldSpec("pinned", FieldType.BOOLEAN),
    ),
)


class _Response(AsyncHttpResponseImpl):  # type: ignore[misc]
    def __init__(
        self,
        request: HttpRequest,
        status: int,
        body: bytes = b"",
        content_type: str = "application/json",
    ) -> None:
        super().__init__(
            request=request,
            internal_response=None,
            block_size=None,
            status_code=status,
            reason="stub",
            content_type=content_type,
            headers={"content-type": content_type},
            stream_download_generator=None,
        )
        self._content = body

    async def read(self) -> bytes:
        return self._content or b""


@dataclass
class Seen:
    method: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: Any


@dataclass
class StubService(AsyncHttpTransport):  # type: ignore[misc,type-arg]
    """A transport that answers like a small Azure AI Search service."""

    seen: list[Seen] = field(default_factory=list)
    index: dict[str, Any] | None = None
    docs: dict[str, dict[str, Any]] = field(default_factory=dict)
    closed: int = 0

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        self.closed += 1

    async def __aenter__(self) -> StubService:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.closed += 1

    def requests(self, suffix: str) -> list[Seen]:
        return [seen for seen in self.seen if seen.path.endswith(suffix)]

    async def send(self, request: HttpRequest, **kwargs: Any) -> _Response:
        parts = urlsplit(request.url)
        body = json.loads(request.body) if request.body else None
        self.seen.append(
            Seen(
                request.method,
                unquote(parts.path),
                parse_qs(parts.query),
                dict(request.headers),
                body,
            )
        )
        return self._answer(request, request.method, unquote(parts.path), body)

    def _answer(
        self, request: HttpRequest, method: str, path: str, body: Any
    ) -> _Response:
        def ok(payload: Any, status: int = 200) -> _Response:
            return _Response(request, status, json.dumps(payload).encode())

        if path == "/indexes" and method == "POST":
            self.index = body
            return ok(body, 201)
        if re.fullmatch(r"/indexes\('[^']+'\)", path):
            if method == "DELETE":
                self.index = None
                return _Response(request, 204)
            if self.index is None:
                return ok({"error": {"code": "", "message": "no such index"}}, 404)
            return ok(self.index)
        if path.endswith("/docs/search.index"):
            results = []
            for action in body["value"]:
                key = action["id"]
                if action["@search.action"] == "delete":
                    self.docs.pop(key, None)
                else:
                    document = {
                        k: v for k, v in action.items() if k != "@search.action"
                    }
                    self.docs.setdefault(key, {}).update(document)
                results.append(
                    {
                        "key": key,
                        "status": True,
                        "errorMessage": None,
                        "statusCode": 200,
                    }
                )
            return ok({"value": results})
        if path.endswith("/docs/search.post.search"):
            return ok({"value": self._search(body)})
        if path.endswith("/docs/$count"):
            return _Response(request, 200, str(len(self.docs)).encode(), "text/plain")
        match = re.search(r"/docs\('([^']+)'\)$", path)
        if match and method == "GET":
            found = self.docs.get(match.group(1))
            if found is None:
                return ok({"error": {"message": "missing"}}, 404)
            return ok(found)
        return ok({"error": {"message": f"unexpected {method} {path}"}}, 400)

    def _search(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        select = [name.strip() for name in body.get("select", "").split(",") if name]
        wanted = re.fullmatch(r"id eq '([^']*)'", body.get("filter") or "")
        rows = (
            [d for k, d in self.docs.items() if k == wanted.group(1)]
            if wanted
            else list(self.docs.values())
        )
        return [
            {"@search.score": 0.75, **{k: v for k, v in row.items() if k in select}}
            for row in rows[: body.get("top", 50)]
        ]


def adapter_on(service: StubService) -> AzureAISearchAdapter:
    def factory(settings: AzureSearchSettings, index_name: str) -> ClientPair:
        assert settings.api_key is not None
        credential = AzureKeyCredential(settings.api_key)
        return ClientPair(
            documents=SearchClient(
                settings.endpoint, index_name, credential, transport=service
            ),
            indexes=SearchIndexClient(settings.endpoint, credential, transport=service),
        )

    return AzureAISearchAdapter(
        SCHEMA,
        AzureSearchSettings(endpoint=ENDPOINT, api_key=API_KEY),
        client_factory=factory,
    )


def record(record_id: str, source: str = "a.md") -> Record:
    return Record(
        id=record_id,
        fields={
            "source_path": source,
            "chunk_index": 0,
            "content": "alpha",
            "mtime": 1.5,
            "pinned": False,
        },
        vector=[1.0, 0.0, 0.0, 0.0],
    )


@pytest.fixture(autouse=True)
def no_poll_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_POLL_START", 0.0)
    monkeypatch.setattr(module, "_POLL_MAX", 0.0)


@pytest.mark.asyncio
async def test_create_index_request() -> None:
    service = StubService()
    adapter = adapter_on(service)

    await adapter.ensure_collection()

    first = service.seen[0]
    assert (first.method, first.path) == ("GET", "/indexes('docs_dim4')")
    (create,) = [s for s in service.seen if s.method == "POST"]
    assert create.path == "/indexes"
    assert create.query == {"api-version": [API_VERSION]}
    assert create.headers["api-key"] == API_KEY
    assert create.body == build_index(SCHEMA, "docs_dim4").as_dict()
    fields = {f["name"]: f for f in create.body["fields"]}
    assert fields["id"]["key"] is True and fields["id"]["sortable"] is True
    assert fields["embedding"]["dimensions"] == 4
    assert fields["embedding"]["vectorSearchProfile"] == "embedding_profile"
    assert create.body["vectorSearch"]["algorithms"][0]["kind"] == "hnsw"
    assert create.body["vectorSearch"]["algorithms"][0]["hnswParameters"] == {
        "metric": "cosine"
    }
    # The service's reply parsed back into a compatible index.
    assert await adapter.collection_exists() is True


@pytest.mark.asyncio
async def test_upload_request() -> None:
    service = StubService()
    adapter = adapter_on(service)

    await adapter.upsert([record("docs/a.md_chunk_0"), record("plain_1")])

    (upload,) = service.requests("/docs/search.index")
    assert upload.method == "POST"
    assert upload.path == "/indexes('docs_dim4')/docs/search.index"
    assert upload.query == {"api-version": [API_VERSION]}
    first, second = upload.body["value"]
    assert first["@search.action"] == "mergeOrUpload"
    assert first["id"] == module.encode_key("docs/a.md_chunk_0")
    assert second["id"] == "plain_1"
    assert {k: v for k, v in second.items() if k != "id"} == {
        "@search.action": "mergeOrUpload",
        "source_path": "a.md",
        "chunk_index": 0,
        "content": "alpha",
        "mtime": 1.5,
        "pinned": False,
        "embedding": [1.0, 0.0, 0.0, 0.0],
    }
    # The wait for visibility is one search for the last document.
    (probe,) = service.requests("/docs/search.post.search")
    assert probe.body["filter"] == "id eq 'plain_1'" and probe.body["top"] == 1


@pytest.mark.asyncio
async def test_hybrid_search_request() -> None:
    service = StubService()
    adapter = adapter_on(service)
    await adapter.upsert([record("a_chunk_0")])
    service.seen.clear()

    hits = await adapter.hybrid_search(
        HybridQuery(
            "o'brien zeppelin",
            [1.0, 0.0, 0.0, 0.0],
            3,
            "content",
            RecordFilter.where(source_path="a.md"),
        )
    )

    (search,) = service.requests("/docs/search.post.search")
    assert search.method == "POST"
    assert search.path == "/indexes('docs_dim4')/docs/search.post.search"
    assert search.query == {"api-version": [API_VERSION]}
    assert search.body == {
        "search": "o brien zeppelin",
        "filter": "source_path eq 'a.md'",
        "queryType": "simple",
        "searchFields": "content",
        "searchMode": "any",
        "select": "id,source_path,chunk_index,content,mtime,pinned",
        "top": 3,
        "vectorQueries": [
            {
                "kind": "vector",
                "vector": [1.0, 0.0, 0.0, 0.0],
                "k": 20,
                "fields": "embedding",
            }
        ],
        "vectorFilterMode": "preFilter",
    }
    assert [h.record.id for h in hits] == ["a_chunk_0"]
    assert hits[0].record.fields["source_path"] == "a.md"


@pytest.mark.asyncio
async def test_vector_and_lexical_search_requests() -> None:
    service = StubService()
    adapter = adapter_on(service)
    await adapter.upsert([record("a")])
    service.seen.clear()

    vector_hits = await adapter.vector_search(VectorQuery([1.0, 0.0, 0.0, 0.0], 2))
    await adapter.lexical_search(LexicalQuery("alpha -beta", 2, "content"))

    vector, lexical = (s.body for s in service.requests("/docs/search.post.search"))
    assert "search" not in vector
    assert vector["vectorQueries"] == [
        {
            "kind": "vector",
            "vector": [1.0, 0.0, 0.0, 0.0],
            "k": 2,
            "fields": "embedding",
        }
    ]
    assert vector["top"] == 2 and vector["vectorFilterMode"] == "preFilter"
    assert lexical["search"] == "alpha beta"
    assert lexical["searchFields"] == "content" and "vectorQueries" not in lexical
    # The stub scored the hit 0.75, which Azure reports for cosine distance 1/3.
    assert vector_hits[0].score == pytest.approx(2 - 1 / 0.75)


@pytest.mark.asyncio
async def test_enumeration_request_uses_a_key_range_and_order() -> None:
    service = StubService()
    adapter = adapter_on(service)
    await adapter.ensure_collection()
    service.seen.clear()

    await adapter.find(RecordFilter.where(source_path="o'brien.md"))
    await adapter.list_page(cursor="id_1", limit=10)

    find, listing = (s.body for s in service.requests("/docs/search.post.search"))
    assert find == {
        "search": "*",
        "filter": "source_path eq 'o''brien.md'",
        "orderby": "id asc",
        "select": "id,source_path,chunk_index,content,mtime,pinned",
        "top": 1000,
    }
    assert listing["filter"] == "id gt 'id_1'" and listing["top"] == 11
    assert "skip" not in find and "skip" not in listing


@pytest.mark.asyncio
async def test_point_reads_delete_count_and_drop_requests() -> None:
    service = StubService()
    adapter = adapter_on(service)
    await adapter.upsert([record("a"), record("b")])
    service.seen.clear()

    found = await adapter.get(["a", "nope"])
    count = await adapter.count()
    await adapter.delete(["a"])
    await adapter.delete_collection()

    assert [r.id for r in found] == ["a"]
    assert count == 2
    gets = [s for s in service.seen if s.method == "GET" and "/docs('" in s.path]
    assert {s.path for s in gets} == {
        "/indexes('docs_dim4')/docs('a')",
        "/indexes('docs_dim4')/docs('nope')",
    }
    assert all(s.query["api-version"] == [API_VERSION] for s in gets)
    assert gets[0].query["$select"] == [
        "id,source_path,chunk_index,content,mtime,pinned"
    ]
    (counted,) = service.requests("/docs/$count")
    assert counted.query == {"api-version": [API_VERSION]}
    (removed,) = [
        s
        for s in service.requests("/docs/search.index")
        if s.body["value"][0]["@search.action"] == "delete"
    ]
    assert removed.body == {"value": [{"@search.action": "delete", "id": "a"}]}
    (dropped,) = [s for s in service.seen if s.method == "DELETE"]
    assert dropped.path == "/indexes('docs_dim4')"
    assert dropped.query == {"api-version": [API_VERSION]}


@pytest.mark.asyncio
async def test_close_releases_the_transport_and_the_adapter_reopens() -> None:
    service = StubService()
    adapter = adapter_on(service)
    await adapter.ensure_collection()

    await adapter.close()

    assert service.closed >= 1
    assert await adapter.collection_exists() is True
