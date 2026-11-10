"""A fake of the Cosmos DB aio container API, for the Cosmos NoSQL adapter.

Why a fake
----------
The Linux emulator (``azure-cosmos-emulator:vnext-preview``, PostgreSQL based)
runs vector search, filters, point reads, paging and counts, and the adapter's
contract tests run there. It does not know ``FullTextScore``,
``FullTextContains`` or ``RRF`` (``'FullTextScore' is not a recognized built-in
function name``), so it cannot run native hybrid search; it also accepts only
the ``diskANN`` vector index and does not echo ``vectorEmbeddingPolicy`` from a
container read. This fake runs the same contract suite everywhere and covers
hybrid search.

What it models
--------------
Only the calls the adapter makes: database and container lifecycle,
``upsert_item``, ``read_item``, ``delete_item``, ``read`` and ``query_items``
with ``by_page`` continuation tokens. It interprets exactly the SQL shapes the
adapter emits and answers anything else with a 400, so a query that
interpolates text or uses an unknown shape fails loudly. Rules it keeps from
what the emulator and the service documentation showed:

* item IDs with ``/ \\ ? #`` are refused (400, "Id contains illegal chars");
* a missing database, container or item is a 404;
* partition key ``/id``: ``read_item`` and ``delete_item`` need the ID as the
  partition key value;
* numbers are normalized like the service does (an integral float reads back as
  an integer), values pass a JSON round trip (non-finite numbers are refused);
* ``VectorDistance`` is the cosine similarity, best first, for documents that
  have an embedding;
* a vector index of ``flat`` holds at most 505 dimensions, any index at most
  4096;
* a query pages at 100 items unless ``max_item_count`` is set; the
  continuation token is opaque.

Not verified against the service: its BM25 and RRF arithmetic. The fake ranks
text with a plain term-frequency score and fuses with RRF (k = 60), so hybrid
results are only checked for the order the contract suite needs.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError

from holodeck.lib.retrieval import CollectionSchema
from holodeck.lib.retrieval.adapters.cosmos_nosql import (
    CosmosNoSqlAdapter,
    CosmosSettings,
    _Client,
    settings_from_connection,
)

DEFAULT_PAGE = 100
ILLEGAL_ID_CHARS = set("/\\?#")
RRF_K = 60


def _not_found(message: str) -> CosmosResourceNotFoundError:
    return CosmosResourceNotFoundError(status_code=404, message=message)


def _bad_request(message: str) -> CosmosHttpResponseError:
    return CosmosHttpResponseError(status_code=400, message=message)


def _normalize(value: object) -> object:
    """Round-trip through JSON and turn integral floats into integers."""
    try:
        text = json.dumps(value, allow_nan=False)
    except ValueError as error:
        raise _bad_request(f"Invalid JSON: {error}") from None

    def fix(item: object) -> object:
        if isinstance(item, float) and item.is_integer():
            return int(item)
        if isinstance(item, list):
            return [fix(part) for part in item]
        if isinstance(item, dict):
            return {key: fix(part) for key, part in item.items()}
        return item

    return fix(json.loads(text))


def _same(left: object, right: object) -> bool:
    """JSON equality: a boolean never equals a number."""
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    return left == right


def _plain(path: str) -> str:
    return path.replace('"', "")


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


def _terms(text: object) -> list[str]:
    return re.findall(r"\w+", text.lower(), re.UNICODE) if isinstance(text, str) else []


@dataclass
class FakeContainerData:
    """One container: its definition and documents."""

    properties: dict[str, object]
    documents: dict[str, dict[str, object]] = field(default_factory=dict)
    clock: int = 0


@dataclass
class FakeAccount:
    """State of the fake account. It outlives clients, like a real account."""

    databases: dict[str, dict[str, FakeContainerData]] = field(default_factory=dict)
    queries: list[tuple[str, list[dict[str, object]]]] = field(default_factory=list)
    in_flight: int = 0
    peak_in_flight: int = 0
    clients_opened: int = 0
    clients_closed: int = 0


class _Page:
    """Async iterator over one page of items."""

    def __init__(self, items: list[object]) -> None:
        self._items = iter(items)

    def __aiter__(self) -> _Page:
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration from None


class _Pager:
    """What ``query_items(...).by_page()`` returns."""

    def __init__(
        self, rows: list[object], size: int, continuation_token: str | None
    ) -> None:
        self._rows = rows
        self._size = size
        self._started = continuation_token is not None
        self._offset = self._decode(continuation_token)
        self._done = False
        self.continuation_token: str | None = continuation_token

    @staticmethod
    def _decode(token: str | None) -> int:
        if token is None:
            return 0
        match = re.fullmatch(r"fake-token:(\d+)", token)
        if match is None:
            raise _bad_request("The continuation token is not valid.")
        return int(match.group(1))

    def __aiter__(self) -> _Pager:
        return self

    async def __anext__(self) -> _Page:
        if self._done:
            raise StopAsyncIteration
        end = self._offset + self._size
        page = self._rows[self._offset : end]
        if end >= len(self._rows):
            self.continuation_token = None
            self._done = True
        else:
            self.continuation_token = f"fake-token:{end}"
            self._offset = end
        return _Page(page)


class _Query:
    """What ``query_items`` returns."""

    def __init__(self, rows: list[object], size: int) -> None:
        self._rows = rows
        self._size = size

    def by_page(self, continuation_token: str | None = None) -> _Pager:
        return _Pager(self._rows, self._size, continuation_token)

    async def __aiter__(self) -> AsyncIterator[object]:
        pager = self.by_page()
        try:
            while True:
                page = await pager.__anext__()
                async for row in page:
                    yield row
        except StopAsyncIteration:
            return


_SELECT = re.compile(
    r"^SELECT (?:TOP @(?P<top>\w+) )?(?P<cols>.+?) FROM c"
    r"(?: WHERE (?P<where>.+?))?(?: ORDER BY (?P<order>.+))?$"
)
_DISTANCE = 'VectorDistance(c["embedding"], @vector)'
_COLUMN = re.compile(r'^(?:c\.id|c\["(?P<name>\w+)"\])$')
_EQUALS = re.compile(r'^c\["(?P<name>\w+)"\] = (?P<param>@\w+)$')
_IS_NULL = re.compile(r'^IS_NULL\(c\["(?P<name>\w+)"\]\)$')
_RRF = re.compile(
    r'^RANK RRF\(VectorDistance\(c\["embedding"\], @vector\), '
    r'FullTextScore\(c\["(?P<field>\w+)"\], (?P<terms>@t\d+(?:, @t\d+)*)\)\)$'
)


class FakeContainer:
    """A fake ``ContainerProxy``."""

    def __init__(self, account: FakeAccount, database: str, name: str) -> None:
        self._account = account
        self._database = database
        self._name = name

    def _data(self) -> FakeContainerData:
        containers = self._account.databases.get(self._database)
        if containers is None or self._name not in containers:
            raise _not_found(f"Container '{self._name}' does not exist.")
        return containers[self._name]

    async def read(self) -> Mapping[str, object]:
        return copy.deepcopy(self._data().properties)

    async def upsert_item(self, body: Mapping[str, object]) -> object:
        account = self._account
        account.in_flight += 1
        account.peak_in_flight = max(account.peak_in_flight, account.in_flight)
        try:
            await asyncio.sleep(0)
            data = self._data()
            item_id = body.get("id")
            if not isinstance(item_id, str) or not item_id:
                raise _bad_request("The input content is invalid: id is required.")
            if ILLEGAL_ID_CHARS.intersection(item_id):
                raise _bad_request("Id contains illegal chars.")
            if len(item_id.encode("utf-8")) > 1023:
                raise _bad_request("Id is too long.")
            stored = _normalize(dict(body))
            assert isinstance(stored, dict)
            data.clock += 1
            stored.update({"_rid": f"rid{data.clock}", "_ts": data.clock})
            data.documents[item_id] = stored
            return copy.deepcopy(stored)
        finally:
            account.in_flight -= 1

    async def read_item(self, item: str, partition_key: str) -> Mapping[str, object]:
        data = self._data()
        document = data.documents.get(item)
        if document is None or partition_key != item:
            raise _not_found(f"Item '{item}' does not exist.")
        return copy.deepcopy(document)

    async def delete_item(self, item: str, partition_key: str) -> None:
        data = self._data()
        if item not in data.documents or partition_key != item:
            raise _not_found(f"Item '{item}' does not exist.")
        del data.documents[item]

    def query_items(
        self,
        query: str,
        *,
        parameters: Sequence[Mapping[str, object]],
        max_item_count: int | None = None,
        enable_scan_in_query: bool = True,
    ) -> _Query:
        self._account.queries.append(
            (query, [dict(parameter) for parameter in parameters])
        )
        values = {str(p["name"]): p["value"] for p in parameters}
        rows = self._run(query, values)
        return _Query(rows, max_item_count or DEFAULT_PAGE)

    # -- the SQL interpreter ------------------------------------------------------

    def _run(self, query: str, values: Mapping[str, object]) -> list[object]:
        data = self._data()
        for name in re.findall(r"@\w+", query):
            if name not in values:
                raise _bad_request(f"Undefined parameter {name}.")
        if query == "SELECT VALUE COUNT(1) FROM c":
            return [len(data.documents)]
        match = _SELECT.match(query)
        if match is None:
            raise _bad_request(f"Syntax error near: {query[:60]}")
        cols = match["cols"]
        with_score = cols.endswith(f", {_DISTANCE} AS score")
        if with_score:
            cols = cols[: -len(f", {_DISTANCE} AS score")]
        names = self._columns(cols)

        documents = list(data.documents.values())
        if match["where"]:
            documents = [
                d for d in documents if self._matches(d, match["where"], values)
            ]

        order = match["order"]
        if order == _DISTANCE:
            if not with_score:
                raise _bad_request("ORDER BY VectorDistance needs the same projection.")
            ranked = self._by_similarity(data, documents, values)
            rows = [
                self._project(doc, names, {"score": score}) for doc, score in ranked
            ]
        elif order is not None:
            documents = self._by_rrf(data, documents, values, order)
            rows = [self._project(doc, names, {}) for doc in documents]
        else:
            if with_score:
                raise _bad_request("VectorDistance needs an ORDER BY.")
            rows = [self._project(doc, names, {}) for doc in documents]

        if match["top"]:
            top = values[f"@{match['top']}"]
            if not isinstance(top, int) or isinstance(top, bool) or top < 0:
                raise _bad_request("TOP needs a non-negative integer.")
            rows = rows[:top]
        return list(rows)

    @staticmethod
    def _columns(cols: str) -> list[str]:
        names: list[str] = []
        for part in cols.split(", "):
            column = _COLUMN.match(part)
            if column is None:
                raise _bad_request(f"Unsupported projection: {part}")
            names.append(column["name"] or "id")
        return names

    @staticmethod
    def _project(
        document: Mapping[str, object], names: list[str], extra: Mapping[str, object]
    ) -> dict[str, object]:
        row = {name: document[name] for name in names if name in document}
        row.update(extra)
        return copy.deepcopy(row)

    @staticmethod
    def _matches(
        document: Mapping[str, object], where: str, values: Mapping[str, object]
    ) -> bool:
        for clause in where.split(" AND "):
            null = _IS_NULL.match(clause)
            if null is not None:
                if not (null["name"] in document and document[null["name"]] is None):
                    return False
                continue
            equals = _EQUALS.match(clause)
            if equals is None:
                raise _bad_request(f"Unsupported filter: {clause}")
            if equals["name"] not in document or not _same(
                document[equals["name"]], values[equals["param"]]
            ):
                return False
        return True

    @staticmethod
    def _vector(document: Mapping[str, object]) -> list[float] | None:
        value = document.get("embedding")
        if isinstance(value, list) and all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in value
        ):
            return [float(item) for item in value]
        return None

    def _by_similarity(
        self,
        data: FakeContainerData,
        documents: list[dict[str, object]],
        values: Mapping[str, object],
    ) -> list[tuple[dict[str, object], float]]:
        self._require_vector_policy(data)
        query = values["@vector"]
        assert isinstance(query, list)
        scored = [
            (doc, _cosine(vector, [float(v) for v in query]))
            for doc in documents
            if (vector := self._vector(doc)) is not None
        ]
        return sorted(scored, key=lambda pair: -pair[1])

    @staticmethod
    def _require_vector_policy(data: FakeContainerData) -> None:
        policy = data.properties.get("vectorEmbeddingPolicy")
        if not isinstance(policy, dict) or not policy.get("vectorEmbeddings"):
            raise _bad_request("The container has no vector embedding policy.")

    def _by_rrf(
        self,
        data: FakeContainerData,
        documents: list[dict[str, object]],
        values: Mapping[str, object],
        order: str,
    ) -> list[dict[str, object]]:
        match = _RRF.match(order)
        if match is None:
            raise _bad_request(f"Unsupported ORDER BY: {order}")
        indexing = data.properties.get("indexingPolicy")
        indexes = (
            indexing.get("fullTextIndexes") if isinstance(indexing, dict) else None
        )
        paths = {_plain(str(item.get("path"))) for item in indexes or []}
        if f"/{match['field']}" not in paths:
            raise _bad_request("FullTextScore needs a full-text index on the property.")
        self._require_vector_policy(data)
        terms = [str(values[name]) for name in match["terms"].split(", ")]
        by_vector = [doc for doc, _ in self._by_similarity(data, documents, values)]
        scores = {
            id(doc): sum(_terms(doc.get(match["field"])).count(term) for term in terms)
            for doc in documents
        }
        by_text = sorted(documents, key=lambda doc: -scores[id(doc)])
        fused: dict[int, float] = {}
        for ranking in (by_vector, by_text):
            for rank, doc in enumerate(ranking, start=1):
                fused[id(doc)] = fused.get(id(doc), 0.0) + 1.0 / (RRF_K + rank)
        return sorted(documents, key=lambda doc: -fused[id(doc)])


class FakeDatabase:
    """A fake ``DatabaseProxy``."""

    def __init__(self, account: FakeAccount, name: str) -> None:
        self._account = account
        self._name = name

    def _containers(self) -> dict[str, FakeContainerData]:
        if self._name not in self._account.databases:
            raise _not_found(f"Database '{self._name}' does not exist.")
        return self._account.databases[self._name]

    async def create_container_if_not_exists(
        self,
        *,
        id: str,
        partition_key: Mapping[str, object],
        indexing_policy: Mapping[str, object],
        vector_embedding_policy: Mapping[str, object],
        full_text_policy: Mapping[str, object] | None = None,
    ) -> FakeContainer:
        containers = self._containers()
        if id not in containers:
            self._validate_vector_policies(indexing_policy, vector_embedding_policy)
            if full_text_policy is not None:
                self._validate_full_text(indexing_policy, full_text_policy)
            properties: dict[str, object] = {
                "id": id,
                "partitionKey": dict(partition_key),
                "indexingPolicy": copy.deepcopy(dict(indexing_policy)),
                "vectorEmbeddingPolicy": copy.deepcopy(dict(vector_embedding_policy)),
            }
            if full_text_policy is not None:
                properties["fullTextPolicy"] = copy.deepcopy(dict(full_text_policy))
            containers[id] = FakeContainerData(properties=properties)
        return FakeContainer(self._account, self._name, id)

    @staticmethod
    def _validate_vector_policies(
        indexing: Mapping[str, object], embedding: Mapping[str, object]
    ) -> None:
        embeddings = embedding.get("vectorEmbeddings")
        if not isinstance(embeddings, list) or not embeddings:
            raise _bad_request("vectorEmbeddings must not be empty.")
        by_path: dict[str, dict[str, object]] = {}
        for item in embeddings:
            if item.get("dataType") != "float32" or item.get(
                "distanceFunction"
            ) not in {
                "cosine",
                "dotproduct",
                "euclidean",
            }:
                raise _bad_request("Invalid vector embedding policy.")
            dimensions = item.get("dimensions")
            if not isinstance(dimensions, int) or not 0 < dimensions <= 4096:
                raise _bad_request("Vector dimensions must be 1 to 4096.")
            by_path[_plain(str(item["path"]))] = item
        indexes = indexing.get("vectorIndexes")
        for index in indexes if isinstance(indexes, list) else []:
            policy = by_path.get(_plain(str(index.get("path"))))
            if policy is None:
                raise _bad_request("A vector index needs a vector embedding policy.")
            if index.get("type") not in {"flat", "quantizedFlat", "diskANN"}:
                raise _bad_request("Unknown vector index type.")
            dimensions = policy["dimensions"]
            assert isinstance(dimensions, int)
            if index["type"] == "flat" and dimensions > 505:
                raise _bad_request("A flat index supports at most 505 dimensions.")

    @staticmethod
    def _validate_full_text(
        indexing: Mapping[str, object], policy: Mapping[str, object]
    ) -> None:
        paths = policy.get("fullTextPaths")
        if not isinstance(paths, list) or not paths:
            raise _bad_request("fullTextPaths must not be empty.")
        planned = {_plain(str(item["path"])) for item in paths}
        indexed = indexing.get("fullTextIndexes")
        wanted = {_plain(str(item["path"])) for item in indexed or []}
        if wanted and not wanted <= planned:
            raise _bad_request("A full-text index needs a full-text policy path.")

    async def delete_container(self, container: str) -> None:
        containers = self._containers()
        if container not in containers:
            raise _not_found(f"Container '{container}' does not exist.")
        del containers[container]

    def get_container_client(self, container: str) -> FakeContainer:
        return FakeContainer(self._account, self._name, container)


class FakeCosmosClient:
    """A fake ``CosmosClient``."""

    def __init__(self, account: FakeAccount) -> None:
        self._account = account
        self._closed = False
        account.clients_opened += 1

    async def create_database_if_not_exists(self, id: str) -> FakeDatabase:
        self._account.databases.setdefault(id, {})
        return FakeDatabase(self._account, id)

    def get_database_client(self, database: str) -> FakeDatabase:
        return FakeDatabase(self._account, database)

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._account.clients_closed += 1


class FakeCosmosAdapter(CosmosNoSqlAdapter):
    """``CosmosNoSqlAdapter`` on the fake account. Only ``_connect`` differs."""

    shared: ClassVar[FakeAccount | None] = None
    """Account that ``create`` (the registry entry point) uses, when set."""

    def __init__(
        self,
        schema: CollectionSchema,
        settings: CosmosSettings,
        account: FakeAccount | None = None,
    ) -> None:
        super().__init__(schema, settings)
        self.account = account or FakeAccount()

    @classmethod
    def create(
        cls, schema: CollectionSchema, connection: Mapping[str, object]
    ) -> FakeCosmosAdapter:
        """Registry entry point: the real settings parser, the shared fake account."""
        return cls(schema, settings_from_connection(connection), account=cls.shared)

    async def _connect(self) -> _Client:
        client: _Client = FakeCosmosClient(self.account)  # type: ignore[assignment]
        return client

    @classmethod
    def build(
        cls, schema: CollectionSchema, *, full_text_search: bool = True
    ) -> FakeCosmosAdapter:
        """Return an adapter with native hybrid search on a new fake account."""
        return cls(
            schema,
            CosmosSettings(
                endpoint="https://fake.documents.azure.com:443/",
                key="fake-key",
                database_name="holodeck",
                create_database=True,
                full_text_search=full_text_search,
            ),
        )
