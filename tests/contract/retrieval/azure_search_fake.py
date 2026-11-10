"""An in-memory stand-in for the two Azure AI Search asynchronous clients.

There is no Azure AI Search emulator, so this fake is the only stateful target
for the adapter. It implements the surface that ``AzureAISearchAdapter`` calls
on ``azure.search.documents.aio.SearchClient`` and
``azure.search.documents.indexes.aio.SearchIndexClient`` and returns the SDK's
own model types (``SearchIndex``, ``IndexingResult``), so a renamed attribute in
the client shows up here.

What it enforces, because the real service does:

* index definitions: a name that follows the naming rules, exactly one string
  key field, vector fields with dimensions and a profile that exists;
* documents: keys of letters, digits, ``_``, ``-`` and ``=`` that do not start
  with ``_`` (a bad key fails that document with HTTP 400 in the per-document
  result, as the service does), known fields, values of the declared type,
  vectors of the declared size, at most 1000 documents per request;
* queries: ``$filter`` only on filterable fields, ``$orderby`` only on sortable
  fields, ``searchFields`` only on searchable fields, ``top`` capped at 1000
  (50 when absent), ``@search.score`` values that follow the service's formulas.

Scoring follows the documentation:

* vector query, cosine metric: ``1 / (1 + (1 - cosine))``;
* text query: a BM25-style score (an order, not the service's numbers);
* text plus vector in one request: Reciprocal Rank Fusion with constant 60;
* ``merge_or_upload``: fields a document leaves out keep their stored value,
  and an explicit ``None`` clears the field.

``search_lag`` delays the visibility of writes by that many ``search`` calls, to
test the adapter's wait for eventual consistency.

``$filter`` is parsed, not pattern-matched, for the subset the adapter emits:
``field op literal`` clauses joined by ``and``, with ``eq ne gt ge lt le``,
single-quoted strings (``''`` is a quote), numbers, ``true``, ``false`` and
``null``. Anything else is HTTP 400.
"""

from __future__ import annotations

import copy
import math
import operator
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

from azure.core.exceptions import (
    HttpResponseError,
    ResourceExistsError,
    ResourceNotFoundError,
)
from azure.search.documents.indexes.models import SearchField, SearchIndex
from azure.search.documents.models import IndexingResult, VectorizedQuery, VectorQuery

from holodeck.lib.retrieval.adapters.azure_ai_search import (
    AzureSearchSettings,
    ClientPair,
)

_KEY = re.compile(r"^[A-Za-z0-9=-][A-Za-z0-9_=-]*$")
_INDEX_NAME = re.compile(r"^[a-z0-9]+([-_][a-z0-9]+)*$")
_WORD = re.compile(r"\w+", re.UNICODE)
_RRF_K: Final = 60
MAX_BATCH: Final = 1000
MAX_TOP: Final = 1000
DEFAULT_TOP: Final = 50

Document = dict[str, object]


def bad_request(message: str) -> HttpResponseError:
    """Return the error the service raises for HTTP 400."""
    error = HttpResponseError(message=message)
    error.status_code = 400
    return error


# -- $filter -------------------------------------------------------------------------

_TOKEN = re.compile(
    r"\s*(?:(?P<str>'(?:[^']|'')*')"
    r"|(?P<num>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"|(?P<word>[A-Za-z_][A-Za-z0-9_]*))"
)
_OPERATORS = {"eq", "ne", "gt", "ge", "lt", "le"}


@dataclass(frozen=True)
class Clause:
    """One ``field op literal`` comparison."""

    field: str
    op: str
    value: str | int | float | bool | None


def _tokens(text: str) -> list[tuple[str, str]]:
    position, out = 0, []
    while position < len(text):
        if not text[position:].strip():
            break
        match = _TOKEN.match(text, position)
        if match is None:
            raise bad_request(f"Invalid $filter near {text[position:position+20]!r}.")
        kind = str(match.lastgroup)
        out.append((kind, match.group(kind)))
        position = match.end()
    return out


def parse_filter(text: str) -> list[Clause]:
    """Parse the adapter's ``$filter`` subset into clauses."""
    tokens = _tokens(text)
    clauses: list[Clause] = []
    index = 0
    while True:
        if index + 3 > len(tokens):
            raise bad_request("Incomplete $filter.")
        name, op, literal = tokens[index], tokens[index + 1], tokens[index + 2]
        if name[0] != "word" or op[0] != "word" or op[1] not in _OPERATORS:
            raise bad_request("Invalid $filter clause.")
        clauses.append(Clause(name[1], op[1], _value(literal)))
        index += 3
        if index == len(tokens):
            return clauses
        if tokens[index] != ("word", "and"):
            raise bad_request("Clauses must be joined by 'and'.")
        index += 1


def _value(token: tuple[str, str]) -> str | int | float | bool | None:
    kind, text = token
    if kind == "str":
        return text[1:-1].replace("''", "'")
    if kind == "num":
        return float(text) if re.search(r"[.eE]", text) else int(text)
    if text in ("true", "false"):
        return text == "true"
    if text == "null":
        return None
    raise bad_request(f"Invalid literal {text!r}.")


_ORDER = {"gt": operator.gt, "ge": operator.ge, "lt": operator.lt, "le": operator.le}


def _compare(stored: object, op: str, literal: object) -> bool:
    if op in ("eq", "ne"):
        same = stored == literal and isinstance(stored, bool) == isinstance(
            literal, bool
        )
        return same if op == "eq" else not same
    if stored is None or literal is None:
        return False
    if isinstance(stored, str) and isinstance(literal, str):
        return bool(_ORDER[op](stored, literal))
    if isinstance(stored, (int, float)) and isinstance(literal, (int, float)):
        return bool(_ORDER[op](float(stored), float(literal)))
    raise bad_request("Cannot compare values of different types.")


# -- index state ---------------------------------------------------------------------


@dataclass
class _Index:
    definition: SearchIndex
    docs: dict[str, Document] = field(default_factory=dict)

    def fields(self) -> dict[str, SearchField]:
        return {declared.name: declared for declared in self.definition.fields}

    def key_name(self) -> str:
        return next(f.name for f in self.definition.fields if f.key)


def _edm(declared: SearchField) -> str:
    """Return the Edm type name of a field."""
    kind = declared.type
    return kind.value if isinstance(kind, Enum) else str(kind)


def _type_ok(declared: SearchField, value: object) -> bool:
    if value is None:
        return True
    edm = _edm(declared)
    if edm == "Edm.String":
        return isinstance(value, str)
    if edm == "Edm.Int64" or edm == "Edm.Int32":
        return isinstance(value, int) and not isinstance(value, bool)
    if edm == "Edm.Double":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if edm == "Edm.Boolean":
        return isinstance(value, bool)
    if edm == "Collection(Edm.Single)":
        return (
            isinstance(value, list)
            and len(value) == declared.vector_search_dimensions
            and all(isinstance(v, (int, float)) for v in value)
        )
    return False


def _tokenize(text: object) -> list[str]:
    return _WORD.findall(text.lower()) if isinstance(text, str) else []


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


@dataclass
class _Pending:
    remaining: int
    apply: Callable[[], None]


class FakeSearchService:
    """One Azure AI Search service: a set of indexes shared by every client."""

    def __init__(self, *, search_lag: int = 0) -> None:
        self.indexes: dict[str, _Index] = {}
        self.search_lag = search_lag
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.closed = 0
        self.built: list[str] = []
        self._pending: list[_Pending] = []

    def names(self, operation: str) -> list[dict[str, object]]:
        """Return the arguments of every logged call to ``operation``."""
        return [args for called, args in self.calls if called == operation]

    def pair(self, settings: AzureSearchSettings, index_name: str) -> ClientPair:
        """Client factory for ``AzureAISearchAdapter(client_factory=...)``."""
        self.built.append(index_name)
        return ClientPair(
            documents=FakeSearchClient(self, index_name),
            indexes=FakeSearchIndexClient(self),
        )

    def _write(self, apply: Callable[[], None]) -> None:
        if self.search_lag > 0:
            self._pending.append(_Pending(self.search_lag, apply))
        else:
            apply()

    def _tick(self) -> None:
        """Called at the start of every search: age pending writes."""
        still: list[_Pending] = []
        for item in self._pending:
            if item.remaining == 0:
                item.apply()
            else:
                item.remaining -= 1
                still.append(item)
        self._pending = still

    def index(self, name: str) -> _Index:
        """Return an index or raise 404."""
        found = self.indexes.get(name)
        if found is None:
            raise ResourceNotFoundError(message=f"Index '{name}' not found.")
        return found


class FakeSearchIndexClient:
    """``SearchIndexClient`` over a ``FakeSearchService``."""

    def __init__(self, service: FakeSearchService) -> None:
        self._service = service

    async def get_index(self, name: str) -> SearchIndex:
        self._service.calls.append(("get_index", {"name": name}))
        return copy.deepcopy(self._service.index(name).definition)

    async def create_index(self, index: SearchIndex) -> SearchIndex:
        self._service.calls.append(("create_index", {"name": index.name}))
        if index.name in self._service.indexes:
            raise ResourceExistsError(message=f"Index '{index.name}' exists.")
        self._validate(index)
        self._service.indexes[index.name] = _Index(copy.deepcopy(index))
        return copy.deepcopy(index)

    @staticmethod
    def _validate(index: SearchIndex) -> None:
        if not (2 <= len(index.name) <= 128 and _INDEX_NAME.match(index.name)):
            raise bad_request(f"Invalid index name '{index.name}'.")
        keys = [f for f in index.fields if f.key]
        if len(keys) != 1 or _edm(keys[0]) != "Edm.String":
            raise bad_request("An index needs exactly one Edm.String key field.")
        profiles = {p.name for p in index.vector_search.profiles or []}
        for declared in index.fields:
            if _edm(declared) == "Collection(Edm.Single)":
                if (
                    not declared.vector_search_dimensions
                    or declared.vector_search_profile_name not in profiles
                ):
                    raise bad_request(f"Vector field '{declared.name}' is incomplete.")
            elif declared.searchable and _edm(declared) != "Edm.String":
                raise bad_request(f"Field '{declared.name}' cannot be searchable.")

    async def delete_index(self, index: str) -> None:
        self._service.calls.append(("delete_index", {"name": index}))
        self._service.index(index)
        del self._service.indexes[index]

    async def close(self) -> None:
        self._service.closed += 1


class FakeSearchClient:
    """``SearchClient`` over a ``FakeSearchService``, bound to one index."""

    def __init__(self, service: FakeSearchService, index_name: str) -> None:
        self._service = service
        self._name = index_name

    # -- documents ---------------------------------------------------------------------

    async def merge_or_upload_documents(
        self, documents: list[dict[str, object]]
    ) -> list[IndexingResult]:
        self._service.calls.append(
            ("merge_or_upload", {"documents": copy.deepcopy(documents)})
        )
        return self._index(documents, delete=False)

    async def delete_documents(
        self, documents: list[dict[str, object]]
    ) -> list[IndexingResult]:
        self._service.calls.append(("delete", {"documents": copy.deepcopy(documents)}))
        return self._index(documents, delete=True)

    def _index(
        self, documents: list[dict[str, object]], *, delete: bool
    ) -> list[IndexingResult]:
        index = self._service.index(self._name)
        if len(documents) > MAX_BATCH:
            raise bad_request(f"A batch holds at most {MAX_BATCH} documents.")
        declared = index.fields()
        key_name = index.key_name()
        results: list[IndexingResult] = []
        for document in documents:
            key = document.get(key_name)
            problem = self._problem(declared, key_name, document, delete=delete)
            if problem is not None:
                results.append(
                    IndexingResult(
                        key=key if isinstance(key, str) else None,
                        succeeded=False,
                        status_code=400,
                        error_message=problem,
                    )
                )
                continue
            assert isinstance(key, str)
            created = key not in index.docs
            self._service._write(self._apply(index, key, document, delete, declared))
            results.append(
                IndexingResult(
                    key=key,
                    succeeded=True,
                    status_code=200 if delete or not created else 201,
                )
            )
        return results

    @staticmethod
    def _problem(
        declared: Mapping[str, SearchField],
        key_name: str,
        document: Mapping[str, object],
        *,
        delete: bool,
    ) -> str | None:
        key = document.get(key_name)
        if not isinstance(key, str) or not _KEY.match(key) or len(key) > 1024:
            return "Invalid document key: only letters, digits, _ - = and no leading _."
        if delete:
            return None
        for name, value in document.items():
            if name not in declared:
                return f"The property '{name}' does not exist on the index."
            if not _type_ok(declared[name], value):
                return (
                    f"The value of '{name}' does not match type {declared[name].type}."
                )
        return None

    @staticmethod
    def _apply(
        index: _Index,
        key: str,
        document: Document,
        delete: bool,
        declared: Mapping[str, SearchField],
    ) -> Callable[[], None]:
        snapshot = copy.deepcopy(document)

        def apply() -> None:
            if delete:
                index.docs.pop(key, None)
                return
            stored = index.docs.setdefault(key, dict.fromkeys(declared))
            stored.update(snapshot)

        return apply

    async def get_document(
        self, key: str, *, selected_fields: list[str] | None = None
    ) -> Document:
        self._service.calls.append(
            ("get_document", {"key": key, "selected_fields": selected_fields})
        )
        index = self._service.index(self._name)
        document = index.docs.get(key)
        if document is None:
            raise ResourceNotFoundError(message=f"Document '{key}' not found.")
        return self._project(index, document, selected_fields)

    async def get_document_count(self) -> int:
        self._service.calls.append(("get_document_count", {}))
        return len(self._service.index(self._name).docs)

    async def close(self) -> None:
        self._service.closed += 1

    @staticmethod
    def _project(
        index: _Index, document: Document, select: list[str] | None
    ) -> Document:
        declared = index.fields()
        names = (
            select
            if select is not None
            else [name for name, f in declared.items() if f.retrievable is not False]
        )
        for name in names:
            if name not in declared or declared[name].retrievable is False:
                raise bad_request(f"Cannot select '{name}'.")
        return {name: copy.deepcopy(document[name]) for name in names}

    # -- search ------------------------------------------------------------------------

    async def search(
        self,
        search_text: str | None = None,
        *,
        filter: str | None = None,
        order_by: list[str] | None = None,
        query_type: str | None = None,
        search_fields: list[str] | None = None,
        search_mode: str | None = None,
        select: list[str] | None = None,
        top: int | None = None,
        vector_queries: list[VectorQuery] | None = None,
        vector_filter_mode: str | None = None,
        skip: int | None = None,
    ) -> _Results:
        self._service.calls.append(
            (
                "search",
                {
                    "search_text": search_text,
                    "filter": filter,
                    "order_by": order_by,
                    "query_type": query_type,
                    "search_fields": search_fields,
                    "search_mode": search_mode,
                    "select": select,
                    "top": top,
                    "vector_queries": vector_queries,
                    "vector_filter_mode": vector_filter_mode,
                    "skip": skip,
                },
            )
        )
        self._service._tick()
        index = self._service.index(self._name)
        declared = index.fields()
        candidates = self._filtered(index, filter)
        ranked = self._rank(
            index,
            declared,
            candidates,
            search_text,
            search_fields,
            search_mode,
            vector_queries,
        )
        ordered = self._order(index, declared, ranked, order_by)
        size = DEFAULT_TOP if top is None else min(top, MAX_TOP)
        window = ordered[skip or 0 :][:size]
        rows: list[Document] = []
        for key, score in window:
            row = self._project(index, index.docs[key], select)
            row["@search.score"] = score
            for name in (
                "@search.reranker_score",
                "@search.highlights",
                "@search.captions",
                "@search.document_debug_info",
                "@search.reranker_boosted_score",
            ):
                row[name] = None
            rows.append(row)
        return _Results(rows)

    def _filtered(self, index: _Index, expression: str | None) -> list[str]:
        clauses = parse_filter(expression) if expression else []
        declared = index.fields()
        for clause in clauses:
            if clause.field not in declared:
                raise bad_request(f"Unknown field '{clause.field}' in $filter.")
            if not declared[clause.field].filterable:
                raise bad_request(f"Field '{clause.field}' is not filterable.")
            self._check_literal(declared[clause.field], clause)
        return [
            key
            for key, document in index.docs.items()
            if all(_compare(document[c.field], c.op, c.value) for c in clauses)
        ]

    @staticmethod
    def _check_literal(declared: SearchField, clause: Clause) -> None:
        if clause.value is None:
            return
        edm = _edm(declared)
        ok = {
            "Edm.String": isinstance(clause.value, str),
            "Edm.Boolean": isinstance(clause.value, bool),
        }.get(
            edm,
            isinstance(clause.value, (int, float))
            and not isinstance(clause.value, bool),
        )
        if not ok:
            raise bad_request(
                f"Literal {clause.value!r} does not match {edm} of '{clause.field}'."
            )

    def _rank(
        self,
        index: _Index,
        declared: Mapping[str, SearchField],
        candidates: list[str],
        text: str | None,
        search_fields: list[str] | None,
        mode: str | None,
        vector_queries: list[VectorQuery] | None,
    ) -> list[tuple[str, float]]:
        lists: list[list[tuple[str, float]]] = []
        has_text = text is not None and text.strip() not in ("", "*")
        if has_text:
            assert text is not None
            lists.append(
                self._text_hits(index, declared, candidates, text, search_fields, mode)
            )
        for query in vector_queries or []:
            lists.append(self._vector_hits(index, declared, candidates, query))
        if not lists:
            return [(key, 1.0) for key in sorted(candidates)]
        if len(lists) == 1:
            return lists[0]
        fused: dict[str, float] = {}
        for hits in lists:
            for rank, (key, _score) in enumerate(hits, start=1):
                fused[key] = fused.get(key, 0.0) + 1.0 / (_RRF_K + rank)
        return sorted(fused.items(), key=lambda item: (-item[1], item[0]))

    def _text_hits(
        self,
        index: _Index,
        declared: Mapping[str, SearchField],
        candidates: list[str],
        text: str,
        search_fields: list[str] | None,
        mode: str | None,
    ) -> list[tuple[str, float]]:
        fields = search_fields or [
            name
            for name, f in declared.items()
            if f.searchable and _edm(f) == "Edm.String"
        ]
        for name in fields:
            if name not in declared or not declared[name].searchable:
                raise bad_request(f"Field '{name}' is not searchable.")
        wanted = list(dict.fromkeys(_tokenize(text)))
        everyone = list(index.docs.values())
        frequency = {
            token: sum(
                1
                for document in everyone
                if token in {t for n in fields for t in _tokenize(document[n])}
            )
            for token in wanted
        }
        total = len(everyone)
        hits: list[tuple[str, float]] = []
        for key in candidates:
            tokens = [t for n in fields for t in _tokenize(index.docs[key][n])]
            matched = [token for token in wanted if token in tokens]
            if not matched or (mode == "all" and len(matched) != len(wanted)):
                continue
            score = sum(
                tokens.count(token)
                * math.log(
                    1 + (total - frequency[token] + 0.5) / (frequency[token] + 0.5)
                )
                for token in matched
            )
            hits.append((key, score))
        return sorted(hits, key=lambda item: (-item[1], item[0]))

    def _vector_hits(
        self,
        index: _Index,
        declared: Mapping[str, SearchField],
        candidates: list[str],
        query: VectorQuery,
    ) -> list[tuple[str, float]]:
        if not isinstance(query, VectorizedQuery):
            raise bad_request("Only raw vector queries are supported.")
        name = query.fields
        if (
            name is None
            or name not in declared
            or not declared[name].vector_search_dimensions
        ):
            raise bad_request(f"'{name}' is not a vector field.")
        if len(query.vector) != declared[name].vector_search_dimensions:
            raise bad_request("The query vector has the wrong number of dimensions.")
        k = query.k_nearest_neighbors or 50
        hits: list[tuple[str, float]] = []
        for key in candidates:
            vector = index.docs[key][name]
            if not isinstance(vector, list):
                continue
            distance = 1.0 - _cosine([float(v) for v in query.vector], vector)
            hits.append((key, 1.0 / (1.0 + distance)))
        hits.sort(key=lambda item: (-item[1], item[0]))
        return hits[:k]

    @staticmethod
    def _order(
        index: _Index,
        declared: Mapping[str, SearchField],
        ranked: list[tuple[str, float]],
        order_by: list[str] | None,
    ) -> list[tuple[str, float]]:
        if not order_by:
            return ranked
        ordered = list(ranked)
        for clause in reversed(order_by):
            name, _, direction = clause.partition(" ")
            if name not in declared or not declared[name].sortable:
                raise bad_request(f"Field '{name}' is not sortable.")
            ordered.sort(
                key=lambda item: index.docs[item[0]][name],  # type: ignore[arg-type,return-value]
                reverse=direction.strip() == "desc",
            )
        return ordered


class _Results:
    """The async iterable that ``SearchClient.search`` returns."""

    def __init__(self, rows: list[Document]) -> None:
        self._rows = iter(rows)

    def __aiter__(self) -> _Results:
        return self

    async def __anext__(self) -> Document:
        try:
            return next(self._rows)
        except StopIteration:
            raise StopAsyncIteration from None
