"""Unit tests for backend-aware collection filter helpers."""

from __future__ import annotations

import dataclasses
import importlib.util
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from holodeck.lib.collection_filter import (
    _PAGE_SIZE,
    find_all_records,
    find_records_by_field,
)

_HAS_PSYCOPG = importlib.util.find_spec("psycopg") is not None
_HAS_WEAVIATE = importlib.util.find_spec("weaviate") is not None
_HAS_QDRANT = importlib.util.find_spec("qdrant_client") is not None


class TestProviderDispatch:
    """Errors for unsupported providers."""

    @pytest.mark.asyncio
    async def test_unknown_provider_raises_for_find_by_field(self) -> None:
        with pytest.raises(NotImplementedError, match="pinecone"):
            await find_records_by_field("pinecone", MagicMock(), "x", "y")

    @pytest.mark.asyncio
    async def test_unknown_provider_raises_for_find_all(self) -> None:
        with pytest.raises(NotImplementedError, match="redis"):
            await find_all_records("redis", MagicMock())


@pytest.mark.skipif(not _HAS_QDRANT, reason="qdrant-client not installed")
class TestQdrant:
    """qdrant scroll-based filter."""

    @pytest.mark.asyncio
    async def test_find_by_field_builds_match_filter(self) -> None:
        """Issues a scroll with a MatchValue Filter and returns payload+id."""
        point = SimpleNamespace(
            id="abc", payload={"source_path": "/a.md", "mtime": 1.0}
        )
        client = MagicMock()
        client.scroll = AsyncMock(return_value=([point], None))
        collection = MagicMock(qdrant_client=client, collection_name="docs")

        records = await find_records_by_field(
            "qdrant", collection, "source_path", "/a.md"
        )

        assert records == [{"source_path": "/a.md", "mtime": 1.0, "id": "abc"}]
        kwargs = client.scroll.call_args.kwargs
        assert kwargs["collection_name"] == "docs"
        assert kwargs["with_vectors"] is False
        assert kwargs["with_payload"] is True
        # Filter is a qdrant_client.models.Filter with one FieldCondition.
        scroll_filter = kwargs["scroll_filter"]
        assert scroll_filter is not None
        assert len(scroll_filter.must) == 1
        assert scroll_filter.must[0].key == "source_path"
        assert scroll_filter.must[0].match.value == "/a.md"

    @pytest.mark.asyncio
    async def test_find_all_omits_filter(self) -> None:
        """No filter is sent when field is None — full scroll."""
        client = MagicMock()
        client.scroll = AsyncMock(return_value=([], None))
        collection = MagicMock(qdrant_client=client, collection_name="docs")

        await find_all_records("qdrant", collection)

        assert client.scroll.call_args.kwargs["scroll_filter"] is None

    @pytest.mark.asyncio
    async def test_pagination_follows_offset(self) -> None:
        """Pages through results until offset is None."""
        page1 = [SimpleNamespace(id=str(i), payload={"n": i}) for i in range(2)]
        page2 = [SimpleNamespace(id=str(i), payload={"n": i}) for i in range(2, 3)]
        client = MagicMock()
        client.scroll = AsyncMock(side_effect=[(page1, "next-token"), (page2, None)])
        collection = MagicMock(qdrant_client=client, collection_name="docs")

        records = await find_all_records("qdrant", collection)

        assert len(records) == 3
        assert client.scroll.call_count == 2
        # Second call must carry the prior offset token.
        assert client.scroll.call_args_list[1].kwargs["offset"] == "next-token"

    @pytest.mark.asyncio
    async def test_limit_caps_page_size_on_last_page(self) -> None:
        """`limit` is honoured even when smaller than the default page."""
        point = SimpleNamespace(id="0", payload={})
        client = MagicMock()
        # Return one point then signal more available; helper should stop at limit=1.
        client.scroll = AsyncMock(return_value=([point], "more"))
        collection = MagicMock(qdrant_client=client, collection_name="docs")

        records = await find_all_records("qdrant", collection, limit=1)

        assert len(records) == 1
        assert client.scroll.call_count == 1
        assert client.scroll.call_args.kwargs["limit"] == 1

    @pytest.mark.asyncio
    async def test_handles_none_payload(self) -> None:
        """A point with no payload still yields an `{id: ...}` dict."""
        point = SimpleNamespace(id="x", payload=None)
        client = MagicMock()
        client.scroll = AsyncMock(return_value=([point], None))
        collection = MagicMock(qdrant_client=client, collection_name="docs")

        records = await find_records_by_field("qdrant", collection, "k", "v")

        assert records == [{"id": "x"}]

    @pytest.mark.asyncio
    async def test_default_page_size(self) -> None:
        """First call uses the module's default page size."""
        client = MagicMock()
        client.scroll = AsyncMock(return_value=([], None))
        collection = MagicMock(qdrant_client=client, collection_name="docs")

        await find_all_records("qdrant", collection)

        assert client.scroll.call_args.kwargs["limit"] == _PAGE_SIZE


class TestChromadb:
    """chroma `get(where=...)` path."""

    def _make_collection(self, raw: dict) -> MagicMock:
        inner = MagicMock()
        inner.get = MagicMock(return_value=raw)
        collection = MagicMock()
        collection._get_collection = MagicMock(return_value=inner)
        return collection

    @pytest.mark.asyncio
    async def test_find_by_field_uses_where_clause(self) -> None:
        """Filter becomes a chroma `where` dict."""
        collection = self._make_collection(
            {
                "ids": ["a", "b"],
                "metadatas": [{"source_path": "/x", "mtime": 1}, {"source_path": "/x"}],
                "documents": ["doc-a", "doc-b"],
            }
        )

        records = await find_records_by_field(
            "chromadb", collection, "source_path", "/x"
        )

        assert len(records) == 2
        assert records[0]["id"] == "a"
        assert records[0]["source_path"] == "/x"
        assert records[0]["document"] == "doc-a"
        kwargs = collection._get_collection().get.call_args.kwargs
        assert kwargs["where"] == {"source_path": "/x"}

    @pytest.mark.asyncio
    async def test_find_all_omits_where(self) -> None:
        """No `where` key is passed when field is None."""
        collection = self._make_collection(
            {"ids": [], "metadatas": [], "documents": []}
        )

        await find_all_records("chromadb", collection)

        # chroma's `where` parameter must not be set for unfiltered fetch
        # (chroma raises ValueError on empty where dicts).
        kwargs = collection._get_collection().get.call_args.kwargs
        assert "where" not in kwargs

    @pytest.mark.asyncio
    async def test_missing_metadata_yields_id_only(self) -> None:
        """If metadata slot is missing, we still surface the id."""
        collection = self._make_collection({"ids": ["solo"], "metadatas": []})

        records = await find_records_by_field("chromadb", collection, "k", "v")

        assert records == [{"id": "solo"}]

    @pytest.mark.asyncio
    async def test_limit_forwarded(self) -> None:
        """`limit` is passed through to chroma."""
        collection = self._make_collection({"ids": [], "metadatas": []})

        await find_all_records("chromadb", collection, limit=42)

        assert collection._get_collection().get.call_args.kwargs["limit"] == 42


@pytest.mark.skipif(not _HAS_PSYCOPG, reason="psycopg not installed (postgres extra)")
class TestPostgres:
    """SK postgres connector via the underlying pool."""

    def _make_collection(self, rows: list[tuple], col_names: list[str]) -> MagicMock:
        # Fields the helper inspects to build the SELECT.
        fields = [SimpleNamespace(name=c, storage_name=c) for c in col_names]
        definition = SimpleNamespace(
            fields=fields,
            key_field_storage_name="id",
        )

        cur = MagicMock()
        cur.execute = AsyncMock()
        cur.fetchall = AsyncMock(return_value=rows)
        cur.description = [SimpleNamespace(name=c) for c in col_names]
        cur.__aenter__ = AsyncMock(return_value=cur)
        cur.__aexit__ = AsyncMock(return_value=None)

        conn = MagicMock()
        conn.cursor = MagicMock(return_value=cur)
        conn.__aenter__ = AsyncMock(return_value=conn)
        conn.__aexit__ = AsyncMock(return_value=None)

        pool = MagicMock()
        pool.connection = MagicMock(return_value=conn)

        return (
            MagicMock(
                connection_pool=pool,
                db_schema="public",
                collection_name="docs",
                definition=definition,
            ),
            cur,
        )

    @pytest.mark.asyncio
    async def test_find_by_field_builds_where_clause(self) -> None:
        """Generates `WHERE field = %s` with the value bound positionally."""
        collection, cur = self._make_collection(
            [("k1", "/a", 1.0)], ["id", "source_path", "mtime"]
        )

        records = await find_records_by_field(
            "postgres", collection, "source_path", "/a"
        )

        assert records == [{"id": "k1", "source_path": "/a", "mtime": 1.0}]
        # The composed SQL is a psycopg.sql.Composed; render to string for assertion.
        sql_obj = cur.execute.call_args.args[0]
        rendered = (
            sql_obj.as_string(None) if hasattr(sql_obj, "as_string") else str(sql_obj)
        )
        assert '"source_path"' in rendered
        assert "WHERE" in rendered
        # Bound params: (value,)
        assert cur.execute.call_args.args[1] == ("/a",)

    @pytest.mark.asyncio
    async def test_find_all_omits_where(self) -> None:
        """No WHERE clause and empty params when no field is set."""
        collection, cur = self._make_collection([], ["id"])

        await find_all_records("postgres", collection)

        sql_obj = cur.execute.call_args.args[0]
        rendered = (
            sql_obj.as_string(None) if hasattr(sql_obj, "as_string") else str(sql_obj)
        )
        assert "WHERE" not in rendered
        assert cur.execute.call_args.args[1] == ()

    @pytest.mark.asyncio
    async def test_limit_appended(self) -> None:
        """`LIMIT %s` is appended and value is in the params."""
        collection, cur = self._make_collection([], ["id"])

        await find_all_records("postgres", collection, limit=7)

        sql_obj = cur.execute.call_args.args[0]
        rendered = (
            sql_obj.as_string(None) if hasattr(sql_obj, "as_string") else str(sql_obj)
        )
        assert "LIMIT" in rendered
        assert cur.execute.call_args.args[1] == (7,)

    @pytest.mark.asyncio
    async def test_raises_when_pool_uninitialised(self) -> None:
        """A clear error if the collection wasn't entered as a context manager."""
        collection = MagicMock(connection_pool=None)
        with pytest.raises(RuntimeError, match="connection_pool"):
            await find_records_by_field("postgres", collection, "x", "y")


@pytest.mark.skipif(not _HAS_WEAVIATE, reason="weaviate-client not installed")
class TestWeaviate:
    """weaviate `fetch_objects` filter path."""

    def _make_collection(self, *pages: list) -> MagicMock:
        wcoll = MagicMock()
        responses = [MagicMock(objects=objs) for objs in pages]
        wcoll.query.fetch_objects = AsyncMock(side_effect=responses)
        async_client = MagicMock()
        async_client.collections.get = MagicMock(return_value=wcoll)
        return MagicMock(async_client=async_client, collection_name="Docs"), wcoll

    @pytest.mark.asyncio
    async def test_find_by_field_applies_property_filter(self) -> None:
        obj = SimpleNamespace(uuid="u1", properties={"source_path": "/a", "mtime": 1.0})
        # Single short page → loop exits because len < page size.
        collection, wcoll = self._make_collection([obj])

        records = await find_records_by_field(
            "weaviate", collection, "source_path", "/a"
        )

        assert records == [{"id": "u1", "source_path": "/a", "mtime": 1.0}]
        # Filter argument is non-None for filtered fetch.
        assert wcoll.query.fetch_objects.call_args.kwargs["filters"] is not None

    @pytest.mark.asyncio
    async def test_find_all_omits_filter(self) -> None:
        collection, wcoll = self._make_collection([])
        await find_all_records("weaviate", collection)
        assert wcoll.query.fetch_objects.call_args.kwargs["filters"] is None

    @pytest.mark.asyncio
    async def test_pagination_advances_offset(self) -> None:
        """A full page triggers a second fetch with offset bumped by page size."""
        full_page = [
            SimpleNamespace(uuid=str(i), properties={}) for i in range(_PAGE_SIZE)
        ]
        last_page = [SimpleNamespace(uuid="last", properties={})]
        collection, wcoll = self._make_collection(full_page, last_page)

        records = await find_all_records("weaviate", collection)

        assert len(records) == _PAGE_SIZE + 1
        assert (
            wcoll.query.fetch_objects.call_args_list[1].kwargs["offset"] == _PAGE_SIZE
        )


class TestInMemory:
    """SK in-memory store iteration."""

    @dataclasses.dataclass
    class FakeRecord:
        id: str
        source_path: str
        mtime: float
        embedding: list[float]

    @pytest.mark.asyncio
    async def test_find_by_field_filters_attribute_equality(self) -> None:
        """Only records whose attribute matches are returned."""
        records_in = {
            "a": self.FakeRecord("a", "/x", 1.0, [0.1]),
            "b": self.FakeRecord("b", "/y", 2.0, [0.2]),
            "c": self.FakeRecord("c", "/x", 3.0, [0.3]),
        }
        collection = MagicMock(inner_storage=records_in)

        results = await find_records_by_field(
            "in-memory", collection, "source_path", "/x"
        )

        ids = sorted(r["id"] for r in results)
        assert ids == ["a", "c"]
        # Vectors are omitted from the projected dict.
        assert "embedding" not in results[0]

    @pytest.mark.asyncio
    async def test_find_all_returns_every_record(self) -> None:
        records_in = {
            "a": self.FakeRecord("a", "/x", 1.0, [0.1]),
            "b": self.FakeRecord("b", "/y", 2.0, [0.2]),
        }
        collection = MagicMock(inner_storage=records_in)

        results = await find_all_records("in-memory", collection)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_limit_truncates(self) -> None:
        records_in = {
            str(i): self.FakeRecord(str(i), "/x", float(i), [0.0]) for i in range(5)
        }
        collection = MagicMock(inner_storage=records_in)

        results = await find_all_records("in-memory", collection, limit=2)

        assert len(results) == 2
