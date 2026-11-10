"""Tests for the vectorstore tool's search result types."""

import pytest

from holodeck.lib.retrieval import QueryResult


def test_query_result_holds_its_fields() -> None:
    result = QueryResult(
        content="Found content",
        score=0.95,
        source_path="/path/to/source.txt",
        chunk_index=2,
        metadata={"file_type": ".txt", "file_size_bytes": 1000},
    )
    assert result.content == "Found content"
    assert result.score == 0.95
    assert result.source_path == "/path/to/source.txt"
    assert result.chunk_index == 2
    assert result.metadata == {"file_type": ".txt", "file_size_bytes": 1000}


def test_query_result_metadata_defaults_to_empty() -> None:
    first = QueryResult("a", score=0.5, source_path="/a", chunk_index=0)
    second = QueryResult("b", score=0.5, source_path="/b", chunk_index=0)
    assert first.metadata == {}
    assert first.metadata is not second.metadata


@pytest.mark.parametrize("score", [0.0, 0.5, 0.99, 1.0])
def test_query_result_accepts_scores_in_range(score: float) -> None:
    result = QueryResult("content", score=score, source_path="/p", chunk_index=0)
    assert result.score == score


@pytest.mark.parametrize("score", [-1.0, -0.1, 1.1, 2.5])
def test_query_result_rejects_scores_out_of_range(score: float) -> None:
    with pytest.raises(ValueError, match="Score must be between 0.0 and 1.0"):
        QueryResult("content", score=score, source_path="/p", chunk_index=0)
