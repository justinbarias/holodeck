"""Offline guarantees for the NLP metrics: no network, no Hugging Face evaluate."""

from __future__ import annotations

import ast
import socket
from pathlib import Path

import pytest

from holodeck.lib.evaluators.nlp_metrics import (
    METEOREvaluator,
    ROUGEEvaluator,
    missing_nltk_resource,
)

SRC = Path(__file__).resolve().parents[4] / "src"


def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse)


@pytest.mark.asyncio
async def test_rouge_and_meteor_use_no_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ROUGE and METEOR compute with every socket connection refused."""
    _block_network(monkeypatch)

    rouge = await ROUGEEvaluator().evaluate(
        response="The cat sat on the mat", ground_truth="The cat is on the mat"
    )
    assert 0.0 < rouge["rougeL"] < 1.0

    missing = missing_nltk_resource()
    if missing is not None:
        pytest.skip(
            f"NLTK resource '{missing}' is not installed; run "
            "`python -m nltk.downloader wordnet punkt_tab omw-1.4`"
        )
    meteor = await METEOREvaluator().evaluate(
        response="The cat sat on the mat", ground_truth="The cat is on the mat"
    )
    assert 0.0 < meteor["meteor"] < 1.0


def test_evaluate_package_is_not_imported_in_src() -> None:
    """Nothing under src/ imports the Hugging Face ``evaluate`` package."""
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            if any(n == "evaluate" or n.startswith("evaluate.") for n in names):
                offenders.append(str(path.relative_to(SRC)))
    assert not offenders, f"Hugging Face evaluate imported in: {offenders}"


@pytest.mark.unit
def test_meteor_names_the_download_command_when_nltk_data_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A data directory with a missing file passes the check but fails later."""
    import nltk

    from holodeck.lib.evaluators import nlp_metrics

    def broken_tokenizer(text: str) -> list[str]:
        raise LookupError("collocations.tab not found")

    monkeypatch.setattr(nlp_metrics, "missing_nltk_resource", lambda: None)
    monkeypatch.setattr(nltk, "word_tokenize", broken_tokenizer)

    score = nlp_metrics.METEOREvaluator()._get_metric()
    with pytest.raises(nlp_metrics.NLPMetricsError, match="nltk.downloader"):
        score("the cat sat", "the cat sat")
