"""Golden BLEU, ROUGE and METEOR scores for ``holodeck.lib.evaluators.nlp_metrics``.

Seven (prediction, reference) pairs are scored with ``BLEUEvaluator``,
``ROUGEEvaluator`` (rouge1, rouge2, rougeL) and ``METEOREvaluator``. Scores are
frozen at six decimals in ``tests/fixtures/golden/nlp_metrics/expected.json``
and compared with ``pytest.approx(abs=1e-6)``. A threshold case freezes the
``passed`` flag for each metric and ROUGE variant.

Network: the current ROUGE and METEOR evaluators call ``evaluate.load``, which
fetches the metric script from the Hugging Face hub and METEOR downloads NLTK
data on first use. When a metric cannot load (offline, no cache) its tests skip.

Regenerate with ``HOLODECK_UPDATE_GOLDEN=1`` (see ``_golden.py``).
"""

from __future__ import annotations

from typing import Any

import pytest

from holodeck.lib.evaluators.nlp_metrics import (
    BLEUEvaluator,
    METEOREvaluator,
    NLPMetricsError,
    ROUGEEvaluator,
)

from ._golden import GOLDEN_ROOT, assert_golden

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

EXPECTED = GOLDEN_ROOT / "nlp_metrics" / "expected.json"
PRECISION = 6

PAIRS: dict[str, tuple[str, str]] = {
    "identical": ("The cat sat on the mat.", "The cat sat on the mat."),
    "one_word_changed": ("The cat sat on the mat", "The cat is on the mat"),
    "synonym": ("The automobile is red", "The car is red"),
    "disjoint": ("Completely different words here", "Nothing overlaps with that text"),
    "summary": (
        "The survey counted 412 tern nests near the lighthouse. A kestrel took "
        "two chicks from the eastern colony.",
        "Rangers counted 412 arctic tern nests along the northern shingle, "
        "densest near the old lighthouse. A kestrel hunting over the dunes took "
        "two chicks before wardens scared it off.",
    ),
    "unicode_punctuation": (
        "Café déjà vu: naïve, résumé!",
        "Cafe deja vu naive resume",
    ),
    "empty_prediction": ("", "A reference with some words"),
}


def _round(value: Any) -> float:
    return round(float(value), PRECISION)


async def _score(
    evaluator_name: str, response: str, reference: str
) -> dict[str, float]:
    if evaluator_name == "bleu":
        result = await BLEUEvaluator().evaluate(
            response=response, ground_truth=reference
        )
        return {"bleu": _round(result["bleu"])}
    if evaluator_name == "rouge":
        result = await ROUGEEvaluator().evaluate(
            response=response, ground_truth=reference
        )
        return {k: _round(result[k]) for k in ("rouge1", "rouge2", "rougeL")}
    result = await METEOREvaluator().evaluate(response=response, ground_truth=reference)
    return {"meteor": _round(result["meteor"])}


async def _scores_or_skip(evaluator_name: str) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    try:
        for pair_id, (response, reference) in PAIRS.items():
            scores[pair_id] = await _score(evaluator_name, response, reference)
    except NLPMetricsError as exc:
        pytest.skip(f"{evaluator_name} cannot run offline: {exc}")
    return scores


@pytest.mark.parametrize("evaluator_name", ["bleu", "rouge", "meteor"])
async def test_nlp_scores_golden(evaluator_name: str) -> None:
    """Scores for every pair match the frozen goldens to six decimals."""
    actual = {
        "metric": evaluator_name,
        "pairs": {k: {"prediction": p, "reference": r} for k, (p, r) in PAIRS.items()},
        "scores": await _scores_or_skip(evaluator_name),
    }
    assert_golden(
        EXPECTED.with_name(f"{evaluator_name}.json"),
        actual,
        abs_tol=1e-6,
        label=f"nlp_metrics/{evaluator_name}",
    )


async def test_nlp_threshold_flags_golden() -> None:
    """The ``passed`` flag for threshold 0.5 matches the golden for each metric."""
    pair_id = "one_word_changed"
    response, reference = PAIRS[pair_id]
    try:
        flags: dict[str, Any] = {}
        for variant in ("rouge1", "rouge2", "rougeL"):
            result = await ROUGEEvaluator(threshold=0.5, variant=variant).evaluate(
                response=response, ground_truth=reference
            )
            flags[variant] = result["passed"]
        flags["bleu"] = (
            await BLEUEvaluator(threshold=0.5).evaluate(
                response=response, ground_truth=reference
            )
        )["passed"]
        flags["meteor"] = (
            await METEOREvaluator(threshold=0.5).evaluate(
                response=response, ground_truth=reference
            )
        )["passed"]
    except NLPMetricsError as exc:
        pytest.skip(f"metrics cannot run offline: {exc}")
    assert_golden(
        EXPECTED.with_name("threshold_flags.json"),
        {"pair": pair_id, "threshold": 0.5, "passed": flags},
        label="nlp_metrics/threshold_flags",
    )
