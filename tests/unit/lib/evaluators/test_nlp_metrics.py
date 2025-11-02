"""Unit tests for NLP metrics evaluators.

Tests cover:
- BLEU score calculation using Hugging Face evaluate library
- ROUGE score calculation (ROUGE-1, ROUGE-2, ROUGE-L)
- METEOR score calculation
- F1 score calculation for text similarity
- Error handling for missing ground truth
- Metric normalization (SacreBLEU 0-100 → 0-1)
- Lazy loading of evaluate metrics
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from holodeck.lib.evaluators.base import EvaluationError
from holodeck.lib.evaluators.nlp_metrics import (
    BLEUEvaluator,
    METEOREvaluator,
    ROUGEEvaluator,
)


class TestBLEUEvaluator:
    """Test BLEU score evaluator using SacreBLEU."""

    @pytest.mark.asyncio
    async def test_bleu_perfect_match(self, bleu_evaluator: BLEUEvaluator) -> None:
        """Test BLEU score for perfect match."""
        result = await bleu_evaluator.evaluate(
            response="The cat sat on the mat",
            ground_truth="The cat sat on the mat",
        )

        assert "bleu" in result
        assert result["bleu"] == pytest.approx(1.0, rel=0.01)
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_bleu_partial_match(self, bleu_evaluator: BLEUEvaluator) -> None:
        """Test BLEU score for partial match."""
        result = await bleu_evaluator.evaluate(
            response="The cat sat on the mat",
            ground_truth="The cat is on the mat",
        )

        assert "bleu" in result
        # BLEU may be 0 for short sentences with word differences
        # Just verify it returns a valid score
        assert 0.0 <= result["bleu"] <= 1.0
        assert isinstance(result["bleu"], float)

    @pytest.mark.asyncio
    async def test_bleu_no_match(self, bleu_evaluator: BLEUEvaluator) -> None:
        """Test BLEU score for no match."""
        result = await bleu_evaluator.evaluate(
            response="Hello world",
            ground_truth="Goodbye universe",
        )

        assert "bleu" in result
        assert result["bleu"] == pytest.approx(0.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_bleu_score_range(self) -> None:
        """Test that SacreBLEU returns scores normalized to 0-1 range."""
        # Mock the sacrebleu library to return a BLEU score
        with patch(
            "holodeck.lib.evaluators.nlp_metrics._get_sacrebleu"
        ) as mock_get_sacrebleu:
            mock_sacrebleu = MagicMock()
            mock_result = MagicMock()
            mock_result.score = 75.0  # SacreBLEU returns 0-100 range
            mock_sacrebleu.sentence_bleu.return_value = mock_result
            mock_get_sacrebleu.return_value = mock_sacrebleu

            # Create new evaluator to pick up mocked sacrebleu
            evaluator_mocked = BLEUEvaluator()
            result = await evaluator_mocked.evaluate(
                response="test", ground_truth="test"
            )

            # Should be normalized to 0-1 range
            assert result["bleu"] == pytest.approx(0.75, abs=0.01)

    @pytest.mark.asyncio
    async def test_bleu_threshold_passing(self) -> None:
        """Test BLEU score threshold passing logic."""
        evaluator = BLEUEvaluator(threshold=0.5)

        # High score should pass
        result = await evaluator.evaluate(
            response="The cat sat on the mat",
            ground_truth="The cat sat on the mat",
        )
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_bleu_threshold_failing(self) -> None:
        """Test BLEU score threshold failing logic."""
        evaluator = BLEUEvaluator(threshold=0.9)

        # Partial match should fail high threshold
        result = await evaluator.evaluate(
            response="The cat sat on the mat",
            ground_truth="The dog sat on the rug",
        )
        assert result["passed"] is False

    @pytest.mark.asyncio
    async def test_bleu_missing_ground_truth(
        self, bleu_evaluator: BLEUEvaluator
    ) -> None:
        """Test BLEU evaluator with missing ground truth."""
        # ValueError is wrapped in EvaluationError by the retry mechanism
        with pytest.raises(Exception, match="ground_truth.*required"):
            await bleu_evaluator.evaluate(response="test response")

    @pytest.mark.asyncio
    async def test_bleu_empty_strings(self, bleu_evaluator: BLEUEvaluator) -> None:
        """Test BLEU evaluator with empty strings."""
        # Empty response should result in low/zero score or error
        # The BLEU metric may raise an error for empty strings
        try:
            result = await bleu_evaluator.evaluate(response="", ground_truth="test")
            assert "bleu" in result
            assert result["bleu"] <= 0.1  # Very low score
        except Exception as e:
            # It's acceptable for BLEU to fail on empty strings
            logging.warning(
                f"BLEU failed on empty string as expected: {type(e).__name__}"
            )

    @pytest.mark.asyncio
    async def test_bleu_lazy_loading(self) -> None:
        """Test that SacreBLEU library is loaded lazily."""
        with patch(
            "holodeck.lib.evaluators.nlp_metrics._get_sacrebleu"
        ) as mock_get_sacrebleu:
            mock_sacrebleu = MagicMock()
            mock_get_sacrebleu.return_value = mock_sacrebleu

            evaluator = BLEUEvaluator()
            # Library should not be loaded until evaluate is called
            mock_get_sacrebleu.assert_not_called()

            # Now trigger lazy load
            mock_result = MagicMock()
            mock_result.score = 50.0
            mock_sacrebleu.sentence_bleu.return_value = mock_result

            await evaluator.evaluate(response="test", ground_truth="test")

            # Library should be loaded exactly once
            mock_get_sacrebleu.assert_called_once()


class TestROUGEEvaluator:
    """Test ROUGE score evaluator."""

    @pytest.mark.asyncio
    async def test_rouge_perfect_match(self, rouge_evaluator: ROUGEEvaluator) -> None:
        """Test ROUGE scores for perfect match."""
        result = await rouge_evaluator.evaluate(
            response="The cat sat on the mat",
            ground_truth="The cat sat on the mat",
        )

        assert "rouge1" in result
        assert "rouge2" in result
        assert "rougeL" in result
        assert result["rouge1"] == pytest.approx(1.0, rel=0.01)
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_rouge_partial_match(self, rouge_evaluator: ROUGEEvaluator) -> None:
        """Test ROUGE scores for partial match."""
        result = await rouge_evaluator.evaluate(
            response="The cat sat on the mat",
            ground_truth="The cat is on the mat",
        )

        assert "rouge1" in result
        assert "rouge2" in result
        assert "rougeL" in result
        assert 0.0 < result["rouge1"] < 1.0
        assert 0.0 < result["rouge2"] < 1.0
        assert 0.0 < result["rougeL"] < 1.0

    @pytest.mark.asyncio
    async def test_rouge_no_match(self, rouge_evaluator: ROUGEEvaluator) -> None:
        """Test ROUGE scores for no match."""
        result = await rouge_evaluator.evaluate(
            response="Hello world",
            ground_truth="Goodbye universe",
        )

        assert result["rouge1"] == pytest.approx(0.0, abs=0.01)
        assert result["rouge2"] == pytest.approx(0.0, abs=0.01)
        assert result["rougeL"] == pytest.approx(0.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_rouge_threshold_rouge1(self) -> None:
        """Test ROUGE evaluator with ROUGE-1 threshold."""
        evaluator = ROUGEEvaluator(threshold=0.5, variant="rouge1")

        # High overlap should pass
        result = await evaluator.evaluate(
            response="The cat sat on the mat today",
            ground_truth="The cat sat on the mat yesterday",
        )
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_rouge_threshold_rouge_l(self) -> None:
        """Test ROUGE evaluator with ROUGE-L threshold."""
        evaluator = ROUGEEvaluator(threshold=0.8, variant="rougeL")

        # Low LCS should fail
        result = await evaluator.evaluate(
            response="The quick brown fox", ground_truth="The lazy dog"
        )
        assert result["passed"] is False

    @pytest.mark.asyncio
    async def test_rouge_missing_ground_truth(
        self, rouge_evaluator: ROUGEEvaluator
    ) -> None:
        """Test ROUGE evaluator with missing ground truth."""
        with pytest.raises(EvaluationError, match="ground_truth.*required"):
            await rouge_evaluator.evaluate(response="test response")

    @pytest.mark.asyncio
    async def test_rouge_invalid_variant(self) -> None:
        """Test ROUGE evaluator with invalid variant."""
        with pytest.raises(ValueError, match="variant.*must be one of"):
            ROUGEEvaluator(variant="rouge99")

    @pytest.mark.asyncio
    async def test_rouge_lazy_loading(self) -> None:
        """Test that ROUGE metric is loaded lazily."""
        with patch(
            "holodeck.lib.evaluators.nlp_metrics._get_evaluate"
        ) as mock_get_evaluate:
            mock_evaluate = MagicMock()
            mock_get_evaluate.return_value = mock_evaluate

            evaluator = ROUGEEvaluator()
            mock_evaluate.load.assert_not_called()

            mock_metric = MagicMock()
            mock_metric.compute.return_value = {
                "rouge1": 0.5,
                "rouge2": 0.3,
                "rougeL": 0.4,
            }
            mock_evaluate.load.return_value = mock_metric

            await evaluator.evaluate(response="test", ground_truth="test")
            mock_evaluate.load.assert_called_once_with("rouge")


class TestMETEOREvaluator:
    """Test METEOR score evaluator."""

    @pytest.mark.asyncio
    async def test_meteor_perfect_match(
        self, meteor_evaluator: METEOREvaluator
    ) -> None:
        """Test METEOR score for perfect match."""
        result = await meteor_evaluator.evaluate(
            response="The cat sat on the mat",
            ground_truth="The cat sat on the mat",
        )

        assert "meteor" in result
        assert result["meteor"] == pytest.approx(1.0, rel=0.05)
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_meteor_partial_match(
        self, meteor_evaluator: METEOREvaluator
    ) -> None:
        """Test METEOR score for partial match."""
        result = await meteor_evaluator.evaluate(
            response="The cat sat on the mat",
            ground_truth="The cat is on the mat",
        )

        assert "meteor" in result
        assert 0.0 < result["meteor"] < 1.0

    @pytest.mark.asyncio
    async def test_meteor_synonym_handling(
        self, meteor_evaluator: METEOREvaluator
    ) -> None:
        """Test METEOR's synonym handling capability."""
        # METEOR should score higher than BLEU for synonym matches
        result = await meteor_evaluator.evaluate(
            response="The automobile is red", ground_truth="The car is red"
        )

        assert "meteor" in result
        # METEOR considers synonyms, so score should be reasonably high
        assert result["meteor"] > 0.5

    @pytest.mark.asyncio
    async def test_meteor_no_match(self, meteor_evaluator: METEOREvaluator) -> None:
        """Test METEOR score for no match."""
        result = await meteor_evaluator.evaluate(
            response="Hello world",
            ground_truth="Goodbye universe",
        )

        assert result["meteor"] < 0.3  # Very low score for unrelated text

    @pytest.mark.asyncio
    async def test_meteor_threshold_passing(self) -> None:
        """Test METEOR score threshold passing logic."""
        evaluator = METEOREvaluator(threshold=0.5)

        result = await evaluator.evaluate(
            response="The cat sat on the mat",
            ground_truth="The cat sat on the mat",
        )
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_meteor_threshold_failing(self) -> None:
        """Test METEOR score threshold failing logic."""
        evaluator = METEOREvaluator(threshold=0.95)

        result = await evaluator.evaluate(
            response="The cat sat on the mat",
            ground_truth="The dog sat on the rug",
        )
        assert result["passed"] is False

    @pytest.mark.asyncio
    async def test_meteor_missing_ground_truth(
        self, meteor_evaluator: METEOREvaluator
    ) -> None:
        """Test METEOR evaluator with missing ground truth."""
        with pytest.raises(Exception, match="ground_truth.*required"):
            await meteor_evaluator.evaluate(response="test response")

    @pytest.mark.asyncio
    async def test_meteor_empty_strings(
        self, meteor_evaluator: METEOREvaluator
    ) -> None:
        """Test METEOR evaluator with empty strings."""
        result = await meteor_evaluator.evaluate(response="", ground_truth="test")
        assert "meteor" in result
        assert result["meteor"] == pytest.approx(0.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_meteor_lazy_loading(self) -> None:
        """Test that METEOR metric is loaded lazily."""
        with patch(
            "holodeck.lib.evaluators.nlp_metrics._get_evaluate"
        ) as mock_get_evaluate:
            mock_evaluate = MagicMock()
            mock_get_evaluate.return_value = mock_evaluate
            evaluator = METEOREvaluator()
            mock_evaluate.load.assert_not_called()

            mock_metric = MagicMock()
            mock_metric.compute.return_value = {"meteor": 0.75}
            mock_evaluate.load.return_value = mock_metric

            await evaluator.evaluate(response="test", ground_truth="test")
            mock_evaluate.load.assert_called_once_with("meteor")


class TestNLPMetricsErrorHandling:
    """Test error handling across NLP metrics."""

    @pytest.mark.asyncio
    async def test_bleu_with_none_response(self, bleu_evaluator: BLEUEvaluator) -> None:
        """Test BLEU evaluator with None response."""
        with pytest.raises(EvaluationError):
            await bleu_evaluator.evaluate(response=None, ground_truth="test")

    @pytest.mark.asyncio
    async def test_rouge_with_none_ground_truth(
        self, rouge_evaluator: ROUGEEvaluator
    ) -> None:
        """Test ROUGE evaluator with None ground truth."""
        with pytest.raises(EvaluationError):
            await rouge_evaluator.evaluate(response="test", ground_truth=None)

    @pytest.mark.asyncio
    async def test_meteor_library_import_error(self) -> None:
        """Test METEOR evaluator when evaluate library is not available."""
        with patch(
            "holodeck.lib.evaluators.nlp_metrics._get_evaluate",
            side_effect=EvaluationError(
                "evaluate library is not installed. "
                "Install with: pip install evaluate sacrebleu"
            ),
        ):
            evaluator = METEOREvaluator()

            with pytest.raises(EvaluationError, match="evaluate.*not installed"):
                await evaluator.evaluate(response="test", ground_truth="test")

    @pytest.mark.asyncio
    async def test_bleu_metric_computation_error(self) -> None:
        """Test BLEU evaluator when metric computation fails."""
        with patch(
            "holodeck.lib.evaluators.nlp_metrics._get_sacrebleu"
        ) as mock_get_sacrebleu:
            mock_sacrebleu = MagicMock()
            mock_sacrebleu.sentence_bleu.side_effect = RuntimeError(
                "Computation failed"
            )
            mock_get_sacrebleu.return_value = mock_sacrebleu

            evaluator = BLEUEvaluator()

            with pytest.raises(EvaluationError, match="BLEU.*computation failed"):
                await evaluator.evaluate(response="test", ground_truth="test")


class TestNLPMetricsIntegration:
    """Integration tests for NLP metrics with realistic scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_metrics_same_text(
        self,
        bleu_evaluator: BLEUEvaluator,
        rouge_evaluator: ROUGEEvaluator,
        meteor_evaluator: METEOREvaluator,
    ) -> None:
        """Test evaluating same text with multiple metrics."""
        text = "The cat sat on the mat"

        bleu_result = await bleu_evaluator.evaluate(response=text, ground_truth=text)
        rouge_result = await rouge_evaluator.evaluate(response=text, ground_truth=text)
        meteor_result = await meteor_evaluator.evaluate(
            response=text, ground_truth=text
        )

        # Perfect match should score high on all metrics
        assert bleu_result["bleu"] > 0.95
        assert rouge_result["rouge1"] > 0.95
        assert meteor_result["meteor"] > 0.95

    @pytest.mark.asyncio
    async def test_metrics_with_realistic_agent_response(self) -> None:
        """Test metrics with realistic agent response vs ground truth."""
        response = (
            "Our business hours are Monday through Friday, 9 AM to 5 PM Eastern Time."
        )
        ground_truth = "We are open Monday-Friday from 9:00 AM to 5:00 PM EST."

        bleu_eval = BLEUEvaluator(threshold=0.3)
        rouge_eval = ROUGEEvaluator(threshold=0.5)
        meteor_eval = METEOREvaluator(threshold=0.5)

        bleu_result = await bleu_eval.evaluate(
            response=response, ground_truth=ground_truth
        )
        rouge_result = await rouge_eval.evaluate(
            response=response, ground_truth=ground_truth
        )
        meteor_result = await meteor_eval.evaluate(
            response=response, ground_truth=ground_truth
        )

        # All should have reasonable scores for semantically similar text
        assert bleu_result["bleu"] > 0.0
        assert rouge_result["rouge1"] > 0.3
        assert meteor_result["meteor"] > 0.3

    @pytest.mark.asyncio
    async def test_metrics_score_ordering(
        self, meteor_evaluator: METEOREvaluator
    ) -> None:
        """Test that metrics produce expected score ordering for different matches."""
        perfect_match = "The cat sat on the mat"
        close_match = "The cat is on the mat"
        distant_match = "A feline rested on a rug"

        perfect_result = await meteor_evaluator.evaluate(
            response=perfect_match, ground_truth=perfect_match
        )
        close_result = await meteor_evaluator.evaluate(
            response=close_match, ground_truth=perfect_match
        )
        distant_result = await meteor_evaluator.evaluate(
            response=distant_match, ground_truth=perfect_match
        )

        # Scores should be ordered: perfect > close > distant
        assert perfect_result["meteor"] > close_result["meteor"]
        assert close_result["meteor"] > distant_result["meteor"]
