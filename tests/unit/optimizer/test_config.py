"""Unit tests for OptimizerConfig parsing and validation (T1)."""

import pytest
from pydantic import ValidationError

from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.optimizer.config import OptimizerConfig


class TestOptimizerConfigParsing:
    """OptimizerConfig parses the evaluations.optimizer block."""

    def test_minimal_config_parses_with_defaults(self) -> None:
        cfg = OptimizerConfig.model_validate(
            {
                "loss": {"groundedness": 1.0},
                "axes": {
                    "numeric": [
                        {
                            "path": "model.temperature",
                            "type": "float",
                            "range": [0.0, 1.0],
                        }
                    ],
                    "textual": [{"path": "instructions.inline", "max_chars": 4000}],
                },
            }
        )

        assert cfg.loss == {"groundedness": 1.0}
        assert cfg.axes.numeric[0].path == "model.temperature"
        assert cfg.axes.numeric[0].type == "float"
        assert cfg.axes.textual[0].max_chars == 4000
        # Defaults are populated.
        assert cfg.max_cycles >= 1
        assert cfg.numeric_phase.max_trials >= 1
        assert cfg.numeric_phase.patience >= 1
        assert cfg.textual_phase.max_trials >= 1
        assert cfg.min_delta >= 0.0
        assert isinstance(cfg.seed, int)

    def test_categorical_axis_parses(self) -> None:
        cfg = OptimizerConfig.model_validate(
            {
                "loss": {"relevance": 1.0},
                "axes": {
                    "numeric": [
                        {
                            "path": "tools[name=kb].top_k",
                            "type": "categorical",
                            "range": [3, 5, 8],
                        }
                    ]
                },
            }
        )
        assert cfg.axes.numeric[0].range == [3, 5, 8]


class TestOptimizerConfigValidation:
    """Invalid optimizer config raises pydantic ValidationError."""

    def test_empty_loss_weights_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OptimizerConfig.model_validate({"loss": {}})

    def test_non_positive_loss_weight_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OptimizerConfig.model_validate({"loss": {"groundedness": 0.0}})

    def test_bad_axis_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OptimizerConfig.model_validate(
                {
                    "loss": {"groundedness": 1.0},
                    "axes": {
                        "numeric": [
                            {
                                "path": "model.temperature",
                                "type": "bogus",
                                "range": [0, 1],
                            }
                        ]
                    },
                }
            )

    def test_float_axis_requires_two_bounds(self) -> None:
        with pytest.raises(ValidationError):
            OptimizerConfig.model_validate(
                {
                    "loss": {"groundedness": 1.0},
                    "axes": {
                        "numeric": [
                            {
                                "path": "model.temperature",
                                "type": "float",
                                "range": [0.0],
                            }
                        ]
                    },
                }
            )

    def test_float_axis_low_must_be_below_high(self) -> None:
        with pytest.raises(ValidationError):
            OptimizerConfig.model_validate(
                {
                    "loss": {"groundedness": 1.0},
                    "axes": {
                        "numeric": [
                            {
                                "path": "model.temperature",
                                "type": "float",
                                "range": [1.0, 0.0],
                            }
                        ]
                    },
                }
            )


class TestEvaluationConfigIntegration:
    """An agent's evaluations.optimizer block parses into OptimizerConfig."""

    def test_agent_evaluations_carries_optimizer(self) -> None:
        agent = Agent(
            name="opt-agent",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o-mini"),
            instructions=Instructions(inline="You are helpful."),
            evaluations={
                "metrics": [{"type": "standard", "metric": "groundedness"}],
                "optimizer": {
                    "loss": {"groundedness": 1.0},
                    "axes": {
                        "numeric": [
                            {
                                "path": "model.temperature",
                                "type": "float",
                                "range": [0.0, 1.0],
                            }
                        ]
                    },
                },
            },
        )
        assert agent.evaluations is not None
        assert agent.evaluations.optimizer is not None
        assert isinstance(agent.evaluations.optimizer, OptimizerConfig)
        assert agent.evaluations.optimizer.loss == {"groundedness": 1.0}
