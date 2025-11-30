"""Unit tests for TestExecutor GEval integration.

Tests cover:
- _build_deepeval_config() helper method for various providers
- _create_evaluators() handling GEvalMetric instances
- End-to-end execution with mocked GEvalEvaluator
- Mixed metrics (standard + GEval) in same configuration
"""

from unittest.mock import AsyncMock, Mock

import pytest

from holodeck.config.loader import ConfigLoader
from holodeck.lib.evaluators.deepeval.config import DeepEvalModelConfig
from holodeck.lib.file_processor import FileProcessor
from holodeck.lib.test_runner.agent_factory import AgentFactory, AgentThreadRun
from holodeck.lib.test_runner.executor import TestExecutor
from holodeck.models.agent import Agent, Instructions
from holodeck.models.config import ExecutionConfig
from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric, GEvalMetric
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.test_case import TestCaseModel


class TestBuildDeepEvalConfig:
    """Tests for _build_deepeval_config() helper method."""

    def _create_executor_with_no_evaluations(self) -> TestExecutor:
        """Create a TestExecutor instance for testing helper methods."""
        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[],
            evaluations=None,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig()

        return TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
        )

    def test_build_deepeval_config_none(self):
        """_build_deepeval_config returns None when input is None."""
        executor = self._create_executor_with_no_evaluations()
        result = executor._build_deepeval_config(None)
        assert result is None

    def test_build_deepeval_config_openai(self):
        """_build_deepeval_config correctly converts OpenAI provider."""
        executor = self._create_executor_with_no_evaluations()

        llm_provider = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-4o",
            api_key="sk-test-key",
        )

        result = executor._build_deepeval_config(llm_provider)

        assert result is not None
        assert isinstance(result, DeepEvalModelConfig)
        assert result.provider == ProviderEnum.OPENAI
        assert result.model_name == "gpt-4o"
        assert result.api_key == "sk-test-key"
        assert result.temperature == 0.0  # Deterministic for evaluation

    def test_build_deepeval_config_anthropic(self):
        """_build_deepeval_config correctly converts Anthropic provider."""
        executor = self._create_executor_with_no_evaluations()

        llm_provider = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-3-5-sonnet-latest",
            api_key="anthropic-test-key",
        )

        result = executor._build_deepeval_config(llm_provider)

        assert result is not None
        assert result.provider == ProviderEnum.ANTHROPIC
        assert result.model_name == "claude-3-5-sonnet-latest"
        assert result.api_key == "anthropic-test-key"

    def test_build_deepeval_config_azure_openai(self):
        """_build_deepeval_config correctly converts Azure OpenAI provider."""
        executor = self._create_executor_with_no_evaluations()

        llm_provider = LLMProvider(
            provider=ProviderEnum.AZURE_OPENAI,
            name="gpt-4",
            api_key="azure-test-key",
            endpoint="https://test.openai.azure.com/",
        )

        result = executor._build_deepeval_config(llm_provider)

        assert result is not None
        assert result.provider == ProviderEnum.AZURE_OPENAI
        assert result.model_name == "gpt-4"
        assert result.api_key == "azure-test-key"
        assert result.endpoint == "https://test.openai.azure.com/"

    def test_build_deepeval_config_ollama(self):
        """_build_deepeval_config correctly converts Ollama provider."""
        executor = self._create_executor_with_no_evaluations()

        llm_provider = LLMProvider(
            provider=ProviderEnum.OLLAMA,
            name="llama3:8b",
            endpoint="http://localhost:11434",
        )

        result = executor._build_deepeval_config(llm_provider)

        assert result is not None
        assert result.provider == ProviderEnum.OLLAMA
        assert result.model_name == "llama3:8b"
        assert result.endpoint == "http://localhost:11434"
        assert result.api_key is None  # Ollama doesn't need API key


class TestCreateEvaluatorsGEval:
    """Tests for _create_evaluators() with GEvalMetric.

    Note: These tests mock the GEvalEvaluator creation to avoid
    actually initializing the DeepEval SDK which requires API keys.
    """

    def test_create_evaluators_geval_metric(self):
        """_create_evaluators creates GEvalEvaluator for GEvalMetric."""
        from unittest.mock import patch

        geval_metric = GEvalMetric(
            name="Professionalism",
            criteria="Evaluate professional language",
            evaluation_params=["input", "actual_output"],
            threshold=0.7,
        )

        eval_config = EvaluationConfig(
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                api_key="test-key",
            ),
            metrics=[geval_metric],
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[],
            evaluations=eval_config,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig()

        # Mock GEvalEvaluator to avoid DeepEval SDK initialization
        with patch(
            "holodeck.lib.test_runner.executor.GEvalEvaluator"
        ) as mock_geval_cls:
            mock_evaluator = Mock()
            mock_evaluator.name = "Professionalism"
            mock_geval_cls.return_value = mock_evaluator

            executor = TestExecutor(
                agent_config_path="test.yaml",
                config_loader=mock_loader,
            )

            # Verify GEvalEvaluator was created with correct name as key
            assert "Professionalism" in executor.evaluators

            # Verify constructor was called with expected arguments
            mock_geval_cls.assert_called_once()
            call_kwargs = mock_geval_cls.call_args.kwargs
            assert call_kwargs["name"] == "Professionalism"
            assert call_kwargs["criteria"] == "Evaluate professional language"
            assert call_kwargs["evaluation_params"] == ["input", "actual_output"]
            assert call_kwargs["threshold"] == 0.7

    def test_create_evaluators_mixed_metrics(self):
        """_create_evaluators handles both standard and GEval metrics."""
        from unittest.mock import patch

        eval_config = EvaluationConfig(
            model=LLMProvider(
                provider=ProviderEnum.AZURE_OPENAI,
                name="gpt-4",
                api_key="test-key",
                endpoint="https://test.openai.azure.com/",
            ),
            metrics=[
                EvaluationMetric(metric="bleu", threshold=0.6),
                GEvalMetric(
                    name="Helpfulness",
                    criteria="Is the response helpful?",
                    threshold=0.7,
                ),
                EvaluationMetric(metric="rouge", threshold=0.65),
                GEvalMetric(
                    name="Accuracy",
                    criteria="Is the response accurate?",
                    strict_mode=True,
                ),
            ],
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[],
            evaluations=eval_config,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig()

        # Mock GEvalEvaluator to avoid DeepEval SDK initialization
        with patch(
            "holodeck.lib.test_runner.executor.GEvalEvaluator"
        ) as mock_geval_cls:
            mock_evaluator1 = Mock()
            mock_evaluator1.name = "Helpfulness"
            mock_evaluator2 = Mock()
            mock_evaluator2.name = "Accuracy"
            mock_geval_cls.side_effect = [mock_evaluator1, mock_evaluator2]

            executor = TestExecutor(
                agent_config_path="test.yaml",
                config_loader=mock_loader,
            )

            # Verify all evaluators were created
            assert "bleu" in executor.evaluators
            assert "Helpfulness" in executor.evaluators
            assert "rouge" in executor.evaluators
            assert "Accuracy" in executor.evaluators

    def test_create_evaluators_geval_with_per_metric_model(self):
        """GEvalMetric uses per-metric model override when specified."""
        from unittest.mock import patch

        geval_metric = GEvalMetric(
            name="CustomMetric",
            criteria="Custom evaluation criteria",
            model=LLMProvider(
                provider=ProviderEnum.ANTHROPIC,
                name="claude-3-5-sonnet-latest",
                api_key="anthropic-key",
            ),
        )

        eval_config = EvaluationConfig(
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                api_key="openai-key",
            ),
            metrics=[geval_metric],
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[],
            evaluations=eval_config,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig()

        # Mock GEvalEvaluator to avoid DeepEval SDK initialization
        with patch(
            "holodeck.lib.test_runner.executor.GEvalEvaluator"
        ) as mock_geval_cls:
            mock_evaluator = Mock()
            mock_evaluator.name = "CustomMetric"
            mock_geval_cls.return_value = mock_evaluator

            executor = TestExecutor(
                agent_config_path="test.yaml",
                config_loader=mock_loader,
            )

            # Verify evaluator was created with per-metric model
            assert "CustomMetric" in executor.evaluators

            # Verify per-metric model was passed (Anthropic, not OpenAI)
            call_kwargs = mock_geval_cls.call_args.kwargs
            model_config = call_kwargs["model_config"]
            assert model_config.provider == ProviderEnum.ANTHROPIC
            assert model_config.model_name == "claude-3-5-sonnet-latest"

    def test_create_evaluators_geval_no_model(self):
        """GEvalMetric without model uses default (None leads to Ollama default)."""
        from unittest.mock import patch

        geval_metric = GEvalMetric(
            name="NoModelMetric",
            criteria="Test criteria",
        )

        eval_config = EvaluationConfig(
            model=None,  # No default model
            metrics=[geval_metric],
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[],
            evaluations=eval_config,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig()

        # Mock GEvalEvaluator to avoid DeepEval SDK initialization
        with patch(
            "holodeck.lib.test_runner.executor.GEvalEvaluator"
        ) as mock_geval_cls:
            mock_evaluator = Mock()
            mock_evaluator.name = "NoModelMetric"
            mock_geval_cls.return_value = mock_evaluator

            executor = TestExecutor(
                agent_config_path="test.yaml",
                config_loader=mock_loader,
            )

            # Evaluator should still be created (will use DeepEval's default Ollama)
            assert "NoModelMetric" in executor.evaluators

            # Verify model_config is None (will use DeepEval defaults)
            call_kwargs = mock_geval_cls.call_args.kwargs
            assert call_kwargs["model_config"] is None


class TestGEvalExecutionFlow:
    """Tests for end-to-end execution with GEval metrics."""

    @pytest.mark.asyncio
    async def test_geval_metric_in_test_execution(self):
        """Test execution with GEval metric invokes evaluator correctly."""
        from holodeck.lib.evaluators.param_spec import EvalParam, ParamSpec

        test_case = TestCaseModel(
            name="test_geval",
            input="What is machine learning?",
            expected_tools=None,
            ground_truth="Machine learning is a subset of AI",
            files=None,
            evaluations=None,
        )

        geval_metric = GEvalMetric(
            name="Comprehensiveness",
            criteria="Evaluate if the response fully addresses the question",
            threshold=0.7,
        )

        eval_config = EvaluationConfig(
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                api_key="test-key",
            ),
            metrics=[geval_metric],
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[test_case],
            evaluations=eval_config,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        mock_file_processor = Mock(spec=FileProcessor)

        # Mock agent response
        mock_chat_history = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "Machine learning is a type of artificial intelligence."
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.tool_results = []
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

        # Mock GEval evaluator to return a score
        # Must use Mock() with proper get_param_spec, not AsyncMock
        mock_geval_evaluator = Mock()
        mock_geval_evaluator.name = "Comprehensiveness"
        mock_geval_evaluator.get_param_spec = Mock(
            return_value=ParamSpec(
                required=frozenset({EvalParam.RESPONSE, EvalParam.QUERY}),
            )
        )
        mock_geval_evaluator.evaluate = AsyncMock(
            return_value={
                "score": 0.85,
                "Comprehensiveness": 0.85,
            }
        )

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            evaluators={"Comprehensiveness": mock_geval_evaluator},
        )

        report = await executor.execute_tests()

        # Verify report
        assert report.summary.total_tests == 1
        assert len(report.results[0].metric_results) == 1

        # Verify the metric result
        metric_result = report.results[0].metric_results[0]
        assert metric_result.metric_name == "Comprehensiveness"
        assert metric_result.score == 0.85
        assert metric_result.passed is True  # 0.85 >= 0.7 threshold

        # Verify evaluator was called with correct arguments
        mock_geval_evaluator.evaluate.assert_called_once()
        call_kwargs = mock_geval_evaluator.evaluate.call_args.kwargs
        assert "response" in call_kwargs
        assert "query" in call_kwargs
        assert call_kwargs["query"] == "What is machine learning?"

    @pytest.mark.asyncio
    async def test_mixed_metrics_execution(self):
        """Test execution with both standard and GEval metrics."""
        from holodeck.lib.evaluators.param_spec import EvalParam, ParamSpec

        test_case = TestCaseModel(
            name="test_mixed",
            input="Translate hello to Spanish",
            expected_tools=None,
            ground_truth="hola",
            files=None,
            evaluations=None,
        )

        eval_config = EvaluationConfig(
            model=None,
            metrics=[
                EvaluationMetric(metric="bleu", threshold=0.5),
                GEvalMetric(
                    name="Correctness",
                    criteria="Is the translation correct?",
                    threshold=0.8,
                ),
            ],
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[test_case],
            evaluations=eval_config,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        mock_file_processor = Mock(spec=FileProcessor)

        mock_chat_history = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "hola"
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.tool_results = []
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

        # Mock evaluators with proper get_param_spec
        mock_bleu = Mock()
        mock_bleu.name = "bleu"
        mock_bleu.get_param_spec = Mock(
            return_value=ParamSpec(
                required=frozenset({EvalParam.RESPONSE, EvalParam.GROUND_TRUTH}),
            )
        )
        mock_bleu.evaluate = AsyncMock(return_value={"bleu": 1.0})

        mock_geval = Mock()
        mock_geval.name = "Correctness"
        mock_geval.get_param_spec = Mock(
            return_value=ParamSpec(
                required=frozenset({EvalParam.RESPONSE, EvalParam.QUERY}),
            )
        )
        mock_geval.evaluate = AsyncMock(return_value={"score": 0.9})

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            evaluators={
                "bleu": mock_bleu,
                "Correctness": mock_geval,
            },
        )

        report = await executor.execute_tests()

        # Verify both metrics were evaluated
        assert len(report.results[0].metric_results) == 2
        metric_names = {m.metric_name for m in report.results[0].metric_results}
        assert metric_names == {"bleu", "Correctness"}

    @pytest.mark.asyncio
    async def test_geval_per_test_override(self):
        """Test that per-test GEval metrics override global metrics."""
        from holodeck.lib.evaluators.param_spec import EvalParam, ParamSpec

        # Per-test metric
        per_test_metric = GEvalMetric(
            name="SpecificCriteria",
            criteria="Specific criteria for this test only",
            threshold=0.9,
        )

        test_case = TestCaseModel(
            name="test_per_test",
            input="Query",
            expected_tools=None,
            ground_truth="Expected",
            files=None,
            evaluations=[per_test_metric],  # Per-test override
        )

        # Global metrics (should be overridden)
        eval_config = EvaluationConfig(
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                api_key="test-key",
            ),
            metrics=[
                EvaluationMetric(metric="bleu", threshold=0.5),
                GEvalMetric(
                    name="GlobalMetric",
                    criteria="Global criteria",
                    threshold=0.7,
                ),
            ],
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[test_case],
            evaluations=eval_config,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        mock_file_processor = Mock(spec=FileProcessor)

        mock_chat_history = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "Response"
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.tool_results = []
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

        # Mock all possible evaluators with proper get_param_spec
        mock_bleu = Mock()
        mock_bleu.name = "bleu"
        mock_bleu.get_param_spec = Mock(
            return_value=ParamSpec(
                required=frozenset({EvalParam.RESPONSE, EvalParam.GROUND_TRUTH}),
            )
        )
        mock_bleu.evaluate = AsyncMock(return_value={"bleu": 0.8})

        mock_global = Mock()
        mock_global.name = "GlobalMetric"
        mock_global.get_param_spec = Mock(
            return_value=ParamSpec(
                required=frozenset({EvalParam.RESPONSE, EvalParam.QUERY}),
            )
        )
        mock_global.evaluate = AsyncMock(return_value={"score": 0.85})

        mock_specific = Mock()
        mock_specific.name = "SpecificCriteria"
        mock_specific.get_param_spec = Mock(
            return_value=ParamSpec(
                required=frozenset({EvalParam.RESPONSE, EvalParam.QUERY}),
            )
        )
        mock_specific.evaluate = AsyncMock(return_value={"score": 0.95})

        mock_evaluators = {
            "bleu": mock_bleu,
            "GlobalMetric": mock_global,
            "SpecificCriteria": mock_specific,
        }

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            evaluators=mock_evaluators,
        )

        report = await executor.execute_tests()

        # Verify only the per-test metric was evaluated
        assert len(report.results[0].metric_results) == 1
        assert report.results[0].metric_results[0].metric_name == "SpecificCriteria"
        assert report.results[0].metric_results[0].score == 0.95


class TestGEvalEvaluatorIntegration:
    """Tests for GEvalEvaluator instantiation verification."""

    def test_geval_evaluator_created_with_correct_params(self):
        """Verify GEvalEvaluator is instantiated with correct parameters."""
        from unittest.mock import patch

        geval_metric = GEvalMetric(
            name="DetailedMetric",
            criteria="Detailed evaluation criteria",
            evaluation_steps=["Step 1", "Step 2", "Step 3"],
            evaluation_params=["input", "actual_output", "expected_output"],
            strict_mode=True,
            threshold=0.8,
        )

        eval_config = EvaluationConfig(
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                api_key="test-key",
            ),
            metrics=[geval_metric],
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[],
            evaluations=eval_config,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig()

        # Mock GEvalEvaluator to capture the initialization parameters
        with patch(
            "holodeck.lib.test_runner.executor.GEvalEvaluator"
        ) as mock_geval_cls:
            mock_evaluator = Mock()
            mock_evaluator.name = "DetailedMetric"
            mock_geval_cls.return_value = mock_evaluator

            executor = TestExecutor(
                agent_config_path="test.yaml",
                config_loader=mock_loader,
            )

            # Verify the evaluator was created
            assert "DetailedMetric" in executor.evaluators

            # Verify GEvalEvaluator was called with all expected parameters
            mock_geval_cls.assert_called_once()
            call_kwargs = mock_geval_cls.call_args.kwargs

            assert call_kwargs["name"] == "DetailedMetric"
            assert call_kwargs["criteria"] == "Detailed evaluation criteria"
            assert call_kwargs["evaluation_steps"] == ["Step 1", "Step 2", "Step 3"]
            assert call_kwargs["evaluation_params"] == [
                "input",
                "actual_output",
                "expected_output",
            ]
            assert call_kwargs["strict_mode"] is True
            assert call_kwargs["threshold"] == 0.8
            assert call_kwargs["model_config"] is not None
            assert call_kwargs["model_config"].provider == ProviderEnum.OPENAI
