"""US3: Snapshot fidelity tests for ``EvalRunMetadata.agent_config``.

These tests enforce the full round-trip contract described in
``specs/031-eval-runs-dashboard/data-model.md`` — every nested field in the
``Agent`` tree (model, embedding provider, tools of every type, claude block,
evaluations of every metric type, test_cases with multimodal files,
instructions) must survive ``model_dump_json()`` → ``model_validate_json()``
without loss.

Feature: 031-eval-runs-dashboard — User Story 3 (Tasks T201–T214).
"""

from __future__ import annotations

import json

import pytest

from holodeck.lib.eval_run.metadata import build_eval_run_metadata
from holodeck.lib.eval_run.redactor import REDACTED_PLACEHOLDER, redact
from holodeck.models.agent import Agent, Instructions
from holodeck.models.claude_config import (
    BashConfig,
    ClaudeConfig,
    ExtendedThinkingConfig,
    FileSystemConfig,
    PermissionMode,
    SubagentConfig,
)
from holodeck.models.eval_run import EvalRun, EvalRunMetadata, PromptVersion
from holodeck.models.evaluation import (
    EvaluationConfig,
    EvaluationMetric,
    GEvalMetric,
    RAGMetric,
    RAGMetricType,
)
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.test_case import FileInput, TestCaseModel
from holodeck.models.test_result import ReportSummary, TestReport
from holodeck.models.tool import (
    ChunkingStrategy,
    CommandType,
    DatabaseConfig,
    FunctionTool,
    HierarchicalDocumentToolConfig,
    MCPTool,
    PromptTool,
    SearchMode,
    TransportType,
    VectorstoreTool,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _make_prompt_version() -> PromptVersion:
    return PromptVersion(
        version="auto-00000000",
        source="inline",
        body_hash="0" * 64,
    )


def _make_report(agent_name: str) -> TestReport:
    return TestReport(
        agent_name=agent_name,
        agent_config_path="agent.yaml",
        results=[],
        summary=ReportSummary(
            total_tests=0,
            passed=0,
            failed=0,
            pass_rate=0.0,
            total_duration_ms=0,
        ),
        timestamp="2026-04-18T14:22:09.812Z",
        holodeck_version="0.1.0",
    )


def _wrap(agent: Agent) -> EvalRun:
    """Build an ``EvalRun`` around an ``Agent`` for round-trip assertions."""
    metadata = EvalRunMetadata(
        agent_config=agent,
        prompt_version=_make_prompt_version(),
        holodeck_version="0.1.0",
        cli_args=["test", "agent.yaml"],
        git_commit=None,
    )
    return EvalRun(report=_make_report(agent.name), metadata=metadata)


def _round_trip(agent: Agent) -> Agent:
    """Serialize an ``Agent`` through a full ``EvalRun`` and return the snapshot."""
    run = _wrap(agent)
    dumped = run.model_dump_json()
    rehydrated = EvalRun.model_validate_json(dumped)
    return rehydrated.metadata.agent_config


# --------------------------------------------------------------------------- #
# T201 — model block round-trip (AC1).                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestModelBlockRoundTrip:
    def test_model_fields_preserved(self) -> None:
        agent = Agent(
            name="t201-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                temperature=0.7,
                max_tokens=1024,
                top_p=0.95,
            ),
            instructions=Instructions(inline="You are helpful."),
        )
        snap = _round_trip(agent)
        assert snap.model.provider == ProviderEnum.OPENAI
        assert snap.model.name == "gpt-4o"
        assert snap.model.temperature == 0.7
        assert snap.model.max_tokens == 1024
        assert snap.model.top_p == 0.95


# --------------------------------------------------------------------------- #
# T202 — embedding_provider non-secret env-substituted values preserved.      #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestEmbeddingProviderRoundTrip:
    def test_azure_embedding_provider_fields_preserved(self) -> None:
        agent = Agent(
            name="t202-agent",
            model=LLMProvider(provider=ProviderEnum.ANTHROPIC, name="claude-sonnet-4"),
            instructions=Instructions(inline="You are helpful."),
            embedding_provider=LLMProvider(
                provider=ProviderEnum.AZURE_OPENAI,
                name="text-embedding-3-large",
                endpoint="https://my-azure.openai.azure.com",
                api_version="2024-10-21",
            ),
        )
        snap = _round_trip(agent)
        assert snap.embedding_provider is not None
        assert snap.embedding_provider.provider == ProviderEnum.AZURE_OPENAI
        assert snap.embedding_provider.name == "text-embedding-3-large"
        # Non-secret env-substituted values MUST be preserved verbatim.
        assert snap.embedding_provider.endpoint == "https://my-azure.openai.azure.com"
        assert snap.embedding_provider.api_version == "2024-10-21"


# --------------------------------------------------------------------------- #
# T203 — VectorstoreTool round-trip (AC3).                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestVectorstoreToolRoundTrip:
    def test_full_vectorstore_config_preserved(self) -> None:
        tool = VectorstoreTool(
            name="kb_search",
            description="Knowledge base",
            source="./docs",
            vector_field="body",
            meta_fields=["title", "url"],
            chunk_size=512,
            chunk_overlap=64,
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
            database=DatabaseConfig(provider="postgres"),
            top_k=7,
            min_similarity_score=0.25,
            id_field="id",
            field_separator="\n\n",
        )
        agent = Agent(
            name="t203-agent",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="x"),
            tools=[tool],
        )
        snap = _round_trip(agent)
        assert snap.tools is not None and len(snap.tools) == 1
        rt = snap.tools[0]
        assert isinstance(rt, VectorstoreTool)
        assert rt.name == "kb_search"
        assert rt.type == "vectorstore"
        assert rt.source == "./docs"
        assert rt.vector_field == "body"
        assert rt.meta_fields == ["title", "url"]
        assert rt.chunk_size == 512
        assert rt.chunk_overlap == 64
        assert rt.embedding_model == "text-embedding-3-small"
        assert rt.embedding_dimensions == 1536
        assert isinstance(rt.database, DatabaseConfig)
        assert rt.database.provider == "postgres"
        assert rt.top_k == 7
        assert rt.min_similarity_score == 0.25
        assert rt.id_field == "id"
        assert rt.field_separator == "\n\n"


# --------------------------------------------------------------------------- #
# T204 — MCPTool round-trip (AC3).                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestMCPToolRoundTrip:
    def test_stdio_mcp_tool_preserved(self) -> None:
        tool = MCPTool(
            name="search_mcp",
            description="An MCP search server",
            transport=TransportType.STDIO,
            command=CommandType.NPX,
            args=["-y", "@modelcontextprotocol/server-brave-search"],
            env={"BRAVE_API_KEY": "resolved-value"},
            load_tools=True,
            load_prompts=False,
            request_timeout=45,
            is_retrieval=True,
            registry_name="io.github.user/search",
        )
        agent = Agent(
            name="t204-agent",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="x"),
            tools=[tool],
        )
        snap = _round_trip(agent)
        rt = snap.tools[0]  # type: ignore[index]
        assert isinstance(rt, MCPTool)
        assert rt.type == "mcp"
        assert rt.transport == TransportType.STDIO
        assert rt.command == CommandType.NPX
        assert rt.args == ["-y", "@modelcontextprotocol/server-brave-search"]
        assert rt.env == {"BRAVE_API_KEY": "resolved-value"}
        assert rt.load_tools is True
        assert rt.load_prompts is False
        assert rt.request_timeout == 45
        assert rt.is_retrieval is True
        assert rt.registry_name == "io.github.user/search"

    def test_http_mcp_tool_preserved(self) -> None:
        tool = MCPTool(
            name="remote_mcp",
            description="A remote HTTP MCP server",
            transport=TransportType.HTTP,
            url="https://example.com/mcp",
            headers={"Authorization": "Bearer redacted-at-source"},
            timeout=30.0,
            terminate_on_close=True,
        )
        agent = Agent(
            name="t204b-agent",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="x"),
            tools=[tool],
        )
        snap = _round_trip(agent)
        rt = snap.tools[0]  # type: ignore[index]
        assert isinstance(rt, MCPTool)
        assert rt.transport == TransportType.HTTP
        assert rt.url == "https://example.com/mcp"
        assert rt.headers == {"Authorization": "Bearer redacted-at-source"}
        assert rt.timeout == 30.0
        assert rt.terminate_on_close is True


# --------------------------------------------------------------------------- #
# T205 — FunctionTool + PromptTool round-trip (AC3).                          #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestFunctionAndPromptToolRoundTrip:
    def test_function_tool_preserved(self) -> None:
        tool = FunctionTool(
            name="lookup_order",
            description="Look up an order",
            file="./tools.py",
            function="lookup_order",
            parameters={
                "order_id": {"type": "string", "description": "Order identifier"}
            },
        )
        agent = Agent(
            name="t205-agent",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="x"),
            tools=[tool],
        )
        snap = _round_trip(agent)
        rt = snap.tools[0]  # type: ignore[index]
        assert isinstance(rt, FunctionTool)
        assert rt.type == "function"
        assert rt.file == "./tools.py"
        assert rt.function == "lookup_order"
        assert rt.parameters == {
            "order_id": {"type": "string", "description": "Order identifier"}
        }

    def test_prompt_tool_preserved(self) -> None:
        tool = PromptTool(
            name="summarize",
            description="Summarize content",
            template="Summarize: {{ content }}",
            parameters={"content": {"type": "string"}},
        )
        agent = Agent(
            name="t205b-agent",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="x"),
            tools=[tool],
        )
        snap = _round_trip(agent)
        rt = snap.tools[0]  # type: ignore[index]
        assert isinstance(rt, PromptTool)
        assert rt.type == "prompt"
        assert rt.template == "Summarize: {{ content }}"
        assert rt.parameters == {"content": {"type": "string"}}


# --------------------------------------------------------------------------- #
# T206 — HierarchicalDocumentToolConfig round-trip (AC3).                     #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestHierarchicalDocumentToolRoundTrip:
    def test_hierarchical_document_tool_preserved(self) -> None:
        tool = HierarchicalDocumentToolConfig(
            name="legal_docs",
            description="Legal corpus",
            source="./legal/",
            chunking_strategy=ChunkingStrategy.STRUCTURE,
            max_chunk_tokens=1200,
            chunk_overlap=100,
            search_mode=SearchMode.HYBRID,
            top_k=20,
            semantic_weight=0.6,
            keyword_weight=0.3,
            exact_weight=0.1,
            contextual_embeddings=True,
            context_max_tokens=150,
            context_concurrency=8,
            extract_definitions=True,
            extract_cross_references=False,
        )
        agent = Agent(
            name="t206-agent",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="x"),
            tools=[tool],
        )
        snap = _round_trip(agent)
        rt = snap.tools[0]  # type: ignore[index]
        assert isinstance(rt, HierarchicalDocumentToolConfig)
        assert rt.type == "hierarchical_document"
        assert rt.chunking_strategy == ChunkingStrategy.STRUCTURE
        assert rt.max_chunk_tokens == 1200
        assert rt.chunk_overlap == 100
        assert rt.search_mode == SearchMode.HYBRID
        assert rt.top_k == 20
        assert rt.semantic_weight == 0.6
        assert rt.keyword_weight == 0.3
        assert rt.exact_weight == 0.1
        assert rt.contextual_embeddings is True
        assert rt.context_max_tokens == 150
        assert rt.context_concurrency == 8
        assert rt.extract_definitions is True
        assert rt.extract_cross_references is False


# --------------------------------------------------------------------------- #
# T207 — full claude block round-trip (AC4).                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestClaudeBlockRoundTrip:
    def test_full_claude_block_preserved(self) -> None:
        claude = ClaudeConfig(
            working_directory="/var/workspace",  # noqa: S108
            permission_mode=PermissionMode.acceptEdits,
            max_turns=25,
            max_concurrent_sessions=8,
            extended_thinking=ExtendedThinkingConfig(
                enabled=True, budget_tokens=50_000
            ),
            web_search=True,
            bash=BashConfig(
                enabled=True,
                excluded_commands=["rm", "sudo"],
                allow_unsafe=False,
            ),
            file_system=FileSystemConfig(read=True, write=True, edit=False),
            subagents=SubagentConfig(enabled=True, max_parallel=4),
            allowed_tools=["Read", "Write", "Bash"],
        )
        agent = Agent(
            name="t207-agent",
            model=LLMProvider(provider=ProviderEnum.ANTHROPIC, name="claude-sonnet-4"),
            instructions=Instructions(inline="x"),
            claude=claude,
        )
        snap = _round_trip(agent)
        assert snap.claude is not None
        assert snap.claude.working_directory == "/var/workspace"
        assert snap.claude.permission_mode == PermissionMode.acceptEdits
        assert snap.claude.max_turns == 25
        assert snap.claude.max_concurrent_sessions == 8
        assert snap.claude.extended_thinking is not None
        assert snap.claude.extended_thinking.enabled is True
        assert snap.claude.extended_thinking.budget_tokens == 50_000
        assert snap.claude.web_search is True
        assert snap.claude.bash is not None
        assert snap.claude.bash.enabled is True
        assert snap.claude.bash.excluded_commands == ["rm", "sudo"]
        assert snap.claude.bash.allow_unsafe is False
        assert snap.claude.file_system is not None
        assert snap.claude.file_system.read is True
        assert snap.claude.file_system.write is True
        assert snap.claude.file_system.edit is False
        assert snap.claude.subagents is not None
        assert snap.claude.subagents.enabled is True
        assert snap.claude.subagents.max_parallel == 4
        assert snap.claude.allowed_tools == ["Read", "Write", "Bash"]


# --------------------------------------------------------------------------- #
# T208 — evaluations block with every metric type round-trip.                 #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestEvaluationsRoundTrip:
    def test_every_metric_kind_plus_overrides_preserved(self) -> None:
        eval_cfg = EvaluationConfig(
            model=LLMProvider(
                provider=ProviderEnum.OPENAI, name="gpt-4o", temperature=0.0
            ),
            metrics=[
                EvaluationMetric(metric="bleu", threshold=0.4),
                GEvalMetric(
                    name="Helpfulness",
                    criteria="Is the response helpful?",
                    evaluation_params=["actual_output", "input"],
                    threshold=0.7,
                    model=LLMProvider(
                        provider=ProviderEnum.OPENAI,
                        name="gpt-4o-mini",
                        temperature=0.1,
                    ),
                ),
                RAGMetric(
                    metric_type=RAGMetricType.FAITHFULNESS,
                    threshold=0.8,
                    include_reason=True,
                ),
            ],
        )
        agent = Agent(
            name="t208-agent",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="x"),
            evaluations=eval_cfg,
        )
        snap = _round_trip(agent)
        assert snap.evaluations is not None
        assert snap.evaluations.model is not None
        assert snap.evaluations.model.name == "gpt-4o"
        metrics = snap.evaluations.metrics
        assert len(metrics) == 3
        assert isinstance(metrics[0], EvaluationMetric)
        assert metrics[0].metric == "bleu"
        assert metrics[0].threshold == 0.4
        assert isinstance(metrics[1], GEvalMetric)
        assert metrics[1].name == "Helpfulness"
        assert metrics[1].criteria == "Is the response helpful?"
        assert metrics[1].evaluation_params == ["actual_output", "input"]
        assert metrics[1].threshold == 0.7
        assert metrics[1].model is not None
        assert metrics[1].model.name == "gpt-4o-mini"
        assert metrics[1].model.temperature == 0.1
        assert isinstance(metrics[2], RAGMetric)
        assert metrics[2].metric_type == RAGMetricType.FAITHFULNESS
        assert metrics[2].threshold == 0.8
        assert metrics[2].include_reason is True


# --------------------------------------------------------------------------- #
# T209 — multimodal files: path-only, NO bytes/hash/text (FR-009a).           #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestMultimodalFilesPathOnly:
    def test_file_input_preserves_only_declared_fields(self) -> None:
        file_in = FileInput(
            path="./data/image.png",
            type="image",
            description="Product image",
        )
        excel = FileInput(
            path="./data/sheet.xlsx",
            type="excel",
            sheet="Summary",
            range="A1:E100",
        )
        ppt = FileInput(
            path="./data/deck.pptx",
            type="powerpoint",
            pages=[1, 2, 4],
        )
        tc = TestCaseModel(
            name="multimodal",
            input="Analyse the files",
            files=[file_in, excel, ppt],
        )
        agent = Agent(
            name="t209-agent",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="x"),
            test_cases=[tc],
        )

        # Serialize and inspect the raw JSON — NO bytes/hash/text fields must appear.
        run = _wrap(agent)
        raw = run.model_dump_json()
        payload = json.loads(raw)
        tc_json = payload["metadata"]["agent_config"]["test_cases"][0]
        assert len(tc_json["files"]) == 3

        allowed_keys = {
            "path",
            "url",
            "type",
            "description",
            "pages",
            "sheet",
            "range",
            "cache",
        }
        forbidden_keys = {"bytes", "content", "content_hash", "hash", "extracted_text"}
        for file_entry in tc_json["files"]:
            keys = set(file_entry.keys())
            leaked = keys & forbidden_keys
            assert (
                not leaked
            ), f"Forbidden multimodal fields leaked into snapshot: {leaked}"
            extra = keys - allowed_keys
            assert not extra, f"Unexpected multimodal keys: {extra}"

        snap = _round_trip(agent)
        assert snap.test_cases is not None
        assert snap.test_cases[0].files is not None
        rt_files = snap.test_cases[0].files
        assert rt_files[0].path == "./data/image.png"
        assert rt_files[0].type == "image"
        assert rt_files[0].description == "Product image"
        assert rt_files[1].sheet == "Summary"
        assert rt_files[1].range == "A1:E100"
        assert rt_files[2].pages == [1, 2, 4]


# --------------------------------------------------------------------------- #
# T212 — redactor preserves non-secret env-substituted values (AC2).          #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestRedactorNonSecretPreservation:
    def test_endpoint_not_masked_but_api_key_is(self) -> None:
        agent = Agent(
            name="t212-agent",
            model=LLMProvider(
                provider=ProviderEnum.AZURE_OPENAI,
                name="gpt-4o",
                endpoint="https://my-azure.openai.azure.com",
                api_key="sk-super-secret-real-value",  # type: ignore[arg-type]
                api_version="2024-10-21",
            ),
            instructions=Instructions(inline="x"),
        )
        redacted = redact(agent)
        # Non-secret fields preserved verbatim.
        assert redacted.model.endpoint == "https://my-azure.openai.azure.com"
        assert redacted.model.api_version == "2024-10-21"
        assert redacted.model.name == "gpt-4o"
        # Secret field must be masked.
        assert redacted.model.api_key is not None
        assert redacted.model.api_key.get_secret_value() == REDACTED_PLACEHOLDER


# --------------------------------------------------------------------------- #
# T213 — instructions.inline fully preserved.                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestInstructionsInlinePreserved:
    def test_full_inline_instruction_string_survives(self) -> None:
        body = (
            "You are a helpful assistant.\n\n"
            "Rules:\n"
            "1. Always cite sources.\n"
            "2. Never fabricate URLs.\n"
            "3. Stay professional.\n"
        )
        agent = Agent(
            name="t213-agent",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline=body),
        )
        snap = _round_trip(agent)
        assert snap.instructions.inline == body
        assert snap.instructions.file is None


# --------------------------------------------------------------------------- #
# T214 — instructions.file path preserved verbatim.                           #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestInstructionsFilePathPreserved:
    def test_file_path_string_roundtrips(self) -> None:
        agent = Agent(
            name="t214-agent",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(file="prompts/instructions.md"),
        )
        snap = _round_trip(agent)
        assert snap.instructions.file == "prompts/instructions.md"
        assert snap.instructions.inline is None


# --------------------------------------------------------------------------- #
# Snapshot isolation — redaction must not mutate the live Agent (T218/T219). #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestSnapshotDeepCopyIsolation:
    def test_build_eval_run_metadata_does_not_mutate_live_agent(self) -> None:
        """The live ``Agent`` instance must survive metadata assembly unchanged.

        If ``build_eval_run_metadata`` eventually performs redaction internally,
        it MUST operate on a deep copy; otherwise the running test executor
        would see its ``Agent.model.api_key`` replaced with ``"***"`` mid-run.
        """
        live = Agent(
            name="isolation-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                api_key="sk-live-secret",  # type: ignore[arg-type]
            ),
            instructions=Instructions(inline="x"),
        )
        live_api_key_id = id(live.model.api_key)

        metadata = build_eval_run_metadata(
            agent=live,
            prompt_version=_make_prompt_version(),
            argv=["test", "agent.yaml"],
        )

        # Live agent untouched: secret still recoverable, object identity intact.
        assert live.model.api_key is not None
        assert live.model.api_key.get_secret_value() == "sk-live-secret"
        assert id(live.model.api_key) == live_api_key_id

        # Snapshot embedded in metadata is a DIFFERENT object from the live one.
        assert metadata.agent_config is not live
        # And the snapshot's secret has been redacted (two-rule policy).
        snap_key = metadata.agent_config.model.api_key
        assert snap_key is not None
        assert snap_key.get_secret_value() == REDACTED_PLACEHOLDER


# --------------------------------------------------------------------------- #
# End-to-end round-trip equivalence with a maximal Agent.                     #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestFullyLoadedAgentRoundTrip:
    def _maximal_agent(self) -> Agent:
        return Agent(
            name="maximal",
            description="A maximal agent",
            author="jane@example.com",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                temperature=0.4,
                max_tokens=2048,
            ),
            instructions=Instructions(inline="You help."),
            tools=[
                VectorstoreTool(
                    name="vs",
                    description="vs",
                    source="./docs",
                    top_k=3,
                ),
                FunctionTool(
                    name="fn",
                    description="fn",
                    file="./t.py",
                    function="f",
                ),
                MCPTool(
                    name="mcp_tool",
                    description="mcp",
                    transport=TransportType.STDIO,
                    command=CommandType.NPX,
                    args=["-y", "pkg"],
                ),
                PromptTool(
                    name="p",
                    description="p",
                    template="t {{x}}",
                    parameters={"x": {"type": "string"}},
                ),
            ],
            evaluations=EvaluationConfig(
                metrics=[EvaluationMetric(metric="bleu", threshold=0.4)]
            ),
            test_cases=[
                TestCaseModel(
                    name="c",
                    input="hi",
                    files=[FileInput(path="./x.png", type="image")],
                )
            ],
        )

    def test_maximal_agent_roundtrip_equal(self) -> None:
        agent = self._maximal_agent()
        run = _wrap(agent)
        dumped = run.model_dump_json()
        rehydrated = EvalRun.model_validate_json(dumped)
        # Re-serialize the rehydrated run; the two dumps must be byte-equivalent.
        redumped = rehydrated.model_dump_json()
        assert dumped == redumped
