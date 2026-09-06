"""Tests for ``type: hosted`` tool entries (spec 035 FR-083 / D06 / D17)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from holodeck.models.agent import Agent
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import (
    COMPUTER_TOOL_DEFERRED_MESSAGE,
    HOSTED_SDK_TOOL_NAMES,
    CodeInterpreterHostedTool,
    FileSearchHostedTool,
    HostedMCPHostedTool,
    ImageGenerationHostedTool,
    WebSearchHostedTool,
)


def _agent(tools: list[dict]) -> Agent:
    return Agent(
        name="a",
        model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
        instructions={"inline": "hi"},
        tools=tools,
    )


def _hosted(tool: str, **params: object) -> dict:
    entry: dict = {"name": tool.lower(), "type": "hosted", "tool": tool}
    if params:
        entry["params"] = params
    return entry


@pytest.mark.unit
class TestHostedToolUnion:
    @pytest.mark.parametrize(
        ("entry", "cls"),
        [
            (_hosted("WebSearchTool"), WebSearchHostedTool),
            (
                _hosted("FileSearchTool", vector_store_ids=["vs_1"]),
                FileSearchHostedTool,
            ),
            (
                _hosted("CodeInterpreterTool", container={"type": "auto"}),
                CodeInterpreterHostedTool,
            ),
            (_hosted("ImageGenerationTool"), ImageGenerationHostedTool),
            (
                _hosted("HostedMCPTool", server_label="docs", server_url="https://x"),
                HostedMCPHostedTool,
            ),
        ],
    )
    def test_tool_selects_class_and_sdk_name(self, entry: dict, cls: type) -> None:
        agent = _agent([entry])
        assert agent.tools is not None
        built = agent.tools[0]
        assert isinstance(built, cls)
        assert built.type == "hosted"
        assert built.sdk_tool_name == HOSTED_SDK_TOOL_NAMES[entry["tool"]]

    def test_unknown_tool_name_lists_expected_tags(self) -> None:
        with pytest.raises(ValidationError, match="Input tag 'Nope'") as info:
            _agent([_hosted("Nope")])
        assert "WebSearchTool" in str(info.value)
        assert "HostedMCPTool" in str(info.value)

    def test_computer_tool_rejected_with_deferred_message(self) -> None:
        with pytest.raises(ValidationError) as info:
            _agent([_hosted("ComputerTool")])
        assert COMPUTER_TOOL_DEFERRED_MESSAGE in str(info.value)

    def test_computer_tool_rejected_even_with_unsafe_opt_in(self) -> None:
        with pytest.raises(ValidationError, match="H-012"):
            Agent(
                name="a",
                model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
                instructions={"inline": "hi"},
                tools=[_hosted("ComputerTool")],
                openai={"i_understand_this_is_unsafe": True},
            )

    def test_extra_top_level_field_rejected(self) -> None:
        entry = _hosted("WebSearchTool")
        entry["bogus"] = 1
        with pytest.raises(ValidationError, match="bogus"):
            _agent([entry])

    def test_duplicate_config_names_rejected(self) -> None:
        a = _hosted("WebSearchTool")
        b = _hosted("ImageGenerationTool")
        b["name"] = a["name"]
        with pytest.raises(ValidationError, match="Duplicate tool names"):
            _agent([a, b])


@pytest.mark.unit
class TestHostedParams:
    def test_web_search_defaults(self) -> None:
        tool = _agent([_hosted("WebSearchTool")]).tools[0]
        assert isinstance(tool, WebSearchHostedTool)
        assert tool.params.search_context_size == "medium"
        assert tool.params.user_location is None
        assert tool.params.allowed_domains is None
        assert tool.params.external_web_access is None

    def test_web_search_rejects_unknown_param(self) -> None:
        with pytest.raises(ValidationError, match="filters"):
            _agent([_hosted("WebSearchTool", filters={"allowed_domains": ["x"]})])

    def test_web_search_rejects_bad_context_size(self) -> None:
        with pytest.raises(ValidationError, match="search_context_size"):
            _agent([_hosted("WebSearchTool", search_context_size="huge")])

    def test_file_search_requires_vector_store_ids(self) -> None:
        with pytest.raises(ValidationError, match="params"):
            _agent([_hosted("FileSearchTool")])
        with pytest.raises(ValidationError, match="vector_store_ids"):
            _agent([_hosted("FileSearchTool", vector_store_ids=[])])

    def test_file_search_bounds_max_results(self) -> None:
        with pytest.raises(ValidationError, match="max_num_results"):
            _agent(
                [_hosted("FileSearchTool", vector_store_ids=["vs"], max_num_results=0)]
            )

    def test_file_search_score_threshold_range(self) -> None:
        with pytest.raises(ValidationError, match="score_threshold"):
            _agent(
                [
                    _hosted(
                        "FileSearchTool",
                        vector_store_ids=["vs"],
                        ranking_options={"score_threshold": 1.5},
                    )
                ]
            )

    def test_code_interpreter_requires_container(self) -> None:
        with pytest.raises(ValidationError, match="container"):
            _agent([_hosted("CodeInterpreterTool", container="")])
        with pytest.raises(ValidationError, match="params"):
            _agent([_hosted("CodeInterpreterTool")])

    def test_code_interpreter_accepts_id_or_auto_spec(self) -> None:
        by_id = _agent([_hosted("CodeInterpreterTool", container="cntr_1")]).tools[0]
        assert isinstance(by_id, CodeInterpreterHostedTool)
        assert by_id.params.container == "cntr_1"
        auto = _agent(
            [
                _hosted(
                    "CodeInterpreterTool",
                    container={
                        "type": "auto",
                        "file_ids": ["f1"],
                        "memory_limit": "4g",
                    },
                )
            ]
        ).tools[0]
        assert isinstance(auto, CodeInterpreterHostedTool)
        assert not isinstance(auto.params.container, str)
        assert auto.params.container.memory_limit == "4g"

    def test_code_interpreter_rejects_bad_memory_limit(self) -> None:
        with pytest.raises(ValidationError, match="memory_limit"):
            _agent(
                [
                    _hosted(
                        "CodeInterpreterTool",
                        container={"type": "auto", "memory_limit": "2g"},
                    )
                ]
            )

    def test_image_generation_validates_enums(self) -> None:
        with pytest.raises(ValidationError, match="quality"):
            _agent([_hosted("ImageGenerationTool", quality="ultra")])
        with pytest.raises(ValidationError, match="partial_images"):
            _agent([_hosted("ImageGenerationTool", partial_images=4)])

    def test_hosted_mcp_requires_exactly_one_endpoint(self) -> None:
        with pytest.raises(ValidationError, match="exactly one of server_url"):
            _agent([_hosted("HostedMCPTool", server_label="d")])
        with pytest.raises(ValidationError, match="exactly one of server_url"):
            _agent(
                [
                    _hosted(
                        "HostedMCPTool",
                        server_label="d",
                        server_url="https://x",
                        connector_id="connector_gmail",
                    )
                ]
            )

    def test_hosted_mcp_requires_server_label(self) -> None:
        with pytest.raises(ValidationError, match="server_label"):
            _agent([_hosted("HostedMCPTool", server_url="https://x")])

    def test_hosted_mcp_rejects_interactive_approval(self) -> None:
        with pytest.raises(ValidationError, match="require_approval"):
            _agent(
                [
                    _hosted(
                        "HostedMCPTool",
                        server_label="d",
                        server_url="https://x",
                        require_approval="always",
                    )
                ]
            )

    def test_hosted_mcp_defaults_to_never_approval(self) -> None:
        tool = _agent(
            [_hosted("HostedMCPTool", server_label="d", server_url="https://x")]
        ).tools[0]
        assert isinstance(tool, HostedMCPHostedTool)
        assert tool.params.require_approval == "never"
