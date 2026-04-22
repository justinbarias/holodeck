"""Unit tests for code-grader invocation harness (US4 Phase 5).

Covers T032–T041 in ``specs/032-multi-turn-test-cases/tasks-us4.md`` — the
``GraderContext`` / ``GraderResult`` immutability and the ``invoke_grader``
normalization + exception policy described in
``specs/032-multi-turn-test-cases/contracts/code-grader-contract.md``.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from my_benchmarks import (
    raises_value_error,
    returns_dict,
    returns_float,
    returns_grader_result,
)

from holodeck.lib.test_runner.code_grader import (
    GraderContext,
    GraderResult,
    build_grader_context,
    invoke_grader,
)


def _ctx(**overrides: object) -> GraderContext:
    base: dict[str, object] = {
        "turn_input": "q",
        "agent_response": "a",
        "ground_truth": "gt",
        "tool_invocations": (),
        "retrieval_context": None,
        "turn_index": 0,
        "test_case_name": "tc",
        "turn_config": {},
    }
    base.update(overrides)
    return GraderContext(**base)  # type: ignore[arg-type]


@pytest.mark.unit
class TestGraderContextImmutability:
    def test_grader_context_is_frozen(self) -> None:
        ctx = _ctx()
        with pytest.raises(FrozenInstanceError):
            ctx.turn_input = "mutated"  # type: ignore[misc]

    def test_grader_context_tuples_immutable(self) -> None:
        ctx = _ctx(
            tool_invocations=(),
            retrieval_context=("a", "b"),
        )
        assert isinstance(ctx.tool_invocations, tuple)
        assert isinstance(ctx.retrieval_context, tuple)

        ctx_none = _ctx(retrieval_context=None)
        assert ctx_none.retrieval_context is None


@pytest.mark.unit
class TestInvokeGraderReturnNormalization:
    def test_invoke_with_grader_result(self) -> None:
        mr, details, captured = invoke_grader(
            returns_grader_result,
            _ctx(),
            metric_name="mygrader",
            threshold=None,
        )
        assert captured is None
        assert mr.kind == "code"
        assert mr.score == 1.0
        assert mr.passed is True
        assert mr.reasoning == "fixture result"
        assert details == {"foo": "bar"}

    def test_return_true_shortcut(self) -> None:
        mr, _d, cap = invoke_grader(
            lambda ctx: True,
            _ctx(),
            metric_name="g",
            threshold=None,
        )
        assert cap is None
        assert mr.score == 1.0
        assert mr.passed is True

    def test_return_false_shortcut(self) -> None:
        mr, _d, cap = invoke_grader(
            lambda ctx: False,
            _ctx(),
            metric_name="g",
            threshold=None,
        )
        assert cap is None
        assert mr.score == 0.0
        assert mr.passed is False

    def test_return_float_threshold_derivation_pass(self) -> None:
        mr, _d, _c = invoke_grader(
            returns_float, _ctx(), metric_name="g", threshold=0.7
        )
        assert mr.score == 0.75
        assert mr.passed is True

    def test_return_float_threshold_derivation_fail(self) -> None:
        mr, _d, _c = invoke_grader(
            returns_float, _ctx(), metric_name="g", threshold=0.8
        )
        assert mr.score == 0.75
        assert mr.passed is False

    def test_return_float_default_gate(self) -> None:
        mr, _d, _c = invoke_grader(
            returns_float, _ctx(), metric_name="g", threshold=None
        )
        # default gate is 0.5; 0.75 passes.
        assert mr.passed is True

    def test_non_standard_return_treated_as_error(self) -> None:
        mr, details, captured = invoke_grader(
            returns_dict, _ctx(), metric_name="g", threshold=None
        )
        # Grader returned a dict — recorded as grader error, not success.
        assert captured is None
        assert details is None
        assert mr.passed is False
        assert mr.score == 0.0
        assert mr.error is not None
        assert "unsupported" in mr.error.lower() or "dict" in mr.error.lower()


@pytest.mark.unit
class TestInvokeGraderExceptionPolicy:
    def test_exception_captured_on_metric_result(self) -> None:
        mr, details, captured = invoke_grader(
            raises_value_error,
            _ctx(),
            metric_name="failing",
            threshold=None,
        )
        assert isinstance(captured, ValueError)
        assert mr.passed is False
        assert mr.score == 0.0
        assert mr.error is not None
        assert "ValueError" in mr.error
        assert details is None


@pytest.mark.unit
class TestInvokeGraderContextFields:
    def test_grader_receives_expected_context_fields(self) -> None:
        seen: dict[str, object] = {}

        def probe(ctx: GraderContext) -> bool:
            seen["turn_input"] = ctx.turn_input
            seen["agent_response"] = ctx.agent_response
            seen["ground_truth"] = ctx.ground_truth
            seen["tool_invocations"] = ctx.tool_invocations
            seen["retrieval_context"] = ctx.retrieval_context
            seen["turn_index"] = ctx.turn_index
            seen["test_case_name"] = ctx.test_case_name
            seen["turn_config"] = ctx.turn_config
            return True

        ctx = build_grader_context(
            turn_input="q",
            agent_response="a",
            ground_truth="gt",
            tool_invocations=[],
            retrieval_context=["snippet"],
            turn_index=2,
            test_case_name="case_x",
            turn_config={"turn_program": "foo"},
        )
        _mr, _d, _c = invoke_grader(probe, ctx, metric_name="probe", threshold=None)
        assert seen["turn_input"] == "q"
        assert seen["agent_response"] == "a"
        assert seen["ground_truth"] == "gt"
        assert seen["tool_invocations"] == ()
        assert seen["retrieval_context"] == ("snippet",)
        assert seen["turn_index"] == 2
        assert seen["test_case_name"] == "case_x"
        assert seen["turn_config"] == {"turn_program": "foo"}


@pytest.mark.unit
class TestGraderDetails:
    def test_details_preserved_on_return(self) -> None:
        mr, details, _c = invoke_grader(
            returns_grader_result, _ctx(), metric_name="g", threshold=None
        )
        assert details == {"foo": "bar"}
        assert mr.passed is True

    def test_details_must_be_json_safe(self) -> None:
        def grader(ctx: GraderContext) -> GraderResult:
            return GraderResult(
                score=1.0,
                passed=True,
                details={"bad": object()},  # not JSON-serializable
            )

        mr, details, captured = invoke_grader(
            grader, _ctx(), metric_name="g", threshold=None
        )
        assert captured is None
        assert details is None
        assert mr.passed is False
        assert mr.score == 0.0
        assert mr.error is not None
        assert "not JSON-serializable" in mr.error


@pytest.mark.unit
class TestTestCaseFatalExceptionClass:
    def test_test_case_fatal_is_exception(self) -> None:
        # TestCaseFatal is executor-local (T042a). We import it only for this
        # isA check — don't re-export.
        from holodeck.lib.test_runner.executor import TestCaseFatal

        assert issubclass(TestCaseFatal, Exception)
