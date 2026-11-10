# Test audit, 2026-11

Audit date: 2026-11-09. Base: `d382992` on `main`. Scope: `tests/` and the coverage of `src/holodeck`.
The audit itself only reported findings. With maintainer approval, H1–H5 were then fixed in test code only, on branch `claude/test-audit-bd1afc`. See [Resolution of H1–H5](#resolution-of-h1h5).

## Method

The procedure adapts the OpenClaw `test-audit` skill
([SKILL.md](https://github.com/openclaw/openclaw/blob/main/.agents/skills/test-audit/SKILL.md),
`CAMPAIGN.md`, and `references/audit.md` from the same directory).
The skill was written for a TypeScript/Vitest repository. This audit translated its checks to pytest as follows.

| Skill check | Python/pytest translation used here |
| --- | --- |
| Assertion-free probes, self-comparisons, constant asserts | AST scan of every `test_*` function: no `assert`, `pytest.raises`/`warns`, or `mock.assert_*`; `x == x`; `assert True` |
| Subject-generated expected values, mocks that supply asserted behavior | AST scan for tests that never reference a `holodeck` import, plus `assert result is mock_x.return_value` greps, then manual reading |
| Negative controls that pass on an unrelated guard | Manual reading of error-path tests, permissive `exit_code in (...)` asserts, and `RuntimeWarning: coroutine ... never awaited` lines in the run log |
| Names that promise behavior the test never exercises | Manual reading of every flagged test |
| Test-only exports and production seams | Grep of `src/holodeck` for reset/for-test helpers, then caller search |
| Duplicate contracts and repeated layers | Duplicate test names across files and per-command copies |
| Campaign step 1 (baseline) | Full suite run with branch coverage and durations |
| Retention bar | Applied before listing a finding. Retained false positives are listed at the end. |

Skipped skill steps:

- Campaign steps 3 to 8 (per-lane ledgers, cutover, mutation proof, PR handoff). They edit code, and this task is report-only.
- OpenClaw-only tooling: `pnpm deadcode:*`, `scripts/check-changed.mjs`, the Vitest proof policy in `openclaw-testing/SKILL.md`, and `docs/help/testing/writing-tests.md`. These do not exist in this repository.
- Commit, push, and land steps. They are outside this task's authority.

## Baseline

Command (Python 3.10.18, the CI version; `uv sync --all-extras --frozen`):

```bash
uv run pytest -n auto -q --cov --cov-branch --durations=40
```

| Measure | Value |
| --- | --- |
| Result | 6,032 passed, 66 skipped, 0 failed, exit 0 |
| Wall time | 133.8 s (10 macOS workers) |
| Collected test functions (static) | 5,362 in 285 files |
| Test LOC / production LOC | 114,947 / 62,595 |
| Coverage (line + branch, `src/holodeck`, dashboard omitted) | **91.0 %**; branches 4,622 / 5,460 |
| Warnings | 1,557, including 80 `PytestCollectionWarning`, 11 `RuntimeWarning` (never-awaited coroutines), 2 `PytestUnknownMarkWarning` |

All 66 skips come from live-credential gates or missing optional clients. None is a failure.
Note: a fresh worktree without `UV_PYTHON` resolves to Python 3.14, and `onnxruntime==1.20.1` has no 3.14 wheel, so `uv run` fails before tests start.

## Findings

Severity: **High** means a test cannot detect the regression its name describes. **Medium** means a real gap in proof or enforcement. **Low** means cost, coupling, or hygiene.

### High

#### H1. The retry cap test cannot fail when the cap is removed, and it sleeps 17 s (resolved)

- Evidence: [test_base.py:183-203](../../tests/unit/lib/evaluators/test_base.py) asserts only `elapsed_time >= 16.0`.
  With `max_delay` removed, the delays are 1+2+4+8+16 = 31 s, which still passes.
  It is the slowest test in the suite (17.01 s).
  `test_azure_ai.py:488-518` (6.0 s; file removed in 042 T18) and [test_base.py:174](../../tests/unit/lib/evaluators/test_base.py) use the same lower-bound timing pattern.
- Contract: `BaseEvaluator._calculate_delay` returns `min(base * exp**attempt, max_delay)` ([base.py:171-194](../../src/holodeck/lib/evaluators/base.py)).
- Fix: patch `holodeck.lib.evaluators.base.asyncio.sleep` with a recorder. Assert the exact delays `[1.0, 2.0, 4.0, 5.0, 5.0]`. This adds the missing upper bound and removes about 25 s of real sleep.

#### H2. The chat error-path tests pass for the wrong reason (resolved)

- Evidence: [test_chat_errors.py:335-368](../../tests/unit/cli/commands/test_chat_errors.py) sets `process_message` to raise `ExecutionError`.
  The chat loop calls `process_message_streaming` ([chat.py:381](../../src/holodeck/cli/commands/chat.py)).
  On an `AsyncMock` that call returns a coroutine, so `async for` raises `TypeError`.
  The generic handler at [chat.py:432](../../src/holodeck/cli/commands/chat.py) catches it.
  The baseline log shows `RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited` at `chat.py:381` and `chat.py:438` for exactly these tests.
  The assertion `result.exit_code in (0, 1, 2) or "Error" in result.output` accepts almost every outcome, and `error_msg` is never checked.
  [test_chat_errors.py:280](../../tests/unit/cli/commands/test_chat_errors.py) and [:314](../../tests/unit/cli/commands/test_chat_errors.py) use the same permissive `or` assertions.
- Effect: the successful streaming path in [chat.py:384-430](../../src/holodeck/cli/commands/chat.py) is not covered (`chat.py` is at 75.2 %).
- Fix: mock `process_message_streaming` as an async generator that raises `ExecutionError(error_msg)`. Assert `error_msg in result.output` and `exit_code == 0`. Add one streaming happy-path test.

#### H3. Middleware logging assertions can never fail (resolved)

- Evidence: [test_middleware.py:131-139](../../tests/unit/serve/test_middleware.py), [:152-160](../../tests/unit/serve/test_middleware.py), and [:488-495](../../tests/unit/serve/test_middleware.py) loop over `mock_logger.info.call_args_list` and run `assert True; break` on a match.
  If no call matches, the loop ends and the test passes.
  The tests claim to prove session-ID extraction from the query, header priority, and 500-status logging.
- Fix: `assert any(c.kwargs.get("extra", {}).get("session_id") == "query-session-456" for c in mock_logger.info.call_args_list)`. Use the same pattern for the other two tests.

#### H4. Hierarchical document tool tests that run no code (resolved)

- Evidence, [test_hierarchical_document_tool.py](../../tests/unit/tools/test_hierarchical_document_tool.py):
  - `test_contextual_embeddings_when_enabled` (:2126-2171) builds mocks and patches, then runs `pass` with the comment "Skip the actual test since chunker is None".
  - `test_contextual_embeddings_disabled_reason_logged` (:2174-2190) never calls the tool. Its name promises a log check.
  - `test_ingest_logs_warning_no_files` (:2019-2038) patches the private `_discover_files` and asserts nothing. Its name promises a warning.
- Fix: delete the first two, or rewrite them through `_ingest_documents` with `caplog`. Add a `caplog` assertion to the third.

#### H5. Observability tests that test Python, not HoloDeck (resolved)

- Evidence, [test_logging_coordination.py](../../tests/unit/lib/observability/test_logging_coordination.py):
  - `test_observability_config_enabled_check` (:20) asserts Pydantic field defaults.
  - `test_observability_enabled_check_pattern` (:32) asserts the short-circuit behavior of Python `and` on `MagicMock`.
  - `test_otel_replaces_setup_logging_when_enabled` (:61) and `test_setup_logging_used_when_otel_disabled` (:75) define the decision function inside the test and assert on it. No production code runs.
  - `test_otel_shutdown_flushes_logs` (:107) has no assertion.
  - `test_otel_logger_provider_initialized` (:93) initializes real global OTel providers and never shuts them down. The baseline log contains repeated `ValueError: I/O operation on closed file` tracebacks from the OTel log export thread. This shows provider state leaking past the test.
- The real contract (CLI commands choose OTel or `setup_logging`) already has owners in `test_{chat,test,optimize}_observability.py` and `serve/test_server_observability.py`.
- Fix: delete the four tautological tests. Add a shutdown, or a fixture that resets OTel global providers, to the two tests that initialize real providers. Give `test_otel_shutdown_flushes_logs` an observable check (for example, an in-memory exporter that receives the flushed record).

### Medium

#### M1. The Docker build integration tests cannot fail, and `--strict-markers` is not enforced

- Evidence: [test_build_image.py:153-168](../../tests/integration/deploy/test_build_image.py) asserts only inside `if result.exit_code == 0:`.
  [:170-184](../../tests/integration/deploy/test_build_image.py) accepts `exit_code in (0, 3)`, where 3 is the Docker-error code.
  `@pytest.mark.docker` is not registered in `pyproject.toml`.
  `addopts` includes `--strict-markers`, but pytest 9.0.3 only warns (`PytestUnknownMarkWarning`, 2 occurrences) and the file passes.
  [test_cli_help.py:95-109](../../tests/unit/cli/test_cli_help.py) has the same shape for `holodeck --version`: it accepts exit 2 ("option not found") and asserts only inside `if result.exit_code == 0`.
- Fix: register `docker`. Gate with `skipif(no docker daemon)` and assert `exit_code == 0` when the test runs. Assert `exit_code == 0` and a version pattern for `--version`. Find out why `--strict-markers` does not reject unknown marks under pytest 9. One option to evaluate is the `strict_markers = true` ini key.

#### M2. The "dual-backend" integration test exercises neither backend

- Evidence: the docstring of [test_multi_turn_dual_backend.py:1-14](../../tests/integration/test_multi_turn_dual_backend.py) says `HOLODECK_IT_OPENAI=1` and `HOLODECK_IT_ANTHROPIC=1` run live backends.
  The code at :101-130 *skips* when those variables are set (`pytest.skip("TODO: wire real ... session")`).
  Otherwise it runs a `Mock` backend with a spoofed `__class__.__name__`.
  So the live path does not exist, and the default path proves only class-name dispatch plus token summation.
- Fix: rename and move it to `tests/unit` as a dispatch test, or implement the live variant behind a `live` marker (see M5).

#### M3. Modules with little or no default-suite proof

| Module | Coverage | Cause |
| --- | --- | --- |
| [cli/commands/test_view.py](../../src/holodeck/cli/commands/test_view.py) | 27.4 % | No test references it |
| [services/mcp_registry.py](../../src/holodeck/services/mcp_registry.py) | 29.0 % | Only the live tests in [test_mcp_registry_client.py](../../tests/integration/test_mcp_registry_client.py) exercise it. `SKIP_NETWORK_TESTS` defaults to `true`, so all 19 are skipped. CLI tests mock the client. |
| [lib/logging_utils.py](../../src/holodeck/lib/logging_utils.py) | 39.7 % | No test references it |
| [cli/commands/chat.py](../../src/holodeck/cli/commands/chat.py) | 75.2 % | See H2 |
| `lib/collection_filter.py` (removed in 042 T4) | 79.8 % | The `TestWeaviate` class (`test_collection_filter.py:294`, removed with the module) skips because `weaviate-client` is not part of `--all-extras`. The same is true in CI. |
| [serve/protocols/agui.py](../../src/holodeck/serve/protocols/agui.py) | 81.7 % | Lines 834-865 are not covered |

- Fix: add `httpx.MockTransport` tests for the `mcp_registry` HTTP and error mapping. Add a `CliRunner` test for `holodeck test view` against a fixture run file. Either add `weaviate-client` to a test extra or accept the gap in writing.

#### M4. The 80 % coverage target is not enforced by the CI job

- Evidence: [ci.yml:59](../../.github/workflows/ci.yml) runs `pytest --cov` without `--cov-fail-under`.
  The Makefile `ci-github` target has `--cov-fail-under=$(MIN_COVERAGE)`, but `ci.yml` does not call it.
  [codecov.yml](../../codecov.yml) sets an 80 % status target with `require_ci_to_pass: false`.
  This audit did not check whether branch protection requires that Codecov status.
  The constitution target is restated in [QUALITY_SCORE.md](../QUALITY_SCORE.md), but no gate enforces it.
- Risk is low today (91 %). Fix: add `fail_under = 80` to `[tool.coverage.report]`.

#### M5. Tier markers are inconsistent, and live gates are copied per file

- Evidence: 130 of 241 unit files carry `unit`. 37 of 43 integration files carry `integration`. [test_pdf_heading_detection.py](../../tests/unit/lib/test_pdf_heading_detection.py) marks `integration` inside `tests/unit`.
  Twelve files define their own `skip_if_no_claude_oauth` / `skip_if_no_openai` / `skip_if_no_azure` / `skip_if_no_ollama` (for example [test_chat_executor_claude.py:31](../../tests/integration/test_chat_executor_claude.py) and [test_claude_instrumentation.py:98](../../tests/integration/test_claude_instrumentation.py)). No `live` marker exists.
- Effect: a `verify-fast` / `verify-live` split cannot select tiers with `-m`. It must rely on paths and environment variables.
- Fix: register `live` (and `docker`). Move the credential gates into a shared conftest helper that applies `live` together with the `skipif`. Optionally, add a `pytest_collection_modifyitems` hook that marks tests by directory.

### Low

#### L1. A test-only production seam

[openai_agents_tracing.py:179](../../src/holodeck/lib/backends/openai_agents_tracing.py) `_reset_tracing_router()` says "test isolation only".
Its only callers are [test_openai_agents_tracing.py:283,289](../../tests/unit/lib/backends/test_openai_agents_tracing.py) and [test_openai_agents_fallback_runner.py:231,236](../../tests/unit/lib/backends/test_openai_agents_fallback_runner.py).
Fix: a fixture that uses `monkeypatch.setattr` to replace `_registry` and `_router`. Then delete the function.

#### L2. Implementation-coupled unit suites

| File | Private attribute assignments | `patch.object(..., "_x")` calls |
| --- | --- | --- |
| [test_hierarchical_document_tool.py](../../tests/unit/tools/test_hierarchical_document_tool.py) | 136 | 27 |
| [test_claude_backend.py](../../tests/unit/lib/backends/test_claude_backend.py) | 65 | 0 |
| [test_vectorstore_tool.py](../../tests/unit/test_vectorstore_tool.py) | 37 | 0 |
| [test_file_processor.py](../../tests/unit/lib/test_file_processor.py) | 0 | 31 |

Renaming a private helper breaks these tests. That conflicts with the skill rule that behavior-preserving refactors must not break tests.
Fix (campaign-sized): drive the tools through `initialize()` / `search()` with a fake embedding service and an in-memory store. Do this one lane at a time.

#### L3. Ownership and layout drift

- [tests/unit/agent/test_executor_coverage.py](../../tests/unit/agent/test_executor_coverage.py) (949 lines) has the docstring "Additional coverage tests for chat/executor.py". It tests `holodeck.chat.executor`, but `tests/unit/chat/` exists. The file name shows it was written to raise coverage, not to own a contract.
- [tests/unit/test_vectorstore_tool.py](../../tests/unit/test_vectorstore_tool.py) (2,494 lines) is at the unit root, but its siblings are in `tests/unit/tools/`.
- Fix: move these files to the owner directories during the next change in each area.

#### L4. Per-command observability tests duplicate copied wiring

`initialize_observability` setup is copied into [serve.py:126](../../src/holodeck/cli/commands/serve.py), [optimize.py:227](../../src/holodeck/cli/commands/optimize.py), [chat.py:190](../../src/holodeck/cli/commands/chat.py), and [test.py:219](../../src/holodeck/cli/commands/test.py).
Four test files (1,072 lines) re-prove it with call-shape asserts.
Thirty-seven tests discard the `runner.invoke(...)` result. Most are in these files and in `test_chat.py` / `test_test.py`. A crash after initialization still passes.
Fix: assert `result.exit_code` where the result is discarded. Longer term, extracting one shared context manager (a production change, so ask first) would allow one owner test plus one smoke test per command.

#### L5. Dead or slow test fragments

- [test_file_processor.py:1261](../../tests/unit/lib/test_file_processor.py) is unconditionally `pytest.skip(...)`. Delete it.
- [test_runs_fingerprint.py:61,89](../../tests/unit/dashboard/test_runs_fingerprint.py) uses `time.sleep(1.1)` twice to bump the mtime. Use `os.utime` instead.
- Exporter tests leave `PeriodicExportingMetricReader` ticker threads running without a provider. After H5, 16 `Cannot call collect on a MetricReader until it is registered` tracebacks still appear per full run (the OTel log-export leak is gone). Shut the readers down in teardown.
- 80 `PytestCollectionWarning`s come from production classes `TestCaseModel`, `TestExecutor`, `TestReport`, and `TestResult` that are imported into tests. Set `__test__ = False` on those classes, or filter the warning.

## Retained false positives

These matched a junk pattern but pass the retention bar:

- [test_tool_hooks.py:232](../../tests/unit/lib/backends/test_tool_hooks.py) `session.tool_events is session.tool_events` checks that the property is memoized. It fails if the property builds a new queue on each access.
- [test_feel_conformance.py](../../tests/unit/workflow/test_feel_conformance.py) never imports `holodeck`. It pins the behavior of the `bkflow-feel` dependency, which is an independent dependency contract.
- `warnings.simplefilter("error")` tests with no `assert` in [test_claude_config.py](../../tests/unit/models/test_claude_config.py) (:451, :458, :590-645) are valid negative controls with paired positive tests.
- Assertion-free validator tests in [test_validators.py](../../tests/unit/lib/backends/test_validators.py), [test_feel.py](../../tests/unit/workflow/test_feel.py), and [test_sandbox_safety.py](../../tests/unit/temporal/test_sandbox_safety.py) prove "does not raise", and a sibling test proves the raise.
- 34 `assert x is mock_y` identity assertions, for example in [test_selector.py](../../tests/unit/lib/backends/test_selector.py), prove dispatch and routing contracts.
- Temporal subprocess and sandbox tests that never import `holodeck` in the test body run production code in a child process.

## Relation to QUALITY_SCORE.md

[QUALITY_SCORE.md](../QUALITY_SCORE.md) (2026-09-06) grades most areas **B** with "not rerun here". This audit adds:

- **New** (not in QUALITY_SCORE or the tech-debt tracker): H1-H5, M1, M2, the M3 module list, M4, M5, and L1-L5.
- **Known, now measured:** the "80 % coverage target, measure explicitly" policy. Measured at 91.0 %, but not gated (M4).
- **Known, unchanged:** "Live stores require service-specific verification" (Tools) and "Cloud behavior requires authorized end-to-end validation" (Serving). The skip list confirms these paths are covered only by credential-gated tests. M5 is the tooling gap that makes them hard to run as a tier.
- **Known, not re-reported:** global import-layer enforcement (C) and spec freshness (C).

This audit does not change any QUALITY_SCORE row.

## Suggested order

1. ~~H1-H5~~: done. See below.
2. M5 and M1: register `live` / `docker` markers and fix strict-marker enforcement before the tiered verify suite depends on markers.
3. M3 and M4: add the missing owner tests, then add `fail_under`.
4. L1-L5 opportunistically. Treat L2 as a separate per-lane campaign.

## Resolution of H1–H5

Fixed on 2026-11-09 on branch `claude/test-audit-bd1afc`. Only test code changed: no change to `src/`, `pyproject.toml`, or `uv.lock`.

Proof method: for each fix, one deliberate change to the production owner made the fixed test fail. Every source file was then restored with `git checkout` and verified clean. For H2, H3, and H5, the HEAD versions of the tests ran against the same mutants and **passed**, which shows the old tests were blind to them.

| Finding | Change | Mutant (production change) | New tests | HEAD tests |
| --- | --- | --- | --- | --- |
| H1 | [test_base.py](../../tests/unit/lib/evaluators/test_base.py): a `recorded_sleeps` fixture replaces `asyncio.sleep` in `holodeck.lib.evaluators.base`. Exact delays are asserted: `[1.0, 2.0]` and `[1.0, 2.0, 4.0, 5.0, 5.0]`. `test_azure_ai.py` (removed in 042 T18) asserted `[2.0, 4.0]` the same way. | `_calculate_delay` returns `delay` (cap removed) | `test_max_delay_cap` fails: `[1, 2, 4, 8, 16] != [1, 2, 4, 5, 5]` | By arithmetic, the old `elapsed >= 16.0` also holds at 31 s |
| H1 | as above | delay × 1.5 | 3 tests fail | — |
| H2 | [test_chat_errors.py](../../tests/unit/cli/commands/test_chat_errors.py): a `_streaming_session` helper streams chunks or raises from `process_message_streaming`. The error tests assert `exit_code == 0`, `Error: <msg>`, and a successful next turn. New `test_streaming_reply_is_displayed` covers the happy path. The Ctrl+C tests patch `click.prompt` to raise `KeyboardInterrupt` and assert exit 130, `terminate` awaited, and `Goodbye!`. The empty-line test asserts the session is never called. | The REPL `except Exception` re-raises instead of logging | 3 fail | **5 passed** |
| H2 | as above | `composer.write_text(chunk)` removed | 4 fail | — |
| H2 | as above | `sys.exit(130)` → `sys.exit(0)` | 1 fails | — |
| H2 | as above | `session_manager.terminate()` removed | 3 fail | — |
| H2 | as above | Empty-input `continue` removed | 1 fails | — |
| H3 | [test_middleware.py](../../tests/unit/serve/test_middleware.py): an `_info_extras` helper asserts the exact `extra` of the "Request started" and "Request completed" log calls. This replaces the five loops that could pass with zero matches. | `"status_code": 200` in the completion log | `test_middleware_order_matters` fails | **2 passed** |
| H3 | as above | Query-parameter session ID ignored | 1 fails | **10 passed** |
| H3 | as above | `session_id` dropped from the start log, then from the completion log | 3 fail each time | — |
| H3 | as above | Query parameter checked before the header | 1 fails | — |
| H4 | [test_hierarchical_document_tool.py](../../tests/unit/tools/test_hierarchical_document_tool.py): both no-op tests now drive `_ingest_documents` and capture the chunks passed to `_embed_chunks`. The enabled test asserts the generated context lands in `contextualized_content`. The disabled test is parametrized over both log reasons. The no-files test uses the real `_discover_files` with `caplog` and asserts `ingested == 0`. | No-files warning downgraded to `info` | 1 fails | Old test had no assertion |
| H4 | as above | Context output ignored, reason text swapped, fallback content blanked, or `contextual_embeddings` flag ignored | 1–2 fail for each mutant | Old tests ran no production code |
| H5 | [test_logging_coordination.py](../../tests/unit/lib/observability/test_logging_coordination.py): the four tautological tests are deleted. The provider test asserts the global LoggerProvider and a single root `LoggingHandler`. The flush test exports through `InMemoryLogRecordExporter` and asserts nothing is exported before shutdown and that `flush me` is exported after it. New [conftest.py](../../tests/unit/lib/observability/conftest.py) `isolated_otel` fixture: it resets the set-once OTel globals, shuts down the created context, and restores root logger handlers and level. It is also applied to `TestInitializeObservability` in [test_providers.py](../../tests/unit/lib/observability/test_providers.py). | `shutdown_observability` returns early | flush test fails | **6 passed** |
| H5 | as above | `set_logger_provider` removed | 1 fails | — |
| H5 | as above | Root `addHandler(otel_handler)` removed | 2 fail | — |

H3 also exposed a wrong test premise. `test_middleware_order_matters` added the middlewares in the reverse of the production order ([server.py:422-423](../../src/holodeck/serve/server.py)). In that order `LoggingMiddleware` never logs the 500, and the old loop hid this. The test now uses the server's order and explains why the order matters.

H2 also rewrote the two `TestValidationErrors` tests, which had the same defect. One mocked the unused `process_message`. The other sent an empty line, which the CLI skips before it reaches the session.

### Verification

| Check | Result |
| --- | --- |
| Changed test files, 3 parallel runs | 337 passed each run |
| Full suite (`uv run pytest -n auto`, Python 3.10) | 6,030 passed, 66 skipped, exit 0, 100.7 s (baseline: 6,032 passed, 133.8 s). Delta: −4 deleted tautologies, +1 streaming test, +1 parametrized case. |
| `RuntimeWarning` never-awaited coroutines at `chat.py` | 0 (baseline 2) |
| OTel log-export `I/O operation on closed file` tracebacks | 0. The 17 that remain come from metric-reader tickers (16) and OpenAI tracing (1); see L5. |
| `ruff check`, `black --check` on changed files | Pass |
| `mypy` on changed files | `tests/` is excluded from the mypy gate in pre-commit. Error counts are at or below HEAD: `test_chat_errors.py` 10 → 5, other files unchanged, new `conftest.py` 0. |
| `make harness-check` | Pass |

The `isolated_otel` fixture replaces private OTel module globals (`_TRACER_PROVIDER`, `_*_SET_ONCE`, and their log and metric equivalents). These names exist in the locked `opentelemetry-sdk` 1.39.1. Recheck them when the OTel SDK upgrade (1.45) lands.
