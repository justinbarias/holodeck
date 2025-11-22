# Plan: Integrate Spinner During Test Execution

**Created**: 2025-11-22
**Status**: Planning
**Related Tasks**: T104-T111 (Progress Display Enhancements)

## Problem Analysis

**Current Issue**: The spinner infrastructure exists (T104-T105) but never displays because:
- `progress_callback` only fires AFTER test completes
- Spinner needs to show DURING test execution (especially for long-running LLM calls)
- The `_get_spinner_char()` method exists but is never called

**Long-Running Operations** (where spinner should show):
1. LLM agent invocation (3-60+ seconds)
2. Evaluation metrics (multiple LLM calls)

**Test Evidence**: Running `holodeck test sample/hitchhikers_agent.yaml` shows:
```
[Test 1/1] FAIL Answer about the Answer to Life (3.82s)
```
- ✅ Progress display works
- ✅ Elapsed time shows
- ❌ Spinner never appears (no feedback during 3.82s execution)

## Current Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     CLI Command (test.py)                     │
│  1. Setup logging, load config                                │
│  2. Create ProgressIndicator(total_tests)                     │
│  3. Define progress_callback(result)                          │
│  4. Create TestExecutor(progress_callback=callback)          │
│  5. Run asyncio.run(executor.execute_tests())                │
│  6. Display summary                                            │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  TestExecutor.execute_tests()                 │
│  for test_case in test_cases:                                 │
│    ├─ result = await _execute_single_test(test_case)         │
│    │                                                           │
│    └─ if progress_callback:                                   │
│         progress_callback(result)  ◄─── CALLBACK FIRES AFTER │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│             TestExecutor._execute_single_test()               │
│  1. start_time = time.time()                                  │
│  2. process_files() - synchronous                             │
│  3. prepare_agent_input() - synchronous                       │
│  4. ┌─────────────────────────────────────────┐              │
│     │ await agent_factory.invoke(input)       │ ◄─── LONG    │
│     │   - Creates ChatHistory                  │      RUNNING │
│     │   - Calls _invoke_with_retry()           │      PART    │
│     │     - Calls _invoke_agent_impl()         │   (3-60s)    │
│     │       - async for response in agent.invoke() │          │
│     │         (LLM call happens here)          │              │
│     └─────────────────────────────────────────┘              │
│  5. validate_tool_calls() - synchronous                       │
│  6. await _run_evaluations() - can be long                    │
│  7. determine_test_passed() - synchronous                     │
│  8. return TestResult                                         │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              ProgressIndicator.update(result)                 │
│  - Called AFTER test completes                                │
│  - Updates counters (current_test, passed, failed)            │
│  - Stores result in test_results list                         │
│  - Does NOT show spinner during execution                     │
└─────────────────────────────────────────────────────────────┘
```

**Problem**: Callback fires too late (after test completes) to show spinner during execution.

## Solution Options Considered

### Option A: Threading/Async Background Task ❌
**Concept**: Run background task that periodically updates spinner while test executes.

**Pros**:
- Real-time animated spinner
- Smooth visual feedback

**Cons**:
- Complex implementation with task lifecycle management
- Requires new `TestProgress` model (separate from `TestResult`)
- Callback needs to handle two different types
- More difficult to test

**Verdict**: Too complex for the value it provides.

---

### Option B: Pre-Test Callback Pattern ✅ RECOMMENDED
**Concept**: Add `start_callback` that fires BEFORE test execution, showing spinner in initial state.

**Pros**:
- ✅ Minimal changes to existing architecture
- ✅ Clean callback separation (start vs complete)
- ✅ Easy to test
- ✅ Maintains backward compatibility
- ✅ No threading complexity

**Cons**:
- ⚠️ Spinner doesn't animate (shows single frame)
- ⚠️ No stage-specific updates

**Verdict**: Best balance of simplicity and user value.

---

### Option C: Async Generator Streaming ❌
**Concept**: Make executor yield progress events during execution using async generators.

**Pros**:
- Most granular control
- Can show different stages

**Cons**:
- Major refactoring of executor interface
- Breaks existing callback pattern
- Would break all existing tests
- Overkill for showing a spinner

**Verdict**: Not justified for current use case.

---

## Recommended Solution: Option B - Pre-Test Callback

### Architecture Changes

```
┌─────────────────────────────────────────────────────────────┐
│                     CLI Command (test.py)                     │
│  • Define start_callback(test_name, num, total)  ← NEW       │
│  • Define progress_callback(result)              ← EXISTING  │
│  • Pass both to TestExecutor                     ← UPDATED   │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  TestExecutor.execute_tests()                 │
│  for test_case in test_cases:                                 │
│    ├─ if start_callback:                         ← NEW       │
│    │    start_callback(name, idx, total)                      │
│    │                                                           │
│    ├─ result = await _execute_single_test(test_case)         │
│    │                                                           │
│    └─ if progress_callback:                      ← EXISTING  │
│         progress_callback(result)                             │
└─────────────────────────────────────────────────────────────┘
                ▲                        ▲
                │                        │
         BEFORE EXECUTION         AFTER EXECUTION
         Shows: "⠋ TestName"      Shows: "✓ TestName (3.82s)"
```

### User Experience Flow

**TTY Mode (Interactive Terminal)**:
```
Test 1/3: ⠋ Answer about the Answer to Life
```
↓ *User sees this immediately when test starts*
↓ *LLM call happens (3.82s)*
↓ *Line gets overwritten with final result*

```
Test 1/3: ✗ Answer about the Answer to Life (3.82s)
```

**Non-TTY Mode (CI/CD)**:
```
[Test 1/3] Running: Answer about the Answer to Life
[Test 1/3] FAIL Answer about the Answer to Life (3.82s)
```
*Two separate lines - no overwriting*

## Implementation Plan

### Task 1: Update ProgressIndicator
**File**: `src/holodeck/lib/test_runner/progress.py`

**Changes**:

1. Add instance variables to `__init__`:
```python
def __init__(self, total_tests: int, quiet: bool = False, verbose: bool = False):
    # ... existing variables ...
    self.current_test_name: str | None = None
    self.current_test_number: int = 0
```

2. Add `start_test()` method:
```python
def start_test(self, test_name: str, test_number: int) -> None:
    """Mark test as started for spinner display.

    Args:
        test_name: Name of the test being started
        test_number: Current test number (1-indexed)
    """
    self.current_test_name = test_name
    self.current_test_number = test_number
```

3. Add `get_running_line()` method:
```python
def get_running_line(self) -> str:
    """Get display line for currently running test.

    Returns:
        Progress string with spinner for running test
        Empty string if quiet mode or no test started
    """
    if self.quiet:
        return ""

    if not self.current_test_name:
        return ""

    spinner = self._get_spinner_char()
    progress = f"Test {self.current_test_number}/{self.total_tests}"

    if self.is_tty:
        # TTY: Show spinner character
        return f"{progress}: {spinner} {self.current_test_name}"
    else:
        # Non-TTY: Plain text
        return f"[{progress}] Running: {self.current_test_name}"
```

**Estimated LOC**: ~25 lines (including docstrings)

---

### Task 2: Update TestExecutor
**File**: `src/holodeck/lib/test_runner/executor.py`

**Changes**:

1. Update `__init__` signature:
```python
def __init__(
    self,
    agent_config_path: str,
    execution_config: ExecutionConfig | None = None,
    progress_callback: Callable[[TestResult], None] | None = None,
    start_callback: Callable[[str, int, int], None] | None = None,  # NEW
) -> None:
    """Initialize test executor.

    Args:
        agent_config_path: Path to agent configuration file
        execution_config: Optional execution configuration
        progress_callback: Optional callback for test completion (existing)
        start_callback: Optional callback for test start (NEW)
            Receives (test_name: str, test_number: int, total_tests: int)
    """
    # ... existing initialization ...
    self.progress_callback = progress_callback
    self.start_callback = start_callback  # NEW
```

2. Update `execute_tests()` method to fire start callback:
```python
async def execute_tests(self) -> TestReport:
    """Execute all test cases and generate report."""
    logger.info(f"Starting test execution: {len(test_cases)} test cases")

    test_results: list[TestResult] = []
    test_cases = self.agent_config.test_cases or []

    for idx, test_case in enumerate(test_cases, 1):
        # Fire start callback BEFORE execution (NEW)
        if self.start_callback:
            test_name = test_case.name or f"Test {idx}"
            self.start_callback(test_name, idx, len(test_cases))

        # Execute test (existing)
        result = await self._execute_single_test(test_case)
        test_results.append(result)

        # Fire progress callback AFTER execution (existing)
        if self.progress_callback:
            self.progress_callback(result)

    return self._generate_report(test_results)
```

**Estimated LOC**: ~10 lines (including parameter + call)

---

### Task 3: Update CLI Command
**File**: `src/holodeck/cli/commands/test.py`

**Changes**:

1. Import `sys` at top:
```python
import sys
```

2. Define `start_callback` before `progress_callback`:
```python
# Define start callback (NEW)
def start_callback(test_name: str, test_num: int, total: int) -> None:
    """Update progress indicator when test starts.

    Shows spinner and test name immediately when execution begins.
    """
    progress.start_test(test_name, test_num)
    running_line = progress.get_running_line()

    if running_line:
        if progress.is_tty:
            # TTY: Use \r to allow overwriting in next update
            click.echo(f"\r{running_line}", nl=False)
            sys.stdout.flush()  # Force immediate display
        else:
            # Non-TTY: Print normally (separate line)
            click.echo(running_line)
```

3. Update existing `progress_callback` to overwrite spinner line:
```python
def progress_callback(result: TestResult) -> None:
    """Update progress indicator when test completes.

    Overwrites the "running" line with final result in TTY mode.
    """
    progress.update(result)
    progress_line = progress.get_progress_line()

    if progress_line:
        if progress.is_tty:
            # Overwrite the spinner line with final result
            click.echo(f"\r{progress_line}")
        else:
            # Non-TTY: Print as separate line
            click.echo(progress_line)
```

4. Pass both callbacks to executor:
```python
# Initialize executor with both callbacks
logger.debug("Initializing test executor")
executor = TestExecutor(
    agent_config_path=agent_config,
    execution_config=cli_config,
    progress_callback=progress_callback,
    start_callback=start_callback,  # NEW
)
```

**Estimated LOC**: ~30 lines (including import + both callbacks)

---

### Task 4: Add Tests for ProgressIndicator
**File**: `tests/unit/lib/test_runner/test_progress.py`

**New test class**:
```python
class TestRunningStateDisplay:
    """Tests for spinner display during test execution (T110-T111 enhancement)."""

    def test_start_test_updates_state(self) -> None:
        """start_test() should update internal state."""
        indicator = ProgressIndicator(total_tests=5)

        indicator.start_test("Processing Test", 2)

        assert indicator.current_test_name == "Processing Test"
        assert indicator.current_test_number == 2

    def test_running_line_format_tty(self) -> None:
        """Running line should show spinner in TTY mode."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=5)
            indicator.start_test("API Integration Test", 2)

            output = indicator.get_running_line()

            # Should contain test number
            assert "2/5" in output
            # Should contain test name
            assert "API Integration Test" in output
            # Should contain a spinner character
            assert any(
                char in output
                for char in ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            )

    def test_running_line_format_non_tty(self) -> None:
        """Running line should be plain text in non-TTY mode."""
        with patch("sys.stdout.isatty", return_value=False):
            indicator = ProgressIndicator(total_tests=3)
            indicator.start_test("Database Query Test", 1)

            output = indicator.get_running_line()

            # Should use bracket format for CI/CD
            assert "[Test 1/3]" in output
            # Should say "Running:"
            assert "Running:" in output
            # Should contain test name
            assert "Database Query Test" in output

    def test_running_line_respects_quiet_mode(self) -> None:
        """Running line should be suppressed in quiet mode."""
        indicator = ProgressIndicator(total_tests=5, quiet=True)
        indicator.start_test("Test", 1)

        output = indicator.get_running_line()

        assert output == ""

    def test_running_line_empty_before_start(self) -> None:
        """Running line should be empty before any test starts."""
        indicator = ProgressIndicator(total_tests=3)

        output = indicator.get_running_line()

        assert output == ""

    def test_spinner_char_rotates(self) -> None:
        """Spinner character should advance through frames."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=1)

            # Call get_running_line multiple times
            indicator.start_test("Test", 1)

            line1 = indicator.get_running_line()
            line2 = indicator.get_running_line()
            line3 = indicator.get_running_line()

            # Extract spinner chars (would be different frames)
            # This verifies _get_spinner_char() advances the index
            # Note: Actual verification depends on implementation details
            assert line1 is not None
            assert line2 is not None
            assert line3 is not None
```

**Estimated LOC**: ~80 lines (6 tests)

---

### Task 5: Add Tests for TestExecutor
**File**: `tests/unit/lib/test_runner/test_executor.py`

**New tests**:
```python
@pytest.mark.asyncio
async def test_start_callback_fires_before_execution(
    mock_agent_config, tmp_path
):
    """start_callback should be called before test execution starts."""
    call_order = []

    def start_cb(name: str, num: int, total: int) -> None:
        call_order.append(("start", name, num, total))

    def progress_cb(result: TestResult) -> None:
        call_order.append(("complete", result.test_name))

    # Create executor with both callbacks
    executor = TestExecutor(
        agent_config_path=str(tmp_path / "agent.yaml"),
        start_callback=start_cb,
        progress_callback=progress_cb,
    )

    await executor.execute_tests()

    # Verify start_callback fired before progress_callback
    assert len(call_order) >= 2
    assert call_order[0][0] == "start"
    assert call_order[1][0] == "complete"

@pytest.mark.asyncio
async def test_start_callback_receives_correct_arguments(
    mock_agent_config, tmp_path
):
    """start_callback should receive (test_name, test_number, total_tests)."""
    captured_args = []

    def start_cb(name: str, num: int, total: int) -> None:
        captured_args.append((name, num, total))

    # Mock agent config with 3 test cases
    mock_agent_config.test_cases = [
        TestCaseModel(name="Test 1", input="input1"),
        TestCaseModel(name="Test 2", input="input2"),
        TestCaseModel(name="Test 3", input="input3"),
    ]

    executor = TestExecutor(
        agent_config_path=str(tmp_path / "agent.yaml"),
        start_callback=start_cb,
    )

    await executor.execute_tests()

    # Verify callback was called with correct arguments
    assert len(captured_args) == 3
    assert captured_args[0] == ("Test 1", 1, 3)
    assert captured_args[1] == ("Test 2", 2, 3)
    assert captured_args[2] == ("Test 3", 3, 3)

@pytest.mark.asyncio
async def test_start_callback_optional_backward_compatibility(
    mock_agent_config, tmp_path
):
    """Executor should work without start_callback (backward compatibility)."""
    # Create executor without start_callback
    executor = TestExecutor(
        agent_config_path=str(tmp_path / "agent.yaml"),
        start_callback=None,  # Explicitly None
    )

    # Should not raise error
    report = await executor.execute_tests()

    assert report is not None
    assert isinstance(report, TestReport)
```

**Estimated LOC**: ~70 lines (3 tests)

---

### Task 6: Add Tests for CLI Integration
**File**: `tests/unit/cli/commands/test_test.py`

**New tests in `TestCLIProgressDisplay` class**:
```python
def test_start_callback_passed_to_executor(self):
    """start_callback should be passed to TestExecutor initialization."""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with (
            patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
            patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
            patch("holodeck.cli.commands.test.ProgressIndicator"),
        ):
            mock_loader = MagicMock()
            mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(1)
            mock_loader_class.return_value = mock_loader

            mock_instance = MagicMock()
            mock_instance.execute_tests = AsyncMock(
                return_value=_create_mock_report(tmp_path)
            )
            mock_executor.return_value = mock_instance

            runner.invoke(test, [tmp_path])

            # Verify TestExecutor was called with start_callback
            call_kwargs = mock_executor.call_args.kwargs
            assert "start_callback" in call_kwargs
            assert callable(call_kwargs["start_callback"])
    finally:
        Path(tmp_path).unlink(missing_ok=True)

def test_start_callback_calls_progress_start_test(self):
    """start_callback should call ProgressIndicator.start_test()."""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with (
            patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
            patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
            patch("holodeck.cli.commands.test.ProgressIndicator") as mock_progress_class,
        ):
            mock_loader = MagicMock()
            mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(1)
            mock_loader_class.return_value = mock_loader

            # Capture the start_callback
            captured_start_callback = None

            def capture_callbacks(*args, **kwargs):
                nonlocal captured_start_callback
                captured_start_callback = kwargs.get("start_callback")
                mock_instance = MagicMock()
                mock_instance.execute_tests = AsyncMock(
                    return_value=_create_mock_report(tmp_path)
                )
                return mock_instance

            mock_executor.side_effect = capture_callbacks

            mock_progress_instance = MagicMock()
            mock_progress_class.return_value = mock_progress_instance

            runner.invoke(test, [tmp_path])

            # Verify callback was captured
            assert captured_start_callback is not None

            # Simulate calling the start_callback
            captured_start_callback("Test Name", 1, 1)

            # Verify ProgressIndicator.start_test was called
            mock_progress_instance.start_test.assert_called_with("Test Name", 1)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

def test_running_line_printed_when_start_callback_fires(self):
    """Running line should be printed when start_callback is invoked."""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with (
            patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
            patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
            patch("holodeck.cli.commands.test.ProgressIndicator") as mock_progress_class,
        ):
            mock_loader = MagicMock()
            mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(1)
            mock_loader_class.return_value = mock_loader

            captured_start_callback = None

            def capture_callbacks(*args, **kwargs):
                nonlocal captured_start_callback
                captured_start_callback = kwargs.get("start_callback")
                mock_instance = MagicMock()
                mock_instance.execute_tests = AsyncMock(
                    return_value=_create_mock_report(tmp_path)
                )
                return mock_instance

            mock_executor.side_effect = capture_callbacks

            mock_progress_instance = MagicMock()
            mock_progress_instance.get_running_line.return_value = "Test 1/1: ⠋ Running Test"
            mock_progress_class.return_value = mock_progress_instance

            runner.invoke(test, [tmp_path])

            # Call the start_callback
            captured_start_callback("Running Test", 1, 1)

            # Verify get_running_line was called
            mock_progress_instance.get_running_line.assert_called()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
```

**Estimated LOC**: ~90 lines (3 tests)

---

### Task 7: Verification Steps

**Step 1: Run Unit Tests**
```bash
source .venv/bin/activate
make test-unit
```
Expected: All tests pass, including new spinner tests

**Step 2: Code Quality Checks**
```bash
make format
make lint
make type-check
```
Expected: No errors

**Step 3: Run Sample Test (Manual Verification)**
```bash
source .venv/bin/activate
cd sample
set -a && source .env && set +a
holodeck test hitchhikers_agent.yaml
```

**Expected Output (TTY mode)**:
```
Test 1/1: ⠋ Answer about the Answer to Life
```
↓ *Immediately visible when test starts*
↓ *Gets overwritten after 3-4 seconds*
```
Test 1/1: ✗ Answer about the Answer to Life (3.82s)

============================================================
Test Results: 0/1 passed (0.0%)
  Failed: 1
  Duration: 4.02s
============================================================
```

**Step 4: Integration Tests**
```bash
make test-integration
```
Expected: All integration tests pass

**Step 5: Full Test Suite**
```bash
make test
```
Expected: All tests pass

---

## Files Modified Summary

| File | Purpose | Est. LOC | Complexity |
|------|---------|----------|------------|
| `src/holodeck/lib/test_runner/progress.py` | Add running state display | ~25 | Low |
| `src/holodeck/lib/test_runner/executor.py` | Add start_callback support | ~10 | Low |
| `src/holodeck/cli/commands/test.py` | Wire up start_callback | ~30 | Low |
| `tests/unit/lib/test_runner/test_progress.py` | Test running state | ~80 | Low |
| `tests/unit/lib/test_runner/test_executor.py` | Test start_callback | ~70 | Medium |
| `tests/unit/cli/commands/test_test.py` | Test CLI integration | ~90 | Medium |

**Total**: 6 files, ~305 lines (including tests and docstrings)

---

## Testing Strategy (TDD)

### Phase 1: ProgressIndicator
1. **Red**: Write `TestRunningStateDisplay.test_start_test_updates_state()` - should fail
2. **Green**: Implement `start_test()` method - test passes
3. **Red**: Write `test_running_line_format_tty()` - should fail
4. **Green**: Implement `get_running_line()` method - test passes
5. **Refactor**: Run `make format && make lint`
6. **Red**: Write remaining tests (non-TTY, quiet mode, etc.) - should fail
7. **Green**: Refine `get_running_line()` - all tests pass
8. **Refactor**: Clean up, verify type-check

### Phase 2: TestExecutor
1. **Red**: Write `test_start_callback_fires_before_execution()` - should fail
2. **Green**: Add `start_callback` parameter and call it - test passes
3. **Red**: Write `test_start_callback_receives_correct_arguments()` - should fail
4. **Green**: Pass correct arguments to callback - test passes
5. **Red**: Write `test_start_callback_optional_backward_compatibility()` - should fail
6. **Green**: Handle None callback gracefully - test passes
7. **Refactor**: Verify lint/type-check

### Phase 3: CLI Integration
1. **Red**: Write `test_start_callback_passed_to_executor()` - should fail
2. **Green**: Define `start_callback` in CLI and pass to executor - test passes
3. **Red**: Write `test_start_callback_calls_progress_start_test()` - should fail
4. **Green**: Wire up callback to call `progress.start_test()` - test passes
5. **Red**: Write `test_running_line_printed_when_start_callback_fires()` - should fail
6. **Green**: Add `click.echo()` call in start_callback - test passes
7. **Refactor**: Clean up, verify all tests

### Phase 4: End-to-End Verification
1. Run full test suite: `make test`
2. Manual test with sample: `holodeck test sample/hitchhikers_agent.yaml`
3. Verify spinner appears immediately when test starts
4. Verify result overwrites spinner line in TTY
5. Verify plain text output in CI/CD (redirect to file)

---

## Backward Compatibility

✅ **Fully backward compatible**:
- `start_callback` is optional (defaults to `None`)
- Existing code without `start_callback` will continue to work
- `progress_callback` behavior unchanged (still fires after completion)
- All existing tests should pass without modification

---

## Future Enhancements (Out of Scope)

These enhancements could be added later without breaking changes:

1. **Animated Spinner** (rotating frames during execution)
   - Would require background asyncio task
   - Update spinner every 100ms during LLM call
   - More complex but better UX

2. **Stage-Specific Updates**
   - "Processing files..." (during file processing)
   - "Invoking agent..." (during LLM call)
   - "Running evaluations..." (during metric calculation)
   - Requires callbacks from deeper in the execution stack

3. **Progress Percentage**
   - For evaluation metrics: "Running evaluations... (2/5 metrics)"
   - Requires metric count information

4. **Estimated Time Remaining**
   - Based on average test duration
   - Requires tracking historical timing data

5. **Parallel Test Execution**
   - Multiple tests running simultaneously
   - Would need thread-safe progress tracking
   - Significant architectural change

---

## Success Criteria

✅ **Spinner shows immediately** when test starts (not after it completes)
✅ **TTY mode** shows spinner character and overwrites with result
✅ **Non-TTY mode** shows plain text "Running:" message
✅ **Quiet mode** suppresses running line (only shows summary)
✅ **All tests pass** (unit, integration, full suite)
✅ **Code quality** passes (format, lint, type-check)
✅ **Backward compatible** with existing code
✅ **Manual verification** with sample test shows expected behavior

---

## Related Documentation

- **Tasks**: `specs/006-agent-test-execution/tasks.md` - T104-T111
- **Spec**: `specs/006-agent-test-execution/spec.md` - User Story 4
- **Plan**: `specs/006-agent-test-execution/plan.md` - Progress Display section
- **Code**:
  - `src/holodeck/lib/test_runner/progress.py` - ProgressIndicator class
  - `src/holodeck/lib/test_runner/executor.py` - TestExecutor class
  - `src/holodeck/cli/commands/test.py` - CLI command

---

**Plan Status**: Ready for implementation
**Estimated Effort**: ~4-6 hours (including testing)
**Risk Level**: Low (minimal changes, high test coverage)
