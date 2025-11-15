"""Test execution framework for HoloDeck agents.

This package orchestrates the complete test execution pipeline,
from configuration resolution through result reporting:

**Pipeline Flow:**
1. `executor`: Main orchestrator that coordinates the entire test flow
   - Resolves test configuration files
   - Processes multimodal file inputs (images, PDFs, Office docs)
   - Creates agent instances
   - Invokes agents with test inputs
   - Runs evaluations on outputs
   - Generates test reports

2. `agent_factory`: Creates agent instances from configuration
   - Resolves LLM provider settings
   - Loads and validates tool configurations
   - Instantiates agents with proper initialization

3. `progress`: Real-time test execution progress indicators
   - Shows test progress during batch execution
   - Displays success/failure statistics
   - Provides visual feedback to users

4. `reporter`: Test result reporting and formatting
   - Generates structured test reports
   - Supports multiple output formats (JSON, HTML, markdown)
   - Summarizes metrics and evaluation results

**Example Usage:**
    from holodeck.lib.test_runner.executor import TestExecutor
    from holodeck.config.loader import ConfigLoader

    loader = ConfigLoader()
    config = loader.load("agent.yaml")

    executor = TestExecutor()
    results = executor.run_tests(config)

    # results contains test execution outcomes and evaluation metrics

**Key Components:**
- Multimodal file handling (OCR, PDF parsing, Office extraction)
- Integration with evaluation framework
- Comprehensive error handling and logging
- Test result persistence and reporting

Classes:
    TestExecutor: Main test execution orchestrator
    AgentFactory: Agent instantiation from configuration
    ProgressIndicator: Real-time progress tracking
    TestReporter: Result report generation

Functions:
    execute_tests: Run test suite for an agent
    format_results: Format test results for output
    save_report: Persist test results to file
"""
