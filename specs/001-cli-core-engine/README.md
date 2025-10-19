# v0.1: CLI & Core Agent Engine - Specification

This directory contains the specification for HoloDeck v0.1, which introduces the foundational CLI commands and core agent execution engine.

## Files

- **spec.md** - Complete feature specification with tool type breakdown
- **checklists/requirements.md** - Quality validation checklist

## Overview

HoloDeck v0.1 delivers the essential tools for developers to:

1. Initialize new agent projects via CLI
2. Define agents entirely through YAML configuration
3. Test agents against predefined test cases with evaluation metrics
4. Chat interactively with agents for debugging
5. Deploy agents as local FastAPI services

## Key Statistics

| Metric                  | Count                                                   |
| ----------------------- | ------------------------------------------------------- |
| User Stories            | 11 (7 base + 4 vectorstore sub-stories; 9 P1, 2 P2)     |
| Functional Requirements | 35 (9 core + 4 vectorstore sub-requirements + 22 other) |
| Success Criteria        | 10                                                      |
| Edge Cases              | 8                                                       |
| Acceptance Scenarios    | 50+ (comprehensive tool coverage)                       |
| Core Entities           | 5                                                       |

## Feature Scope

### In Scope (v0.1)

- CLI commands: `init`, `test`, `chat`, `deploy`, `--version`
- Agent YAML configuration with multiple LLM providers
- Tool system with four explicit types:
  - **Vectorstore tools** (type: vectorstore) - Semantic search with dual support:
    - **Unstructured data**: Single files, directories, multiple text formats (.txt, .md, .pdf)
    - **Structured data**: CSV, JSON with field mapping, metadata handling, and pandas.json_normalize pattern support
  - **Function tools** (type: function) - Custom Python functions from tools/ directory
  - **MCP tools** (type: mcp) - Model Context Protocol server integrations
  - **Prompt-based tools** (type: prompt) - LLM-powered semantic functions with template substitution
- Testing framework with test case execution
- Evaluation metrics: AI-powered (groundedness, relevance) and NLP-based (F1, BLEU, ROUGE)
- Local API deployment with FastAPI
- Comprehensive error handling and validation

### Out of Scope (v0.2+)

- Cloud deployment (Azure, AWS, GCP)
- Multi-agent orchestration (sequential, concurrent, handoff, group chat, magentic)
- Web UI for no-code editing
- Advanced RAG and chunking strategies
- API authentication and rate limiting
- Enterprise features (SSO, RBAC, audit logs)
- Plugin marketplace
- OpenTelemetry observability integration
- Multimodal test inputs (images, PDFs, Office docs)

## Priority Breakdown

**Foundation (P1) - Must Have**

1. Project initialization
2. Agent configuration via YAML
3. Test case execution
4. Tool operations with explicit types:
   - Vector search tools (type: vectorstore)
   - Custom function tools (type: function)
   - MCP tools (type: mcp)
   - Prompt-based tools (type: prompt)
5. System instructions loading

**Enhancement (P2) - Nice to Have** 6. Interactive chat mode 7. Local API deployment

## Success Criteria Highlights

- **Simplicity**: Developers define agents in YAML without writing code
- **Speed**: Project init < 30 seconds, test execution < 5 seconds per test
- **Accuracy**: Tool tracking with 100% accuracy
- **Usability**: Clear, actionable error messages
- **Performance**: Chat responses < 3 seconds, API supports 10+ concurrent requests

## Status

✅ **Specification Complete and Validated**

All mandatory sections completed. No clarifications needed. Ready for planning phase.

Run `/speckit.plan` to generate implementation plan and task breakdown.

---

Created: 2025-10-19
Branch: 001-cli-core-engine
