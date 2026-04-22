"""Integration test: US2 does NOT alter the string handed to the LLM.

T115 (031-eval-runs-dashboard US2, FR-015). The backend-facing prompt string
is produced by :func:`holodeck.lib.instruction_resolver.resolve_instructions`.
US2 introduces a *separate*, additive ``resolve_prompt_version`` call; it must
not change the body that reaches the backend.

This test is a behavioural snapshot: for an instruction file with frontmatter,
the string returned by ``resolve_instructions`` remains byte-equivalent to the
raw file contents, unchanged by the landing of US2. Both SK and Claude
backends call this helper to feed their chat completion, so any regression
here would propagate to both.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from holodeck.lib.instruction_resolver import resolve_instructions
from holodeck.models.agent import Instructions


@pytest.mark.integration
def test_prompt_body_unchanged_by_us2(tmp_path: Path) -> None:
    raw = dedent("""\
        ---
        version: "1.2"
        author: jane
        tags:
          - support
        ---
        You are a helpful support agent.
        Always greet the user politely.
        """)
    md = tmp_path / "instructions.md"
    md.write_text(raw)

    body = resolve_instructions(Instructions(file="instructions.md"), base_dir=tmp_path)

    # The exact string handed to the backend: byte-equivalent to file contents.
    # Frontmatter is NOT stripped at resolve time — the split happens only
    # inside the (separate) resolve_prompt_version call.
    assert body == raw


@pytest.mark.integration
def test_inline_prompt_body_unchanged_by_us2() -> None:
    raw = "You are a helpful agent. Keep answers short."
    body = resolve_instructions(Instructions(inline=raw), base_dir=None)
    assert body == raw
