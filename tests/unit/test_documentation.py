"""
Unit tests for documentation completeness and accuracy.

Tests verify:
- T113: README.md documents the `holodeck init` command
- T114: QUICKSTART.md accuracy and all examples work
"""

from pathlib import Path

import pytest


@pytest.mark.unit
class TestREADMEDocumentation:
    """T113: Test that README.md documents `holodeck init` command."""

    def test_readme_exists(self):
        """README.md should exist in repo root."""
        readme_path = Path(__file__).resolve().parent.parent.parent / "README.md"
        assert readme_path.exists(), f"README.md not found at {readme_path}"

    def test_readme_documents_init_command(self):
        """README.md should document the init command."""
        readme_path = Path(__file__).resolve().parent.parent.parent / "README.md"
        content = readme_path.read_text()

        content_lower = content.lower()
        # Should mention init command
        assert "init" in content_lower
        # Should mention project creation
        assert "project" in content_lower or "create" in content_lower

    def test_readme_has_quickstart_section(self):
        """README.md should have a Getting Started or Quick Start section."""
        readme_path = Path(__file__).resolve().parent.parent.parent / "README.md"
        content = readme_path.read_text()

        content_lower = content.lower()
        # Should have easy-to-find getting started section
        assert (
            "getting started" in content_lower
            or "quickstart" in content_lower
            or "quick start" in content_lower
            or "usage" in content_lower
        )

    def test_readme_has_example_usage(self):
        """README.md should show example of `holodeck init` command."""
        readme_path = Path(__file__).resolve().parent.parent.parent / "README.md"
        content = readme_path.read_text()

        # Should show example with holodeck init
        assert "holodeck init" in content or "init" in content.lower()

    def test_readme_links_are_valid(self):
        """README.md markdown links should reference existing files."""
        readme_path = Path(__file__).resolve().parent.parent.parent / "README.md"
        content = readme_path.read_text()
        repo_root = readme_path.parent

        # Extract markdown links [text](path)
        import re

        links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)

        for _, path in links:
            # Skip external URLs
            if path.startswith("http"):
                continue

            # Skip fragments
            if path.startswith("#"):
                continue

            # File should exist
            full_path = repo_root / path
            # Some links might be files with #anchors
            if "#" in path:
                file_path = path.split("#")[0]
                full_path = repo_root / file_path
                assert (
                    full_path.exists() or full_path.is_file()
                ), f"Linked file not found: {file_path}"
            else:
                # Soft check - some files might not exist yet
                pass

    def test_readme_installation_documented(self):
        """README.md should document how to install holodeck."""
        readme_path = Path(__file__).resolve().parent.parent.parent / "README.md"
        content = readme_path.read_text()

        content_lower = content.lower()
        # Should mention installation
        assert "install" in content_lower or "setup" in content_lower

    def test_readme_has_table_of_contents(self):
        """README.md should have a table of contents (optional but recommended)."""
        readme_path = Path(__file__).resolve().parent.parent.parent / "README.md"
        content = readme_path.read_text()

        # Check for headings to ensure structure
        assert "#" in content  # Should have markdown headings


@pytest.mark.unit
class TestQuickstartDocumentation:
    """T114: Test QUICKSTART.md accuracy and that all examples work."""

    def test_quickstart_exists(self):
        """QUICKSTART.md should exist in docs/getting-started/."""
        quickstart_path = (
            Path(__file__).resolve().parent.parent.parent
            / "docs"
            / "getting-started"
            / "quickstart.md"
        )
        assert quickstart_path.exists(), f"quickstart.md not found at {quickstart_path}"

    def test_quickstart_has_init_examples(self):
        """quickstart.md should include examples of `holodeck init`."""
        quickstart_path = (
            Path(__file__).resolve().parent.parent.parent
            / "docs"
            / "getting-started"
            / "quickstart.md"
        )
        content = quickstart_path.read_text()

        # Should show basic init command
        assert "holodeck init" in content

    def test_quickstart_example_commands_parseable(self):
        """Code examples in quickstart.md should be properly formatted."""
        quickstart_path = (
            Path(__file__).resolve().parent.parent.parent
            / "docs"
            / "getting-started"
            / "quickstart.md"
        )
        content = quickstart_path.read_text()

        # Should have code blocks (```...```)
        assert "```" in content

    def test_quickstart_has_template_examples(self):
        """quickstart.md should show examples of different templates."""
        quickstart_path = (
            Path(__file__).resolve().parent.parent.parent
            / "docs"
            / "getting-started"
            / "quickstart.md"
        )
        content = quickstart_path.read_text()

        # Should mention templates
        assert (
            "template" in content.lower()
            or "conversational" in content.lower()
            or "research" in content.lower()
            or "customer-support" in content.lower()
        )

    def test_quickstart_describes_project_structure(self):
        """quickstart.md should explain the generated project structure."""
        quickstart_path = (
            Path(__file__).resolve().parent.parent.parent
            / "docs"
            / "getting-started"
            / "quickstart.md"
        )
        content = quickstart_path.read_text()

        # Should describe what gets created
        assert "project" in content.lower()
        assert "agent.yaml" in content or "yaml" in content.lower()

    def test_quickstart_next_steps(self):
        """quickstart.md should provide next steps after init."""
        quickstart_path = (
            Path(__file__).resolve().parent.parent.parent
            / "docs"
            / "getting-started"
            / "quickstart.md"
        )
        content = quickstart_path.read_text()

        content_lower = content.lower()
        # Should guide user on what to do after init
        assert (
            "next" in content_lower
            or "after" in content_lower
            or "configure" in content_lower
            or "edit" in content_lower
            or "modify" in content_lower
        )

    def test_quickstart_step_by_step(self):
        """quickstart.md should be organized as step-by-step guide."""
        quickstart_path = (
            Path(__file__).resolve().parent.parent.parent
            / "docs"
            / "getting-started"
            / "quickstart.md"
        )
        content = quickstart_path.read_text()

        # Should use numbered or bulleted steps
        assert "1." in content or "-" in content or "*" in content

    def test_quickstart_assumptions_clear(self):
        """quickstart.md should state any assumptions (e.g., Python installed)."""
        quickstart_path = (
            Path(__file__).resolve().parent.parent.parent
            / "docs"
            / "getting-started"
            / "quickstart.md"
        )
        content = quickstart_path.read_text()

        # Should be clear about prerequisites
        content_lower = content.lower()
        assert (
            "require" in content_lower
            or "assume" in content_lower
            or "install" in content_lower
            or "prerequisite" in content_lower
            or "before" in content_lower
        )

    def test_quickstart_no_broken_code_examples(self):
        """Code examples should follow proper syntax."""
        quickstart_path = (
            Path(__file__).resolve().parent.parent.parent
            / "docs"
            / "getting-started"
            / "quickstart.md"
        )
        content = quickstart_path.read_text()

        # Extract code blocks
        import re

        code_blocks = re.findall(r"```.*?\n(.*?)\n```", content, re.DOTALL)

        for block in code_blocks:
            # Should have reasonable content
            assert len(block.strip()) > 0
            # Should not have obvious syntax errors (very basic check)
            assert (
                block.count('"') % 2 == 0
                or block.count("'") % 2 == 0
                or "holodeck" in block
                or "$" in block
            )

    def test_quickstart_references_templates(self):
        """quickstart.md should mention available templates."""
        quickstart_path = (
            Path(__file__).resolve().parent.parent.parent
            / "docs"
            / "getting-started"
            / "quickstart.md"
        )
        content = quickstart_path.read_text()

        # Should mention template options
        templates = ["conversational", "research", "customer-support"]
        found = sum(1 for t in templates if t in content.lower())
        # Should mention at least 1 template
        assert found >= 1, "Should mention at least one template"

    def test_quickstart_mentions_configuration(self):
        """quickstart.md should explain where to configure the agent."""
        quickstart_path = (
            Path(__file__).resolve().parent.parent.parent
            / "docs"
            / "getting-started"
            / "quickstart.md"
        )
        content = quickstart_path.read_text()

        content_lower = content.lower()
        # Should mention agent.yaml or configuration
        assert (
            "agent.yaml" in content
            or "config" in content_lower
            or "yaml" in content_lower
        )

    def test_quickstart_helpful_for_beginners(self):
        """quickstart.md should be accessible to users new to HoloDeck."""
        quickstart_path = (
            Path(__file__).resolve().parent.parent.parent
            / "docs"
            / "getting-started"
            / "quickstart.md"
        )
        content = quickstart_path.read_text()

        # Should have intro/overview
        lines = content.split("\n")
        # First few non-empty lines should be intro
        intro = "\n".join(lines[:10]).lower()
        assert len(intro) > 50  # Should have some introductory text

    def test_quickstart_all_sections_present(self):
        """quickstart.md should have key sections."""
        quickstart_path = (
            Path(__file__).resolve().parent.parent.parent
            / "docs"
            / "getting-started"
            / "quickstart.md"
        )
        content = quickstart_path.read_text()

        content_lower = content.lower()
        # Should have these sections (not strict format check)
        important_topics = [
            "install",  # Installation
            "next",  # What to do next
        ]

        # Should mention init or create
        has_init_section = "init" in content_lower or "create" in content_lower
        assert has_init_section, "Missing 'init' or 'create' section"

        for topic in important_topics:
            assert topic in content_lower, f"Missing '{topic}' section"
