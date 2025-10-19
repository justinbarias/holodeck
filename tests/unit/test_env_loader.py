"""Tests for environment variable loading and substitution."""

import os

import pytest

from holodeck.config.env_loader import substitute_env_vars
from holodeck.lib.errors import ConfigError


class TestSubstituteEnvVars:
    """Tests for substitute_env_vars() function."""

    def test_substitute_env_vars_with_existing_var(self) -> None:
        """Test substitution with existing environment variable."""
        os.environ["TEST_API_KEY"] = "secret-key-123"
        text = "api_key: ${TEST_API_KEY}"
        result = substitute_env_vars(text)
        assert result == "api_key: secret-key-123"

    def test_substitute_env_vars_multiple_vars(self) -> None:
        """Test substitution with multiple environment variables."""
        os.environ["DB_HOST"] = "localhost"
        os.environ["DB_PORT"] = "5432"
        text = "connection: ${DB_HOST}:${DB_PORT}"
        result = substitute_env_vars(text)
        assert result == "connection: localhost:5432"

    def test_substitute_env_vars_with_missing_var_raises_error(self) -> None:
        """Test that missing env var raises ConfigError."""
        if "NONEXISTENT_VAR_XYZ" in os.environ:
            del os.environ["NONEXISTENT_VAR_XYZ"]
        text = "value: ${NONEXISTENT_VAR_XYZ}"
        with pytest.raises(ConfigError) as exc_info:
            substitute_env_vars(text)
        assert "NONEXISTENT_VAR_XYZ" in str(exc_info.value)

    def test_substitute_env_vars_no_vars_returns_unchanged(self) -> None:
        """Test that text without vars is returned unchanged."""
        text = "plain text without vars"
        result = substitute_env_vars(text)
        assert result == text

    def test_substitute_env_vars_pattern_matching(self) -> None:
        """Test that only ${VAR_NAME} pattern is recognized."""
        os.environ["VALID_VAR"] = "value"
        text = "${VALID_VAR} and $INVALID_VAR and {ALSO_INVALID}"
        result = substitute_env_vars(text)
        assert result == "value and $INVALID_VAR and {ALSO_INVALID}"

    def test_substitute_env_vars_with_empty_string(self) -> None:
        """Test substitution with empty environment variable."""
        os.environ["EMPTY_VAR"] = ""
        text = "value: ${EMPTY_VAR}"
        result = substitute_env_vars(text)
        assert result == "value: "

    def test_substitute_env_vars_preserves_special_chars(self) -> None:
        """Test that special characters in env values are preserved."""
        os.environ["SPECIAL_VAR"] = "value-with-special!@#$%"
        text = "data: ${SPECIAL_VAR}"
        result = substitute_env_vars(text)
        assert result == "data: value-with-special!@#$%"

    def test_substitute_env_vars_case_sensitive(self) -> None:
        """Test that variable names are case-sensitive."""
        os.environ["TEST_VAR"] = "uppercase"
        os.environ["TEST_VAR_LOWER"] = "lowercase"
        text = "${TEST_VAR} and ${TEST_VAR_LOWER}"
        result = substitute_env_vars(text)
        assert result == "uppercase and lowercase"

    def test_substitute_env_vars_yaml_format(self) -> None:
        """Test substitution in typical YAML format."""
        os.environ["OPENAI_API_KEY"] = "sk-12345"
        os.environ["OPENAI_ORG_ID"] = "org-67890"
        yaml_content = """
model:
  provider: openai
  api_key: ${OPENAI_API_KEY}
  org_id: ${OPENAI_ORG_ID}
"""
        result = substitute_env_vars(yaml_content)
        assert "sk-12345" in result
        assert "org-67890" in result
        assert "${OPENAI_API_KEY}" not in result

    def test_substitute_env_vars_with_spaces_in_var_value(self) -> None:
        """Test substitution where env value contains spaces."""
        os.environ["DESCRIPTION"] = "This is a long description with spaces"
        text = "description: ${DESCRIPTION}"
        result = substitute_env_vars(text)
        assert result == "description: This is a long description with spaces"

    def test_substitute_env_vars_adjacent_vars(self) -> None:
        """Test substitution with adjacent variables."""
        os.environ["VAR1"] = "first"
        os.environ["VAR2"] = "second"
        text = "${VAR1}${VAR2}"
        result = substitute_env_vars(text)
        assert result == "firstsecond"

    def test_substitute_env_vars_ignores_nested_patterns(self) -> None:
        """Test that nested patterns are not recursively substituted.

        Test that ${${INNER}} patterns are not recursively substituted.
        """
        # This is a realistic edge case: malformed variable reference
        text = "value: ${OUTER_${INNER}}"
        # The regex matches ${OUTER_${INNER}} but the var name contains $,
        # which doesn't match [A-Za-z_], so this should fail to find a
        # matching variable
        with pytest.raises(ConfigError):
            substitute_env_vars(text)
