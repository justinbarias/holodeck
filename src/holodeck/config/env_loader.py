"""Environment variable substitution and loading utilities."""

import os
import re
from typing import Any

from holodeck.lib.errors import ConfigError


def substitute_env_vars(text: str) -> str:
    """Substitute environment variables in text using ${VAR_NAME} pattern.

    Replaces all occurrences of ${VAR_NAME} with the corresponding environment
    variable value. Raises ConfigError if a referenced variable does not exist.

    Args:
        text: Text potentially containing ${VAR_NAME} patterns

    Returns:
        Text with all environment variables substituted

    Raises:
        ConfigError: If a referenced environment variable does not exist

    Example:
        >>> import os
        >>> os.environ["API_KEY"] = "secret123"
        >>> substitute_env_vars("key: ${API_KEY}")
        'key: secret123'
    """
    # Pattern to match ${VAR_NAME} - captures alphanumeric, underscore
    pattern = r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}"

    def replace_var(match: re.Match[str]) -> str:
        """Replace a single ${VAR_NAME} pattern with env value.

        Args:
            match: Regex match object for ${VAR_NAME}

        Returns:
            Environment variable value

        Raises:
            ConfigError: If variable does not exist
        """
        var_name = match.group(1)
        if var_name not in os.environ:
            raise ConfigError(
                var_name,
                f"Environment variable '{var_name}' not found. "
                f"Please set it and try again.",
            )
        return os.environ[var_name]

    return re.sub(pattern, replace_var, text)


def get_env_var(key: str, default: Any = None) -> Any:
    """Get environment variable with optional default.

    Args:
        key: Environment variable name
        default: Default value if variable not set

    Returns:
        Environment variable value or default
    """
    return os.environ.get(key, default)


def load_env_file(path: str) -> dict[str, str]:
    """Load environment variables from a .env file.

    Args:
        path: Path to .env file

    Returns:
        Dictionary of loaded environment variables

    Raises:
        ConfigError: If file cannot be read
    """
    try:
        env_vars = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
        return env_vars
    except OSError as e:
        raise ConfigError("env_file", f"Cannot read environment file: {e}") from e
