"""Offline function tool for the verify suite agents."""


def smoke_echo(text: str) -> str:
    """Return a fixed marker plus the given text.

    Args:
        text: Any short string.
    """
    return f"SMOKE-TOOL-RESULT: {text}"
