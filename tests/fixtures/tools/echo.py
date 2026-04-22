"""Echo function-tool fixture used by FunctionTool runtime tests."""


def echo(message: str) -> str:
    """Return the input message unchanged.

    Args:
        message: The string to echo back.

    Returns:
        The same string, unchanged.
    """
    return message


async def async_echo(message: str) -> str:
    """Async variant of echo — used to cover the coroutine branch in the adapter.

    Args:
        message: The string to echo back.

    Returns:
        The same string, unchanged.
    """
    return message


NOT_CALLABLE = "I am a string, not a function."
