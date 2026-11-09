"""Function tools used only by the live verify agents (make verify-live)."""

# Built at import time so no credential-shaped literal sits in the source.
# It matches HoloDeck's github-token redaction pattern but is obviously fake.
FAKE_GITHUB_TOKEN = "ghp_" + "x" * 36
LEAK_PROBE_RESULT = "LEAK-PROBE-OK"


def leak_probe() -> str:
    """Return a fixed marker plus a fake GitHub token for the trace check.

    The exported span for this tool must show ``[REDACTED:github-token]``
    in place of the token.
    """
    return f"{LEAK_PROBE_RESULT} token={FAKE_GITHUB_TOKEN}"
