"""Unit tests for the CLI root module (``holodeck.cli.main``)."""

import os


def test_grpc_verbosity_defaulted_to_error() -> None:
    """Importing the CLI entrypoint defaults gRPC's C-core to ERROR verbosity.

    This silences the INFO ``FD from fork parent still in poll list`` lines that
    grpcio prints straight to stderr after a fork (multiprocessing / pytest-xdist)
    when a gRPC channel exists. The default is applied at import time via
    ``os.environ.setdefault`` in ``holodeck.cli.main``.
    """
    import holodeck.cli.main  # noqa: F401  (import triggers the env defaults)

    assert os.environ.get("GRPC_VERBOSITY") == "ERROR"
