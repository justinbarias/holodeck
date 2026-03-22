"""HoloDeck deployment engine.

This package provides the deployment functionality for HoloDeck agents,
including Dockerfile generation, container building, and cloud deployment.

Note: ContainerBuilder and BuildResult require the 'docker' optional
dependency (pip install holodeck-ai[deploy]). They are lazily imported
to avoid ImportError when docker is not installed.
"""

from holodeck.deploy.builder import generate_tag, get_oci_labels
from holodeck.deploy.dockerfile import generate_dockerfile


def __getattr__(name: str) -> object:
    """Lazy import for docker-dependent symbols."""
    if name in ("BuildResult", "ContainerBuilder"):
        from holodeck.deploy.builder import BuildResult, ContainerBuilder

        return {"BuildResult": BuildResult, "ContainerBuilder": ContainerBuilder}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BuildResult",
    "ContainerBuilder",
    "generate_dockerfile",
    "generate_tag",
    "get_oci_labels",
]
