"""Generated Dockerfiles install an exact, constrained HoloDeck (spec 042, FR-029)."""

from __future__ import annotations

import re

import pytest

from holodeck.deploy.dockerfile import (
    CONSTRAINTS_CONTEXT_FILE,
    MANIFEST_IMAGE_PATH,
    generate_dockerfile,
)


def _gen(**overrides: object) -> str:
    kwargs: dict = {
        "agent_name": "t",
        "port": 8080,
        "protocol": "rest",
        "holodeck_version": "1.2.3",
    }
    kwargs.update(overrides)
    return generate_dockerfile(**kwargs)


def _install_block(dockerfile: str) -> str:
    start = dockerfile.index("RUN uv pip install")
    return dockerfile[start : dockerfile.index("\n\n", start)]


@pytest.mark.unit
def test_version_is_required() -> None:
    with pytest.raises(TypeError):
        generate_dockerfile("t", 8080, "rest")  # type: ignore[call-arg]


@pytest.mark.unit
def test_exact_pin_with_extras_in_sorted_order() -> None:
    df = _gen(extras=["s3", "chromadb", "azure-blob", "chromadb"])

    assert '"holodeck-ai[azure-blob,chromadb,s3]==1.2.3"' in df
    assert 'LABEL com.holodeck.extras="azure-blob,chromadb,s3"' in df
    assert 'LABEL com.holodeck.version="1.2.3"' in df


@pytest.mark.unit
def test_exact_pin_without_extras_still_installs_the_version() -> None:
    df = _gen()

    assert '"holodeck-ai==1.2.3"' in df
    assert "holodeck-ai[" not in df


@pytest.mark.unit
def test_constraints_are_copied_and_used() -> None:
    df = _gen(extras=["qdrant"])
    block = _install_block(df)

    assert f"COPY {CONSTRAINTS_CONTEXT_FILE} /tmp/holodeck-constraints.txt" in df
    assert "--constraint /tmp/holodeck-constraints.txt" in block
    assert "COPY holodeck-overrides.txt /tmp/holodeck-overrides.txt" in df
    assert "--override /tmp/holodeck-overrides.txt" in block
    assert df.index("COPY holodeck-constraints.txt") < df.index("RUN uv pip install")


@pytest.mark.unit
def test_no_prerelease_resolution() -> None:
    df = _gen(extras=["qdrant", "openai-agents"])

    assert "--prerelease" not in df
    assert not re.search(r"--pre\b", df)


@pytest.mark.unit
def test_package_manifest_is_written_into_the_image() -> None:
    df = _gen(extras=["qdrant"])

    assert "uv pip freeze --system" in df
    assert f"> {MANIFEST_IMAGE_PATH}" in df
    assert "# holodeck-ai 1.2.3; extras: qdrant" in df


@pytest.mark.unit
def test_dev_version_installs_the_wheel_from_the_build_context() -> None:
    df = _gen(
        holodeck_version="0.7.2.dev61",
        holodeck_wheel="holodeck_ai-0.7.2.dev61-py3-none-any.whl",
        extras=["qdrant"],
    )
    block = _install_block(df)

    assert (
        "COPY holodeck_ai-0.7.2.dev61-py3-none-any.whl "
        "/tmp/wheels/holodeck_ai-0.7.2.dev61-py3-none-any.whl"
    ) in df
    assert '"/tmp/wheels/holodeck_ai-0.7.2.dev61-py3-none-any.whl[qdrant]"' in block
    assert "holodeck-ai[qdrant]==" not in df
    assert "--constraint" in block
    assert "--prerelease" not in df


@pytest.mark.unit
def test_release_build_has_no_wheel_copy() -> None:
    assert ".whl" not in _gen()


@pytest.mark.unit
def test_install_happens_before_the_user_switch() -> None:
    df = _gen(extras=["qdrant"])

    assert df.index("RUN uv pip install") < df.index("USER holodeck")


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"holodeck_version": "1.0; rm -rf /"},
        {"holodeck_version": ""},
        {"holodeck_wheel": "../evil.whl"},
        {"holodeck_wheel": "x.whl; id"},
        {"extras": ["qdrant]==9"]},
        {"extras": ["Qdrant"]},
    ],
)
def test_values_that_change_the_install_line_are_rejected(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _gen(**overrides)
