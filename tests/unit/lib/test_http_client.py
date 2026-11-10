"""HTTPX keeps the CA bundle and netrc settings that Requests honoured."""

from __future__ import annotations

import ssl
from pathlib import Path

import certifi
import httpx
import pytest

from holodeck.lib.http_client import netrc_auth, tls_verify


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "NETRC"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.mark.unit
def test_tls_verify_defaults_to_httpx_behaviour() -> None:
    assert tls_verify() is True


@pytest.mark.unit
@pytest.mark.parametrize("variable", ["REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"])
def test_tls_verify_uses_the_requests_ca_bundle(
    monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    monkeypatch.setenv(variable, certifi.where())
    assert isinstance(tls_verify(), ssl.SSLContext)


@pytest.mark.unit
def test_tls_verify_ignores_a_missing_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/does/not/exist.pem")
    assert tls_verify() is True


@pytest.mark.unit
def test_netrc_auth_is_absent_without_a_file() -> None:
    assert netrc_auth() is None


@pytest.mark.unit
def test_netrc_auth_sends_credentials_for_a_listed_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    netrc_file = tmp_path / "creds"
    netrc_file.write_text("machine files.example login alice password s3cret\n")
    netrc_file.chmod(0o600)
    monkeypatch.setenv("NETRC", str(netrc_file))

    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200)

    with httpx.Client(
        transport=httpx.MockTransport(handler), auth=netrc_auth()
    ) as client:
        client.get("https://files.example/doc.txt")
        client.get("https://other.example/doc.txt")

    assert seen[0] is not None and seen[0].startswith("Basic ")
    assert seen[1] is None


@pytest.mark.unit
def test_netrc_auth_ignores_an_unreadable_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    broken = tmp_path / "broken"
    broken.write_text("machine\n")
    monkeypatch.setenv("NETRC", str(broken))
    assert netrc_auth() is None
