"""HTTPX settings that keep the environment contract HoloDeck had on Requests.

HTTPX reads ``SSL_CERT_FILE`` and ``SSL_CERT_DIR`` but not the Requests
variables, and it does not read ``.netrc`` unless asked. Deployments that set
``REQUESTS_CA_BUNDLE`` or ``CURL_CA_BUNDLE``, or keep credentials in
``.netrc``, must keep working.
"""

from __future__ import annotations

import logging
import netrc
import os
import ssl
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_CA_VARIABLES = ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")


def tls_verify() -> ssl.SSLContext | bool:
    """Return the TLS verification setting for an HTTPX client.

    A CA bundle named by ``REQUESTS_CA_BUNDLE`` or ``CURL_CA_BUNDLE`` is
    used when it exists. Otherwise HTTPX applies its own defaults.
    """
    for variable in _CA_VARIABLES:
        location = os.environ.get(variable)
        if not location:
            continue
        path = Path(location)
        if path.is_dir():
            return ssl.create_default_context(capath=str(path))
        if path.is_file():
            return ssl.create_default_context(cafile=str(path))
        logger.warning(f"{variable} names a path that does not exist: {location}")
    return True


def netrc_auth() -> httpx.Auth | None:
    """Return netrc authentication when a readable netrc file exists."""
    location = os.environ.get("NETRC")
    path = Path(location) if location else Path.home() / ".netrc"
    if not path.is_file():
        return None
    try:
        return httpx.NetRCAuth(str(path))
    except (netrc.NetrcParseError, OSError) as error:
        logger.debug(f"Ignoring unreadable netrc file {path}: {error}")
        return None
