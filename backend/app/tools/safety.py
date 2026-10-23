"""URL / path safety. Web content and model output are untrusted; every fetch target is validated here."""
from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from urllib.parse import urlparse

from ..config import get_settings


class UnsafeURL(ValueError):
    pass


class UnsafePath(ValueError):
    pass


def _ip_is_blocked(ip: ipaddress._BaseAddress) -> bool:
    return (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
            or ip.is_reserved or ip.is_unspecified)


def check_url_syntax(url: str) -> str:
    """Cheap, offline validation (scheme, credentials, IP literals, localhost)."""
    p = urlparse(url.strip())
    if p.scheme not in ("http", "https"):
        raise UnsafeURL(f"unsupported URL scheme: {p.scheme or '(none)'}")
    if not p.hostname:
        raise UnsafeURL("URL has no host")
    if p.username or p.password:
        raise UnsafeURL("URLs with embedded credentials are not allowed")
    if get_settings().allow_private_hosts:
        return url.strip()
    host = p.hostname.lower()
    if host == "localhost" or host.endswith((".local", ".internal", ".localhost")):
        raise UnsafeURL("local hostnames are not allowed")
    try:
        if _ip_is_blocked(ipaddress.ip_address(host)):
            raise UnsafeURL("private/loopback addresses are not allowed")
    except ValueError:
        pass  # not an IP literal
    return url.strip()


def resolve_and_check(url: str) -> str:
    """Full validation incl. DNS resolution (blocking; run in a thread)."""
    check_url_syntax(url)
    if get_settings().allow_private_hosts:
        return url
    host = urlparse(url).hostname or ""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise UnsafeURL(f"cannot resolve host {host}: {e}") from e
    for info in infos:
        if _ip_is_blocked(ipaddress.ip_address(info[4][0])):
            raise UnsafeURL(f"host {host} resolves to a non-public address")
    return url


def safe_join(root: Path, *parts: str) -> Path:
    """Join paths under root, refusing anything that escapes it."""
    root = root.resolve()
    target = root.joinpath(*parts).resolve()
    if target != root and root not in target.parents:
        raise UnsafePath("path escapes the allowed directory")
    return target


def registrable_domain(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    parts = host.split(".")
    if len(parts) >= 3 and parts[-2] in {"co", "com", "org", "net", "gov", "ac"} and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host
