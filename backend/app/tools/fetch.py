"""HTTP page fetching with robots.txt, redirect validation, size caps and text extraction."""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from ..config import get_settings
from ..schemas.domain import PageRecord
from .safety import UnsafeURL, check_url_syntax, resolve_and_check

USER_AGENT = "JOSE-Agent/1.0 (+research; respects robots.txt)"
MAX_BYTES = 2_000_000
MAX_TEXT_CHARS = 40_000
ACCESS_WALL = re.compile(
    r"(verify you are (a )?human|captcha|are you a robot|access denied|enable javascript and cookies|"
    r"checking your browser|attention required)", re.I)


class FetchError(RuntimeError):
    def __init__(self, message: str, kind: str = "network"):
        super().__init__(message)
        self.kind = kind  # network | blocked | robots | unsafe | content | timeout


def parse_html(html: str, base_url: str) -> tuple[str, str, list[dict]]:
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    links: list[dict] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        if urlparse(href).scheme in ("http", "https") and href not in seen:
            seen.add(href)
            links.append({"text": " ".join(a.get_text(" ").split())[:100], "url": href})
        if len(links) >= 250:
            break
    for tag in soup(["script", "style", "noscript", "svg", "template", "iframe"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ").split())
    return title, text[:MAX_TEXT_CHARS], links


def looks_blocked(text: str, status: int | None) -> bool:
    if status in (401, 403, 429):
        return True
    return len(text) < 1500 and bool(ACCESS_WALL.search(text))


class RobotsCache:
    def __init__(self) -> None:
        self._cache: dict[str, RobotFileParser | None] = {}

    async def allowed(self, client: httpx.AsyncClient, url: str) -> bool:
        p = urlparse(url)
        origin = f"{p.scheme}://{p.netloc}"
        if origin not in self._cache:
            rp = RobotFileParser()
            try:
                r = await client.get(f"{origin}/robots.txt", timeout=8)
                if r.status_code == 200 and "html" not in r.headers.get("content-type", "").lower():
                    rp.parse(r.text.splitlines())
                    self._cache[origin] = rp
                else:
                    self._cache[origin] = None  # no usable robots.txt -> allowed
            except Exception:  # noqa: BLE001
                self._cache[origin] = None
        rp = self._cache[origin]
        return True if rp is None else rp.can_fetch(USER_AGENT, url)


class HttpFetcher:
    simulated = False

    def __init__(self) -> None:
        self.robots = RobotsCache()

    async def fetch(self, url: str, role: str = "homepage") -> PageRecord:
        s = get_settings()
        timeout = s.http_timeout_seconds
        try:
            current = await asyncio.to_thread(resolve_and_check, url)
        except UnsafeURL as e:
            raise FetchError(str(e), "unsafe") from e
        headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
        async with httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=False) as client:
            if s.respect_robots and not await self.robots.allowed(client, current):
                raise FetchError(f"robots.txt disallows fetching {current}", "robots")
            for _hop in range(6):
                try:
                    async with client.stream("GET", current) as resp:
                        if resp.is_redirect:
                            nxt = urljoin(current, resp.headers.get("location", ""))
                            try:
                                check_url_syntax(nxt)
                                await asyncio.to_thread(resolve_and_check, nxt)
                            except UnsafeURL as e:
                                raise FetchError(f"redirect blocked: {e}", "unsafe") from e
                            current = nxt
                            continue
                        ctype = resp.headers.get("content-type", "").lower()
                        body = b""
                        async for chunk in resp.aiter_bytes():
                            body += chunk
                            if len(body) > MAX_BYTES:
                                break
                        status = resp.status_code
                except httpx.TimeoutException as e:
                    raise FetchError(f"timeout fetching {current}", "timeout") from e
                except httpx.HTTPError as e:
                    raise FetchError(f"network error fetching {current}: {e}", "network") from e
                if status >= 400:
                    kind = "blocked" if status in (401, 403, 429) else "network"
                    raise FetchError(f"HTTP {status} for {current}", kind)
                if "html" not in ctype and "text" not in ctype:
                    raise FetchError(f"unsupported content type '{ctype}' for {current}", "content")
                html = body.decode(resp.encoding or "utf-8", errors="replace")
                title, text, links = parse_html(html, current)
                if looks_blocked(text, status):
                    raise FetchError(f"access restriction detected at {current}; not bypassing", "blocked")
                return PageRecord(url=url, final_url=current, title=title, text=text, links=links,
                                  retrieved_at=datetime.now(timezone.utc), via="http", role=role,  # type: ignore[arg-type]
                                  status_code=status)
        raise FetchError(f"too many redirects for {url}", "network")
