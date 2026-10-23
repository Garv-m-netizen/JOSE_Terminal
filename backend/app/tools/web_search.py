"""Pluggable web-search providers. No provider -> explicit error; results are never invented."""
from __future__ import annotations

from typing import Protocol
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel

from ..config import get_settings


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""


class SearchUnavailable(RuntimeError):
    pass


class SearchProvider(Protocol):
    name: str
    simulated: bool

    async def search(self, query: str, n: int = 8) -> list[SearchResult]: ...


class TavilySearch:
    name = "tavily"
    simulated = False

    async def search(self, query: str, n: int = 8) -> list[SearchResult]:
        s = get_settings()
        async with httpx.AsyncClient(timeout=s.http_timeout_seconds) as c:
            r = await c.post("https://api.tavily.com/search",
                             json={"api_key": s.tavily_api_key, "query": query, "max_results": n})
            r.raise_for_status()
        return [SearchResult(title=x.get("title", ""), url=x["url"], snippet=x.get("content", "")[:400])
                for x in r.json().get("results", []) if x.get("url")]


class BraveSearch:
    name = "brave"
    simulated = False

    async def search(self, query: str, n: int = 8) -> list[SearchResult]:
        s = get_settings()
        async with httpx.AsyncClient(timeout=s.http_timeout_seconds) as c:
            r = await c.get("https://api.search.brave.com/res/v1/web/search", params={"q": query, "count": n},
                            headers={"X-Subscription-Token": s.brave_api_key, "Accept": "application/json"})
            r.raise_for_status()
        return [SearchResult(title=x.get("title", ""), url=x["url"], snippet=x.get("description", "")[:400])
                for x in r.json().get("web", {}).get("results", []) if x.get("url")]


class DuckDuckGoSearch:
    """Keyless best-effort search through DuckDuckGo's HTML endpoint (may be rate limited)."""

    name = "duckduckgo"
    simulated = False

    async def search(self, query: str, n: int = 8) -> list[SearchResult]:
        s = get_settings()
        async with httpx.AsyncClient(timeout=s.http_timeout_seconds, follow_redirects=True,
                                     headers={"User-Agent": "Mozilla/5.0 (compatible; JOSE-Agent/1.0)"}) as c:
            r = await c.post("https://html.duckduckgo.com/html/", data={"q": query})
            if r.status_code != 200:
                raise SearchUnavailable(f"DuckDuckGo returned HTTP {r.status_code}")
        soup = BeautifulSoup(r.text, "html.parser")
        out: list[SearchResult] = []
        for res in soup.select(".result"):
            a = res.select_one("a.result__a")
            if not a or not a.get("href"):
                continue
            href = a["href"]
            q = parse_qs(urlparse(href).query)
            url = unquote(q["uddg"][0]) if "uddg" in q else href
            if not url.startswith("http"):
                continue
            sn = res.select_one(".result__snippet")
            out.append(SearchResult(title=a.get_text(" ", strip=True), url=url,
                                    snippet=sn.get_text(" ", strip=True)[:400] if sn else ""))
            if len(out) >= n:
                break
        if not out:
            raise SearchUnavailable("DuckDuckGo returned no parseable results (possibly rate limited)")
        return out


def get_search_provider() -> SearchProvider:
    s = get_settings()
    p = s.search_provider.lower()
    if p == "tavily":
        if not s.tavily_api_key:
            raise SearchUnavailable("SEARCH_PROVIDER=tavily but TAVILY_API_KEY is not set")
        return TavilySearch()
    if p == "brave":
        if not s.brave_api_key:
            raise SearchUnavailable("SEARCH_PROVIDER=brave but BRAVE_API_KEY is not set")
        return BraveSearch()
    if p == "duckduckgo":
        return DuckDuckGoSearch()
    raise SearchUnavailable("no search provider configured (SEARCH_PROVIDER=none)")
