"""Playwright browser tool: read-only navigation, text/link extraction and screenshots.

Used as a fallback for JavaScript-rendered pages. It never fills forms, logs in, or bypasses
CAPTCHAs / access walls.
"""
from __future__ import annotations

import asyncio
import ipaddress
from datetime import datetime, timezone
from urllib.parse import urlparse

from ..config import get_settings
from ..schemas.domain import PageRecord
from .fetch import MAX_TEXT_CHARS, FetchError, looks_blocked, parse_html
from .safety import UnsafeURL, resolve_and_check


class BrowserUnavailable(RuntimeError):
    pass


def _literal_private(host: str) -> bool:
    if host in ("localhost",) or host.endswith((".local", ".internal")):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except ValueError:
        return False


class BrowserTool:
    simulated = False

    def __init__(self) -> None:
        self._pw = None
        self._browser = None
        self._lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(2)

    @staticmethod
    def available() -> tuple[bool, str]:
        s = get_settings()
        if not s.browser_enabled:
            return False, "disabled in settings"
        try:
            import playwright  # noqa: F401
        except ImportError:
            return False, "playwright package not installed"
        return True, "ok"

    async def _ensure(self):
        async with self._lock:
            if self._browser:
                return self._browser
            ok, why = self.available()
            if not ok:
                raise BrowserUnavailable(why)
            from playwright.async_api import async_playwright

            try:
                self._pw = await async_playwright().start()
                self._browser = await self._pw.chromium.launch(headless=get_settings().browser_headless)
            except Exception as e:  # noqa: BLE001
                raise BrowserUnavailable(f"chromium could not start ({e}); run `python -m playwright install chromium`") from e
            return self._browser

    async def close(self) -> None:
        async with self._lock:
            try:
                if self._browser:
                    await self._browser.close()
                if self._pw:
                    await self._pw.stop()
            finally:
                self._browser = self._pw = None

    async def render(self, url: str, role: str = "homepage", screenshot: bool = False) -> tuple[PageRecord, bytes | None]:
        try:
            await asyncio.to_thread(resolve_and_check, url)
        except UnsafeURL as e:
            raise FetchError(str(e), "unsafe") from e
        browser = await self._ensure()
        async with self._sem:
            context = await browser.new_context(user_agent="JOSE-Agent/1.0 (+research)", java_script_enabled=True)
            try:
                page = await context.new_page()

                async def guard(route):  # block private targets and non-http schemes at request level
                    u = urlparse(route.request.url)
                    if u.scheme in ("http", "https") and not (
                            _literal_private(u.hostname or "") and not get_settings().allow_private_hosts):
                        await route.continue_()
                    elif u.scheme in ("data", "blob", "about"):
                        await route.continue_()
                    else:
                        await route.abort()

                await page.route("**/*", guard)
                timeout_ms = int(get_settings().http_timeout_seconds * 1000)
                try:
                    resp = await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                    try:
                        await page.wait_for_load_state("networkidle", timeout=5000)
                    except Exception:  # noqa: BLE001 - networkidle is best effort
                        pass
                except Exception as e:  # noqa: BLE001
                    raise FetchError(f"browser navigation failed for {url}: {e}", "timeout" if "Timeout" in str(e) else "network") from e
                html = await page.content()
                final = page.url
                title, text, links = parse_html(html, final)
                status = resp.status if resp else None
                if status and status >= 400:
                    raise FetchError(f"HTTP {status} for {final}", "blocked" if status in (401, 403, 429) else "network")
                if looks_blocked(text, status):
                    raise FetchError(f"access restriction detected at {final}; not bypassing", "blocked")
                shot = await page.screenshot(type="png", full_page=False) if screenshot else None
                rec = PageRecord(url=url, final_url=final, title=title, text=text[:MAX_TEXT_CHARS], links=links,
                                 retrieved_at=datetime.now(timezone.utc), via="browser", role=role,  # type: ignore[arg-type]
                                 status_code=status)
                return rec, shot
            finally:
                await context.close()


_browser_tool: BrowserTool | None = None


def get_browser_tool() -> BrowserTool:
    global _browser_tool
    if _browser_tool is None:
        _browser_tool = BrowserTool()
    return _browser_tool
