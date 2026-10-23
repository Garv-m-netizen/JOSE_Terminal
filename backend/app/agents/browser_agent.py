"""Browser Agent: retrieves competitor pages (HTTP first, Playwright fallback) as untrusted, read-only evidence."""
from __future__ import annotations

from urllib.parse import urlparse

from ..schemas.domain import Candidate, PageRecord
from ..services import artifact_service
from ..tools.extraction import keyword_links
from ..tools.fetch import FetchError
from ..tools.safety import registrable_domain
from .base import AgentContext, TaskError

MIN_TEXT = 300  # below this a page is probably JS-rendered; try the browser


def candidate_for(ctx: AgentContext) -> Candidate:
    disc = ctx.upstream("research.discover")
    if not disc:
        raise TaskError("candidate list is missing", "validation", retryable=False)
    cands = disc[0][1]["candidates"]
    idx = int(ctx.task.params.get("candidate_index", ctx.task.params["slot"]))
    if idx >= len(cands):
        raise TaskError("no unused backup competitors remain", "exhausted", retryable=False)
    return Candidate.model_validate(cands[idx])


def _kind(e: FetchError) -> str:
    return {"blocked": "blocked", "robots": "blocked", "unsafe": "blocked", "timeout": "timeout"}.get(e.kind, "network")


async def _get(ctx: AgentContext, url: str, role: str, screenshot: bool = False) -> PageRecord:
    prov = ctx.providers
    force_browser = bool(ctx.task.params.get("force_browser")) and prov.browser is not None
    page: PageRecord | None = None
    http_err: FetchError | None = None
    if not force_browser:
        try:
            page = await ctx.tool("http.fetch", {"url": url, "role": role}, lambda: prov.fetcher.fetch(url, role),
                                  lambda p: f"{len(p.text)} chars, title '{p.title[:60]}'", simulated=prov.simulated)
        except FetchError as e:
            if e.kind in ("blocked", "robots", "unsafe"):
                raise  # access restrictions are respected, never bypassed
            http_err = e
    if page is None or (len(page.text) < MIN_TEXT and not prov.simulated):
        if prov.browser is not None:
            try:
                rec, shot = await ctx.tool("browser.render", {"url": url, "role": role},
                                           lambda: prov.browser.render(url, role, screenshot=screenshot),
                                           lambda r: f"{len(r[0].text)} chars rendered", simulated=False)
                if shot:
                    from ..services.artifact_service import run_dir

                    path = run_dir(ctx.run_id) / "evidence" / f"{ctx.task.key}-{role}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(shot)
                    art = artifact_service.register(ctx.run_id, ctx.task.id, path, "screenshot", False)
                    rec.screenshot_artifact_id = art["id"]
                if page is None or len(rec.text) > len(page.text):
                    page = rec
            except FetchError as e:
                if page is None:
                    raise
                http_err = http_err or e
            except Exception as e:  # noqa: BLE001 - e.g. BrowserUnavailable: keep the HTTP result if we have one
                if page is None:
                    raise FetchError(f"{url}: {http_err or e}", "network") from e
        elif page is None and http_err:
            raise http_err
    assert page is not None
    return page


async def collect(ctx: AgentContext) -> dict:
    cand = candidate_for(ctx)
    prov = ctx.providers
    warnings: list[str] = []
    pages: list[PageRecord] = []
    async with ctx.activity(f"visiting {urlparse(cand.url).hostname}"):
        try:
            home = await _get(ctx, cand.url, "homepage", screenshot=True)
        except FetchError as e:
            raise TaskError(f"{cand.name}: {e}", _kind(e)) from e
        pages.append(home)
        # pricing page: follow a discovered link first, then the conventional path
        tried = {home.final_url or home.url}
        links = keyword_links(home.links, ("pricing", "plans", "price"), same_host_of=home.final_url or cand.url, limit=2)
        guess = f"{cand.url.rstrip('/')}/pricing"
        for url in [*links, guess]:
            if url in tried:
                continue
            tried.add(url)
            try:
                pages.append(await _get(ctx, url, "pricing"))
                break
            except FetchError as e:
                if e.kind in ("blocked", "robots", "unsafe"):
                    warnings.append(f"pricing page not retrievable: {e}")
                    break
        else:
            warnings.append("no pricing page could be retrieved")
        # independent source to cross-check claims
        if prov.search is not None:
            try:
                res = await ctx.tool("web_search", {"query": f"{cand.name} pricing"},
                                     lambda: prov.search.search(f"{cand.name} pricing", 6), lambda r: f"{len(r)} results")
                own = registrable_domain(cand.url) if not prov.simulated else urlparse(cand.url).hostname
                for r in res:
                    other = registrable_domain(r.url) if not prov.simulated else urlparse(r.url).hostname
                    if other == own or r.url in tried:
                        continue
                    try:
                        pages.append(await _get(ctx, r.url, "corroboration"))
                        break
                    except FetchError as e:
                        warnings.append(f"corroborating source skipped: {e}")
                else:
                    warnings.append("no independent corroborating source was retrievable")
            except Exception as e:  # noqa: BLE001 - corroboration is best effort
                warnings.append(f"corroboration search failed: {e}")
        else:
            warnings.append("no search provider: claims could not be cross-checked against an independent source")
    return {"competitor": cand.model_dump(), "pages": [p.model_dump(mode="json") for p in pages], "warnings": warnings,
            "via": sorted({p.via for p in pages})}
