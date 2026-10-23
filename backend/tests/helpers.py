from __future__ import annotations

from datetime import datetime, timezone

from app.schemas.domain import PageRecord
from app.services.llm import Providers
from app.tools.fetch import FetchError
from app.tools.web_search import SearchResult

NAMES = ["Acme Flow", "Borealis AI", "Cobalt Pilot", "Delta Orchestrate", "Ember Agents", "Fjord Robotics", "Gamma Works"]


def host(name: str) -> str:
    return name.lower().replace(" ", "") + ".test"


def site(name: str, pricing: str = "paid") -> dict[str, PageRecord]:
    h = host(name)
    now = datetime.now(timezone.utc)
    feats = [f"{name} offers workflow automation", f"{name} provides audit logging", f"{name} supports human approvals"]
    home = f"{name} - automation platform. " + " ".join(feats) + ". " + "Lorem ipsum dolor sit amet. " * 20
    price = {"paid": f"{name} Starter costs $25 per user per month. Pro costs $60 per user per month.",
             "contact": f"{name} pricing: contact sales for a custom quote.",
             "none": f"{name} - see how we can help your team succeed with a walkthrough."}[pricing] + " Filler text. " * 20
    mk = lambda url, text, role: PageRecord(url=url, final_url=url, title=f"{name} {role}", text=text, retrieved_at=now,  # noqa: E731
                                            via="http", role=role, status_code=200, links=[{"text": "Pricing", "url": f"https://{h}/pricing"}])
    return {"home": mk(f"https://{h}", home, "homepage"), "pricing": mk(f"https://{h}/pricing", price, "pricing"),
            "review": mk(f"https://reviews.example.org/{h}", f"Review: {name} costs $25 per user per month, say users. " + "x " * 200, "corroboration"),
            "feats": feats}


class FakeLLM:
    simulated = False

    def __init__(self, handler):
        self.handler, self.calls = handler, []

    async def call(self, name, description, schema, system, prompt, payload=None):
        self.calls.append(name)
        return schema.model_validate(self.handler(name, payload or {}, prompt))


class FakeSearch:
    name = "fake"
    simulated = False

    def __init__(self, names=NAMES):
        self.names = names

    async def search(self, query, n=8):
        if query.endswith(" pricing"):
            nm = query[: -len(" pricing")]
            return [SearchResult(title="review", url=f"https://reviews.example.org/{host(nm)}", snippet="")]
        return [SearchResult(title=f"{x} - home", url=f"https://{host(x)}", snippet="") for x in self.names]


class FakeFetcher:
    simulated = False

    def __init__(self, pages: dict[str, PageRecord], failures: dict[str, FetchError] | None = None, delay: float = 0):
        self.pages, self.failures, self.delay, self.calls = pages, failures or {}, delay, []

    async def fetch(self, url, role="homepage"):
        import asyncio

        self.calls.append(url)
        if self.delay:
            await asyncio.sleep(self.delay)
        if url in self.failures:
            raise self.failures[url]
        pg = self.pages.get(url)
        if pg is None:
            raise FetchError(f"404 for {url}", "network")
        return pg.model_copy(update={"role": role})


def default_handler(sites: dict[str, dict]):
    """FakeLLM behaviour: pick candidates from search results, then extract verbatim claims from page text."""
    def handle(name: str, payload: dict, prompt: str):
        if name == "extract_intent":
            return {"goal": "g", "industry": "AI automation", "competitor_count": 3, "outputs": ["xlsx", "pptx"],
                    "success_criteria": ["ok"], "risk_level": "LOW"}
        if name == "select_competitors":
            return {"competitors": [{"name": n, "url": f"https://{host(n)}", "rationale": "x"} for n in sites]}
        if name == "extract_competitor_facts":
            nm = payload["name"]
            s = sites[nm]
            feats = [{"text": f.split(" ", 1)[1][:40], "evidence": f, "source_url": s["home"].url} for f in s["feats"]]
            feats.append({"text": "Quantum telepathy", "evidence": "offers quantum telepathy to every user", "source_url": ""})  # hallucination
            ptext = s["pricing"].text
            plans, status = [], "not_found"
            if "$25" in ptext:
                plans = [{"name": "Starter", "price_text": "$25 per user per month", "price_usd": 25,
                          "evidence": f"{nm} Starter costs $25 per user per month.", "source_url": s["pricing"].url}]
                status = "paid"
            elif "contact sales" in ptext:
                status = "contact_sales"
            return {"product_name": nm, "summary": "", "features": feats, "pricing_status": status, "plans": plans}
        raise AssertionError(name)
    return handle


def make_providers(sites: dict[str, dict], failures=None, delay: float = 0, handler=None):
    pages = {}
    for s in sites.values():
        for k in ("home", "pricing", "review"):
            pages[s[k].url] = s[k]
    fetcher = FakeFetcher(pages, failures, delay)
    llm = FakeLLM(handler or default_handler(sites))
    prov = Providers(llm=llm, search=FakeSearch(list(sites)), fetcher=fetcher, browser=None, simulated=False)
    return prov, fetcher, llm
