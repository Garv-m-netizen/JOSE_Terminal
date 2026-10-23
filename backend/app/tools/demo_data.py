"""Deterministic SIMULATED data for demo mode. Companies, pages and prices here are fictional."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from ..schemas.domain import Candidate, PageRecord
from .web_search import SearchResult

SIM = "[SIMULATED SAMPLE DATA]"
GREEK = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa", "Lambda", "Sigma"]
FEATURE_POOL = [
    "Visual workflow builder", "Natural-language task creation", "Built-in audit logging", "Role-based access control",
    "API and webhook triggers", "Human-in-the-loop approvals", "Prebuilt connector library", "Scheduled and event-driven runs",
    "Usage analytics dashboard", "On-premise deployment option", "Versioned workflow templates", "Automatic error recovery",
]
PRICING_VARIANTS = ["paid", "freemium", "contact_sales", "not_found", "paid_annual"]
FIXED_TIME = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def _h(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest(), 16)


def slug(industry: str) -> str:
    return "".join(c for c in industry.lower() if c.isalnum())[:16] or "industry"


def demo_candidates(industry: str, n: int) -> list[Candidate]:
    start = _h(industry) % 4
    out = []
    for i in range(n):
        g = GREEK[(start + i) % len(GREEK)]
        out.append(Candidate(name=f"{g} Demo Co", url=f"https://{g.lower()}-demo.example.com",
                             rationale=f"{SIM} fictional competitor generated for the '{industry}' demo"))
    return out


def _variant(name: str) -> str:
    return PRICING_VARIANTS[_h(name) % len(PRICING_VARIANTS)]


def _features(name: str) -> list[str]:
    base = _h(name)
    idx = [(base >> (3 * k)) % len(FEATURE_POOL) for k in range(12)]
    uniq = list(dict.fromkeys(idx))[:5]
    return [FEATURE_POOL[i] for i in uniq]


def _price(name: str, salt: str, lo: int, hi: int) -> int:
    return lo + _h(name, salt) % (hi - lo)


def demo_pages(name: str, base_url: str, industry: str) -> dict[str, PageRecord]:
    """Fictional homepage / pricing / third-party pages whose text drives the normal extraction pipeline."""
    feats = _features(name)
    variant = _variant(name)
    home = (f"{name} - {industry} platform {SIM}. {name} Automate is a sample product for {industry}. "
            + " ".join(f"Feature: {f}." for f in feats))
    star, pro = _price(name, "s", 9, 40), _price(name, "p", 50, 120)
    if variant == "paid":
        pricing = f"{name} pricing {SIM}. Starter plan: ${star} per user per month. Pro plan: ${pro} per user per month. Enterprise: contact sales."
    elif variant == "freemium":
        pricing = f"{name} pricing {SIM}. Free plan: $0 forever for small teams. Team plan: ${star} per user per month."
    elif variant == "contact_sales":
        pricing = f"{name} pricing {SIM}. Pricing is tailored to your organisation. Contact sales for a custom quote."
    elif variant == "paid_annual":
        pricing = f"{name} pricing {SIM}. Business plan: ${pro * 5} per month billed annually."
    else:
        pricing = f"{name} {SIM}. See how {name} can fit your team. Book a walkthrough."
    third = (f"Independent review of {name} {SIM}. Reviewers liked the {feats[0].lower()}. "
             + (f"Reported pricing: starts at ${star} per user per month." if variant in ("paid", "freemium") else
                "No reliable public pricing was listed."))
    mk = lambda url, title, text, role: PageRecord(  # noqa: E731
        url=url, final_url=url, title=title, text=text, retrieved_at=FIXED_TIME, via="demo", role=role,
        status_code=200, simulated=True, links=[{"text": "Pricing", "url": f"{base_url}/pricing"}])
    return {
        "homepage": mk(base_url, f"{name} {SIM}", home, "homepage"),
        "pricing": mk(f"{base_url}/pricing", f"{name} Pricing {SIM}", pricing, "pricing"),
        "corroboration": mk(f"https://reviews-demo.example.com/{name.split()[0].lower()}", f"{name} review {SIM}", third, "corroboration"),
    }


def demo_search(query: str, n: int = 8) -> list[SearchResult]:
    if query.endswith(" pricing"):  # per-competitor corroboration lookup
        first = query.split()[0].lower()
        return [SearchResult(title=f"{SIM} review of {query}", url=f"https://reviews-demo.example.com/{first}",
                             snippet=f"{SIM} simulated independent review.")]
    return [SearchResult(title=f"{SIM} {query} result {i + 1}", url=f"https://search-demo.example.com/{slug(query)}/{i + 1}",
                         snippet=f"{SIM} simulated search result for '{query}'.") for i in range(min(n, 5))]
