"""Deterministic statistics + wording shared by the Excel and PowerPoint generators."""
from __future__ import annotations

from collections import Counter

from ..schemas.domain import CompetitorRecord

STATUS_LABEL = {
    "paid": "Paid", "free": "Free", "freemium": "Free tier + paid plans",
    "contact_sales": "Contact sales (no public price)", "not_found": "Not available",
}
SIM_BANNER = "SIMULATED SAMPLE DATA - fictional companies, not real research"


def compute_stats(records: list[CompetitorRecord]) -> dict:
    priced = [(r.starting_price_usd, r.name) for r in records if r.starting_price_usd is not None]
    feat_counter = Counter(f.text.strip().lower() for r in records for f in r.features)
    display = {f.text.strip().lower(): f.text.strip() for r in records for f in r.features}
    sources = {s.url for r in records for s in r.sources}
    return {
        "count": len(records),
        "with_public_price": sum(1 for r in records if r.pricing_status in ("paid", "freemium")),
        "free_tier": sum(1 for r in records if r.pricing_status in ("free", "freemium")),
        "contact_sales": sum(1 for r in records if r.pricing_status == "contact_sales"),
        "not_found": sum(1 for r in records if r.pricing_status == "not_found"),
        "lowest_price": min(priced) if priced else None,
        "highest_price": max(priced) if priced else None,
        "common_features": [(display[k], c) for k, c in feat_counter.most_common(3) if c > 1],
        "verified": sum(1 for r in records if r.verification == "verified"),
        "unique_sources": len(sources),
        "total_features": sum(len(r.features) for r in records),
    }


def key_findings(records: list[CompetitorRecord]) -> list[str]:
    st, n = compute_stats(records), len(records)
    out = [f"{st['with_public_price']} of {n} competitors publish a price on a page we retrieved."]
    if st["lowest_price"]:
        lo, hi = st["lowest_price"], st["highest_price"]
        out.append(f"Lowest listed starting price: ${lo[0]:,.0f} ({lo[1]}); highest: ${hi[0]:,.0f} ({hi[1]}). Prices are not normalised across billing models.")
    if st["free_tier"]:
        out.append(f"{st['free_tier']} of {n} advertise a free tier or free plan.")
    if st["contact_sales"] or st["not_found"]:
        out.append(f"{st['contact_sales']} direct buyers to sales and {st['not_found']} had no public pricing found - "
                   "these are shown as unavailable, not as zero.")
    if st["common_features"]:
        out.append("Most common capabilities: " + "; ".join(f"{t} ({c} of {n})" for t, c in st["common_features"]) + ".")
    else:
        out.append("Feature sets differ widely; no capability appears at more than one competitor under exact matching.")
    return out


def limitations(records: list[CompetitorRecord], simulated: bool) -> list[str]:
    out = [
        "Only information on pages retrieved at run time is included; absence of a claim does not mean the vendor lacks the feature.",
        "Pricing reflects public list prices on the retrieval date and may vary by region, billing period or negotiated discount.",
        "Claims are kept only when a quoted excerpt was found verbatim in the retrieved page; paraphrased claims are discarded.",
    ]
    partial = [r.name for r in records if r.verification != "verified"]
    if partial:
        out.append("Partially verified records: " + ", ".join(partial) + ".")
    if simulated:
        out.insert(0, SIM_BANNER + ". No live web research was performed.")
    return out
