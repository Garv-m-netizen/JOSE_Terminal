"""Deterministic text helpers used to *verify* model-produced claims against fetched page text."""
from __future__ import annotations

import re

PRICE_RE = re.compile(r"(?:US\s?\$|\$)\s?(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)")
_TRANS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.translate(_TRANS)).strip().casefold()


def evidence_in_text(evidence: str, text: str) -> bool:
    """True if the (whitespace/case/quote-normalised) evidence literally appears in the page text."""
    e = norm(evidence).strip(" .…")
    return len(e) >= 3 and e in norm(text)


def parse_price_usd(price_text: str) -> float | None:
    m = PRICE_RE.search(price_text)
    return float(m.group(1).replace(",", "")) if m else None


def find_prices(text: str, window: int = 70) -> list[dict]:
    out = []
    for m in PRICE_RE.finditer(text):
        a, b = max(0, m.start() - window), min(len(text), m.end() + window)
        out.append({"amount": float(m.group(1).replace(",", "")), "match": m.group(0), "context": " ".join(text[a:b].split())})
    return out


def has_free_marker(text: str) -> bool:
    return bool(re.search(r"\b(free (plan|forever|tier)|forever free|\$0\b|free to start)", text, re.I))


def has_contact_sales_marker(text: str) -> bool:
    return bool(re.search(r"(contact (our )?sales|custom pricing|request a quote|talk to sales|get a quote)", text, re.I))


def keyword_links(links: list[dict], words: tuple[str, ...], same_host_of: str | None = None, limit: int = 3) -> list[str]:
    """Pick likely pricing/plans links from discovered anchors."""
    from urllib.parse import urlparse

    host = urlparse(same_host_of).hostname if same_host_of else None
    scored: list[tuple[int, str]] = []
    for ln in links:
        u = ln["url"]
        if host and urlparse(u).hostname != host:
            continue
        hay = (ln.get("text", "") + " " + urlparse(u).path).lower()
        score = sum(2 if w in urlparse(u).path.lower() else 1 for w in words if w in hay)
        if score:
            scored.append((score, u))
    scored.sort(key=lambda t: (-t[0], len(t[1])))
    seen, res = set(), []
    for _, u in scored:
        if u not in seen:
            seen.add(u)
            res.append(u)
    return res[:limit]
