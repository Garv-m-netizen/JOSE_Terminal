"""LLM access (Gemini function calling), a deterministic Demo LLM, and the provider bundle used by agents."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import get_settings
from ..schemas.domain import Candidate, ExamPaper, ExtractedFacts, FeatureClaim, PageRecord, PlanClaim, SolvedAnswer, SolvedSet
from ..tools import demo_data, demo_math
from ..tools.extraction import find_prices, has_contact_sales_marker, has_free_marker, parse_price_usd
from ..tools.web_search import SearchProvider, SearchResult

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger("jose.llm")


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    simulated: bool

    async def call(self, name: str, description: str, schema: type[T], system: str, prompt: str,
                   payload: dict | None = None) -> T: ...


def gemini_schema(schema: type[BaseModel]) -> dict:
    """Gemini rejects schemas with array size limits (minItems/maxItems) as 'too complex'. Drop them for the request;
    our own Pydantic validation still enforces them on the reply."""
    def clean(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k not in ("minItems", "maxItems")}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o

    return clean(schema.model_json_schema())


class GeminiLLM:
    simulated = False

    def __init__(self) -> None:
        from google import genai

        s = get_settings()
        self._client = genai.Client(api_key=s.gemini_api_key)
        self._model = s.gemini_model

    async def call(self, name: str, description: str, schema: type[T], system: str, prompt: str,
                   payload: dict | None = None) -> T:
        from google.genai import types

        decl = types.FunctionDeclaration(name=name, description=description,
                                         parameters_json_schema=gemini_schema(schema))
        cfg = types.GenerateContentConfig(
            system_instruction=system, temperature=0.1,
            tools=[types.Tool(function_declarations=[decl])],
            tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.ANY, allowed_function_names=[name])))
        last_err = ""
        for attempt in range(2):
            content = prompt if not last_err else f"{prompt}\n\nYour previous answer was invalid: {last_err}. Fix it."
            resp = None
            for wait in (0, 2, 6):  # temporary overload (503 / 429 / 500) is retried before we give up on Gemini
                if wait:
                    await asyncio.sleep(wait * get_settings().llm_retry_scale)
                try:
                    resp = await asyncio.wait_for(
                        self._client.aio.models.generate_content(model=self._model, contents=content, config=cfg),
                        timeout=get_settings().task_timeout_seconds)
                    break
                except asyncio.TimeoutError as e:
                    raise LLMError("Gemini request timed out") from e
                except Exception as e:  # noqa: BLE001
                    msg = f"{type(e).__name__}: {str(e)[:300]}"
                    if wait != 6 and re.search(r"\b(503|500|429)\b|UNAVAILABLE|RESOURCE_EXHAUSTED|overloaded|high demand", msg):
                        log.warning("Gemini temporarily unavailable (%s); retrying", msg[:80])
                        continue
                    raise LLMError(f"Gemini request failed: {msg}") from e
            calls = resp.function_calls or []
            if not calls:
                last_err = "no function call was returned"
                continue
            try:
                return schema.model_validate(dict(calls[0].args or {}))
            except ValidationError as e:
                last_err = str(e)[:400]
        raise LLMError(f"Gemini returned no valid '{name}' call: {last_err}")


CURRENT_RUN: ContextVar[tuple[str, str | None] | None] = ContextVar("jose_current_run", default=None)


_GROQ_PREFERRED = ["llama-3.3-70b-versatile", "qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]
_GROQ_SKIP = ("whisper", "orpheus", "guard", "safeguard", "allam", "tts", "embed")
_groq_model_cache: dict[str, str] = {}


class GroqLLM:
    """Groq's OpenAI-compatible chat API with forced tool (function) calling. Used as the fallback provider."""

    simulated = False
    name = "groq"

    def __init__(self, transport: Any = None) -> None:
        s = get_settings()
        self._key, self._transport = s.groq_api_key.strip(), transport
        self._model = _groq_model_cache.get(s.groq_model, s.groq_model)
        self._configured = s.groq_model

    async def _pick_model(self) -> str | None:
        """The configured model may not exist for this account: choose the best chat model it can actually use."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=20, transport=self._transport) as c:
                r = await c.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {self._key}"})
            ids = [m["id"] for m in r.json().get("data", []) if not any(x in m["id"].lower() for x in _GROQ_SKIP)]
        except Exception:  # noqa: BLE001
            return None
        for pref in _GROQ_PREFERRED:
            if pref in ids and pref != self._model:
                return pref
        return next((i for i in ids if i != self._model), None)

    async def call(self, name: str, description: str, schema: type[T], system: str, prompt: str,
                   payload: dict | None = None) -> T:
        import httpx

        body: dict = {"model": self._model, "temperature": 0.1,
                "tools": [{"type": "function", "function": {"name": name, "description": description, "parameters": schema.model_json_schema()}}],
                "tool_choice": {"type": "function", "function": {"name": name}}}
        last_err = ""
        waited = 0
        for _ in range(5):
            body["model"] = self._model
            content = prompt if not last_err else prompt + "\n\nYour previous answer was invalid: " + last_err + ". Fix it."
            body["messages"] = [{"role": "system", "content": system}, {"role": "user", "content": content}]
            try:
                async with httpx.AsyncClient(timeout=get_settings().task_timeout_seconds, transport=self._transport) as c:
                    r = await c.post("https://api.groq.com/openai/v1/chat/completions", json=body,
                                     headers={"Authorization": f"Bearer {self._key}"})
            except httpx.HTTPError as e:
                raise LLMError(f"Groq request failed: {type(e).__name__}") from None
            if r.status_code in (400, 403, 404) and re.search(r"model_not_found|does not exist|decommissioned|not have access", r.text):
                new = await self._pick_model()
                if new:
                    log.warning("Groq model %s is not available for this key; using %s", self._model, new)
                    self._model = _groq_model_cache[self._configured] = new
                    body["model"] = new
                    last_err = ""
                    continue
            if r.status_code == 429 and waited < 2:  # rate-limit window: wait as long as Groq asks (bounded), then retry
                m = re.search(r"try again in ([0-9.]+)s", r.text)
                waited += 1
                await asyncio.sleep(min(float(m.group(1)) + 0.5 if m else 5.0, 20.0) * get_settings().llm_retry_scale)
                continue
            if r.status_code == 400 and ("tool_use_failed" in r.text or "validation failed" in r.text):
                last_err = ("the tool call was malformed JSON" if "tool_use_failed" in r.text else
                            "it did not match the schema: " + re.sub(r"\s+", " ", r.text)[60:420])
                continue
            if r.status_code >= 400:
                raise LLMError(f"Groq returned HTTP {r.status_code}: {r.text[:200].replace(self._key, '[key]')}")
            try:
                args = json.loads(r.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
                return schema.model_validate(args)
            except (KeyError, IndexError, TypeError, ValueError) as e:
                last_err = f"{type(e).__name__}: {str(e)[:300]}"
        raise LLMError(f"Groq returned no valid '{name}' call: {last_err}")


class FallbackLLM:
    """Try the primary provider; on any LLMError use the fallback and record a visible warning on the run."""

    simulated = False

    def __init__(self, primary: Any, fallback: Any, primary_name: str = "gemini", fallback_name: str = "groq") -> None:
        self.primary, self.fallback = primary, fallback
        self.primary_name, self.fallback_name = primary_name, fallback_name
        self.fallbacks_used = 0

    async def call(self, name: str, description: str, schema: type[T], system: str, prompt: str,
                   payload: dict | None = None) -> T:
        try:
            return await self.primary.call(name, description, schema, system, prompt, payload)
        except LLMError as e:
            primary_err = str(e)[:200]
        self.fallbacks_used += 1
        log.warning("%s failed (%s); using %s", self.primary_name, primary_err, self.fallback_name)
        try:
            out = await self.fallback.call(name, description, schema, system, prompt, payload)
        except LLMError as e2:
            raise LLMError(f"{self.primary_name} failed ({primary_err}) and {self.fallback_name} failed too ({str(e2)[:200]})") from e2
        cur = CURRENT_RUN.get()
        if cur:
            from . import store

            store.add_warning(cur[0], f"{self.primary_name} failed ({primary_err[:120]}); {self.fallback_name} answered instead for '{name}'.")
        return out


def build_llm() -> Any:
    s = get_settings()
    has_g, has_q = bool(s.gemini_api_key.strip()), bool(s.groq_api_key.strip())
    if has_g and has_q:
        return FallbackLLM(GeminiLLM(), GroqLLM())
    if has_q:
        return GroqLLM()
    return GeminiLLM()


# --- deterministic demo LLM ------------------------------------------------------------------

def heuristic_facts(name: str, pages: list[PageRecord]) -> ExtractedFacts:
    """Rule-based extraction used by the Demo LLM. Evidence is always a verbatim excerpt of page text."""
    feats: list[FeatureClaim] = []
    home = next((p for p in pages if p.role == "homepage"), pages[0])
    for m in re.finditer(r"Feature:\s*([^.]+)\.", home.text):
        feats.append(FeatureClaim(text=m.group(1).strip(), evidence=m.group(0).strip(), source_url=home.url))
    pricing_pages = [p for p in pages if p.role in ("pricing", "homepage")]
    plans: list[PlanClaim] = []
    saw_free = saw_contact = False
    for p in pricing_pages:
        saw_free |= has_free_marker(p.text)
        saw_contact |= has_contact_sales_marker(p.text)
        for sent in re.split(r"(?<=\.)\s+", p.text):
            prices = find_prices(sent, window=0)
            if not prices:
                continue
            pm = prices[0]["match"]
            label = sent.split(":")[0].replace(" plan", "").strip()[:60] if ":" in sent else "Listed plan"
            tail = re.search(re.escape(pm) + r"[^.]*", sent)
            price_text = tail.group(0).strip()[:78] if tail else pm
            billing = "annual" if "annual" in sent.lower() else ("per user / month" if "per user" in sent else "")
            plans.append(PlanClaim(name=label or "Listed plan", price_text=price_text, price_usd=parse_price_usd(pm),
                                   billing=billing, evidence=sent.strip()[:580], source_url=p.url))
        if plans:
            break
    paid = [x for x in plans if (x.price_usd or 0) > 0]
    if plans and any((x.price_usd or 0) == 0 for x in plans) and paid:
        status = "freemium"
    elif plans and not paid:
        status = "free"
    elif paid:
        status = "paid"
    elif saw_contact:
        status = "contact_sales"
    else:
        status = "not_found"
    summ = re.search(rf"{re.escape(name)}\s+Automate is [^.]+\.", home.text)
    return ExtractedFacts(product_name=f"{name} Automate", summary=(summ.group(0) if summ else ""),
                          summary_evidence=(summ.group(0) if summ else ""), features=feats,
                          pricing_status=status, plans=plans,  # type: ignore[arg-type]
                          pricing_note="" if plans else ("Vendor directs buyers to contact sales." if saw_contact
                                                         else "No public pricing found on fetched pages."))


class DemoLLM:
    simulated = True

    async def call(self, name: str, description: str, schema: type[T], system: str, prompt: str,
                   payload: dict | None = None) -> T:
        payload = payload or {}
        if name == "select_competitors":
            cands = demo_data.demo_candidates(payload["industry"], int(payload["count"]) + 2)
            return schema.model_validate({"competitors": [c.model_dump() for c in cands]})
        if name == "extract_competitor_facts":
            pages = [PageRecord.model_validate(p) for p in payload["pages"]]
            return schema.model_validate(heuristic_facts(payload["name"], pages).model_dump())
        if name == "generate_exam":
            try:
                paper = demo_math.generate(payload["subject"], int(payload["grade"]), int(payload["n"]), payload.get("difficulty", "mixed"),
                                       payload.get("marks"), int(payload.get("paper", 1)), payload.get("avoid"))
            except ValueError as e:
                raise LLMError(f"{e}. Add GEMINI_API_KEY to generate other subjects.") from e
            return schema.model_validate(paper.model_dump())
        if name == "solve_exam":
            return schema.model_validate(SolvedSet(answers=[SolvedAnswer(id=q["id"], answer=demo_math.solve(q["text"], q.get("options", [])))
                                                           for q in payload["questions"]]).model_dump())
        raise LLMError(f"demo LLM has no behaviour for '{name}'")


class DemoSearch:
    name = "demo"
    simulated = True

    async def search(self, query: str, n: int = 8) -> list[SearchResult]:
        return demo_data.demo_search(query, n)


class DemoFetcher:
    simulated = True

    def __init__(self, industry: str) -> None:
        self.industry = industry

    async def fetch(self, url: str, role: str = "homepage") -> PageRecord:
        from urllib.parse import urlparse

        from ..tools.fetch import FetchError

        p = urlparse(url)
        host = p.hostname or ""
        if host.endswith("-demo.example.com") and not host.startswith("reviews"):
            greek = host.split("-")[0].capitalize()
            name = f"{greek} Demo Co"
            pages = demo_data.demo_pages(name, f"https://{host}", self.industry)
            return pages["pricing"] if p.path.rstrip("/") == "/pricing" else pages["homepage"]
        if host == "reviews-demo.example.com":
            greek = p.path.strip("/").capitalize()
            return demo_data.demo_pages(f"{greek} Demo Co", f"https://{greek.lower()}-demo.example.com", self.industry)["corroboration"]
        raise FetchError(f"simulated fetch: unknown demo URL {url}", "network")


@dataclass
class Providers:
    llm: Any
    search: SearchProvider | None
    fetcher: Any
    browser: Any | None
    simulated: bool
    search_error: str | None = None
    messenger: Any | None = None
    extra: dict = field(default_factory=dict)


_override: Any = None


def set_providers_override(factory) -> None:
    """Tests inject a factory(simulated: bool, industry: str) -> Providers."""
    global _override
    _override = factory


def build_providers(simulated: bool, industry: str = "") -> Providers:
    if _override is not None:
        return _override(simulated, industry)
    if simulated:
        from ..tools.telegram import SimulatedTelegram

        return Providers(llm=DemoLLM(), search=DemoSearch(), fetcher=DemoFetcher(industry), browser=None, simulated=True, messenger=SimulatedTelegram())
    from ..tools.browser import get_browser_tool
    from ..tools.fetch import HttpFetcher
    from ..tools.web_search import SearchUnavailable, get_search_provider

    search, err = None, None
    try:
        search = get_search_provider()
    except SearchUnavailable as e:
        err = str(e)
    from ..tools.telegram import TelegramClient, TelegramError

    messenger, merr = None, None
    try:
        messenger = TelegramClient()
    except TelegramError as e:
        merr = str(e)
    return Providers(llm=build_llm(), search=search, fetcher=HttpFetcher(), browser=get_browser_tool(),
                     simulated=False, search_error=err, messenger=messenger, extra={"messenger_error": merr})
