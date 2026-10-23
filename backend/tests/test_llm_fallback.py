from __future__ import annotations

import json

import httpx
import pytest

from app.config import get_settings
from app.database import session_scope
from app.models import Run
from app.schemas.domain import CandidateList
from app.services import store
from app.services.llm import CURRENT_RUN, FallbackLLM, GeminiLLM, GroqLLM, LLMError, Providers, build_llm, set_providers_override

from .helpers import FakeLLM
from .test_exam_workflow import TEACHER, FakeTelegram, _approve_all, exam_handler

GOOD = {"competitors": [{"name": "A", "url": "https://a.test"}]}


class Boom:
    simulated = False

    def __init__(self):
        self.calls = 0

    async def call(self, *a, **k):
        self.calls += 1
        raise LLMError("Gemini request failed: 429 quota exceeded")


async def test_fallback_used_and_recorded_on_the_run():
    with session_scope() as s:
        r = Run(objective="x")
        s.add(r)
        s.flush()
        rid = r.id
    CURRENT_RUN.set((rid, None))
    primary, fb = Boom(), FakeLLM(lambda n, p, pr: GOOD)
    out = await FallbackLLM(primary, fb).call("select_competitors", "d", CandidateList, "s", "p")
    assert out.competitors[0].name == "A" and primary.calls == 1 and fb.calls == ["select_competitors"]
    assert any("groq answered instead" in w and "429" in w for w in store.get_run(rid).warnings)


async def test_primary_success_never_touches_fallback():
    fb = FakeLLM(lambda n, p, pr: GOOD)
    out = await FallbackLLM(FakeLLM(lambda n, p, pr: GOOD), fb).call("select_competitors", "d", CandidateList, "s", "p")
    assert out.competitors and fb.calls == []


async def test_both_failing_reports_both_errors():
    class Boom2(Boom):
        async def call(self, *a, **k):
            raise LLMError("Groq returned HTTP 503")

    with pytest.raises(LLMError) as e:
        await FallbackLLM(Boom(), Boom2()).call("select_competitors", "d", CandidateList, "s", "p")
    assert "gemini failed" in str(e.value) and "groq failed too" in str(e.value) and "503" in str(e.value)


def groq(handler, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_SECRETSECRETSECRET")
    get_settings.cache_clear()
    return GroqLLM(transport=httpx.MockTransport(handler))


def tool_reply(args):
    return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [{"function": {"name": "x", "arguments": json.dumps(args)}}]}}]})


async def test_groq_function_calling_request_and_parse(monkeypatch):
    seen = {}

    def handler(req: httpx.Request):
        seen["auth"], seen["body"] = req.headers["authorization"], json.loads(req.content)
        return tool_reply(GOOD)

    out = await groq(handler, monkeypatch).call("select_competitors", "desc", CandidateList, "sys", "prompt")
    assert out.competitors[0].url == "https://a.test" and seen["auth"] == "Bearer gsk_SECRETSECRETSECRET"
    b = seen["body"]
    assert b["tool_choice"]["function"]["name"] == "select_competitors" and b["tools"][0]["function"]["name"] == "select_competitors"
    assert b["messages"][0] == {"role": "system", "content": "sys"}


async def test_groq_retries_malformed_tool_call_then_succeeds(monkeypatch):
    n = {"i": 0}

    def handler(req):
        n["i"] += 1
        if n["i"] == 1:
            return httpx.Response(400, text='{"error":{"code":"tool_use_failed"}}')
        assert "invalid" in json.loads(req.content)["messages"][1]["content"]
        return tool_reply(GOOD)

    assert (await groq(handler, monkeypatch).call("select_competitors", "d", CandidateList, "s", "p")).competitors and n["i"] == 2


async def test_groq_errors_do_not_leak_the_key(monkeypatch):
    llm = groq(lambda req: httpx.Response(401, text="bad key gsk_SECRETSECRETSECRET"), monkeypatch)
    with pytest.raises(LLMError) as e:
        await llm.call("select_competitors", "d", CandidateList, "s", "p")
    assert "gsk_SECRET" not in str(e.value) and "401" in str(e.value)
    bad = groq(lambda req: tool_reply({"competitors": []}), monkeypatch)  # violates min_length twice
    with pytest.raises(LLMError):
        await bad.call("select_competitors", "d", CandidateList, "s", "p")


def test_provider_selection_and_mode(monkeypatch):
    def setup(g, q):
        monkeypatch.setenv("GEMINI_API_KEY", g)
        monkeypatch.setenv("GROQ_API_KEY", q)
        monkeypatch.setenv("JOSE_MODE", "auto")
        get_settings.cache_clear()

    setup("g" * 12, "")
    assert isinstance(build_llm(), GeminiLLM)
    setup("g" * 12, "q" * 12)
    assert isinstance(build_llm(), FallbackLLM)
    setup("", "q" * 12)
    assert isinstance(build_llm(), GroqLLM) and not get_settings().demo_forced  # Groq alone is enough for real mode
    setup("", "")
    assert get_settings().demo_forced


async def test_whole_workflow_survives_gemini_outage(client, monkeypatch):
    handler, _ = exam_handler()
    monkeypatch.setenv("JOSE_MODE", "real")
    monkeypatch.setenv("GEMINI_API_KEY", "x" * 12)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-1001234567890")
    get_settings.cache_clear()
    tg = FakeTelegram()
    prov = Providers(llm=FallbackLLM(Boom(), FakeLLM(handler)), search=None, fetcher=None, browser=None, simulated=False, messenger=tg)
    set_providers_override(lambda s, i: prov)
    r = (await client.post("/api/objectives", json={"objective": TEACHER, "mode": "autonomous"})).json()
    final = await _approve_all(client, r["id"])
    assert final["status"] == "completed" and len(tg.sent) == 1
    assert any("groq answered instead" in w for w in final["warnings"])


def test_gemini_schema_has_no_array_size_limits():
    from app.schemas.domain import ExamPaper, ExtractedFacts
    from app.services.llm import gemini_schema

    for model in (ExamPaper, ExtractedFacts, CandidateList):
        text = json.dumps(gemini_schema(model))
        assert "minItems" not in text and "maxItems" not in text  # Gemini answers 400 INVALID_ARGUMENT otherwise
    assert "maxItems" in json.dumps(ExamPaper.model_json_schema())  # the Pydantic model still enforces them
