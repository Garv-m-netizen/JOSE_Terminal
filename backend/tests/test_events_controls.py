from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from app.agents import intent_engine
from app.schemas.domain import CandidateList, Intent
from app.services.event_service import EventService, redact
from app.services.llm import GeminiLLM, LLMError, build_providers, set_providers_override

from .conftest import OBJECTIVE, wait_status
from .helpers import NAMES, make_providers, site


def parse_sse(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ") and line != "data: {}"]


async def test_sse_replay_resume_and_end(client):
    r = (await client.post("/api/objectives", json={"objective": OBJECTIVE, "mode": "autonomous"})).json()
    await wait_status(client, r["id"], {"completed"})
    async with client.stream("GET", f"/api/runs/{r['id']}/events/stream") as resp:
        assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
        body = (await resp.aread()).decode()
    evs = parse_sse(body)
    ids = [e["id"] for e in evs]
    assert evs[0]["type"] == "run.created" and evs[-1]["type"] == "run.completed"
    assert ids == sorted(set(ids)) and "event: end" in body
    assert all(e["run_id"] == r["id"] and e["ts"] and isinstance(e["details"], dict) for e in evs)
    # reconnect with Last-Event-ID: only newer events, none duplicated
    mid = ids[len(ids) // 2]
    async with client.stream("GET", f"/api/runs/{r['id']}/events/stream", headers={"Last-Event-ID": str(mid)}) as resp:
        again = parse_sse((await resp.aread()).decode())
    assert [e["id"] for e in again] == [i for i in ids if i > mid]
    # the JSON history endpoint agrees
    hist = (await client.get(f"/api/runs/{r['id']}/events?after={mid}")).json()
    assert [e["id"] for e in hist] == [e["id"] for e in again]
    assert (await client.get("/api/runs/run_nope/events/stream")).status_code == 404


async def test_sse_live_subscription_receives_events_without_duplicates(client):
    r = (await client.post("/api/objectives", json={"objective": OBJECTIVE, "mode": "guided"})).json()
    assert r["status"] == "awaiting_plan_approval"

    async def consume():
        async with client.stream("GET", f"/api/runs/{r['id']}/events/stream") as resp:
            return parse_sse((await resp.aread()).decode())

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0.2)
    await client.post(f"/api/runs/{r['id']}/plan/approve")
    evs = await asyncio.wait_for(consumer, 30)
    ids = [e["id"] for e in evs]
    assert len(ids) == len(set(ids)) and evs[-1]["type"] == "run.completed"
    types = [e["type"] for e in evs]
    for needed in ("run.created", "plan.generated", "plan.approved", "task.started", "tool.started", "tool.completed",
                   "observation.recorded", "verification.passed", "artifact.created", "run.completed"):
        assert needed in types, needed


def test_redaction_masks_secrets_and_truncates():
    out = redact({"api_key": "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ12345", "note": "token AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ12345 here",
                  "headers": {"Authorization": "Bearer abcdefghijklmnopqrstuvwxyz"}, "big": "x" * 5000, "n": 3})
    s = json.dumps(out)
    assert "AIza" not in s and "abcdefghijklmnop" not in s and out["api_key"] == "[redacted]"
    assert out["n"] == 3 and len(out["big"]) < 1700


def test_unknown_event_type_is_rejected():
    with pytest.raises(ValueError):
        EventService().emit("run_x", "made.up.event")


async def test_pause_resume_and_cancel(client, monkeypatch):
    monkeypatch.setenv("JOSE_MODE", "real")
    monkeypatch.setenv("GEMINI_API_KEY", "x" * 12)
    from app.config import get_settings

    get_settings.cache_clear()
    sites = {n: site(n) for n in NAMES[:5]}
    prov, fetcher, _ = make_providers(sites, delay=0.15)
    set_providers_override(lambda s, i: prov)
    obj = "Research three competitors in the AI automation market and create an Excel comparison"
    r = (await client.post("/api/objectives", json={"objective": obj, "mode": "autonomous"})).json()
    await asyncio.sleep(0.2)
    p = await client.post(f"/api/runs/{r['id']}/pause")
    assert p.status_code == 200 and p.json()["status"] == "paused"
    await asyncio.sleep(0.8)  # in-flight work drains, nothing new starts
    n1 = len(fetcher.calls)
    await asyncio.sleep(0.6)
    assert len(fetcher.calls) == n1
    assert (await client.post(f"/api/runs/{r['id']}/pause")).status_code == 409
    assert (await client.post(f"/api/runs/{r['id']}/resume")).json()["status"] == "running"
    final = await wait_status(client, r["id"], {"completed", "failed", "halted", "partial"})
    assert final["status"] == "completed"

    r2 = (await client.post("/api/objectives", json={"objective": obj, "mode": "autonomous"})).json()
    await asyncio.sleep(0.3)
    c = await client.post(f"/api/runs/{r2['id']}/cancel")
    assert c.status_code in (200,)
    final2 = await wait_status(client, r2["id"], {"cancelled", "completed"})
    assert final2["status"] == "cancelled"
    assert not any(t["status"] in ("running", "verifying", "pending") for t in final2["tasks"])
    types = [e["type"] for e in (await client.get(f"/api/runs/{r2['id']}/events")).json()]
    assert types[-1] == "run.cancelled" and "run.completed" not in types
    assert (await client.post(f"/api/runs/{r2['id']}/cancel")).status_code == 409


async def test_cancel_parked_run(client):
    r = (await client.post("/api/objectives", json={"objective": OBJECTIVE, "mode": "guided"})).json()
    c = (await client.post(f"/api/runs/{r['id']}/cancel")).json()
    assert c["status"] == "cancelled" and all(t["status"] == "cancelled" for t in c["tasks"])


async def test_demo_mode_is_explicit_and_secrets_never_leak(client, monkeypatch):
    secret = "AIzaSyFAKEKEYFAKEKEYFAKEKEYFAKEKEY1234"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-SECRETSECRETSECRET")
    monkeypatch.setenv("JOSE_MODE", "demo")
    from app.config import get_settings

    get_settings.cache_clear()
    st = (await client.get("/api/status")).json()
    assert st["mode"] == "demo" and st["demo_forced"] and st["ai"]["configured"] is True
    r = (await client.post("/api/objectives", json={"objective": OBJECTIVE, "mode": "autonomous"})).json()
    assert r["simulated"] is True
    final = await wait_status(client, r["id"], {"completed"})
    blobs = [json.dumps(x) for x in [st, (await client.get("/api/settings")).json(),
                                    (await client.get(f"/api/runs/{r['id']}/trace")).json(), final]]
    assert all(secret not in b and "tvly-SECRET" not in b for b in blobs)
    assert all(a["simulated"] for a in final["artifacts"])
    assert all(e["simulated"] for e in (await client.get(f"/api/runs/{r['id']}/evidence")).json())
    tr = (await client.get(f"/api/runs/{r['id']}/trace")).json()
    assert all(t["simulated"] for t in tr["tool_calls"] if t["tool"] in ("web_search", "http.fetch", "llm.extract_competitor_facts"))
    assert tr["events"][0]["details"]["simulated"] is True


async def test_no_key_means_demo_and_settings_validation(client, monkeypatch):
    monkeypatch.setenv("JOSE_MODE", "auto")
    from app.config import get_settings

    get_settings.cache_clear()
    st = (await client.get("/api/status")).json()
    assert st["mode"] == "demo" and "GEMINI_API_KEY" in st["demo_reason"]
    ok = await client.put("/api/settings", json={"max_retries": 4, "browser_headless": False})
    assert ok.status_code == 200 and ok.json()["execution"]["max_retries"] == 4
    assert (await client.get("/api/settings")).json()["execution"]["browser_headless"] is False
    assert (await client.put("/api/settings", json={"max_retries": 99})).status_code == 422
    assert '"gemini_api_key"' not in json.dumps((await client.get("/api/settings")).json()).lower()


async def test_real_mode_intent_uses_llm_but_safety_rules_still_apply():
    sites = {n: site(n) for n in NAMES[:3]}
    prov, _, llm = make_providers(sites)
    ok = await intent_engine.understand("Research three competitors in the AI automation market and make slides", prov)
    assert "extract_intent" in llm.calls and ok.feasible
    llm.calls.clear()
    bad = await intent_engine.understand("Research three competitors in the AI automation market and publish the findings", prov)
    assert not bad.feasible and bad.risk_level == "HIGH" and llm.calls == []  # the model is never consulted for unsafe asks


async def test_gemini_function_calling_validates_and_retries(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x" * 12)
    from app.config import get_settings

    get_settings.cache_clear()
    replies = [SimpleNamespace(function_calls=[SimpleNamespace(args={"competitors": []})]),  # violates min_length -> retry
               SimpleNamespace(function_calls=[SimpleNamespace(args={"competitors": [{"name": "A", "url": "https://a.test"}]})])]
    seen = []

    class Models:
        async def generate_content(self, model, contents, config):
            seen.append((model, contents, config))
            return replies.pop(0)

    llm = GeminiLLM.__new__(GeminiLLM)
    llm._client, llm._model = SimpleNamespace(aio=SimpleNamespace(models=Models())), "gemini-test"
    out = await llm.call("select_competitors", "d", CandidateList, "sys", "prompt")
    assert out.competitors[0].name == "A" and len(seen) == 2
    assert "invalid" in seen[1][1]
    cfg = seen[0][2]
    assert cfg.tools[0].function_declarations[0].name == "select_competitors"
    assert cfg.tool_config.function_calling_config.allowed_function_names == ["select_competitors"]
    replies[:] = [SimpleNamespace(function_calls=None), SimpleNamespace(function_calls=None)]
    with pytest.raises(LLMError):
        await llm.call("select_competitors", "d", CandidateList, "sys", "prompt")


def test_real_providers_report_missing_search_key(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "x" * 12)
    from app.config import get_settings

    get_settings.cache_clear()
    prov = build_providers(False)
    assert prov.search is None and "TAVILY_API_KEY" in prov.search_error
    assert isinstance(Intent(industry="x y", feasible=True), Intent)
