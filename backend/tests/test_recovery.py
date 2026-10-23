from __future__ import annotations

from app.agents import planner
from app.agents.recovery_agent import diagnose
from app.services.llm import set_providers_override
from app.services.store import TaskView
from app.tools.fetch import FetchError

from .conftest import wait_status
from .helpers import NAMES, host, make_providers, site

OBJ = "Research three competitors in the AI automation market and create an Excel comparison"


def tv(action="browser.collect", params=None, key="T2"):
    return TaskView(id="r:" + key, run_id="r", key=key, title="t", agent="browser", action=action, params=params or {"slot": 0},
                    depends_on=[], resources=[], status="failed", attempts=1, risk_level="LOW")


def test_diagnose_routes_failures():
    d = diagnose(tv(), "HTTP 403", "blocked", max_retries=2, can_use_browser=True, backoff_base=0)
    assert d.decision == "swap_candidate"  # access restrictions are respected, not bypassed
    assert diagnose(tv(), "x", "timeout", max_retries=2, can_use_browser=True, backoff_base=0).decision == "retry"
    assert diagnose(tv(params={"slot": 0, "_recoveries": 1}), "x", "timeout", max_retries=2, can_use_browser=True,
                    backoff_base=0).decision == "retry_with_browser"
    assert diagnose(tv(), "no key", "search_unavailable", max_retries=2, can_use_browser=True).needs_human


def test_retry_limit_gives_up():
    d = diagnose(tv(params={"slot": 0, "_recoveries": 2}), "x", "network", max_retries=2, can_use_browser=False)
    assert d.decision == "give_up" and "retry limit" in d.reason


def test_replan_swaps_to_unused_backup_candidate():
    collects = [tv(params={"slot": i}, key=f"C{i}") for i in range(3)]
    d = diagnose(collects[1], "403", "blocked", max_retries=2, can_use_browser=False, backoff_base=0)
    patches = planner.replan([d], collects)
    assert patches[0].set_params["candidate_index"] == 3  # 0..2 are in use; first backup is 3
    assert patches[0].set_params["_tried"] == [1]


async def _run_real(client, monkeypatch, prov):
    monkeypatch.setenv("JOSE_MODE", "real")
    monkeypatch.setenv("GEMINI_API_KEY", "x" * 12)
    from app.config import get_settings

    get_settings.cache_clear()
    set_providers_override(lambda sim, ind: prov)
    r = await client.post("/api/objectives", json={"objective": OBJ, "mode": "autonomous"})
    assert r.json()["simulated"] is False, r.json()
    return r.json()


async def test_blocked_site_is_replaced_by_backup_and_run_completes(client, monkeypatch):
    sites = {n: site(n) for n in NAMES[:5]}
    first = NAMES[0]
    prov, fetcher, _ = make_providers(sites, failures={f"https://{host(first)}": FetchError("HTTP 403", "blocked")})
    run = await _run_real(client, monkeypatch, prov)
    final = await wait_status(client, run["id"], {"completed", "failed", "partial", "halted"})
    assert final["status"] == "completed", final["error"]
    assert final["recovery_count"] >= 1
    trace = (await client.get(f"/api/runs/{run['id']}/trace")).json()
    types = [e["type"] for e in trace["events"]]
    assert "recovery.started" in types and "recovery.completed" in types and "task.failed" in types
    assert fetcher.calls.count(f"https://{host(first)}") == 1  # the blocked site was tried once, never hammered or bypassed
    ev = (await client.get(f"/api/runs/{run['id']}/evidence")).json()
    assert ev and all(e["simulated"] is False for e in ev)
    assert first not in {e["subject"] for e in ev}


async def test_retries_are_bounded_when_everything_fails(client, monkeypatch):
    sites = {n: site(n) for n in NAMES[:5]}
    fails = {f"https://{host(n)}": FetchError("connection reset", "network") for n in NAMES[:5]}
    prov, fetcher, _ = make_providers(sites, failures=fails)
    run = await _run_real(client, monkeypatch, prov)
    final = await wait_status(client, run["id"], {"completed", "failed", "partial", "halted"})
    assert final["status"] == "failed"  # never "completed" when collection cannot succeed
    assert len(fetcher.calls) < 40  # bounded, not an infinite retry loop
    assert any(t["status"] in ("failed", "blocked") for t in final["tasks"])
    assert final["error"]


async def test_missing_search_provider_halts_for_human(client, monkeypatch):
    sites = {n: site(n) for n in NAMES[:5]}
    prov, _, _ = make_providers(sites)
    prov.search, prov.search_error = None, "SEARCH_PROVIDER=tavily but TAVILY_API_KEY is not set"
    run = await _run_real(client, monkeypatch, prov)
    final = await wait_status(client, run["id"], {"completed", "failed", "partial", "halted"})
    assert final["status"] == "halted" and final["halt_reason"] == "needs_human"
    assert "TAVILY_API_KEY" in final["error"]
