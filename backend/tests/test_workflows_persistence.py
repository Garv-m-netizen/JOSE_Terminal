from __future__ import annotations

from app import database
from app.config import get_settings
from app.services import execution_service as ex

from .conftest import OBJECTIVE, wait_status


async def _done(client, mode="autonomous", objective=OBJECTIVE):
    r = (await client.post("/api/objectives", json={"objective": objective, "mode": mode})).json()
    return await wait_status(client, r["id"], {"completed", "failed", "partial", "halted"})


async def test_state_survives_restart(client):
    run = await _done(client)
    assert run["status"] == "completed"
    # simulate a process restart: new engine on the same database file
    database.init_engine(get_settings().db_url)
    again = (await client.get(f"/api/runs/{run['id']}")).json()
    assert again["status"] == "completed" and len(again["tasks"]) == 15 and len(again["artifacts"]) == 2
    trace = (await client.get(f"/api/runs/{run['id']}/trace")).json()
    assert trace["events"][0]["type"] == "run.created" and trace["events"][-1]["type"] == "run.completed"
    assert trace["tool_calls"] and trace["agent_activity"] and trace["evidence"]
    ids = [e["id"] for e in trace["events"]]
    assert ids == sorted(ids) and len(ids) == len(set(ids))


async def test_interrupted_runs_are_marked_failed_on_startup(client):
    r = (await client.post("/api/objectives", json={"objective": OBJECTIVE, "mode": "guided"})).json()
    assert r["status"] == "awaiting_plan_approval"
    assert ex.manager.mark_interrupted() == 0  # parked runs are not "in flight"
    from app.database import session_scope
    from app.models import Run

    with session_scope() as s:
        s.get(Run, r["id"]).status = "running"
    assert ex.manager.mark_interrupted() == 1
    got = (await client.get(f"/api/runs/{r['id']}")).json()
    assert got["status"] == "failed" and got["halt_reason"] == "server_restart"


async def test_save_and_rerun_workflow_with_new_inputs(client):
    run = await _done(client)
    saved = (await client.post("/api/workflows", json={"run_id": run["id"], "name": "AI automation scan", "description": "d"})).json()
    assert saved["params"]["industry"] == "AI automation" and len(saved["task_config"]) == 15
    assert saved["task_config"][1]["action"] == "browser.collect"
    assert (await client.get("/api/workflows")).json()[0]["task_count"] == 15
    # same shape, new industry -> saved task configuration is reused
    r2 = await client.post(f"/api/workflows/{saved['id']}/run", json={"params": {"industry": "CRM software"}, "mode": "autonomous"})
    assert r2.status_code == 201
    final = await wait_status(client, r2.json()["id"], {"completed", "failed", "halted"})
    assert final["status"] == "completed" and final["template_id"] == saved["id"]
    assert final["params"]["industry"] == "CRM software"
    assert "CRM software" in final["tasks"][0]["title"] and "AI automation" not in final["tasks"][0]["title"]
    # different competitor count -> plan rebuilt
    r3 = await client.post(f"/api/workflows/{saved['id']}/run", json={"params": {"competitor_count": 2, "outputs": ["xlsx"]}, "mode": "autonomous"})
    f3 = await wait_status(client, r3.json()["id"], {"completed", "failed", "halted"})
    assert f3["status"] == "completed" and len(f3["tasks"]) == 1 + 4 + 1 + 1 + 1
    assert [a["kind"] for a in f3["artifacts"] if a["kind"] != "screenshot"] == ["xlsx"]
    # invalid parameters are rejected
    bad = await client.post(f"/api/workflows/{saved['id']}/run", json={"params": {"competitor_count": 99}})
    assert bad.status_code == 422
    assert (await client.delete(f"/api/workflows/{saved['id']}")).status_code == 204
    assert (await client.get(f"/api/workflows/{saved['id']}")).status_code == 404


async def test_only_completed_runs_can_be_saved(client):
    r = (await client.post("/api/objectives", json={"objective": OBJECTIVE, "mode": "guided"})).json()
    res = await client.post("/api/workflows", json={"run_id": r["id"], "name": "nope"})
    assert res.status_code == 409


async def test_stats_come_from_persisted_data(client):
    empty = (await client.get("/api/stats")).json()
    assert empty["total_runs"] == 0 and empty["success_rate"] is None
    await _done(client)
    await client.post("/api/objectives", json={"objective": "Research competitors in CRM and email everyone", "mode": "autonomous"})
    st = (await client.get("/api/stats")).json()
    assert st["total_runs"] == 2 and st["completed"] == 1 and st["by_status"]["rejected"] == 1
    assert st["success_rate"] == 0.5 and st["artifacts"] == 2 and st["avg_duration_seconds"] is not None
    assert len((await client.get("/api/runs")).json()) == 2
