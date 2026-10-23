from __future__ import annotations

import asyncio

from app.services import risk
from app.services.risk import assess, needs_approval

from .conftest import OBJECTIVE, wait_status


def test_risk_classification():
    assert assess("research.discover", {}).level == "LOW"
    assert assess("browser.collect", {}, {"url": "https://example.com"}).level == "LOW"
    assert assess("browser.collect", {}, {"url": "http://example.com"}).level == "MEDIUM"
    assert assess("shell.exec", {}).level == "HIGH"  # unknown / non-allow-listed actions are never LOW


def test_overwrite_is_high_risk_only_when_target_exists():
    p = {"export_name": "q3", "filename": "competitors.xlsx"}
    assert assess("workspace.build_excel", p).level == "LOW"
    dst = risk.export_path("q3", "competitors.xlsx")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(b"previous")
    a = assess("workspace.build_excel", p)
    assert a.level == "HIGH" and "replaced" in a.consequences and str(dst) == a.target


def test_approval_policy():
    assert needs_approval("HIGH", "autonomous") and needs_approval("HIGH", "guided")
    assert not needs_approval("LOW", "guided")
    assert needs_approval("MEDIUM", "guided") and needs_approval("MEDIUM", "autonomous")
    from app.services import runtime_settings

    runtime_settings.update({"auto_approve_medium": True})
    assert not needs_approval("MEDIUM", "autonomous") and needs_approval("MEDIUM", "guided")
    assert needs_approval("HIGH", "autonomous")  # policy can never auto-approve HIGH


async def _run(client, export="latest"):
    r = await client.post("/api/objectives", json={"objective": OBJECTIVE, "mode": "autonomous", "export_name": export})
    return r.json()


async def _wait_approval(client, run_id, timeout=20):
    end = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < end:
        aps = (await client.get(f"/api/approvals?status=pending&run_id={run_id}")).json()
        if aps:
            return aps[0]
        await asyncio.sleep(0.05)
    raise AssertionError("no approval requested")


async def test_overwrite_requires_approval_and_denial_stops_the_action(client):
    first = await _run(client)
    assert (await wait_status(client, first["id"], {"completed", "failed", "halted"}))["status"] == "completed"
    exported = risk.export_path("latest", "competitors.xlsx")
    before = exported.read_bytes()
    second = await _run(client)
    ap = await _wait_approval(client, second["id"])
    assert ap["risk_level"] == "HIGH" and ap["action"] == "workspace.build_excel"
    assert "exports" in ap["target"] and "cannot be restored" in ap["consequences"]
    mid = (await client.get(f"/api/runs/{second['id']}")).json()
    assert mid["status"] == "awaiting_approval"
    res = await client.post(f"/api/approvals/{ap['id']}/resolve", json={"decision": "denied", "note": "keep the old file"})
    assert res.status_code == 200
    final = await wait_status(client, second["id"], {"completed", "failed", "halted"})
    assert final["status"] == "halted" and final["halt_reason"] == "approval_denied"
    assert exported.read_bytes() == before  # the denied action did not run
    assert "blocked" in {x["status"] for x in final["tasks"]}
    trace = (await client.get(f"/api/runs/{second['id']}/trace")).json()
    types = [e["type"] for e in trace["events"]]
    assert "approval.required" in types and "approval.resolved" in types and "run.halted" in types
    # a resolved approval cannot be resolved again
    assert (await client.post(f"/api/approvals/{ap['id']}/resolve", json={"decision": "approved"})).status_code == 409


async def test_approved_overwrite_proceeds(client):
    first = await _run(client, "reports")
    await wait_status(client, first["id"], {"completed"})
    second = await _run(client, "reports")
    seen = set()
    for _ in range(2):  # the Excel and the PowerPoint export each need their own approval
        ap = await _wait_approval(client, second["id"])
        seen.add(ap["action"])
        await client.post(f"/api/approvals/{ap['id']}/resolve", json={"decision": "approved"})
    assert seen == {"workspace.build_excel", "workspace.build_pptx"}
    final = await wait_status(client, second["id"], {"completed", "failed", "halted"})
    assert final["status"] == "completed"
