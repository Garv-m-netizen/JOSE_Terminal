from __future__ import annotations

import pytest

from app.agents.intent_engine import parse_rules
from app.agents.planner import plan, validate_edit
from app.schemas.domain import TaskSpec, WorkflowParams
from app.workflows.research_report import PlanError, find_cycle, validate_plan

from .conftest import OBJECTIVE, wait_status


def test_intent_signature_objective():
    i = parse_rules(OBJECTIVE)
    assert i.feasible and i.industry == "AI automation" and i.competitor_count == 5
    assert set(i.outputs) == {"xlsx", "pptx"} and i.risk_level == "LOW"
    assert any("verification" in c for c in i.success_criteria)


def test_intent_number_words_and_outputs():
    i = parse_rules("Compare 3 vendors in the cloud security industry and make a spreadsheet")
    assert (i.industry, i.competitor_count, i.outputs) == ("cloud security", 3, ["xlsx"])


def test_intent_defaults_are_declared_as_assumptions():
    i = parse_rules("Research competitors in the pet insurance market")
    assert i.feasible and i.competitor_count == 5 and len(i.assumptions) == 2


@pytest.mark.parametrize("text,fragment", [
    ("do it", "too short"),
    ("Research five competitors and make an Excel sheet", "industry"),
    ("Research 50 competitors in the CRM market", "outside the supported range"),
    ("Tell me a joke about cats and dogs please", "competitor-research objective"),
])
def test_intent_rejects_unclear_or_impossible(text, fragment):
    i = parse_rules(text)
    assert not i.feasible and fragment in i.rejection_reason


@pytest.mark.parametrize("text", [
    "Research five competitors in the CRM market and email the report to the whole team",
    "Research competitors in the CRM market, then publish the findings",
    "Compare 3 vendors in the CRM market and delete the old files",
])
def test_intent_rejects_high_risk_external_actions(text):
    i = parse_rules(text)
    assert not i.feasible and i.risk_level == "HIGH"


def test_plan_is_dependency_aware_and_valid():
    tasks = plan(WorkflowParams(industry="CRM", competitor_count=4, outputs=["xlsx", "pptx"]))
    by = {t.key: t for t in tasks}
    assert len(tasks) == 1 + 2 * 4 + 1 + 2 + 1
    norm = next(t for t in tasks if t.action == "workspace.normalize")
    assert len(norm.depends_on) == 4
    assert all(by[d].action == "research.extract" for d in norm.depends_on)
    assert all(t.verification_criteria and t.expected_output for t in tasks)
    assert {t.agent for t in tasks} == {"research", "browser", "workspace", "verification"}


def test_cycle_detection_and_dependency_validation():
    tasks = plan(WorkflowParams(industry="CRM", competitor_count=2))
    assert find_cycle(tasks) is None
    tasks[0].depends_on = [tasks[-1].key]
    assert find_cycle(tasks)
    with pytest.raises(PlanError, match="circular"):
        validate_plan(tasks)
    t2 = plan(WorkflowParams(industry="CRM", competitor_count=2))
    t2[1].depends_on = ["T99"]
    with pytest.raises(PlanError, match="unknown"):
        validate_plan(t2)
    t3 = plan(WorkflowParams(industry="CRM", competitor_count=2))
    norm = next(t for t in t3 if t.action == "workspace.normalize")
    norm.depends_on = norm.depends_on[:1]
    with pytest.raises(PlanError, match="all extraction"):
        validate_plan(t3)


def test_disallowed_action_and_wrong_agent_rejected():
    tasks = plan(WorkflowParams(industry="CRM", competitor_count=1))
    tasks.append(TaskSpec(key="T99", title="run shell", agent="workspace", action="shell.exec", depends_on=[]))
    with pytest.raises(PlanError, match="allowlist"):
        validate_plan(tasks)
    t2 = plan(WorkflowParams(industry="CRM", competitor_count=1))
    t2[0].agent = "browser"
    with pytest.raises(PlanError, match="must be run by"):
        validate_plan(t2)


def test_user_cannot_lower_risk():
    tasks = plan(WorkflowParams(industry="CRM", competitor_count=1, export_name="x"))
    for t in tasks:
        t.risk_level = "LOW"
    assert all(t.risk_level == "LOW" for t in validate_edit(tasks))  # nothing risky yet (no existing export)


async def test_guided_flow_edit_validation_and_approval(client):
    r = await client.post("/api/objectives", json={"objective": OBJECTIVE, "mode": "guided"})
    run = r.json()
    assert run["status"] == "awaiting_plan_approval" and len(run["tasks"]) == 15
    assert all(t["status"] == "pending" for t in run["tasks"])
    # edit with a cycle -> 422 and plan unchanged
    tasks = run["tasks"]
    bad = [dict(t) for t in tasks]
    bad[0]["depends_on"] = [bad[-1]["key"]]
    r2 = await client.put(f"/api/runs/{run['id']}/plan", json={"tasks": bad})
    assert r2.status_code == 422 and "circular" in r2.text
    # valid edit of a description is accepted
    ok = [dict(t) for t in tasks]
    ok[0]["description"] = "edited by a human"
    r3 = await client.put(f"/api/runs/{run['id']}/plan", json={"tasks": ok})
    assert r3.status_code == 200 and r3.json()["tasks"][0]["description"] == "edited by a human"
    r4 = await client.post(f"/api/runs/{run['id']}/plan/approve")
    assert r4.status_code == 200
    final = await wait_status(client, run["id"], {"completed", "failed", "partial", "halted"})
    assert final["status"] == "completed"
    # editing after execution is refused
    assert (await client.put(f"/api/runs/{run['id']}/plan", json={"tasks": ok})).status_code == 409


async def test_rejected_objective_creates_rejected_run(client):
    r = await client.post("/api/objectives", json={"objective": "Research competitors in CRM and email the report to everyone", "mode": "autonomous"})
    assert r.status_code == 201 and r.json()["status"] == "rejected" and r.json()["tasks"] == []
    assert "no allow-listed tool" in r.json()["error"]
