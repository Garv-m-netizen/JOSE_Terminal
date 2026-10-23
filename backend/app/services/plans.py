"""Plan versions and plan-bound approval.

Every distinct task graph of a run is stored as a numbered version with a content hash. A human approval is recorded
against one hash: changing the graph (user edit, regeneration, replan) produces a new hash, and execution refuses to
start unless the current hash is the approved one.
"""
from __future__ import annotations

import hashlib
import json

from ..database import session_scope
from ..models import PlanVersion, Run, utcnow
from ..schemas.domain import TaskSpec

RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
BOOKKEEPING = {"_recoveries", "_tried", "candidate_index", "force_browser", "broaden"}  # recovery nudges, not part of what was approved


class StalePlan(ValueError):
    """Approval refers to a plan that is no longer the current one."""


def canonical(specs: list[TaskSpec]) -> list[dict]:
    """Only the parts that define *what will happen*. Runtime bookkeeping params (leading underscore) are excluded."""
    return [{"key": s.key, "agent": s.agent, "action": s.action, "depends_on": sorted(s.depends_on),
             "params": {k: v for k, v in sorted(s.params.items()) if not k.startswith("_") and k not in BOOKKEEPING}} for s in specs]


def hash_specs(specs: list[TaskSpec]) -> str:
    blob = json.dumps(canonical(specs), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def max_risk(specs: list[TaskSpec]) -> str:
    return max((s.risk_level for s in specs), key=RISK_ORDER.get, default="LOW")  # type: ignore[arg-type]


def record_version(db, run_id: str, specs: list[TaskSpec], reason: str) -> dict | None:
    """Called inside the transaction that replaces the tasks. Returns event details when a new version was created."""
    run = db.get(Run, run_id)
    if run is None:
        return None
    h = hash_specs(specs)
    if run.plan_hash == h:
        return None
    run.plan_version = (run.plan_version or 0) + 1
    run.plan_hash = h
    mr = max_risk(specs)
    db.add(PlanVersion(run_id=run_id, version=run.plan_version, plan_hash=h, reason=reason, max_risk=mr,
                       tasks=[{**c, "title": s.title, "risk_level": s.risk_level} for c, s in zip(canonical(specs), specs)]))
    invalidated = False
    if run.plan_approved and run.status in ("created", "planning", "awaiting_plan_approval"):
        run.plan_approved, invalidated = False, True  # an approval never survives a change of the plan it was given for
    return {"version": run.plan_version, "plan_hash": h, "reason": reason, "max_risk": mr, "tasks": len(specs), "approval_invalidated": invalidated}


def approve(run_id: str, by: str, expected_hash: str | None = None) -> dict:
    """Bind the approval to the current plan hash. `expected_hash` (what the approver actually saw) must match."""
    with session_scope() as db:
        run = db.get(Run, run_id)
        if expected_hash is not None and expected_hash != run.plan_hash:
            raise StalePlan(f"the plan changed after you reviewed it (you approved {expected_hash}, the current plan is {run.plan_hash}); review it again")
        run.plan_approved, run.approved_plan_hash, run.approved_plan_version = True, run.plan_hash, run.plan_version
        pv = db.query(PlanVersion).filter_by(run_id=run_id, version=run.plan_version).first()
        if pv:
            pv.approved_at, pv.approved_by = utcnow(), by
        return {"version": run.plan_version, "plan_hash": run.plan_hash, "by": by}


def is_approved_plan(run_id: str) -> bool:
    with session_scope() as db:
        run = db.get(Run, run_id)
        return bool(run and run.plan_approved and run.approved_plan_hash and run.approved_plan_hash == run.plan_hash)


def carry_approval(run_id: str) -> dict:
    """After a recovery replan: keep the approval only when the new plan is no riskier than the approved one.

    Consequential tasks still need their own per-action approval at execution time, so carrying is safe for equal/lower risk.
    """
    with session_scope() as db:
        run = db.get(Run, run_id)
        cur = db.query(PlanVersion).filter_by(run_id=run_id, version=run.plan_version).first()
        old = db.query(PlanVersion).filter_by(run_id=run_id, version=run.approved_plan_version).first() if run.approved_plan_version else None
        old_high = sorted(t["action"] for t in (old.tasks if old else []) if t.get("risk_level") == "HIGH")
        new_high = sorted(t["action"] for t in (cur.tasks if cur else []) if t.get("risk_level") == "HIGH")
        escalated = bool(old is None or cur is None or RISK_ORDER[cur.max_risk] > RISK_ORDER[old.max_risk]
                         or any(a not in old_high for a in new_high))
        if not escalated:
            run.approved_plan_hash, run.approved_plan_version = run.plan_hash, run.plan_version
            cur.approved_at, cur.approved_by = utcnow(), "carried"
        else:
            run.plan_approved = False
        return {"carried": not escalated, "escalated": escalated, "version": run.plan_version}


def list_versions(run_id: str) -> list[dict]:
    with session_scope() as db:
        run = db.get(Run, run_id)
        rows = db.query(PlanVersion).filter_by(run_id=run_id).order_by(PlanVersion.version).all()
        return [{"version": v.version, "plan_hash": v.plan_hash, "reason": v.reason, "max_risk": v.max_risk, "tasks": v.tasks,
                 "created_at": v.created_at.isoformat(), "approved_at": v.approved_at.isoformat() if v.approved_at else None,
                 "approved_by": v.approved_by, "current": bool(run and v.version == run.plan_version)} for v in rows]


def sync_current(run_id: str, reason: str) -> dict | None:
    """Re-hash the tasks as they are now (after in-place argument repairs) and record a new version when they differ."""
    from . import store
    from .event_service import events

    specs = [TaskSpec(key=t.key, title=t.title, description=t.description, agent=t.agent, action=t.action,  # type: ignore[arg-type]
                      params=t.params, depends_on=t.depends_on, resources=t.resources, expected_output=t.expected_output,
                      verification_criteria=t.verification_criteria, risk_level=t.risk_level)  # type: ignore[arg-type]
             for t in store.get_tasks(run_id)]
    with session_scope() as db:
        ver = record_version(db, run_id, specs, reason)
    if ver:
        events.emit(run_id, "plan.version", **ver)
    return ver
