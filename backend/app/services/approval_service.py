"""Human-approval requests. The orchestrator blocks on these; no code path lets a model resolve one."""
from __future__ import annotations

import asyncio

from sqlalchemy import select

from ..database import session_scope
from ..models import Approval, utcnow
from .event_service import events
from .store import NotFound, iso

_waiters: dict[str, asyncio.Event] = {}


def to_dict(a: Approval) -> dict:
    return {"id": a.id, "run_id": a.run_id, "task_id": a.task_id, "risk_level": a.risk_level, "action": a.action,
            "target": a.target, "details": a.details, "consequences": a.consequences, "status": a.status,
            "note": a.note, "created_at": iso(a.created_at), "resolved_at": iso(a.resolved_at)}


def create(run_id: str, task_id: str | None, risk_level: str, action: str, target: str, details: dict,
           consequences: str) -> dict:
    with session_scope() as s:
        a = Approval(run_id=run_id, task_id=task_id, risk_level=risk_level, action=action, target=target,
                     details=details, consequences=consequences)
        s.add(a)
        s.flush()
        d = to_dict(a)
    _waiters[d["id"]] = asyncio.Event()
    events.emit(run_id, "approval.required", task_id, approval_id=d["id"], risk_level=risk_level, action=action,
                target=target, details=details, consequences=consequences)
    return d


def resolve(approval_id: str, decision: str, note: str | None = None) -> dict:
    with session_scope() as s:
        a = s.get(Approval, approval_id)
        if not a:
            raise NotFound("approval not found")
        if a.status != "pending":
            raise ValueError(f"approval already {a.status}")
        a.status, a.note, a.resolved_at = decision, note, utcnow()
        s.flush()
        d = to_dict(a)
    events.emit(d["run_id"], "approval.resolved", d["task_id"], approval_id=approval_id, decision=decision, note=note)
    ev = _waiters.get(approval_id)
    if ev:
        ev.set()
    return d


async def wait(approval_id: str, cancel: asyncio.Event) -> str:
    """Block until a human decides (or the run is cancelled). Returns approved | denied | cancelled."""
    ev = _waiters.setdefault(approval_id, asyncio.Event())
    while True:
        with session_scope() as s:
            a = s.get(Approval, approval_id)
            if a and a.status != "pending":
                return a.status
        if cancel.is_set():
            return "cancelled"
        try:
            await asyncio.wait_for(ev.wait(), timeout=0.5)
        except asyncio.TimeoutError:
            pass


def list_approvals(status: str | None = None, run_id: str | None = None) -> list[dict]:
    with session_scope() as s:
        q = select(Approval).order_by(Approval.created_at.desc())
        if status:
            q = q.where(Approval.status == status)
        if run_id:
            q = q.where(Approval.run_id == run_id)
        return [to_dict(a) for a in s.scalars(q)]
