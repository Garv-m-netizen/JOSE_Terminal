"""External-action ledger: the record of things JOSE did (or may have done) to the outside world.

It exists so that a retry, a recovery, an interruption or a "continue" can never send the same email / Telegram post twice,
and so the final report can state what a provider actually returned.

Lifecycle:  claim() -> started -> accepted | failed | ambiguous
  * accepted  - the provider confirmed (message id / response recorded). Never repeated.
  * failed    - the provider definitely did not perform it (e.g. authentication rejected). May be attempted again.
  * ambiguous - the request may have reached the provider. Never repeated automatically; a human must check.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from sqlalchemy import select

from ..database import session_scope
from ..models import ExternalAction, utcnow
from .event_service import events
from .store import iso


@dataclass
class Claim:
    action: dict
    proceed: bool  # True: the caller owns this attempt and may perform the action
    reason: str = ""  # why the caller must not proceed


def make_key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]


def to_dict(a: ExternalAction) -> dict:
    return {"id": a.id, "run_id": a.run_id, "task_id": a.task_id, "kind": a.kind, "idempotency_key": a.idempotency_key,
            "target": a.target, "summary": a.summary, "status": a.status, "attempts": a.attempts, "reference": a.reference,
            "response": a.response or {}, "error": a.error, "simulated": a.simulated,
            "created_at": iso(a.created_at), "updated_at": iso(a.updated_at)}


def claim(run_id: str, task_id: str | None, kind: str, key: str, target: str, summary: str, simulated: bool = False) -> Claim:
    """Atomically decide whether this caller may perform the action."""
    with session_scope() as s:
        row = s.scalar(select(ExternalAction).where(ExternalAction.idempotency_key == key))
        if row is None:
            row = ExternalAction(run_id=run_id, task_id=task_id, kind=kind, idempotency_key=key, target=target[:300],
                                 summary=summary[:1000], status="started", simulated=simulated)
            s.add(row)
            s.flush()
            d = to_dict(row)
            fresh = True
        else:
            fresh = False
            if row.status == "failed":  # definitely not performed last time: a deliberate new attempt is allowed
                row.status, row.attempts, row.updated_at, row.error = "started", row.attempts + 1, utcnow(), None
                s.flush()
                d = to_dict(row)
                fresh = True
            else:
                d = to_dict(row)
    if fresh:
        events.emit(run_id, "external.action", task_id, kind=kind, status="started", target=target[:200], attempt=d["attempts"])
        return Claim(d, True)
    why = {"accepted": "this exact action was already performed and accepted by the service; it will not be repeated",
           "started": "this action is already in flight or was interrupted before an outcome was recorded; it may have happened",
           "ambiguous": "an earlier attempt may or may not have reached the service; check the destination before repeating it"}.get(d["status"], d["status"])
    return Claim(d, False, why)


def _finish(action_id: str, status: str, *, reference: str | None = None, response: dict | None = None, error: str | None = None) -> dict:
    with session_scope() as s:
        row = s.get(ExternalAction, action_id)
        row.status, row.updated_at = status, utcnow()
        if reference is not None:
            row.reference = reference
        if response is not None:
            row.response = response
        row.error = error
        s.flush()
        d = to_dict(row)
    events.emit(d["run_id"], "external.action", d["task_id"], kind=d["kind"], status=status, reference=reference, error=error, target=d["target"][:200])
    return d


def accepted(action_id: str, reference: str | None, response: dict) -> dict:
    return _finish(action_id, "accepted", reference=reference, response=response)


def failed(action_id: str, error: str) -> dict:
    return _finish(action_id, "failed", error=error[:500])


def ambiguous(action_id: str, error: str) -> dict:
    return _finish(action_id, "ambiguous", error=error[:500])


def interrupt_in_flight() -> int:
    """On startup: anything still 'started' lost its outcome when the process died. It is now ambiguous, not retryable."""
    n = 0
    with session_scope() as s:
        for row in s.scalars(select(ExternalAction).where(ExternalAction.status == "started")):
            row.status, row.updated_at, row.error = "ambiguous", utcnow(), "the server stopped before the outcome was recorded"
            n += 1
    return n


def for_run(run_id: str) -> list[dict]:
    with session_scope() as s:
        return [to_dict(a) for a in s.scalars(select(ExternalAction).where(ExternalAction.run_id == run_id).order_by(ExternalAction.created_at))]


def for_task(run_id: str, task_id: str) -> list[dict]:
    return [a for a in for_run(run_id) if a["task_id"] == task_id]


def resolve_manually(action_id: str, outcome: str, note: str | None = None) -> dict:
    """A person checked the destination. performed -> accepted (never repeated); not_performed -> failed (may be attempted again)."""
    with session_scope() as s:
        row = s.get(ExternalAction, action_id)
        if row is None:
            raise LookupError("external action not found")
        if row.status not in ("started", "ambiguous"):
            raise ValueError(f"only an action with an unknown outcome can be settled (this one is {row.status})")
        prev = dict(row.response or {})
    if outcome == "performed":
        return _finish(action_id, "accepted", reference=row.reference or f"manual:{action_id}", response={**prev, "status": "confirmed_by_user", "manual": True, "note": note})
    return _finish(action_id, "failed", error=f"user confirmed it did not happen{': ' + note if note else ''}")
