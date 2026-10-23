"""Persisted audit events + live fan-out for SSE. Secrets are redacted before anything is stored or sent."""
from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from typing import Any

from sqlalchemy import select

from ..database import session_scope
from ..models import Event, utcnow

log = logging.getLogger("jose.events")

EVENT_TYPES = {
    "run.created", "run.started", "run.paused", "run.resumed", "intent.parsed", "plan.generated", "plan.updated",
    "plan.approved", "task.started", "task.completed", "task.failed", "task.blocked", "agent.action",
    "tool.started", "tool.completed", "tool.failed", "observation.recorded", "verification.passed",
    "verification.failed", "recovery.started", "recovery.completed", "approval.required", "approval.resolved",
    "artifact.created", "warning", "run.completed", "run.failed", "run.cancelled", "run.halted", "run.rejected",
    "run.partial", "plan.version", "run.continued", "run.needs_input", "external.action",
}
_SECRET_KEY = re.compile(r"(api[_-]?key|token|secret|password|authorization|cookie)", re.I)
_SECRET_VAL = re.compile(r"(AIza[0-9A-Za-z_\-]{20,}|sk-[A-Za-z0-9]{20,}|tvly-[A-Za-z0-9]{10,}|Bearer\s+[A-Za-z0-9._\-]{16,})")
MAX_STR = 1500


def redact(obj: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "…"
    if isinstance(obj, dict):
        return {k: ("[redacted]" if _SECRET_KEY.search(str(k)) else redact(v, depth + 1)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(v, depth + 1) for v in list(obj)[:60]]
    if isinstance(obj, str):
        s = _SECRET_VAL.sub("[redacted]", obj)
        return s if len(s) <= MAX_STR else s[:MAX_STR] + f"… [{len(s) - MAX_STR} more chars]"
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    return redact(str(obj), depth + 1)


def serialize(ev: Event) -> dict:
    return {"id": ev.id, "run_id": ev.run_id, "type": ev.type, "task_id": ev.task_id,
            "ts": ev.ts.isoformat(), "details": ev.details or {}}


class EventService:
    def __init__(self) -> None:
        self._subs: dict[str, set[asyncio.Queue]] = defaultdict(set)

    def emit(self, run_id: str, type_: str, task_id: str | None = None, **details: Any) -> dict:
        if type_ not in EVENT_TYPES:
            raise ValueError(f"unknown event type {type_}")
        with session_scope() as s:
            ev = Event(run_id=run_id, type=type_, task_id=task_id, ts=utcnow(), details=redact(details))
            s.add(ev)
            s.flush()
            data = serialize(ev)
        for q in list(self._subs.get(run_id, ())):
            q.put_nowait(data)
        log.info("event run=%s type=%s task=%s", run_id, type_, task_id)
        return data

    def subscribe(self, run_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs[run_id].add(q)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        self._subs[run_id].discard(q)
        if not self._subs[run_id]:
            self._subs.pop(run_id, None)

    @staticmethod
    def replay(run_id: str, after_id: int = 0, limit: int = 5000) -> list[dict]:
        with session_scope() as s:
            rows = s.scalars(select(Event).where(Event.run_id == run_id, Event.id > after_id)
                             .order_by(Event.id).limit(limit)).all()
            return [serialize(r) for r in rows]


events = EventService()
