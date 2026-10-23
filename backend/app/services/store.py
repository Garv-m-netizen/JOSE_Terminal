"""Database helpers with validated state transitions. All writes are short, atomic transactions."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import select

from ..database import session_scope
from ..models import AgentActivity, Run, Task, ToolCall, VerificationResult, utcnow
from ..models.state import RUN_TERMINAL, check_run_transition, check_task_transition
from ..schemas.domain import TaskSpec
from .event_service import events


class NotFound(LookupError):
    pass


@dataclass
class TaskView:
    id: str
    run_id: str
    key: str
    title: str
    agent: str
    action: str
    params: dict
    depends_on: list[str]
    resources: list[str]
    status: str
    attempts: int
    risk_level: str
    description: str = ""
    expected_output: str = ""
    verification_criteria: list[str] = field(default_factory=list)
    output: dict | None = None
    error: str | None = None


def task_view(t: Task) -> TaskView:
    return TaskView(id=t.id, run_id=t.run_id, key=t.key, title=t.title, agent=t.agent, action=t.action,
                    params=dict(t.params or {}), depends_on=list(t.depends_on or []), resources=list(t.resources or []),
                    status=t.status, attempts=t.attempts, risk_level=t.risk_level, description=t.description,
                    expected_output=t.expected_output, verification_criteria=list(t.verification_criteria or []),
                    output=t.output, error=t.error)


def iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def run_dict(r: Run) -> dict:
    return {"id": r.id, "objective": r.objective, "mode": r.mode, "simulated": r.simulated, "status": r.status,
            "risk_level": r.risk_level, "workflow_type": r.workflow_type, "intent": r.intent, "params": r.params,
            "plan_approved": r.plan_approved, "plan_version": r.plan_version, "plan_hash": r.plan_hash,
            "approved_plan_hash": r.approved_plan_hash, "template_id": r.template_id, "halt_reason": r.halt_reason,
            "error": r.error, "warnings": r.warnings or [], "recovery_count": r.recovery_count,
            "created_at": iso(r.created_at), "started_at": iso(r.started_at), "finished_at": iso(r.finished_at),
            "duration_seconds": ((r.finished_at or utcnow()) - r.started_at).total_seconds() if r.started_at else None}


def task_dict(t: Task) -> dict:
    return {"id": t.id, "run_id": t.run_id, "key": t.key, "position": t.position, "title": t.title,
            "description": t.description, "agent": t.agent, "action": t.action, "params": t.params,
            "depends_on": t.depends_on, "resources": t.resources, "status": t.status, "risk_level": t.risk_level,
            "expected_output": t.expected_output, "verification_criteria": t.verification_criteria,
            "attempts": t.attempts, "error": t.error, "started_at": iso(t.started_at), "finished_at": iso(t.finished_at),
            "output_summary": summarize_output(t.output)}


def summarize_output(out: dict | None) -> dict | None:
    if not out:
        return None
    return {k: (v if isinstance(v, (int, float, bool, str)) and len(str(v)) < 200 else f"<{type(v).__name__}>")
            for k, v in out.items() if k != "pages"}


def get_run(run_id: str) -> Run:
    with session_scope() as s:
        r = s.get(Run, run_id)
        if not r:
            raise NotFound(f"run {run_id} not found")
        return r


def set_run_status(run_id: str, new: str, *, error: str | None = None, halt_reason: str | None = None,
                   event: str | None = None, resume: bool = False, **event_details: Any) -> None:
    with session_scope() as s:
        r = s.get(Run, run_id)
        if not r:
            raise NotFound(run_id)
        if r.status in RUN_TERMINAL and r.status != new and not resume:
            return  # terminal states are final (only an explicit, audited "continue" may leave them)
        check_run_transition(r.status, new, resume=resume)
        r.status = new
        if resume:
            r.finished_at, r.error, r.halt_reason = None, None, None
        if error is not None:
            r.error = error
        if halt_reason is not None:
            r.halt_reason = halt_reason
        if new == "running" and r.started_at is None:
            r.started_at = utcnow()
        if new in RUN_TERMINAL:
            r.finished_at = utcnow()
            if r.started_at is None:
                r.started_at = r.created_at
    if event:
        events.emit(run_id, event, status=new, **event_details)


def add_warning(run_id: str, text: str) -> None:
    with session_scope() as s:
        r = s.get(Run, run_id)
        if r and text not in (r.warnings or []):
            r.warnings = [*(r.warnings or []), text]
    events.emit(run_id, "warning", message=text)


def get_tasks(run_id: str) -> list[TaskView]:
    with session_scope() as s:
        rows = s.scalars(select(Task).where(Task.run_id == run_id).order_by(Task.position)).all()
        return [task_view(t) for t in rows]


def get_task(task_id: str) -> Task:
    with session_scope() as s:
        t = s.get(Task, task_id)
        if not t:
            raise NotFound(f"task {task_id} not found")
        return t


def replace_tasks(run_id: str, specs: list[TaskSpec], reason: str = "generated") -> None:
    """Persist a (re)generated plan as a new plan version (when it differs). Approval is bound to the version hash."""
    from . import plans

    ver = None
    with session_scope() as s:
        for t in s.scalars(select(Task).where(Task.run_id == run_id)).all():
            s.delete(t)
        s.flush()
        for i, sp in enumerate(specs):
            s.add(Task(id=f"{run_id}:{sp.key}", run_id=run_id, key=sp.key, position=i, title=sp.title,
                       description=sp.description, agent=sp.agent, action=sp.action, params=sp.params,
                       depends_on=sp.depends_on, resources=sp.resources, expected_output=sp.expected_output,
                       verification_criteria=sp.verification_criteria, risk_level=sp.risk_level))
        s.flush()
        ver = plans.record_version(s, run_id, specs, reason)
    if ver:
        events.emit(run_id, "plan.version", **ver)


def set_task_status(task_id: str, new: str, *, error: str | None = None, output: dict | None = None,
                    event: str | None = None, **event_details: Any) -> None:
    run_id = None
    with session_scope() as s:
        t = s.get(Task, task_id)
        if not t:
            raise NotFound(task_id)
        check_task_transition(t.status, new)
        t.status = new
        run_id = t.run_id
        if new == "running":
            t.started_at, t.finished_at, t.error = utcnow(), None, None
            t.attempts += 1
        if new in ("completed", "failed", "blocked", "cancelled"):
            t.finished_at = utcnow()
        if new == "pending":
            t.error = None
        if error is not None:
            t.error = error[:2000]
        if output is not None:
            t.output = output
    if event:
        events.emit(run_id, event, task_id, status=new, **({"error": error} if error else {}), **event_details)


def save_task_output(task_id: str, output: dict) -> None:
    with session_scope() as s:
        t = s.get(Task, task_id)
        if t:
            t.output = output


def update_task_params(task_id: str, params: dict) -> None:
    with session_scope() as s:
        t = s.get(Task, task_id)
        if t:
            t.params = params


def record_verification(run_id: str, task_id: str | None, check: str, passed: bool, reason: str, attempt: int = 1,
                        details: dict | None = None) -> None:
    with session_scope() as s:
        s.add(VerificationResult(run_id=run_id, task_id=task_id, check_name=check, passed=passed, reason=reason,
                                 attempt=attempt, details=details or {}))
    events.emit(run_id, "verification.passed" if passed else "verification.failed", task_id, check=check, reason=reason)


def begin_activity(run_id: str, task_id: str | None, agent: str, activity: str) -> str:
    with session_scope() as s:
        a = AgentActivity(run_id=run_id, task_id=task_id, agent=agent, activity=activity)
        s.add(a)
        s.flush()
        aid = a.id
    events.emit(run_id, "agent.action", task_id, agent=agent, activity=activity, activity_id=aid, state="started")
    return aid


def end_activity(run_id: str, aid: str, task_id: str | None, agent: str) -> None:
    with session_scope() as s:
        a = s.get(AgentActivity, aid)
        if a:
            a.ended_at = utcnow()
    events.emit(run_id, "agent.action", task_id, agent=agent, activity_id=aid, state="finished")


def begin_tool(run_id: str, task_id: str | None, tool: str, args: dict, simulated: bool) -> str:
    from .event_service import redact

    with session_scope() as s:
        tc = ToolCall(run_id=run_id, task_id=task_id, tool=tool, args=redact(args), simulated=simulated)
        s.add(tc)
        s.flush()
        tid = tc.id
    events.emit(run_id, "tool.started", task_id, tool=tool, args=args, tool_call_id=tid, simulated=simulated)
    return tid


def end_tool(run_id: str, tool_id: str, task_id: str | None, tool: str, *, ok: bool, summary: str = "",
             error: str | None = None, duration_ms: int = 0) -> None:
    with session_scope() as s:
        tc = s.get(ToolCall, tool_id)
        if tc:
            tc.status = "completed" if ok else "failed"
            tc.result_summary, tc.error, tc.duration_ms = summary[:1000], (error or None), duration_ms
    events.emit(run_id, "tool.completed" if ok else "tool.failed", task_id, tool=tool, tool_call_id=tool_id,
                summary=summary[:500], error=error, duration_ms=duration_ms)


def close_open_activities(run_id: str) -> None:
    with session_scope() as s:
        for a in s.scalars(select(AgentActivity).where(AgentActivity.run_id == run_id, AgentActivity.ended_at.is_(None))):
            a.ended_at = utcnow()
