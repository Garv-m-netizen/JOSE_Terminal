"""REST + SSE API."""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlalchemy import select

from ..config import get_settings
from ..database import session_scope
from ..models import Task, Upload, Workflow
from ..models.state import RUN_TERMINAL
from ..schemas.domain import (ApprovalResolve, ExternalActionResolve, ObjectiveCreate, PlanApprove, PlanUpdate, SettingsUpdate, WorkflowCreate,
                              WorkflowRunRequest)
from ..services import approval_service, artifact_service, capabilities, execution_service as ex, ledger, plans, report, runtime_settings, store
from ..services.event_service import events
from ..services.store import NotFound, iso
from ..tools.browser import BrowserTool
from ..tools.email_provider import email_status
from ..tools.file_tools import FileRejected, validate_upload

router = APIRouter(prefix="/api")
ID_RE = re.compile(r"^[A-Za-z0-9_:\-]{1,80}$")
TERMINAL_EVENTS = {"run.completed", "run.failed", "run.cancelled", "run.halted", "run.rejected", "run.partial"}


def _id(value: str) -> str:
    if not ID_RE.match(value):
        raise HTTPException(422, "malformed identifier")
    return value


# ---------------------------------------------------------------- health / status / settings
@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


def _pdf_converter() -> str | None:
    from ..tools.template_fill import find_converter

    return find_converter()


def _system_status() -> dict:
    s = get_settings()
    ok, why = BrowserTool.available()
    provider = s.search_provider.lower()
    key_ok = {"tavily": bool(s.tavily_api_key), "brave": bool(s.brave_api_key), "duckduckgo": True}.get(provider, False)
    return {
        "mode": "demo" if s.demo_forced else "real",
        "demo_forced": s.demo_forced,
        "demo_reason": (None if not s.demo_forced else "JOSE_MODE=demo" if s.jose_mode.lower() == "demo" else "neither GEMINI_API_KEY nor GROQ_API_KEY is configured"),
        "ai": {"provider": "gemini", "model": s.gemini_model, "configured": bool(s.gemini_api_key.strip()),
               "fallback": {"provider": "groq", "model": s.groq_model, "configured": bool(s.groq_api_key.strip())}},
        "search": {"provider": provider, "configured": key_ok,
                   "note": None if key_ok else f"{provider} requires an API key" if provider in ("tavily", "brave") else "no provider"},
        "browser": {"engine": "playwright-chromium", "available": ok, "detail": why},
        "email": email_status(),
        "workspace_roots": [str(r) for r in s.workspace_roots()],
        "telegram": {"configured": bool(s.telegram_bot_token and s.telegram_chat_id),
                     "note": None if (s.telegram_bot_token and s.telegram_chat_id) else "set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to post for real"},
        "pdf_converter": _pdf_converter(),
        "database": "sqlite", "output_dir": str(runtime_settings.load().output_path),
    }


@router.get("/status")
async def status() -> dict:
    return _system_status()


@router.get("/settings")
async def get_settings_api() -> dict:
    return {"execution": runtime_settings.load().__dict__, "system": _system_status()}


@router.put("/settings")
async def put_settings(body: SettingsUpdate) -> dict:
    try:
        runtime_settings.update(body.model_dump(exclude_none=True))
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return await get_settings_api()


# ---------------------------------------------------------------- uploads
@router.post("/uploads", status_code=201)
async def upload(file: UploadFile) -> dict:
    s = get_settings()
    data = await file.read(s.max_upload_mb * 1024 * 1024 + 1)
    try:
        name = validate_upload(file.filename or "", len(data), s.max_upload_mb)
    except FileRejected as e:
        raise HTTPException(422, str(e)) from e
    s.upload_path.mkdir(parents=True, exist_ok=True)
    stored = f"{uuid.uuid4().hex}{name[name.rfind('.'):].lower()}"
    (s.upload_path / stored).write_bytes(data)
    with session_scope() as db:
        up = Upload(filename=name, stored_name=stored, size_bytes=len(data), content_type=file.content_type or "")
        db.add(up)
        db.flush()
        return {"id": up.id, "filename": name, "size_bytes": len(data)}


# ---------------------------------------------------------------- objectives / runs
@router.post("/objectives", status_code=201)
async def create_objective(body: ObjectiveCreate) -> dict:
    return await ex.manager.create_run(body)


@router.get("/objectives/{run_id}")
async def get_objective(run_id: str) -> dict:
    d = ex.manager.detail(_id(run_id))
    return {"id": d["id"], "objective": d["objective"], "intent": d["intent"], "params": d["params"], "status": d["status"],
            "risk_level": d["risk_level"]}


@router.get("/runs")
async def list_runs(limit: int = 50) -> list[dict]:
    return ex.list_runs(min(max(limit, 1), 200))


@router.get("/stats")
async def get_stats() -> dict:
    return ex.stats()


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    return ex.manager.detail(_id(run_id))


@router.post("/runs/{run_id}/plan/regenerate")
async def regenerate_plan(run_id: str) -> dict:
    return await ex.manager.regenerate_plan(_id(run_id))


@router.put("/runs/{run_id}/plan")
async def update_plan(run_id: str, body: PlanUpdate) -> dict:
    return ex.manager.update_plan(_id(run_id), body.tasks)


@router.post("/runs/{run_id}/plan/approve")
async def approve_plan(run_id: str, body: PlanApprove | None = None) -> dict:
    return ex.manager.approve_plan(_id(run_id), body.plan_hash if body else None)


@router.post("/runs/{run_id}/continue")
async def continue_run(run_id: str) -> dict:
    """Resume a failed / halted / partial run from its persisted, re-verified state."""
    return ex.manager.continue_run(_id(run_id))


@router.get("/runs/{run_id}/plans")
async def plan_versions(run_id: str) -> list[dict]:
    store.get_run(_id(run_id))
    return plans.list_versions(run_id)


@router.get("/runs/{run_id}/report")
async def run_report(run_id: str) -> dict:
    store.get_run(_id(run_id))
    return report.build(run_id)


@router.get("/runs/{run_id}/external-actions")
async def external_actions(run_id: str) -> list[dict]:
    store.get_run(_id(run_id))
    return ledger.for_run(run_id)


@router.post("/external-actions/{action_id}/resolve")
async def resolve_external_action(action_id: str, body: ExternalActionResolve) -> dict:
    try:
        return ledger.resolve_manually(_id(action_id), body.outcome, body.note)
    except LookupError as e:
        raise NotFound(str(e)) from e
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


@router.get("/tools")
async def tools() -> dict:
    """The tool registry as the planner and execution engine see it, including what is currently unavailable and why."""
    from ..workflows.custom import TOOLS, tool_status

    out = []
    for t in TOOLS.values():
        st = tool_status(t)
        out.append({"name": t.name, "category": t.category, "agent": t.agent, "description": t.description, "risk": t.risk, "requires_approval": t.needs_approval,
                    "permissions": list(t.permissions), "timeout_seconds": t.timeout_s, "idempotent": t.idempotent, "verification": t.verification,
                    "error_types": list(t.error_types), "output": t.output, "input_schema": t.args.model_json_schema(), "example": t.example,
                    "available": st["available"], "missing": st["missing"], "enforced": t.enforce_permissions})
    return {"tools": out, **capabilities.describe()}


@router.post("/runs/{run_id}/start")
async def start_run(run_id: str) -> dict:
    """Alias for plan approval (guided) - autonomous runs start themselves."""
    return ex.manager.approve_plan(_id(run_id))


@router.post("/runs/{run_id}/pause")
async def pause(run_id: str) -> dict:
    return ex.manager.pause(_id(run_id))


@router.post("/runs/{run_id}/resume")
async def resume(run_id: str) -> dict:
    return ex.manager.resume(_id(run_id))


@router.post("/runs/{run_id}/cancel")
async def cancel(run_id: str) -> dict:
    return ex.manager.cancel(_id(run_id))


@router.get("/runs/{run_id}/tasks")
async def list_tasks(run_id: str) -> list[dict]:
    store.get_run(_id(run_id))
    with session_scope() as db:
        return [store.task_dict(t) for t in db.scalars(select(Task).where(Task.run_id == run_id).order_by(Task.position))]


@router.get("/runs/{run_id}/tasks/{task_key}")
async def task_detail(run_id: str, task_key: str) -> dict:
    store.get_run(_id(run_id))
    with session_scope() as db:
        t = db.get(Task, f"{run_id}:{_id(task_key)}")
        if not t:
            raise NotFound("task not found")
        d = store.task_dict(t)
        d["output"] = _trim(t.output)
        return d


def _trim(obj, depth=0):
    """Task outputs can embed page text; return them truncated for the inspector."""
    if isinstance(obj, dict):
        return {k: _trim(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_trim(v, depth + 1) for v in obj[:50]]
    if isinstance(obj, str) and len(obj) > 600:
        return obj[:600] + f"… [{len(obj) - 600} more chars]"
    return obj


@router.get("/runs/{run_id}/agents")
async def agents(run_id: str) -> dict:
    store.get_run(_id(run_id))
    return ex.agent_states(run_id)


@router.get("/runs/{run_id}/events")
async def event_history(run_id: str, after: int = 0) -> list[dict]:
    store.get_run(_id(run_id))
    return events.replay(run_id, after)


@router.get("/runs/{run_id}/events/stream")
async def event_stream(run_id: str, request: Request, after: int = 0):
    _id(run_id)
    run = store.get_run(run_id)
    last = int(request.headers.get("last-event-id") or after or 0)

    def fmt(ev: dict) -> str:
        return f"id: {ev['id']}\ndata: {json.dumps(ev)}\n\n"

    async def gen():
        q = events.subscribe(run_id)  # subscribe before replay so nothing is missed; duplicates are filtered by id
        sent = last
        try:
            yield "retry: 2000\n\n"
            for ev in events.replay(run_id, last):
                sent = ev["id"]
                yield fmt(ev)
                if ev["type"] in TERMINAL_EVENTS:
                    yield "event: end\ndata: {}\n\n"
                    return
            if store.get_run(run_id).status in RUN_TERMINAL and q.empty():
                yield "event: end\ndata: {}\n\n"
                return
            while True:
                if await request.is_disconnected():
                    return
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=10)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    if store.get_run(run_id).status in RUN_TERMINAL and q.empty():
                        yield "event: end\ndata: {}\n\n"
                        return
                    continue
                if ev["id"] <= sent:
                    continue
                sent = ev["id"]
                yield fmt(ev)
                if ev["type"] in TERMINAL_EVENTS:
                    yield "event: end\ndata: {}\n\n"
                    return
        finally:
            events.unsubscribe(run_id, q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})


@router.get("/runs/{run_id}/verification")
async def verification(run_id: str) -> dict:
    return ex.verification_report(_id(run_id))


@router.get("/runs/{run_id}/artifacts")
async def run_artifacts(run_id: str) -> list[dict]:
    store.get_run(_id(run_id))
    return artifact_service.list_for_run(run_id)


@router.get("/runs/{run_id}/evidence")
async def run_evidence(run_id: str) -> list[dict]:
    store.get_run(_id(run_id))
    return ex.evidence_list(run_id)


@router.get("/runs/{run_id}/trace")
async def run_trace(run_id: str) -> dict:
    return ex.trace(_id(run_id))


@router.get("/runs/{run_id}/artifacts.zip")
async def download_zip(run_id: str):
    """All deliverables of a run in one ZIP, keeping the organised folder layout. Each file is integrity-checked first."""
    import io
    import zipfile

    store.get_run(_id(run_id))
    arts = [a for a in artifact_service.list_for_run(run_id) if a["kind"] != "screenshot"]
    if not arts:
        raise HTTPException(404, "this run has no deliverables")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for a in arts:
            path, _ = artifact_service.resolve_download(a["id"])
            z.write(path, arcname="/".join(a["path"].split("/")[1:]))  # drop the run-id prefix
    return Response(buf.getvalue(), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{run_id}_deliverables.zip"'})


@router.get("/artifacts/{artifact_id}/download")
async def download(artifact_id: str):
    if not re.fullmatch(r"art_[0-9a-f]{12}", artifact_id):
        raise HTTPException(404, "artifact not found")
    path, art = artifact_service.resolve_download(artifact_id)
    return FileResponse(path, filename=art.filename, media_type=artifact_service.MEDIA.get(art.kind, "application/octet-stream"))


# ---------------------------------------------------------------- approvals
@router.get("/approvals")
async def approvals(status: str | None = None, run_id: str | None = None) -> list[dict]:
    return approval_service.list_approvals(status, run_id)


@router.post("/approvals/{approval_id}/resolve")
async def resolve_approval(approval_id: str, body: ApprovalResolve) -> dict:
    try:
        return approval_service.resolve(_id(approval_id), body.decision, body.note)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


# ---------------------------------------------------------------- workflows
@router.post("/workflows", status_code=201)
async def save_workflow(body: WorkflowCreate) -> dict:
    return ex.manager.save_workflow(_id(body.run_id), body.name, body.description)


@router.get("/workflows")
async def list_workflows() -> list[dict]:
    with session_scope() as db:
        out = []
        from ..models import Run

        for w in db.scalars(select(Workflow).order_by(Workflow.created_at.desc())):
            d = ex.workflow_dict(w)
            d["task_count"] = len(w.task_config)
            d["run_count"] = db.query(Run).filter(Run.template_id == w.id).count()
            d.pop("task_config")
            out.append(d)
        return out


@router.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str) -> dict:
    with session_scope() as db:
        w = db.get(Workflow, _id(workflow_id))
        if not w:
            raise NotFound("workflow not found")
        return ex.workflow_dict(w)


@router.delete("/workflows/{workflow_id}", status_code=204, response_class=Response)
async def delete_workflow(workflow_id: str):
    with session_scope() as db:
        w = db.get(Workflow, _id(workflow_id))
        if not w:
            raise NotFound("workflow not found")
        db.delete(w)
    return Response(status_code=204)


@router.post("/workflows/{workflow_id}/run", status_code=201)
async def run_workflow(workflow_id: str, body: WorkflowRunRequest) -> dict:
    ups = body.params.pop("upload_ids", None) if isinstance(body.params, dict) else None
    return await ex.manager.run_workflow(_id(workflow_id), body.params, body.mode, body.simulate, ups)
