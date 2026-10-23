"""Run lifecycle management: creation, plan approval/editing, background execution, pause/resume/cancel,
saved workflows, statistics and traces."""
from __future__ import annotations

import asyncio
import logging
from collections import Counter

from sqlalchemy import func, select

from ..agents import planner
from ..agents.orchestrator import Orchestrator, RunHandle
from ..config import get_settings
from ..database import session_scope
from ..models import (AgentActivity, Approval, Artifact, Evidence, Event, Run, Task, ToolCall, Upload,
                      VerificationResult, Workflow, utcnow)
from ..models.state import RUN_TERMINAL
from ..schemas.domain import ExamParams, Intent, ObjectiveCreate, TaskSpec, WorkflowParams
from ..services import approval_service, artifact_service, ledger, plans, store
from ..services.event_service import events
from ..services.store import NotFound, iso
from ..workflows.research_report import PlanError, WORKFLOW_TYPE, validate_plan

log = logging.getLogger("jose.execution")
RUN_SPECIFIC_PARAMS = ("_recoveries", "_tried", "candidate_index", "force_browser", "broaden")


class Conflict(ValueError):
    """The requested operation is not valid in the run's current state."""


class RunManager:
    def __init__(self) -> None:
        self.handles: dict[str, RunHandle] = {}

    # ------------------------------------------------------------ lifecycle
    def _launch(self, run_id: str) -> None:
        old = self.handles.get(run_id)
        if old and old.task and not old.task.done() and store.get_run(run_id).status != "awaiting_plan_approval":
            raise Conflict("run is already executing")  # (a parked run's planner task is only winding down; it may be replaced)
        h = RunHandle(run_id)
        self.handles[run_id] = h

        async def _go() -> None:
            try:
                await Orchestrator(run_id, h).run()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - already recorded by Orchestrator.run
                log.exception("run %s crashed", run_id)
            finally:
                events_done = self.handles.get(run_id)
                if events_done is h and store.get_run(run_id).status in RUN_TERMINAL:
                    self.handles.pop(run_id, None)

        h.task = asyncio.get_running_loop().create_task(_go())

    async def create_run(self, payload: ObjectiveCreate) -> dict:
        s = get_settings()
        simulated = payload.simulate or s.demo_forced
        with session_scope() as db:
            for uid in payload.upload_ids:
                if db.get(Upload, uid) is None:
                    raise NotFound(f"upload {uid} not found")
            run = Run(objective=payload.objective.strip(), mode=payload.mode, simulated=simulated,
                      params={"export_name": payload.export_name}, upload_ids=payload.upload_ids)
            db.add(run)
            db.flush()
            run_id = run.id
        events.emit(run_id, "run.created", objective=payload.objective[:500], mode=payload.mode, simulated=simulated,
                    demo_reason=("no GEMINI_API_KEY configured" if s.demo_forced and not payload.simulate else
                                 "requested by user" if payload.simulate else None))
        self._launch(run_id)
        await self._wait_planned(run_id)
        return self.detail(run_id)

    async def _wait_planned(self, run_id: str, timeout: float = 30.0) -> None:
        end = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < end:
            if store.get_run(run_id).status not in ("created", "planning"):
                return
            h = self.handles.get(run_id)
            if h and h.task and h.task.done():
                return
            await asyncio.sleep(0.05)

    def detail(self, run_id: str) -> dict:
        with session_scope() as db:
            r = db.get(Run, run_id)
            if not r:
                raise NotFound(f"run {run_id} not found")
            tasks = db.scalars(select(Task).where(Task.run_id == run_id).order_by(Task.position)).all()
            d = store.run_dict(r)
            d["tasks"] = [store.task_dict(t) for t in tasks]
            d["artifacts"] = artifact_service.list_for_run(run_id)
            d["pending_approvals"] = approval_service.list_approvals("pending", run_id)
            return d

    # ------------------------------------------------------------ plan
    def update_plan(self, run_id: str, specs: list[TaskSpec]) -> dict:
        r = store.get_run(run_id)
        if r.status != "awaiting_plan_approval":
            raise Conflict(f"the plan can only be edited while awaiting approval (run is {r.status})")
        try:
            planner.validate_edit(specs)
            self._check_files(r, specs)
        except PlanError as e:
            raise ValueError("; ".join(e.problems)) from e
        store.replace_tasks(run_id, specs, reason="edited by user")
        events.emit(run_id, "plan.updated", reason="user edit", count=len(specs))
        return self.detail(run_id)

    @staticmethod
    def _check_files(r: Run, specs: list[TaskSpec]) -> None:
        if r.workflow_type == "custom":
            from ..workflows.custom import validate_custom_plan

            with session_scope() as db:
                names = {u.filename for i in (r.upload_ids or []) if (u := db.get(Upload, i))}
            validate_custom_plan(specs, names, r.simulated)

    def approve_plan(self, run_id: str, plan_hash: str | None = None) -> dict:
        """Approve exactly the plan version the approver saw. `plan_hash` (from the plan they reviewed) must still be current."""
        r = store.get_run(run_id)
        if r.status != "awaiting_plan_approval":
            raise Conflict(f"plan cannot be approved while run is {r.status}")
        specs = [TaskSpec(key=t.key, title=t.title, description=t.description, agent=t.agent, action=t.action,  # type: ignore[arg-type]
                          params=t.params, depends_on=t.depends_on, resources=t.resources,
                          expected_output=t.expected_output, verification_criteria=t.verification_criteria)
                 for t in store.get_tasks(run_id)]
        try:
            validate_plan(specs)  # re-validate at approval time; never execute an invalid graph
            self._check_files(r, specs)
        except PlanError as e:
            raise ValueError("; ".join(e.problems)) from e
        try:
            bound = plans.approve(run_id, "user", plan_hash)
        except plans.StalePlan as e:
            raise Conflict(str(e)) from e
        events.emit(run_id, "plan.approved", automatic=False, plan_version=bound["version"], plan_hash=bound["plan_hash"])
        self._launch(run_id)
        return self.detail(run_id)

    async def regenerate_plan(self, run_id: str) -> dict:
        r = store.get_run(run_id)
        if r.status != "awaiting_plan_approval":
            raise Conflict("the plan can only be regenerated while awaiting approval")
        if r.workflow_type == "custom":
            from ..agents import custom_planner
            from ..services.llm import build_providers

            try:
                specs, assumptions = await custom_planner.plan(r.objective, (r.params or {}).get("files", []), build_providers(r.simulated))
            except custom_planner.PlanRejected as e:
                raise ValueError(e.reason) from e
            store.replace_tasks(run_id, specs, reason="regenerated")
            events.emit(run_id, "plan.generated", count=len(specs), regenerated=True, assumptions=assumptions)
            return self.detail(run_id)
        model = ExamParams if r.workflow_type == "test_paper" else WorkflowParams
        specs = planner.plan(model.model_validate(r.params))
        store.replace_tasks(run_id, specs, reason="regenerated")
        events.emit(run_id, "plan.generated", count=len(specs), regenerated=True)
        return self.detail(run_id)

    # ------------------------------------------------------------ controls
    def pause(self, run_id: str) -> dict:
        r, h = store.get_run(run_id), self.handles.get(run_id)
        if r.status != "running" or h is None:
            raise Conflict(f"only a running run can be paused (run is {r.status})")
        h.running.clear()
        store.set_run_status(run_id, "paused", event="run.paused")
        return self.detail(run_id)

    def resume(self, run_id: str) -> dict:
        r, h = store.get_run(run_id), self.handles.get(run_id)
        if r.status != "paused" or h is None:
            raise Conflict(f"only a paused run can be resumed (run is {r.status})")
        store.set_run_status(run_id, "running", event="run.resumed")
        h.running.set()
        return self.detail(run_id)

    def cancel(self, run_id: str) -> dict:
        r = store.get_run(run_id)
        if r.status in RUN_TERMINAL:
            raise Conflict(f"run already {r.status}")
        h = self.handles.get(run_id)
        if h and h.task and not h.task.done() and r.status != "awaiting_plan_approval":
            h.cancel.set()
            h.running.set()
            # pending approvals of this run are closed so nothing stays actionable
            for a in approval_service.list_approvals("pending", run_id):
                try:
                    approval_service.resolve(a["id"], "denied", "run cancelled")
                except ValueError:
                    pass
        else:  # parked (e.g. awaiting plan approval): no live graph
            for t in store.get_tasks(run_id):
                if t.status in ("pending", "failed", "blocked"):
                    store.set_task_status(t.id, "cancelled")
            store.set_run_status(run_id, "cancelled", event="run.cancelled", reason="cancelled by user")
        return self.detail(run_id)

    def mark_interrupted(self) -> int:
        """On startup: runs that were mid-flight when the server stopped cannot be resumed in-process."""
        n = 0
        ledger.interrupt_in_flight()
        with session_scope() as db:
            for r in db.scalars(select(Run).where(Run.status.in_(["created", "planning", "running", "paused", "awaiting_approval"]))):
                r.status, r.finished_at = "failed", utcnow()
                r.error = "Interrupted: the server restarted while this run was in progress. Use Continue to resume from the verified state."
                r.halt_reason = "server_restart"
                for t in db.scalars(select(Task).where(Task.run_id == r.id, Task.status.in_(["running", "verifying"]))):
                    t.status, t.error = "failed", "interrupted by server restart"
                for a in db.scalars(select(Approval).where(Approval.run_id == r.id, Approval.status == "pending")):
                    a.status, a.note, a.resolved_at = "denied", "server restarted", utcnow()
                n += 1
        return n

    # ------------------------------------------------------------ continue (resume from persisted evidence)
    def continue_run(self, run_id: str) -> dict:
        """Resume an interrupted / halted / partially failed run.

        Nothing is assumed: each completed task keeps its result only if the artifacts it produced still exist on disk with the
        recorded hash; everything else is redone. Tasks whose external action (email, post) may already have happened are NOT redone
        automatically - a person has to say whether it happened (POST /api/external-actions/{id}/resolve).
        """
        r = store.get_run(run_id)
        if r.status not in ("failed", "halted", "partial"):
            raise Conflict(f"only a failed, halted or partial run can be continued (run is {r.status})")
        # the run is already in a final state, so any old orchestrator task is only winding down; a new handle replaces it
        self.handles.pop(run_id, None)
        tasks = store.get_tasks(run_id)
        if not tasks:
            raise Conflict("this run has no plan to continue")
        if not plans.is_approved_plan(run_id):
            # the plan on record is not the plan that was approved (e.g. recovery made it riskier): ask again, do not execute
            store.set_run_status(run_id, "awaiting_plan_approval", resume=True, event="run.continued", needs_plan_approval=True)
            with session_scope() as db:
                db.get(Run, run_id).plan_approved = False
            return self.detail(run_id)
        by_key = {t.key: t for t in tasks}
        acts = ledger.for_run(run_id)
        unresolved = {a["task_id"]: a for a in acts if a["status"] in ("started", "ambiguous") and a["task_id"]}
        audit: dict = {"kept": [], "redone": [], "needs_human": []}

        def reset_with_dependents(key: str, why: str) -> None:
            affected, grew = {key}, True
            while grew:
                grew = False
                for t in tasks:
                    if t.key not in affected and set(t.depends_on) & affected:
                        affected.add(t.key)
                        grew = True
            for k in sorted(affected):
                t = by_key[k]
                if t.id in unresolved:
                    continue
                cur = store.get_tasks(run_id)
                cur_t = next(x for x in cur if x.key == k)
                if cur_t.status in ("completed", "failed", "blocked"):
                    store.set_task_status(cur_t.id, "pending")
                    store.save_task_output(cur_t.id, None)  # type: ignore[arg-type]
                    audit["redone"].append({"task": k, "why": why if k == key else f"depends on {key}"})

        for t in tasks:
            if t.id in unresolved:
                audit["needs_human"].append({"task": t.key, "action_id": unresolved[t.id]["id"], "status": unresolved[t.id]["status"],
                                             "why": "an external action may already have happened; confirm whether it did, then continue again"})
                continue
            if t.status == "completed":
                problem = self._evidence_problem(t)
                if problem:
                    reset_with_dependents(t.key, problem)
                else:
                    audit["kept"].append(t.key)
        for t in store.get_tasks(run_id):
            if t.status in ("failed", "blocked") and t.id not in unresolved:
                reset_with_dependents(t.key, f"was {t.status}: {(t.error or '')[:120]}")
        events.emit(run_id, "run.continued", audit=audit, needs_plan_approval=False)
        store.set_run_status(run_id, "running", resume=True)
        self._launch(run_id)
        return self.detail(run_id)

    @staticmethod
    def _evidence_problem(t) -> str | None:
        """None when everything the completed task claims to have produced is still intact on disk."""
        import json as _json
        import re as _re

        if not t.output:
            return "completed but its result is missing"
        for art_id in sorted(set(_re.findall(r"art_[0-9a-f]{12}", _json.dumps(t.output, default=str)))):
            try:
                artifact_service.resolve_download(art_id)
            except NotFound as e:
                return f"its output {art_id} is no longer valid: {e}"
        return None

    # ------------------------------------------------------------ workflows
    def save_workflow(self, run_id: str, name: str, description: str) -> dict:
        r = store.get_run(run_id)
        if r.status != "completed":
            raise Conflict("only a completed run can be saved as a reusable workflow")
        specs = [TaskSpec(key=t.key, title=t.title, description=t.description, agent=t.agent, action=t.action,  # type: ignore[arg-type]
                          params={k: v for k, v in t.params.items() if k not in RUN_SPECIFIC_PARAMS}, depends_on=t.depends_on,
                          resources=t.resources, expected_output=t.expected_output,
                          verification_criteria=t.verification_criteria, risk_level=t.risk_level)  # type: ignore[arg-type]
                 for t in store.get_tasks(run_id)]
        try:
            validate_plan(specs)
        except PlanError as e:
            raise ValueError("; ".join(e.problems)) from e
        if r.workflow_type == "custom":
            wparams: dict = {"files": [f["name"] for f in (r.params or {}).get("files", [])]}
        else:
            wparams = (ExamParams if r.workflow_type == "test_paper" else WorkflowParams).model_validate(r.params).model_dump()
        with session_scope() as db:
            wf = Workflow(name=name.strip(), description=description, workflow_type=r.workflow_type or WORKFLOW_TYPE,
                          params=wparams, task_config=[s.model_dump() for s in specs], source_run_id=run_id,
                          objective=r.objective)
            db.add(wf)
            db.flush()
            return workflow_dict(wf)

    async def run_workflow(self, workflow_id: str, overrides: dict, mode: str, simulate: bool | None,
                           upload_ids: list[str] | None = None) -> dict:
        with session_scope() as db:
            wf = db.get(Workflow, workflow_id)
            if not wf:
                raise NotFound("workflow not found")
            saved, config, name, objective, wtype = dict(wf.params), list(wf.task_config), wf.name, wf.objective, wf.workflow_type
        if wtype == "custom":
            return await self._run_custom_workflow(workflow_id, config, saved, name, objective, mode, simulate, upload_ids)
        if wtype == "test_paper":
            return await self._run_exam_workflow(workflow_id, saved, overrides, name, objective, mode, simulate)
        merged = {**saved, **{k: v for k, v in overrides.items() if v is not None}}
        merged["has_inputs"] = bool(upload_ids)
        params = WorkflowParams.model_validate(merged)  # validates the new input parameters
        old = WorkflowParams.model_validate(saved)
        same_shape = (params.competitor_count == old.competitor_count and params.outputs == old.outputs
                      and params.has_inputs == old.has_inputs)
        if same_shape:
            specs = [TaskSpec.model_validate(c) for c in config]
            for sp in specs:
                for field_ in ("title", "description"):
                    setattr(sp, field_, getattr(sp, field_).replace(old.industry, params.industry))
                if "industry" in sp.params:
                    sp.params["industry"] = params.industry
                if "export_name" in sp.params:
                    sp.params["export_name"] = params.export_name
            validate_plan(specs)
        else:
            specs = planner.plan(params)
        for sp in specs:
            sp.risk_level = planner.risk.assess(sp.action, sp.params).level
        s = get_settings()
        sim = bool(simulate) or s.demo_forced
        intent = Intent(goal=objective, industry=params.industry, competitor_count=params.competitor_count, outputs=params.outputs,
                        inputs=[], constraints=[], assumptions=[f"Re-run of saved workflow '{name}'."],
                        success_criteria=["all verification checks pass"])
        with session_scope() as db:
            run = Run(objective=f"[Saved workflow: {name}] Research {params.competitor_count} competitors in {params.industry}",
                      mode=mode, simulated=sim, status="created", intent=intent.model_dump(), params=params.model_dump(),
                      upload_ids=upload_ids or [], template_id=workflow_id)
            db.add(run)
            db.flush()
            run_id = run.id
        store.replace_tasks(run_id, specs)
        events.emit(run_id, "run.created", objective=f"saved workflow {name}", mode=mode, simulated=sim, template_id=workflow_id)
        events.emit(run_id, "plan.generated", count=len(specs), source="saved workflow", reused_config=same_shape)
        store.set_run_status(run_id, "planning")
        self._launch(run_id)
        await self._wait_planned(run_id)
        return self.detail(run_id)

    async def _run_exam_workflow(self, workflow_id: str, saved: dict, overrides: dict, name: str, objective: str, mode: str,
                                 simulate: bool | None) -> dict:
        s = get_settings()
        sim = bool(simulate) or s.demo_forced
        merged = {**saved, **{k: v for k, v in overrides.items() if v is not None}, "dry_run": sim}
        params = ExamParams.model_validate(merged)  # validates the new input parameters
        specs = planner.plan(params)
        intent = Intent(feasible=True, workflow_type="test_paper", goal=objective, subject=params.subject, grade=params.grade, topics=params.topics,
                        num_questions=params.num_questions, difficulty=params.difficulty, duration_minutes=params.duration_minutes,
                        post_to_telegram=params.post_to_telegram, outputs=["xlsx"], assumptions=[f"Re-run of saved workflow '{name}'."],
                        risk_level="HIGH" if params.post_to_telegram else "LOW", success_criteria=["all verification checks pass"])
        with session_scope() as db:
            run = Run(objective=f"[Saved workflow: {name}] {params.subject} test for class {params.grade}", mode=mode, simulated=sim, status="created",
                      intent=intent.model_dump(), params=params.model_dump(), workflow_type="test_paper", template_id=workflow_id)
            db.add(run)
            db.flush()
            run_id = run.id
        store.replace_tasks(run_id, specs)
        events.emit(run_id, "run.created", objective=f"saved workflow {name}", mode=mode, simulated=sim, template_id=workflow_id)
        events.emit(run_id, "plan.generated", count=len(specs), source="saved workflow")
        store.set_run_status(run_id, "planning")
        self._launch(run_id)
        await self._wait_planned(run_id)
        return self.detail(run_id)

    async def _run_custom_workflow(self, workflow_id: str, config: list, saved: dict, name: str, objective: str, mode: str,
                                   simulate: bool | None, upload_ids: list[str] | None) -> dict:
        import asyncio

        from ..workflows.custom import validate_custom_plan
        from ..workflows.research_report import PlanError

        if not upload_ids:
            raise ValueError(f"this saved workflow works on attached files ({', '.join(saved.get('files', [])) or 'none recorded'}); attach them to re-run it")
        s = get_settings()
        sim = bool(simulate) or s.demo_forced
        specs = [TaskSpec.model_validate(c) for c in config]
        with session_scope() as db:
            names = [u.filename for i in upload_ids if (u := db.get(Upload, i))]
        try:
            validate_custom_plan(specs, set(names))
        except PlanError as e:
            raise ValueError("the attached files do not match the saved workflow: " + "; ".join(e.problems[:3])) from e
        from ..agents.orchestrator import Orchestrator as _O

        files = await asyncio.to_thread(_O(workflow_id, RunHandle(workflow_id))._inspect_uploads, upload_ids)
        intent = Intent(feasible=True, workflow_type="custom", goal=objective, inputs=names, risk_level="LOW", assumptions=[f"Re-run of saved workflow '{name}'."],
                        success_criteria=["every task passes its independent verification checks"])
        with session_scope() as db:
            run = Run(objective=f"[Saved workflow: {name}] {objective}"[:3900], mode=mode, simulated=sim, status="created", intent=intent.model_dump(),
                      params={"files": files}, upload_ids=upload_ids, workflow_type="custom", template_id=workflow_id)
            db.add(run)
            db.flush()
            run_id = run.id
        store.replace_tasks(run_id, specs)
        events.emit(run_id, "run.created", objective=f"saved workflow {name}", mode=mode, simulated=sim, template_id=workflow_id)
        events.emit(run_id, "plan.generated", count=len(specs), source="saved workflow")
        store.set_run_status(run_id, "planning")
        self._launch(run_id)
        await self._wait_planned(run_id)
        return self.detail(run_id)


def workflow_dict(w: Workflow) -> dict:
    return {"id": w.id, "name": w.name, "description": w.description, "workflow_type": w.workflow_type, "params": w.params,
            "task_config": w.task_config, "source_run_id": w.source_run_id, "objective": w.objective, "created_at": iso(w.created_at)}


# ------------------------------------------------------------ read models

def list_runs(limit: int = 50) -> list[dict]:
    with session_scope() as db:
        rows = db.scalars(select(Run).order_by(Run.created_at.desc()).limit(limit)).all()
        out = []
        for r in rows:
            d = store.run_dict(r)
            counts = Counter(t for (t,) in db.execute(select(Task.status).where(Task.run_id == r.id)))
            d["task_counts"] = dict(counts)
            d["artifact_count"] = db.scalar(select(func.count()).select_from(Artifact).where(Artifact.run_id == r.id, Artifact.kind != "screenshot")) or 0
            out.append(d)
        return out


def stats() -> dict:
    with session_scope() as db:
        by_status = dict(db.execute(select(Run.status, func.count()).group_by(Run.status)).all())
        total = sum(by_status.values())
        done = by_status.get("completed", 0)
        durations = [(r.finished_at - r.started_at).total_seconds() for r in db.scalars(select(Run).where(Run.status == "completed")) if r.started_at and r.finished_at]
        tasks = dict(db.execute(select(Task.status, func.count()).group_by(Task.status)).all())
        active = db.scalars(select(Run).where(Run.status.in_(["running", "paused", "awaiting_approval", "awaiting_plan_approval", "planning"]))).all()
        return {
            "total_runs": total, "by_status": by_status, "completed": done,
            "success_rate": (done / total) if total else None,
            "avg_duration_seconds": (sum(durations) / len(durations)) if durations else None,
            "tasks_by_status": tasks, "workflows": db.scalar(select(func.count()).select_from(Workflow)) or 0,
            "artifacts": db.scalar(select(func.count()).select_from(Artifact).where(Artifact.kind != "screenshot")) or 0,
            "active_runs": [store.run_dict(r) for r in active], "simulated_runs": db.scalar(select(func.count()).select_from(Run).where(Run.simulated.is_(True))) or 0,
        }


def verification_report(run_id: str) -> dict:
    store.get_run(run_id)
    with session_scope() as db:
        rows = db.scalars(select(VerificationResult).where(VerificationResult.run_id == run_id).order_by(VerificationResult.created_at)).all()
        tasks = {t.id: t for t in db.scalars(select(Task).where(Task.run_id == run_id))}
        checks = [{"id": v.id, "task_id": v.task_id, "task_key": tasks[v.task_id].key if v.task_id in tasks else None,
                   "check": v.check_name, "passed": v.passed, "reason": v.reason, "attempt": v.attempt, "created_at": iso(v.created_at)} for v in rows]
        # the latest attempt per (task, check) decides the current state; earlier failures stay visible as history
        latest: dict[tuple, dict] = {}
        for c in checks:
            latest[(c["task_id"], c["check"])] = c
        cur = list(latest.values())
        return {"checks": checks, "current": cur, "passed": sum(1 for c in cur if c["passed"]),
                "failed": sum(1 for c in cur if not c["passed"]), "all_passed": bool(cur) and all(c["passed"] for c in cur)}


def trace(run_id: str) -> dict:
    run = RunManager_detail(run_id)
    with session_scope() as db:
        evs = [{"id": e.id, "type": e.type, "task_id": e.task_id, "ts": iso(e.ts), "details": e.details}
               for e in db.scalars(select(Event).where(Event.run_id == run_id).order_by(Event.id))]
        tools = [{"id": t.id, "task_id": t.task_id, "tool": t.tool, "args": t.args, "status": t.status, "summary": t.result_summary,
                  "error": t.error, "simulated": t.simulated, "duration_ms": t.duration_ms, "started_at": iso(t.started_at)}
                 for t in db.scalars(select(ToolCall).where(ToolCall.run_id == run_id).order_by(ToolCall.started_at))]
        acts = [{"id": a.id, "task_id": a.task_id, "agent": a.agent, "activity": a.activity, "started_at": iso(a.started_at), "ended_at": iso(a.ended_at)}
                for a in db.scalars(select(AgentActivity).where(AgentActivity.run_id == run_id).order_by(AgentActivity.started_at))]
    return {"run": run, "events": evs, "tool_calls": tools, "agent_activity": acts, "approvals": approval_service.list_approvals(run_id=run_id),
            "verification": verification_report(run_id), "evidence": evidence_list(run_id),
            "plan_versions": plans.list_versions(run_id), "external_actions": ledger.for_run(run_id)}


def RunManager_detail(run_id: str) -> dict:
    return manager.detail(run_id)


def evidence_list(run_id: str) -> list[dict]:
    with session_scope() as db:
        return [{"id": e.id, "subject": e.subject, "claim_type": e.claim_type, "claim": e.claim, "source_url": e.source_url,
                 "page_title": e.page_title, "retrieved_at": iso(e.retrieved_at), "evidence_text": e.evidence_text,
                 "status": e.status, "simulated": e.simulated}
                for e in db.scalars(select(Evidence).where(Evidence.run_id == run_id).order_by(Evidence.subject, Evidence.claim_type))]


def agent_states(run_id: str) -> dict:
    """Which agents are doing something *right now*, derived from open activity rows (not animation)."""
    with session_scope() as db:
        rows = db.scalars(select(AgentActivity).where(AgentActivity.run_id == run_id, AgentActivity.ended_at.is_(None))).all()
        return {"active": [{"agent": a.agent, "task_id": a.task_id, "activity": a.activity, "since": iso(a.started_at)} for a in rows]}


manager = RunManager()
