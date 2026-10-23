"""Orchestrator: the LangGraph state machine
UNDERSTAND -> PLAN -> RISK CHECK -> EXECUTE -> OBSERVE -> VERIFY, with DIAGNOSE -> RECOVERY -> (replan) -> EXECUTE on failure.

SQLite is the source of truth; the graph state mirrors it and carries routing information.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..config import get_settings
from ..database import session_scope
from ..models import Run, Upload
from ..schemas.domain import ExamParams, TaskSpec, WorkflowParams
from ..workflows.custom import tool_of
from ..services import approval_service, artifact_service, plans, risk, runtime_settings, store
from ..services.event_service import events
from ..services.llm import Providers, build_providers
from ..services.store import TaskView
from . import (browser_agent, custom_agent, custom_planner, exam_agent, messenger_agent, intent_engine, planner, recovery_agent, research_agent, verification_agent,
               workspace_agent)
from .base import AgentContext, Cancelled, TaskError

log = logging.getLogger("jose.orchestrator")

DISPATCH = {
    "file.read_inputs": workspace_agent.read_inputs,
    "research.discover": research_agent.discover,
    "browser.collect": browser_agent.collect,
    "research.extract": research_agent.extract,
    "workspace.normalize": workspace_agent.normalize,
    "workspace.build_excel": workspace_agent.build_excel,
    "workspace.build_pptx": workspace_agent.build_pptx,
    "verify.deliverables": verification_agent.run_deliverable_verification,
    "test.generate": exam_agent.generate,
    "test.solve_check": exam_agent.solve_check,
    "test.build_documents": exam_agent.build_documents,
    "test.verify_documents": exam_agent.verify_documents,
    "test.verify_set": exam_agent.verify_set,
    "telegram.send": messenger_agent.send,
}


class ExecState(TypedDict, total=False):
    run_id: str
    objective: str
    constraints: list
    intent: dict
    task_outputs: dict  # key -> short summary of completed outputs
    current_tasks: list
    tool_results: list
    observations: list
    verification_findings: list
    failures: list
    recovery_attempts: int
    approval_status: str
    artifacts: list
    status: str
    halt_reason: str | None
    iteration: int
    decisions: list


@dataclass
class RunHandle:
    run_id: str
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    running: asyncio.Event = field(default_factory=asyncio.Event)  # cleared while paused
    task: asyncio.Task | None = None

    def __post_init__(self) -> None:
        self.running.set()


class Orchestrator:
    def __init__(self, run_id: str, handle: RunHandle):
        self.run_id, self.h = run_id, handle
        self._providers: Providers | None = None
        self.graph = self._build()

    # ---------------------------------------------------------------- graph
    def _build(self):
        g = StateGraph(ExecState)
        for name in ("understand", "plan", "plan_gate", "risk_check", "execute", "observe", "verify", "diagnose",
                     "recovery", "finalize"):
            g.add_node(name, getattr(self, f"n_{name}"))
        g.add_conditional_edges(START, self.r_entry, {"understand": "understand", "plan_gate": "plan_gate", "risk_check": "risk_check"})
        g.add_conditional_edges("understand", lambda s: "end" if s.get("status") == "rejected" else "plan", {"end": END, "plan": "plan"})
        g.add_conditional_edges("plan", lambda s: "end" if s.get("status") == "rejected" else "gate", {"end": END, "gate": "plan_gate"})
        g.add_conditional_edges("plan_gate", lambda s: "park" if s.get("status") == "awaiting_plan_approval" else "go",
                                {"park": END, "go": "risk_check"})
        g.add_conditional_edges("risk_check", lambda s: "finalize" if s.get("status") == "halted" else "execute", {"finalize": "finalize", "execute": "execute"})
        g.add_conditional_edges("execute", lambda s: "finalize" if s.get("status") in ("cancelled", "halted") else "observe",
                                {"finalize": "finalize", "observe": "observe"})
        g.add_edge("observe", "verify")
        g.add_conditional_edges("verify", self.r_after_verify, {"finalize": "finalize", "diagnose": "diagnose", "execute": "execute"})
        g.add_edge("diagnose", "recovery")
        g.add_conditional_edges("recovery", lambda s: "finalize" if s.get("status") in ("cancelled", "halted") or s.get("halt_reason") else "execute",
                                {"finalize": "finalize", "execute": "execute"})
        g.add_edge("finalize", END)
        return g.compile()

    async def run(self, state: ExecState | None = None) -> ExecState:
        from ..services.llm import CURRENT_RUN

        CURRENT_RUN.set((self.run_id, None))
        st: ExecState = {"run_id": self.run_id, "iteration": 0, "recovery_attempts": 0, "observations": [],
                         "verification_findings": [], "failures": [], "tool_results": [], "artifacts": [], **(state or {})}
        try:
            return await self.graph.ainvoke(st, {"recursion_limit": 400})
        except Exception as e:  # noqa: BLE001 - an orchestrator crash must never leave the run "running"
            log.exception("orchestrator crashed")
            store.set_run_status(self.run_id, "failed", error=f"internal error: {type(e).__name__}: {e}", event="run.failed", reason="internal_error")
            raise

    # ---------------------------------------------------------------- helpers
    def run_row(self) -> Run:
        return store.get_run(self.run_id)

    def providers(self) -> Providers:
        if self._providers is None:
            r = self.run_row()
            self._providers = build_providers(r.simulated, (r.params or {}).get("industry", ""))
        return self._providers

    async def checkpoint(self) -> None:
        """Pause / cancel gate between steps."""
        if not self.h.running.is_set() and not self.h.cancel.is_set():
            await self._wait_either(self.h.running)
            if not self.h.cancel.is_set() and self.run_row().status == "paused":
                store.set_run_status(self.run_id, "running", event="run.resumed")

    async def _wait_either(self, ev: asyncio.Event) -> None:
        a, b = asyncio.ensure_future(ev.wait()), asyncio.ensure_future(self.h.cancel.wait())
        await asyncio.wait({a, b}, return_when=asyncio.FIRST_COMPLETED)
        for f in (a, b):
            f.cancel()

    def outputs(self) -> dict[str, dict]:
        with session_scope() as s:
            from sqlalchemy import select

            from ..models import Task
            return {t.key: t.output for t in s.scalars(select(Task).where(Task.run_id == self.run_id, Task.status == "completed")) if t.output}

    def make_ctx(self, tv: TaskView, tasks: list[TaskView], outputs: dict[str, dict]) -> AgentContext:
        r = self.run_row()
        return AgentContext(run_id=self.run_id, simulated=r.simulated, providers=self.providers(), cancel=self.h.cancel,
                            settings=runtime_settings.load(), run_params={**(r.params or {}), "_upload_ids": r.upload_ids or []},
                            task=tv, outputs=outputs, tasks=tasks)

    def resolved_for(self, tv: TaskView, tasks: list[TaskView], outputs: dict[str, dict]) -> dict:
        if tv.action == "browser.collect":
            try:
                ctx = self.make_ctx(tv, tasks, outputs)
                return {"url": browser_agent.candidate_for(ctx).url}
            except TaskError:
                return {}
        if tv.action == "telegram.send":
            return messenger_agent.preview(self.make_ctx(tv, tasks, outputs))
        if tv.action in ("tool.telegram.send_files", "tool.email.send"):
            ctx = self.make_ctx(tv, tasks, outputs)
            return {**custom_agent.preview(ctx), "dry_run": ctx.simulated}
        return {}

    # ---------------------------------------------------------------- nodes
    def r_entry(self, st: ExecState) -> str:
        r = self.run_row()
        tasks = store.get_tasks(self.run_id)
        if not tasks:
            return "understand"
        if not r.plan_approved:
            return "plan_gate"
        return "risk_check"

    async def n_understand(self, st: ExecState) -> ExecState:
        r = self.run_row()
        store.set_run_status(self.run_id, "planning")
        with session_scope() as s:
            names = [u.filename for uid in (r.upload_ids or []) if (u := s.get(Upload, uid))]
        intent = await intent_engine.understand(r.objective, self.providers(), names)
        with session_scope() as s:
            row = s.get(Run, self.run_id)
            row.intent, row.risk_level = intent.model_dump(), intent.risk_level
        events.emit(self.run_id, "intent.parsed", intent=intent.model_dump(), planner="llm" if not r.simulated else "rule-based")
        if not intent.feasible:
            store.set_run_status(self.run_id, "rejected", error=intent.rejection_reason, event="run.rejected", reason=intent.rejection_reason)
            return {"status": "rejected", "intent": intent.model_dump()}
        if intent.workflow_type == "custom":
            files = await asyncio.to_thread(self._inspect_uploads, r.upload_ids or [])
            with session_scope() as s:
                row = s.get(Run, self.run_id)
                row.params, row.workflow_type = {"files": files}, "custom"
            return {"intent": intent.model_dump(), "objective": r.objective, "constraints": []}
        if intent.workflow_type == "test_paper":
            ep = ExamParams(subject=intent.subject, grade=intent.grade, topics=intent.topics, num_questions=intent.num_questions,  # type: ignore[arg-type]
                            difficulty=intent.difficulty, duration_minutes=intent.duration_minutes or 60,  # type: ignore[arg-type]
                            post_to_telegram=intent.post_to_telegram, dry_run=r.simulated, num_papers=intent.num_papers, total_marks=intent.total_marks)
            with session_scope() as s:
                row = s.get(Run, self.run_id)
                row.params, row.workflow_type = ep.model_dump(), "test_paper"
            return {"intent": intent.model_dump(), "objective": r.objective, "constraints": intent.constraints}
        old = (r.params or {})
        params = WorkflowParams(industry=intent.industry, competitor_count=intent.competitor_count, outputs=intent.outputs,
                                export_name=old.get("export_name"), has_inputs=bool(r.upload_ids))
        with session_scope() as s:
            s.get(Run, self.run_id).params = params.model_dump()
        return {"intent": intent.model_dump(), "objective": r.objective, "constraints": intent.constraints}

    def _inspect_uploads(self, ids: list[str]) -> list[dict]:
        from ..tools.inspect_files import inspect
        from ..tools.safety import safe_join

        out = []
        for uid in ids:
            with session_scope() as s:
                up = s.get(Upload, uid)
                name, stored = (up.filename, up.stored_name) if up else (None, None)
            if name:
                out.append(inspect(safe_join(get_settings().upload_path, stored), name))
        return out

    async def n_plan(self, st: ExecState) -> ExecState:
        r = self.run_row()
        if r.workflow_type == "custom":
            try:
                tasks, assumptions = await custom_planner.plan(r.objective, (r.params or {}).get("files", []), self.providers())
            except custom_planner.PlanRejected as e:
                msg = e.reason if not e.questions or e.reason.startswith("More information") else f"{e.reason}\nQuestions: " + " | ".join(e.questions)
                if e.questions:
                    events.emit(self.run_id, "run.needs_input", questions=e.questions, reason=e.reason)
                store.set_run_status(self.run_id, "rejected", error=msg, event="run.rejected", reason=msg)
                return {"status": "rejected"}
            with session_scope() as s:
                row = s.get(Run, self.run_id)
                row.risk_level = max((t.risk_level for t in tasks), key=risk.ORDER.get, default="LOW")  # type: ignore[arg-type]
                if row.intent:
                    row.intent = {**row.intent, "assumptions": assumptions, "risk_level": row.risk_level}
            store.replace_tasks(self.run_id, tasks)
            events.emit(self.run_id, "plan.generated", tasks=[_spec_summary(t) for t in tasks], count=len(tasks), planner="llm", assumptions=assumptions)
            return {}
        params = ExamParams.model_validate(r.params) if r.workflow_type == "test_paper" else WorkflowParams.model_validate(r.params)
        tasks = planner.plan(params)
        store.replace_tasks(self.run_id, tasks)
        events.emit(self.run_id, "plan.generated", tasks=[_spec_summary(t) for t in tasks], count=len(tasks))
        return {}

    async def n_plan_gate(self, st: ExecState) -> ExecState:
        r = self.run_row()
        if not r.plan_approved:
            if r.mode == "guided":
                if r.status != "awaiting_plan_approval":
                    store.set_run_status(self.run_id, "awaiting_plan_approval")
                return {"status": "awaiting_plan_approval"}
            bound = plans.approve(self.run_id, "auto")
            events.emit(self.run_id, "plan.approved", automatic=True, mode="autonomous", plan_version=bound["version"], plan_hash=bound["plan_hash"])
        return {}

    async def n_risk_check(self, st: ExecState) -> ExecState:
        if not plans.is_approved_plan(self.run_id):  # fail closed: the approval must be for exactly this plan version
            r = self.run_row()
            events.emit(self.run_id, "warning", message=f"execution refused: plan {r.plan_hash} is not the plan that was approved ({r.approved_plan_hash})")
            store.set_run_status(self.run_id, "failed", halt_reason="plan_not_approved", error="The plan to execute is not the plan that was approved. Review and approve it again.",
                                 event="run.failed", reason="plan_not_approved")
            return {"status": "halted", "halt_reason": "plan_not_approved"}
        tasks = store.get_tasks(self.run_id)
        top = "LOW"
        for t in tasks:
            a = risk.assess(t.action, t.params)
            if risk.ORDER[a.level] > risk.ORDER[top]:
                top = a.level
        r = self.run_row()
        with session_scope() as s:
            s.get(Run, self.run_id).risk_level = max(top, r.risk_level, key=risk.ORDER.get)  # type: ignore[arg-type]
        store.set_run_status(self.run_id, "running", event="run.started", risk_level=top, simulated=r.simulated, mode=r.mode)
        return {"status": "running"}

    async def _approve(self, tv: TaskView, a: risk.RiskAssessment) -> str:
        r = self.run_row()
        ap = approval_service.create(self.run_id, tv.id, a.level, tv.action, a.target,
                                     {"task": tv.title, "reasons": a.reasons, **a.details, "plan_version": r.plan_version, "plan_hash": r.plan_hash,
                                      "content_fingerprint": fingerprint(a)}, a.consequences or "; ".join(a.reasons))
        store.set_run_status(self.run_id, "awaiting_approval")
        decision = await approval_service.wait(ap["id"], self.h.cancel)
        if decision == "approved":
            store.set_run_status(self.run_id, "running")
        return decision

    async def n_execute(self, st: ExecState) -> ExecState:
        await self.checkpoint()
        if self.h.cancel.is_set():
            return {"status": "cancelled"}
        rt = runtime_settings.load()
        tasks = store.get_tasks(self.run_id)
        outputs = self.outputs()
        done = {t.key for t in tasks if t.status == "completed"}
        ready = [t for t in tasks if t.status == "pending" and set(t.depends_on) <= done]
        limit = max(len(tasks) * (rt.max_retries + 2) + 10, 30)
        if st.get("iteration", 0) > limit:
            return {"status": "halted", "halt_reason": "iteration_limit"}
        batch: list[TaskView] = []
        used: set[str] = set()
        for t in ready:
            if len(batch) >= rt.max_parallel_tasks:
                break
            if set(t.resources) & used:
                continue  # resource conflict: runs in a later round
            a = risk.assess(t.action, t.params, self.resolved_for(t, tasks, outputs))
            with session_scope() as s:
                from ..models import Task

                s.get(Task, t.id).risk_level = a.level
            if risk.needs_approval(a.level, self.run_row().mode):
                decision = await self._approve(t, a)
                if decision == "cancelled" or self.h.cancel.is_set():
                    return {"status": "cancelled"}
                if decision == "denied":
                    store.set_task_status(t.id, "blocked", error="human approval was denied", event="task.blocked", reason="approval_denied")
                    self._block_dependents(t.key)
                    return {"status": "halted", "halt_reason": "approval_denied", "approval_status": "denied"}
                if a.level == "HIGH" or risk.needs_approval(a.level, self.run_row().mode):
                    # the human approved what the card showed; make sure that is still exactly what would run
                    again = risk.assess(t.action, t.params, self.resolved_for(t, tasks, self.outputs()))
                    if fingerprint(again) != fingerprint(a) or not plans.is_approved_plan(self.run_id):
                        store.set_task_status(t.id, "blocked", error="what would run changed after approval was given", event="task.blocked", reason="approval_stale")
                        self._block_dependents(t.key)
                        return {"status": "halted", "halt_reason": "approval_stale", "approval_status": "denied"}
            batch.append(t)
            used |= set(t.resources)
            if a.level == "HIGH":
                break  # consequential actions run alone
        if not batch:
            return {"current_tasks": [], "tool_results": []}
        results = await asyncio.gather(*[self._run_task(t, tasks, outputs) for t in batch])
        if self.h.cancel.is_set():
            return {"status": "cancelled"}
        return {"current_tasks": [t.key for t in batch], "iteration": st.get("iteration", 0) + 1, "approval_status": "approved",
                "tool_results": [{"task": k, "ok": err is None, "error": err} for k, _o, err in results],
                "failures": [{"key": k, "kind": err[0], "error": err[1]} for k, _o, err in results if err]}

    async def _run_task(self, tv: TaskView, tasks: list[TaskView], outputs: dict) -> tuple[str, dict | None, tuple[str, str] | None]:
        store.set_task_status(tv.id, "running", event="task.started", agent=tv.agent, action=tv.action, title=tv.title)
        tv = next(t for t in store.get_tasks(self.run_id) if t.key == tv.key)  # refresh (attempts)
        ctx = self.make_ctx(tv, tasks, outputs)
        fn = custom_agent.run_tool if tv.action.startswith("tool.") else DISPATCH[tv.action]
        timeout = ctx.settings.task_timeout_seconds
        spec = tool_of(tv.action)
        if spec is not None and spec.timeout_s:
            timeout = min(timeout, spec.timeout_s)  # a tool may be stricter than the run-wide limit, never looser
        work = asyncio.ensure_future(fn(ctx))
        cancel_w = asyncio.ensure_future(self.h.cancel.wait())
        try:
            done, _ = await asyncio.wait({work, cancel_w}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if work in done:
                out = work.result()
                store.save_task_output(tv.id, out)
                return tv.key, out, None
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)
            if self.h.cancel.is_set():
                return tv.key, None, ("cancelled", "cancelled by user")
            return tv.key, None, ("timeout", f"task exceeded {timeout:.0f}s")
        except Cancelled:
            return tv.key, None, ("cancelled", "cancelled by user")
        except TaskError as e:
            return tv.key, None, (e.kind, str(e))
        except Exception as e:  # noqa: BLE001
            log.exception("task %s crashed", tv.key)
            return tv.key, None, ("unknown", f"{type(e).__name__}: {e}")
        finally:
            cancel_w.cancel()

    def _block_dependents(self, key: str) -> None:
        tasks = store.get_tasks(self.run_id)
        by_key = {t.key: t for t in tasks}
        changed = True
        blocked = {key}
        while changed:
            changed = False
            for t in tasks:
                if t.key not in blocked and set(t.depends_on) & blocked and by_key[t.key].status in ("pending", "failed"):
                    if t.status == "pending":
                        store.set_task_status(t.id, "blocked", error=f"blocked: upstream task {sorted(set(t.depends_on) & blocked)[0]} did not complete",
                                              event="task.blocked", reason="upstream_unavailable")
                    blocked.add(t.key)
                    changed = True

    async def n_observe(self, st: ExecState) -> ExecState:
        obs = []
        outputs = {t.key: t for t in store.get_tasks(self.run_id)}
        for key in st.get("current_tasks", []):
            tv = outputs[key]
            if tv.status != "running" or not tv.output:
                continue
            o = observe_output(tv, tv.output)
            events.emit(self.run_id, "observation.recorded", tv.id, observation=o)
            obs.append({"task": key, **o})
        return {"observations": [*st.get("observations", []), *obs]}

    async def n_verify(self, st: ExecState) -> ExecState:
        failures = [dict(f) for f in st.get("failures", []) if f["kind"] != "cancelled"]
        findings = list(st.get("verification_findings", []))
        tasks = store.get_tasks(self.run_id)
        by_key = {t.key: t for t in tasks}
        failed_keys = {f["key"] for f in st.get("failures", [])}
        for f in st.get("failures", []):
            tv = by_key[f["key"]]
            if f["kind"] == "cancelled":
                continue
            store.set_task_status(tv.id, "failed", error=f["error"], event="task.failed", kind=f["kind"], attempt=tv.attempts)
        outputs = self.outputs()
        for key in st.get("current_tasks", []):
            if key in failed_keys:
                continue
            tv = by_key[key]
            fresh = store.get_tasks(self.run_id)
            tv = next(t for t in fresh if t.key == key)
            if tv.status != "running":
                continue
            store.set_task_status(tv.id, "verifying")
            ctx = self.make_ctx(tv, fresh, {**outputs, **({tv.key: tv.output} if tv.output else {})})
            ctx.simulated = self.run_row().simulated
            checks = await asyncio.to_thread(verification_agent.verify_task, tv, tv.output or {}, ctx)
            for c in checks:
                store.record_verification(self.run_id, tv.id, c.name, c.passed, c.reason, attempt=tv.attempts, details=c.details)
                findings.append({"task": key, "check": c.name, "passed": c.passed, "reason": c.reason})
            bad = [c for c in checks if not c.passed]
            if bad:
                msg = "verification failed: " + "; ".join(f"{c.name}: {c.reason}" for c in bad)[:900]
                store.set_task_status(tv.id, "failed", error=msg, event="task.failed", kind="verification", attempt=tv.attempts)
                failures.append({"key": key, "kind": "verification", "error": msg})
            else:
                store.set_task_status(tv.id, "completed", event="task.completed", checks=len(checks), attempt=tv.attempts)
                outputs[key] = tv.output or {}
        return {"verification_findings": findings, "failures": failures}

    def r_after_verify(self, st: ExecState) -> str:
        if self.h.cancel.is_set() or st.get("status") in ("cancelled", "halted"):
            return "finalize"
        if any(f["kind"] != "cancelled" for f in st.get("failures", [])):
            return "diagnose"
        tasks = store.get_tasks(self.run_id)
        if all(t.status == "completed" for t in tasks):
            return "finalize"
        if not any(t.status == "pending" and set(t.depends_on) <= {x.key for x in tasks if x.status == "completed"} for t in tasks):
            return "finalize"  # nothing runnable and not everything completed
        return "execute"

    async def n_diagnose(self, st: ExecState) -> ExecState:
        rt = runtime_settings.load()
        tasks = store.get_tasks(self.run_id)
        by_key = {t.key: t for t in tasks}
        outputs = self.outputs()
        decisions = []
        for f in st["failures"]:
            tv = by_key[f["key"]]
            extra: dict = {}
            if tv.action == "test.solve_check":  # the draft is what must change, not the checker
                disputed = (store.get_task(tv.id).output or {}).get("mismatch_ids", [])
                tv = by_key[tv.depends_on[0]]  # the draft of the same paper
                gen_out = outputs.get(tv.key) or {}
                if disputed and gen_out.get("paper"):  # replace only the disputed questions, keep the verified ones
                    extra = {"previous_paper": gen_out["paper"], "disputed": disputed}
            d = recovery_agent.diagnose(tv, f["error"], f["kind"], max_retries=rt.max_retries,
                                        can_use_browser=not self.run_row().simulated and rt.browser_enabled,
                                        backoff_base=get_settings().retry_backoff_seconds)
            d.extra = extra
            if f["key"] and by_key[f["key"]].action == "test.verify_set":  # regenerate the papers that repeat earlier questions
                bad = (store.get_task(by_key[f["key"]].id).output or {}).get("bad_papers", [])
                gens = [t for t in tasks if t.action == "test.generate" and int(t.params.get("paper", 1)) in bad]
                if gens:
                    tv = gens[0]
                    d = recovery_agent.diagnose(tv, f["error"], f["kind"], max_retries=rt.max_retries,
                                                can_use_browser=False, backoff_base=get_settings().retry_backoff_seconds)
                    for extra_gen in gens[1:]:  # all but the first are regenerated through the dependency chain
                        pass
            if d.kind == "duplicate":
                dup = recovery_agent.duplicate_slot_task(tasks, outputs)
                if dup:
                    d.task_key = dup.key
            decisions.append(d)
        events.emit(self.run_id, "recovery.started", None, diagnoses=[
            {"task": d.task_key, "kind": d.kind, "decision": d.decision, "reason": d.reason, "error": d.error[:300]} for d in decisions])
        return {"verification_findings": st.get("verification_findings", []), "decisions": decisions}

    async def n_recovery(self, st: ExecState) -> ExecState:
        decisions = st.get("decisions", [])
        tasks = store.get_tasks(self.run_id)
        by_key = {t.key: t for t in tasks}
        rt = runtime_settings.load()
        full = next((d for d in decisions if d.decision == "replan_all"), None)
        if full is not None:
            replans = int((self.run_row().params or {}).get("replans", 0))
            try:
                if replans >= 1:
                    raise custom_planner.PlanRejected("a full replan was already tried once in this run")
                files = (self.run_row().params or {}).get("files", [])
                specs, assumptions = await custom_planner.plan(self.run_row().objective, files, self.providers(),
                                                               failure=f"task '{by_key[full.task_key].title}' ({by_key[full.task_key].action}) failed: {full.error}")
                store.replace_tasks(self.run_id, specs, reason="full replan after failure")
                with session_scope() as s:
                    row = s.get(Run, self.run_id)
                    row.params = {**(row.params or {}), "replans": replans + 1}
                    row.recovery_count += 1
                events.emit(self.run_id, "plan.updated", None, reason="full replan after failure", count=len(specs), assumptions=assumptions)
                carry = plans.carry_approval(self.run_id)
                if carry["escalated"]:
                    events.emit(self.run_id, "recovery.completed", None, outcome="gave_up", reasons=["the new plan is riskier than the one you approved; it needs your approval first"])
                    return {"halt_reason": "plan_changed_needs_approval", "failures": [], "recovery_attempts": st.get("recovery_attempts", 0) + 1}
                events.emit(self.run_id, "recovery.completed", None, outcome="replanned", patches=[{"task": "ALL", "reason": full.reason}], approval_carried=True)
                return {"failures": [], "recovery_attempts": st.get("recovery_attempts", 0) + 1, "current_tasks": []}
            except custom_planner.PlanRejected as e:
                full.decision, full.needs_human, full.reason = "give_up", True, f"a new plan could not be made: {e.reason[:200]}"
        revise_patches: list[planner.PlanPatch] = []
        for d in decisions:
            if d.decision != "revise":
                continue
            tv = by_key[d.task_key]
            try:
                new_args = await custom_planner.revise_args(tv, d.error, (self.run_row().params or {}).get("files", []), self.outputs(), self.providers())
                revise_patches.append(planner.PlanPatch(tv.key, {"args": new_args, "_recoveries": int(tv.params.get("_recoveries", 0)) + 1},
                                                        reason=f"arguments revised by the planner after: {d.error[:140]}"))
            except Exception as e:  # noqa: BLE001 - if the planner cannot repair it, a human has to
                d.decision, d.needs_human, d.reason = "give_up", True, f"the planner could not repair the arguments ({type(e).__name__}: {str(e)[:120]})"
        gave_up = [d for d in decisions if d.decision == "give_up"]
        if gave_up:
            for d in gave_up:
                tv = by_key.get(d.task_key)
                if tv:
                    self._block_dependents(tv.key)
            human = any(d.needs_human for d in gave_up)
            events.emit(self.run_id, "recovery.completed", None, outcome="gave_up", reasons=[f"{d.task_key}: {d.reason}" for d in gave_up])
            reason = "needs_human" if human else "retries_exhausted"
            return {"halt_reason": reason, "failures": [], "recovery_attempts": st.get("recovery_attempts", 0) + 1}
        total = self.run_row().recovery_count + len(decisions)
        if total > max(rt.max_retries * len(tasks), 1):
            events.emit(self.run_id, "recovery.completed", None, outcome="gave_up", reasons=["global recovery budget exhausted"])
            return {"halt_reason": "retries_exhausted", "failures": []}
        wait_s = max((d.backoff_s for d in decisions), default=0)
        if wait_s > 0:
            try:
                await asyncio.wait_for(self.h.cancel.wait(), timeout=wait_s)
            except asyncio.TimeoutError:
                pass
        if self.h.cancel.is_set():
            return {"status": "cancelled"}
        patches = [*planner.replan(decisions, tasks), *revise_patches]
        applied = []
        for p in patches:
            tv = by_key[p.key]
            if p.set_params:
                store.update_task_params(tv.id, {**tv.params, **p.set_params})
            self._reset(p.key)
            applied.append({"task": p.key, "set_params": {k: v for k, v in p.set_params.items() if not k.startswith("_")}, "reason": p.reason})
        for d in decisions:  # a failed document verification rebuilds the documents it checked
            if d.task_key in by_key and by_key[d.task_key].action == "test.verify_documents":
                for dep in by_key[d.task_key].depends_on:
                    self._reset(dep)
        # a failed verify.deliverables must rebuild the specific files that failed, not just re-run verification
        for d in decisions:
            if d.task_key in by_key and by_key[d.task_key].action == "verify.deliverables":
                out = store.get_task(by_key[d.task_key].id).output or {}
                for bk in out.get("failed_builds", []):
                    self._reset(bk)
        with session_scope() as s:
            s.get(Run, self.run_id).recovery_count += len(decisions)
        if plans.sync_current(self.run_id, "recovery patch"):
            carry = plans.carry_approval(self.run_id)
            if carry["escalated"]:
                events.emit(self.run_id, "plan.updated", None, reason="recovery", patches=applied)
                events.emit(self.run_id, "recovery.completed", None, outcome="gave_up", reasons=["the repaired plan is riskier than the one you approved; it needs your approval first"])
                return {"halt_reason": "plan_changed_needs_approval", "failures": [], "recovery_attempts": st.get("recovery_attempts", 0) + 1}
        events.emit(self.run_id, "plan.updated", None, reason="recovery", patches=applied)
        events.emit(self.run_id, "recovery.completed", None, outcome="replanned", patches=applied)
        return {"failures": [], "recovery_attempts": st.get("recovery_attempts", 0) + 1, "current_tasks": []}

    def _reset(self, key: str) -> None:
        tasks = store.get_tasks(self.run_id)
        affected = {key}
        grew = True
        while grew:
            grew = False
            for t in tasks:
                if t.key not in affected and set(t.depends_on) & affected:
                    affected.add(t.key)
                    grew = True
        for t in tasks:
            if t.key in affected and t.status in ("completed", "failed", "blocked"):
                store.set_task_status(t.id, "pending")
                store.save_task_output(t.id, None)  # type: ignore[arg-type]

    async def n_finalize(self, st: ExecState) -> ExecState:
        store.close_open_activities(self.run_id)
        tasks = store.get_tasks(self.run_id)
        arts = artifact_service.list_for_run(self.run_id)
        r = self.run_row()
        cancelled = self.h.cancel.is_set() or st.get("status") == "cancelled"
        if cancelled:
            for t in tasks:
                if t.status in ("pending", "running", "verifying", "failed", "blocked"):
                    store.set_task_status(t.id, "cancelled", event="task.blocked", reason="run cancelled")
            store.set_run_status(self.run_id, "cancelled", event="run.cancelled", reason="cancelled by user")
            return {"status": "cancelled", "artifacts": arts}
        halt = st.get("halt_reason")
        failed = [t for t in tasks if t.status in ("failed", "blocked")]
        done = [t for t in tasks if t.status == "completed"]
        if halt == "approval_denied":
            store.set_run_status(self.run_id, "halted", halt_reason=halt, error="A required human approval was denied; execution stopped.",
                                 event="run.halted", reason=halt)
            return {"status": "halted", "artifacts": arts}
        if halt == "plan_changed_needs_approval":
            store.set_run_status(self.run_id, "halted", halt_reason=halt, event="run.halted", reason=halt,
                                 error="Recovery produced a plan that is riskier than the one you approved. Review and approve the new plan to continue.")
            return {"status": "halted", "artifacts": arts}
        if halt == "plan_not_approved" or halt == "approval_stale":
            if halt == "approval_stale":
                store.set_run_status(self.run_id, "halted", halt_reason=halt, event="run.halted", reason=halt,
                                     error="The action changed after it was approved, so it was not executed. Review it and continue.")
            return {"status": "halted", "artifacts": arts}
        if halt == "needs_human" or st.get("status") == "halted":
            msgs = "; ".join(f"{t.key}: {t.error}" for t in failed if t.error)[:800] or halt or ""
            store.set_run_status(self.run_id, "halted", halt_reason=halt or "needs_human", error=f"Human intervention required. {msgs}",
                                 event="run.halted", reason=halt or "needs_human")
            return {"status": "halted", "artifacts": arts}
        if len(done) == len(tasks) and tasks:
            store.set_run_status(self.run_id, "completed", event="run.completed", artifacts=len(arts), simulated=r.simulated,
                                 tasks=len(tasks), recoveries=r.recovery_count)
            return {"status": "completed", "artifacts": arts}
        msgs = "; ".join(f"{t.key}: {t.error}" for t in failed if t.error)[:900]
        real_arts = [a for a in arts if a["kind"] != "screenshot"]
        if real_arts:
            store.set_run_status(self.run_id, "partial", halt_reason=halt, error=f"Not all required steps passed verification. {msgs}",
                                 event="run.partial", artifacts=len(real_arts), failed=[t.key for t in failed])
            return {"status": "partial", "artifacts": arts}
        store.set_run_status(self.run_id, "failed", halt_reason=halt, error=msgs or "execution failed", event="run.failed",
                             failed=[t.key for t in failed])
        return {"status": "failed", "artifacts": arts}


def _spec_summary(t: TaskSpec) -> dict:
    return {"key": t.key, "title": t.title, "agent": t.agent, "action": t.action, "depends_on": t.depends_on,
            "risk_level": t.risk_level}


def observe_output(tv: TaskView, out: dict) -> dict:
    """Independent description of what a task actually produced (read from the output / disk, not from the agent's claim)."""
    a = tv.action
    if a == "research.discover":
        return {"candidates": [c["name"] for c in out.get("candidates", [])], "search_results": len(out.get("search_results", []))}
    if a == "browser.collect":
        return {"competitor": out["competitor"]["name"], "pages": [{"role": p["role"], "url": p["url"], "chars": len(p["text"]), "via": p["via"]} for p in out["pages"]],
                "warnings": out.get("warnings", [])}
    if a == "research.extract":
        r = out["record"]
        return {"competitor": r["name"], "features": len(r["features"]), "plans": len(r["plans"]), "pricing_status": r["pricing_status"],
                "rejected_claims": len(out.get("rejected_claims", []))}
    if a == "workspace.normalize":
        return {"records": len(out["records"]), "stats": out.get("stats", {})}
    if a in ("workspace.build_excel", "workspace.build_pptx"):
        try:
            path, art = artifact_service.resolve_download(out["artifact_id"])
            return {"file": art.filename, "size_on_disk": path.stat().st_size}
        except Exception as e:  # noqa: BLE001
            return {"file": out.get("filename"), "problem": str(e)}
    if a == "verify.deliverables":
        return {"checks": len(out.get("checks", [])), "failed": [c["name"] for c in out.get("checks", []) if not c["passed"]]}
    if a.startswith("tool."):
        return {"tool": out.get("tool"), **{k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in out.items() if k not in ("tool", "text", "rows", "values")
                                              and isinstance(v, (int, float, bool, str, list, dict)) and len(str(v)) < 400}}
    if a == "test.generate":
        qs = out["paper"]["questions"]
        return {"questions": len(qs), "by_type": {t: sum(q["type"] == t for q in qs) for t in ("mcq", "short", "long")}, "total_marks": out["total_marks"]}
    if a in ("test.solve_check", "test.verify_documents", "test.verify_set"):
        return {"checks": len(out.get("checks", [])), "failed": [c["name"] for c in out.get("checks", []) if not c["passed"]]}
    if a == "test.build_documents":
        return {"files": [out["paper_file"], out["key_file"]]}
    if a == "telegram.send":
        return {"delivered": out["delivered"], "simulated": out["simulated"], "message_id": out["message_id"], "file": out["file_sent"]}
    if a == "file.read_inputs":
        return {"files": [{"name": f["name"], "chars": f["chars"]} for f in out.get("files", [])]}
    return {}


def fingerprint(a: "risk.RiskAssessment") -> str:
    """Stable hash of what an approval card shows (target + details): the approval is only valid for exactly this."""
    import hashlib
    import json

    return hashlib.sha256(json.dumps({"t": a.target, "d": a.details, "l": a.level}, sort_keys=True, default=str).encode()).hexdigest()[:20]
