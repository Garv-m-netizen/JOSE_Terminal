"""Planner: builds dependency-aware task graphs, validates edits, and turns recovery diagnoses into plan patches."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas.domain import ExamParams, TaskSpec, WorkflowParams
from ..services import risk
from ..services.store import TaskView
from ..workflows.research_report import PlanError, build_plan, validate_plan  # noqa: F401  (re-exported)


@dataclass
class PlanPatch:
    key: str  # task to modify
    set_params: dict = field(default_factory=dict)
    reset: bool = True  # move the task (and anything depending on it) back to pending
    reason: str = ""


def plan(params: WorkflowParams | ExamParams) -> list[TaskSpec]:
    if isinstance(params, ExamParams):
        from ..workflows.test_paper import build_plan as build_exam_plan

        tasks = build_exam_plan(params)
    else:
        tasks = build_plan(params)
    for t in tasks:
        t.risk_level = risk.assess(t.action, t.params).level
    return tasks


def validate_edit(edited: list[TaskSpec]) -> list[TaskSpec]:
    """Validate a user-edited plan (cycle detection, allow-list, required dependencies). Risk is always recomputed."""
    validate_plan(edited)
    for t in edited:
        t.risk_level = risk.assess(t.action, t.params).level  # a user cannot lower a task's risk
    return edited


def next_candidate_index(collects: list[TaskView]) -> int:
    used: set[int] = set()
    for c in collects:
        used.add(int(c.params.get("candidate_index", c.params["slot"])))
        used.update(int(i) for i in c.params.get("_tried", []))
    i = 0
    while i in used:
        i += 1
    return i


def replan(decisions: list, tasks: list[TaskView]) -> list[PlanPatch]:
    """Translate recovery decisions into concrete, safe modifications of the existing plan (no free-form edits)."""
    by_key = {t.key: t for t in tasks}
    collects = [t for t in tasks if t.action == "browser.collect"]
    patches: list[PlanPatch] = []
    for d in decisions:
        t = by_key[d.task_key]
        bump = {"_recoveries": int(t.params.get("_recoveries", 0)) + 1}
        if d.decision == "give_up":
            continue
        if d.decision == "retry":
            if t.action == "test.generate":
                bump["feedback"] = d.error[:800]  # tell the next draft exactly what was wrong
                bump.update(d.extra)
            patches.append(PlanPatch(t.key, bump, reason=d.reason))
        elif d.decision == "retry_with_browser":
            patches.append(PlanPatch(t.key, {**bump, "force_browser": True}, reason=d.reason))
        elif d.decision == "broaden_search":
            patches.append(PlanPatch(t.key, {**bump, "broaden": True}, reason=d.reason))
        elif d.decision == "swap_candidate":
            slot_task = t
            if t.action == "research.extract":  # swap on the matching collection task
                slot_task = next(c for c in collects if c.params["slot"] == t.params["slot"])
            idx = next_candidate_index(collects)
            tried = [*slot_task.params.get("_tried", []), int(slot_task.params.get("candidate_index", slot_task.params["slot"]))]
            patches.append(PlanPatch(slot_task.key, {"candidate_index": idx, "_tried": tried, "force_browser": False,
                                                     "_recoveries": int(slot_task.params.get("_recoveries", 0)) + 1},
                                     reason=d.reason))
            if slot_task.key != t.key:
                patches.append(PlanPatch(t.key, bump, reason=d.reason))
            collects = [slot_task if c.key == slot_task.key else c for c in collects]
            slot_task.params = {**slot_task.params, "candidate_index": idx, "_tried": tried}
        elif d.decision == "revise":
            continue  # arguments are revised by the orchestrator (needs an async model call)
        elif d.decision == "rebuild":
            patches.append(PlanPatch(t.key, bump, reason=d.reason))
    return patches
