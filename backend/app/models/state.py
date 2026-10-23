"""Validated run/task state machines."""
from __future__ import annotations

RUN_TERMINAL = {"completed", "partial", "failed", "halted", "cancelled", "rejected"}

RUN_TRANSITIONS: dict[str, set[str]] = {
    "created": {"planning", "rejected", "cancelled", "failed"},
    "planning": {"awaiting_plan_approval", "running", "rejected", "failed", "cancelled"},
    "awaiting_plan_approval": {"running", "cancelled", "planning", "failed"},
    "running": {"paused", "awaiting_approval", "completed", "partial", "failed", "halted", "cancelled"},
    "paused": {"running", "cancelled", "failed"},
    "awaiting_approval": {"running", "halted", "cancelled", "failed", "paused"},
    "completed": set(), "partial": set(), "failed": set(), "halted": set(), "cancelled": set(), "rejected": set(),
}

# Only an explicit "continue" (after inspecting persisted state) may leave a terminal state; never the normal path.
RESUME_TRANSITIONS: dict[str, set[str]] = {
    "failed": {"running", "awaiting_plan_approval"}, "halted": {"running", "awaiting_plan_approval"}, "partial": {"running", "awaiting_plan_approval"},
}

TASK_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"running", "blocked", "cancelled"},
    "running": {"verifying", "failed", "cancelled"},
    "verifying": {"completed", "failed", "cancelled"},
    "completed": {"pending"},  # invalidated by an upstream redo
    "failed": {"pending", "blocked", "cancelled"},  # recovery resets to pending
    "blocked": {"pending", "cancelled"},
    "cancelled": set(),
}


class InvalidTransition(ValueError):
    pass


def check_run_transition(old: str, new: str, resume: bool = False) -> None:
    allowed = RUN_TRANSITIONS.get(old, set()) | (RESUME_TRANSITIONS.get(old, set()) if resume else set())
    if old != new and new not in allowed:
        raise InvalidTransition(f"run: {old} -> {new} is not allowed")


def check_task_transition(old: str, new: str) -> None:
    if old != new and new not in TASK_TRANSITIONS.get(old, set()):
        raise InvalidTransition(f"task: {old} -> {new} is not allowed")
