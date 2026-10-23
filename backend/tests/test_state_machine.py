import pytest

from app.models.state import InvalidTransition, check_run_transition, check_task_transition
from app.services import store


def test_valid_run_transitions():
    for a, b in [("created", "planning"), ("planning", "awaiting_plan_approval"), ("awaiting_plan_approval", "running"),
                 ("running", "paused"), ("paused", "running"), ("running", "completed"), ("running", "awaiting_approval")]:
        check_run_transition(a, b)


@pytest.mark.parametrize("a,b", [("completed", "running"), ("created", "completed"), ("cancelled", "running"),
                                 ("awaiting_plan_approval", "completed"), ("failed", "running")])
def test_invalid_run_transitions(a, b):
    with pytest.raises(InvalidTransition):
        check_run_transition(a, b)


@pytest.mark.parametrize("a,b", [("pending", "completed"), ("running", "completed"), ("cancelled", "pending"), ("blocked", "running")])
def test_invalid_task_transitions(a, b):
    with pytest.raises(InvalidTransition):
        check_task_transition(a, b)


def test_task_cannot_complete_without_verification():
    # running -> completed is illegal: a task must pass through "verifying"
    with pytest.raises(InvalidTransition):
        check_task_transition("running", "completed")
    check_task_transition("running", "verifying")
    check_task_transition("verifying", "completed")


def test_store_enforces_transitions_and_terminal_is_final():
    from app.database import session_scope
    from app.models import Run

    with session_scope() as s:
        r = Run(objective="x")
        s.add(r)
        s.flush()
        rid = r.id
    with pytest.raises(InvalidTransition):
        store.set_run_status(rid, "completed")
    store.set_run_status(rid, "cancelled")
    store.set_run_status(rid, "failed")  # ignored: terminal states are final
    assert store.get_run(rid).status == "cancelled"
