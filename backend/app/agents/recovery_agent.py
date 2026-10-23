"""Recovery Agent: classify failures and decide what (if anything) may be safely retried."""
from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse

from ..services.store import TaskView

HUMAN_KINDS = {"search_unavailable", "browser_unavailable", "config", "file", "needs_input", "permission", "ambiguous"}
TRANSIENT = {"network", "timeout", "llm", "empty_result", "unknown"}


@dataclass
class Diagnosis:
    task_key: str
    kind: str
    error: str
    decision: str  # retry | retry_with_browser | broaden_search | swap_candidate | rebuild | give_up
    reason: str
    needs_human: bool = False
    backoff_s: float = 0.0
    extra: dict = field(default_factory=dict)  # extra params the planner should merge into the retried task


def classify(error: str, kind: str | None) -> str:
    if kind:
        return kind
    e = error.lower()
    if "timeout" in e or "timed out" in e:
        return "timeout"
    if "403" in e or "robots" in e or "access restriction" in e:
        return "blocked"
    return "unknown"


def diagnose(task: TaskView, error: str, kind: str | None, *, max_retries: int, can_use_browser: bool,
             backoff_base: float = 1.0) -> Diagnosis:
    k = classify(error, kind)
    n = int(task.params.get("_recoveries", 0))
    backoff = min(backoff_base * (2 ** n), 8.0)
    mk = lambda decision, reason, human=False: Diagnosis(task.key, k, error, decision, reason, human, backoff)  # noqa: E731
    if k in HUMAN_KINDS:
        return mk("give_up", f"'{k}' cannot be fixed automatically; human intervention is required", True)
    if k == "exhausted":
        return mk("give_up", "no alternative competitors remain")
    if n >= max_retries:
        return mk("give_up", f"retry limit reached ({n}/{max_retries})")
    a = task.action
    if a == "tool.telegram.send_files":
        return mk("give_up", "the post may or may not have reached the group; check the group, then run it again deliberately", True)
    if a.startswith("tool."):
        from ..workflows.custom import tool_of

        spec = tool_of(a)
        if spec is not None and not spec.idempotent:  # email, moves, posts: never blindly repeat something that may have taken effect
            return mk("give_up", f"{spec.name} may already have taken effect, so it is not repeated automatically; check the outcome, then run it again deliberately", True)
        if k in ("network", "timeout", "llm"):
            return mk("retry", "transient failure; retrying")
        if n >= 1:  # fixing arguments did not help: rebuild the whole plan, told exactly what failed
            return mk("replan_all", "correcting the arguments was not enough; asking the planner for a new plan that avoids this failure")
        return mk("revise", "asking the planner to correct this task's arguments using the failure details")
    if a == "telegram.send":  # never blindly repeat an outward-facing, irreversible action
        return mk("give_up", "the post may or may not have reached the group; check the group, then run it again deliberately", True)
    if a == "test.generate":
        return mk("retry", "redrafting the paper with the failure details as feedback")
    if a in ("test.build_documents", "test.verify_documents", "test.solve_check"):
        return mk("rebuild" if a != "test.solve_check" else "retry", "rebuilding and re-verifying the documents")
    if a == "browser.collect":
        if k in ("blocked", "unsafe"):
            return mk("swap_candidate", "access to this site is restricted; using an alternative competitor instead of bypassing it")
        if k in ("network", "timeout") and n == 0:
            return mk("retry", "transient network failure; retrying with backoff")
        if can_use_browser and not task.params.get("force_browser"):
            return mk("retry_with_browser", "plain HTTP retrieval failed or was too thin; retrying with the browser")
        return mk("swap_candidate", "site could not be retrieved reliably; using an alternative competitor")
    if a == "research.extract":
        if k == "llm" and n == 0:
            return mk("retry", "model call failed; retrying")
        return mk("swap_candidate", "no verifiable facts for this competitor; using an alternative competitor")
    if a == "research.discover":
        return mk("broaden_search" if k in ("empty_result", "verification") else "retry", "discovery failed; retrying with a broader search")
    if a == "workspace.normalize" and k == "duplicate":
        return mk("swap_candidate", "two slots resolved to the same company; replacing one")
    if a in ("workspace.build_excel", "workspace.build_pptx", "verify.deliverables"):
        return mk("rebuild", "regenerating the deliverable and re-verifying")
    if k == "verification":
        return mk("retry" if n == 0 else "swap_candidate" if a in ("research.extract", "browser.collect") else "retry",
                  "verification failed; retrying")
    if k in TRANSIENT:
        return mk("retry", "treating as transient")
    return mk("give_up", f"unrecognised failure kind '{k}'", True)


def duplicate_slot_task(tasks: list[TaskView], outputs: dict[str, dict]) -> TaskView | None:
    """For a normalisation duplicate: return the later extraction task whose website repeats."""
    seen: dict[str, str] = {}
    for t in sorted((x for x in tasks if x.action == "research.extract" and x.key in outputs), key=lambda x: x.params["slot"]):
        host = (urlparse(outputs[t.key]["record"]["website"]).hostname or "").removeprefix("www.")
        if host in seen:
            return t
        seen[host] = t.key
    return None
