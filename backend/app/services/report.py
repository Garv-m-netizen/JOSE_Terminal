"""Honest completion report, derived only from persisted facts (task states, verification rows, artifacts, the action ledger).

Five different things are reported separately and never merged into one "success":
  1. tool call completed        - the tool ran and returned a result
  2. output generated           - a file / result exists
  3. output verified            - independent checks on that result passed (task status `completed`)
  4. accepted by external service - a provider confirmed an outward action (email accepted, Telegram message id)
  5. workflow completed         - every task verified; only then is the run `completed`
"""
from __future__ import annotations

from collections import defaultdict

from . import artifact_service, ledger, plans, store


def build(run_id: str) -> dict:
    from .execution_service import verification_report

    run = store.get_run(run_id)
    tasks = store.get_tasks(run_id)
    ver = verification_report(run_id)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for c in ver["current"]:
        by_task[c["task_key"]].append(c)
    acts = ledger.for_run(run_id)
    acts_by_task = defaultdict(list)
    for a in acts:
        acts_by_task[a["task_id"]].append(a)
    arts = [a for a in artifact_service.list_for_run(run_id) if a["kind"] != "screenshot"]

    rows = []
    for t in tasks:
        checks = by_task.get(t.key, [])
        failed = [c for c in checks if not c["passed"]]
        rows.append({
            "key": t.key, "title": t.title, "action": t.action, "status": t.status, "risk": t.risk_level, "attempts": t.attempts,
            "tool_completed": t.output is not None,
            "output_verified": t.status == "completed" and bool(checks) and not failed,
            "checks_passed": sum(1 for c in checks if c["passed"]), "checks_failed": len(failed),
            "failed_checks": [f"{c['check']}: {c['reason']}" for c in failed][:5],
            "error": t.error,
            "external": [{"kind": a["kind"], "status": a["status"], "target": a["target"], "reference": a["reference"], "simulated": a["simulated"]} for a in acts_by_task.get(t.id, [])],
        })

    done = [r for r in rows if r["status"] == "completed"]
    external_ok = [a for a in acts if a["status"] == "accepted" and not a["simulated"]]
    external_sim = [a for a in acts if a["status"] == "accepted" and a["simulated"]]
    external_unknown = [a for a in acts if a["status"] in ("started", "ambiguous")]
    external_failed = [a for a in acts if a["status"] == "failed"]
    complete = run.status == "completed" and len(done) == len(rows) and bool(rows)

    caveats = []
    if run.simulated:
        caveats.append("This was a SIMULATED run: no real web data, email or Telegram message was used or sent.")
    if any(r["action"] == "tool.llm.generate" for r in done):
        caveats.append("Some text was written by an AI model and has not been fact-checked.")
    if external_ok:
        caveats.append("'Accepted' means the external service confirmed receipt of the request. For email it is not proof the message reached an inbox.")
    for a in external_ok:
        if (a["response"] or {}).get("manual"):
            caveats.append(f"{a['kind']} to {a['target']} was confirmed by a person, not by the provider.")
    for a in external_unknown:
        caveats.append(f"UNKNOWN OUTCOME: {a['kind']} to {a['target']} may or may not have happened ({a['error'] or a['status']}). Check the destination before repeating it.")

    n = len(rows)
    if complete:
        head = f"Completed: all {n} steps passed their independent verification."
    elif run.status in ("failed", "halted", "partial", "cancelled", "rejected"):
        head = f"Not completed ({run.status}): {len(done)} of {n} steps verified."
    else:
        head = f"In progress ({run.status}): {len(done)} of {n} steps verified so far."
    lines = [head]
    if arts:
        lines.append(f"Files produced: {', '.join(a['filename'] for a in arts)}.")
    for a in external_ok:
        lines.append(f"Accepted by the external service: {a['kind']} -> {a['target']} (reference {a['reference']}).")
    for a in external_sim:
        lines.append(f"Simulated (nothing was sent): {a['kind']} -> {a['target']}.")
    if run.error:
        lines.append(f"Reason: {run.error}")
    for r in rows:
        if r["status"] in ("failed", "blocked") and r["error"]:
            lines.append(f"Step {r['key']} ({r['title']}) {r['status']}: {r['error'][:200]}")

    return {
        "run_id": run_id, "status": run.status, "workflow_completed": complete, "headline": head, "summary": "\n".join(lines),
        "levels": {
            "tool_calls_completed": sum(1 for r in rows if r["tool_completed"]),
            "outputs_generated": len(arts),
            "outputs_verified": len(done),
            "external_actions_accepted": len(external_ok),
            "external_actions_simulated": len(external_sim),
            "external_actions_unknown": len(external_unknown),
            "external_actions_failed": len(external_failed),
            "tasks_total": n,
        },
        "tasks": rows, "artifacts": [{"id": a["id"], "filename": a["filename"], "kind": a["kind"], "size_bytes": a["size_bytes"], "sha256": a["sha256"], "simulated": a["simulated"]} for a in arts],
        "external_actions": acts, "caveats": caveats, "assumptions": (run.intent or {}).get("assumptions", []),
        "plan": {"version": run.plan_version, "hash": run.plan_hash, "approved_hash": run.approved_plan_hash, "versions": len(plans.list_versions(run_id))},
        "recoveries": run.recovery_count,
    }
