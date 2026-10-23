"""JOSE terminal interface. Same engine as the website: it calls the same RunManager, tool registry, approval and ledger services.

    python -m app.cli                       voice mode: listens for "hi jose" (falls back to the typed prompt if there is no audio)
    python -m app.cli repl                  typed prompt (type an objective)
    python -m app.cli run "objective" [-f file ...] [--auto] [--simulate] [-y]
    python -m app.cli history | show RUN | graph RUN | report RUN | artifacts RUN | continue RUN | cancel RUN
    python -m app.cli tools | resolve ACTION_ID performed|not_performed

Nothing here decides anything: plan approval, risk approvals, verification and recovery all happen in the engine.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from .config import get_settings
from .database import init_engine, session_scope
from .models import Upload
from .models.state import RUN_TERMINAL
from .schemas.domain import ObjectiveCreate, TaskSpec
from .services import approval_service, artifact_service, execution_service as ex, ledger, plans, report, runtime_settings, store
from .services.event_service import events
from .tools.file_tools import FileRejected, validate_upload

VERBOSE = False
USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
C = {"r": "31", "g": "32", "y": "33", "b": "34", "m": "35", "c": "36", "d": "2", "B": "1"}
STATUS_COLOR = {"completed": "g", "verified": "g", "failed": "r", "halted": "y", "partial": "y", "cancelled": "d", "rejected": "r", "running": "c", "pending": "d",
                "blocked": "y", "verifying": "m", "awaiting_approval": "y", "awaiting_plan_approval": "y"}


def paint(text: str, c: str) -> str:
    return f"\033[{C[c]}m{text}\033[0m" if USE_COLOR else text


def out(text: str = "") -> None:
    print(text, flush=True)


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


# ------------------------------------------------------------------ rendering

def render_plan(run: dict) -> None:
    out(paint(f"\nPlan v{run['plan_version']} ({run['plan_hash']}) - {len(run['tasks'])} tasks - highest risk {run['risk_level']}", "B"))
    intent = run.get("intent") or {}
    for a in intent.get("assumptions", []):
        out(paint(f"  assumption: {a}", "y"))
    for t in run["tasks"]:
        deps = ",".join(t["depends_on"]) or "-"
        risk = paint(t["risk_level"], "r" if t["risk_level"] == "HIGH" else "y" if t["risk_level"] == "MEDIUM" else "d")
        out(f"  {paint(t['key'], 'B'):>12}  {t['title'][:60]:<60} [{t['action']}] after {deps}  risk {risk}")
        args = (t.get("params") or {}).get("args")
        if args:
            out(paint("        " + json.dumps(args, ensure_ascii=False)[:150], "d"))
    render_graph(run)


GLYPH = {"completed": "[x]", "running": "[>]", "verifying": "[?]", "failed": "[!]", "blocked": "[~]", "cancelled": "[-]", "pending": "[ ]"}


def graph_levels(tasks: list[dict]) -> list[list[dict]]:
    """Group tasks into dependency layers: a task sits one layer below its deepest prerequisite, so each layer can run in parallel."""
    by_key = {t["key"]: t for t in tasks}
    depth: dict[str, int] = {}

    def level(key: str, seen: frozenset = frozenset()) -> int:
        if key in depth:
            return depth[key]
        if key in seen or key not in by_key:  # cycles are rejected by the planner; unknown deps are ignored here
            return 0
        depth[key] = 1 + max((level(d, seen | {key}) for d in by_key[key]["depends_on"] if d in by_key), default=-1)
        return depth[key]

    layers: dict[int, list[dict]] = {}
    for t in tasks:
        layers.setdefault(level(t["key"]), []).append(t)
    return [layers[i] for i in sorted(layers)]


def render_graph(run: dict) -> None:
    """Task graph as dependency layers with the real status of each task (taken from the engine, never inferred here)."""
    out(paint(f"\nTask graph - run {run['id']} - {run['status']}", "B"))
    for i, layer in enumerate(graph_levels(run["tasks"])):
        out(paint(f"  step {i + 1}" + ("  (these can run in parallel)" if len(layer) > 1 else ""), "d"))
        for t in layer:
            deps = ",".join(t["depends_on"]) or "-"
            out(f"    {paint(GLYPH.get(t['status'], '[ ]'), STATUS_COLOR.get(t['status'], 'd'))} {t['key']:<4} {t['title'][:56]:<56} {t['action']}  <- {deps}")
    out(paint("  [ ] pending  [>] running  [?] verifying  [x] verified  [!] failed  [~] blocked  [-] cancelled", "d"))


def render_event(e: dict) -> str | None:
    d, t = e["details"], (e["task_id"] or "").split(":")[-1]
    ty = e["type"]
    if ty == "task.started":
        return paint(f"  > {t} {d.get('title', '')}", "c")
    if ty == "task.completed":
        return paint(f"  + {t} verified ({d.get('checks')} checks)", "g")
    if ty == "task.failed":
        return paint(f"  x {t} FAILED [{d.get('kind')}] {str(d.get('error', ''))[:200]}", "r")
    if ty == "task.blocked":
        return paint(f"  ! {t} blocked: {d.get('reason') or d.get('error')}", "y")
    if ty == "verification.passed":
        return paint(f"      pass {d.get('check')}", "d") if VERBOSE else None
    if ty == "verification.failed":
        return paint(f"      FAIL {d.get('check')}: {d.get('reason')}", "r")
    if ty == "tool.completed":
        return paint(f"      tool {d.get('tool')}: {d.get('summary')}", "d")
    if ty == "tool.failed":
        return paint(f"      tool {d.get('tool')} failed: {str(d.get('error'))[:160]}", "r")
    if ty == "recovery.started":
        return paint("  ~ recovery: " + "; ".join(f"{x['task']} [{x['kind']}] -> {x['decision']}" for x in d.get("diagnoses", [])), "y")
    if ty == "recovery.completed":
        return paint(f"  ~ recovery {d.get('outcome')}", "y")
    if ty == "plan.version":
        return paint(f"  plan v{d.get('version')} {d.get('plan_hash')} ({d.get('reason')})", "b")
    if ty == "external.action":
        return paint(f"  external {d.get('kind')} -> {d.get('status')} {d.get('reference') or ''}", "g" if d.get("status") == "accepted" else "y")
    if ty == "artifact.created":
        return paint(f"  file {d.get('filename')} ({d.get('size_bytes')} bytes)", "g")
    if ty == "run.continued":
        a = d.get("audit") or {}
        return paint(f"  continue: kept {a.get('kept', [])} redo {[x['task'] for x in a.get('redone', [])]} waiting {[x['task'] for x in a.get('needs_human', [])]}", "c")
    if ty == "warning":
        return paint(f"  warning: {d.get('message')}", "y")
    if ty.startswith("run.") and ty in ("run.completed", "run.failed", "run.halted", "run.partial", "run.cancelled", "run.rejected", "run.needs_input"):
        return paint(f"  {ty}", "B")
    return None


def render_approval(a: dict) -> None:
    d = a["details"]
    out(paint(f"\n=== APPROVAL REQUIRED ({a['risk_level']}) ===", "y"))
    out(f"  action : {a['action']}\n  target : {a['target']}")
    if a["action"] == "tool.email.send":
        out(f"  from   : {d.get('from')}\n  to     : {d.get('to')}" + (f"\n  cc     : {d['cc']}" if d.get("cc") else ""))
        out(f"  subject: {d.get('subject')}\n  body   :")
        for line in str(d.get("body", "")).splitlines():
            out(f"    | {line}")
        out(f"  attach : {', '.join(d.get('attachments') or []) or 'none'}")
        if d.get("recipient_basis"):
            out(f"  source : {'; '.join(f'{k}: {v}' for k, v in d['recipient_basis'].items())}")
    else:
        for k in ("file", "caption", "source", "destination", "excluded"):
            if d.get(k):
                out(f"  {k:7}: {d[k]}")
    out(paint(f"  consequences: {a['consequences']}", "y"))
    if d.get("dry_run"):
        out(paint("  (SIMULATED: nothing will really be sent)", "m"))


def render_report(run_id: str) -> None:
    r = report.build(run_id)
    lv = r["levels"]
    out(paint("\n" + r["headline"], "g" if r["workflow_completed"] else "y"))
    out(f"  tool calls completed {lv['tool_calls_completed']}/{lv['tasks_total']} | outputs generated {lv['outputs_generated']} | outputs verified "
        f"{lv['outputs_verified']}/{lv['tasks_total']} | accepted by external services {lv['external_actions_accepted']} | workflow completed: "
        f"{'yes' if r['workflow_completed'] else 'NO'}")
    for line in r["summary"].splitlines()[1:]:
        out("  " + line)
    for c in r["caveats"]:
        out(paint("  ! " + c, "y"))
    for a in r["artifacts"]:
        out(paint(f"  file: {a['filename']}  ({a['size_bytes']} bytes, sha256 {a['sha256'][:12]}...)", "g"))
    if r["artifacts"]:
        out(paint(f"  folder: {artifact_service.run_dir(run_id)}", "d"))


# ------------------------------------------------------------------ actions

def register_upload(path: str) -> str:
    src = Path(path).expanduser()
    if not src.is_file():
        raise SystemExit(f"file not found: {path}")
    s = get_settings()
    try:
        name = validate_upload(src.name, src.stat().st_size, s.max_upload_mb)
    except FileRejected as e:
        raise SystemExit(str(e)) from e
    s.upload_path.mkdir(parents=True, exist_ok=True)
    stored = f"{uuid.uuid4().hex}{src.suffix.lower()}"
    shutil.copyfile(src, s.upload_path / stored)
    with session_scope() as db:
        up = Upload(filename=name, stored_name=stored, size_bytes=src.stat().st_size, content_type="")
        db.add(up)
        db.flush()
        return up.id


def edit_plan(run_id: str) -> str | None:
    """Edit the task list as JSON in $EDITOR. The engine re-validates it and creates a new plan version (the old approval is void)."""
    tasks = ex.manager.detail(run_id)["tasks"]
    keep = ("key", "title", "description", "agent", "action", "params", "depends_on", "resources", "expected_output", "verification_criteria", "risk_level")
    fd, name = tempfile.mkstemp(suffix=".json", prefix=f"{run_id}_plan_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump([{k: t[k] for k in keep} for t in tasks], f, indent=2, ensure_ascii=False)
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or ("notepad" if os.name == "nt" else "vi")
    try:
        subprocess.run([*editor.split(), name], check=False)
        specs = [TaskSpec.model_validate(t) for t in json.loads(Path(name).read_text(encoding="utf-8"))]
        ex.manager.update_plan(run_id, specs)
        return None
    except Exception as e:  # noqa: BLE001 - show the engine's validation message, keep the old plan
        return str(e)
    finally:
        Path(name).unlink(missing_ok=True)


async def follow(run_id: str, auto_yes: bool = False, approver=None, on_event=None) -> dict:
    """Print engine events as they happen and answer approval requests, until the run reaches a final state.

    `approver(approval_dict) -> bool` and `on_event(event_dict)` let another interface (voice) take over prompting and narration."""
    last = max((e["id"] for e in events.replay(run_id, 0)), default=0)
    seen_approvals: set[str] = set()
    while True:
        for e in events.replay(run_id, last):
            last = e["id"]
            line = render_event(e)
            if line:
                out(line)
            if on_event:
                await on_event(e)
        run = ex.manager.detail(run_id)
        for a in run["pending_approvals"]:
            if a["id"] in seen_approvals:
                continue
            seen_approvals.add(a["id"])
            render_approval(a)
            if approver:
                yes = await approver(a)
            else:
                ans = "y" if auto_yes else (await asyncio.to_thread(ask, paint("  approve? [y]es / [n]o (stops the run): ", "B"))).lower()
                yes = ans.startswith("y")
            approval_service.resolve(a["id"], "approved" if yes else "denied", "voice" if approver else "terminal")
            out(paint("  approved" if yes else "  denied - the action will not run", "g" if yes else "r"))
        if run["status"] in RUN_TERMINAL:
            for e in events.replay(run_id, last):
                line = render_event(e)
                if line:
                    out(line)
                if on_event:
                    await on_event(e)
            return run
        await asyncio.sleep(0.15)


async def plan_gate(run: dict, auto_yes: bool) -> dict | None:
    """Show the plan and let the human approve (bound to this exact version), edit, regenerate or cancel."""
    while run["status"] == "awaiting_plan_approval":
        render_plan(run)
        if auto_yes:
            choice = "a"
        else:
            choice = (await asyncio.to_thread(ask, paint("\n[a]pprove  [e]dit plan  [r]egenerate  [c]ancel > ", "B"))).lower()[:1]
        if choice == "a":
            try:
                return ex.manager.approve_plan(run["id"], run["plan_hash"])
            except Exception as e:  # noqa: BLE001
                out(paint(f"  not approved: {e}", "r"))
                run = ex.manager.detail(run["id"])
        elif choice == "e":
            err = await asyncio.to_thread(edit_plan, run["id"])
            out(paint(f"  plan not changed: {err}", "r") if err else paint("  plan updated - review the new version (earlier approval is void)", "y"))
            run = ex.manager.detail(run["id"])
        elif choice == "r":
            run = await ex.manager.regenerate_plan(run["id"])
        elif choice == "c":
            ex.manager.cancel(run["id"])
            return None
    return run


async def run_objective(objective: str, files: list[str], auto: bool, simulate: bool, yes: bool) -> int:
    ids = [register_upload(f) for f in files]
    run = await ex.manager.create_run(ObjectiveCreate(objective=objective, mode="autonomous" if auto else "guided", simulate=simulate, upload_ids=ids))
    out(paint(f"run {run['id']}  status {run['status']}" + ("  [SIMULATED]" if run["simulated"] else ""), "B"))
    if run["status"] == "rejected":
        out(paint(f"\nJOSE did not start: {run['error']}", "y"))
        return 2
    if run["status"] == "awaiting_plan_approval":
        run = await plan_gate(run, yes)
        if run is None:
            out("cancelled")
            return 1
    final = await follow(run["id"], yes)
    render_report(final["id"])
    return 0 if final["status"] == "completed" else 1


async def continue_run(run_id: str, yes: bool) -> int:
    run = ex.manager.continue_run(run_id)
    if run["status"] == "awaiting_plan_approval":
        run = await plan_gate(run, yes)
        if run is None:
            return 1
    final = await follow(run_id, yes)
    render_report(run_id)
    return 0 if final["status"] == "completed" else 1


def show_history() -> None:
    rows = ex.list_runs(30)
    if not rows:
        out("no runs yet")
    for r in rows:
        done = r["task_counts"].get("completed", 0)
        out(f"{r['id']}  {paint(f'{r['status']:<22}', STATUS_COLOR.get(r['status'], 'd'))} {done}/{sum(r['task_counts'].values()):<3} {r['objective'][:70]}")


def show_run(run_id: str) -> None:
    run = ex.manager.detail(run_id)
    out(paint(f"{run['id']}  {run['status']}  mode {run['mode']}  plan v{run['plan_version']}", "B"))
    out(run["objective"])
    for t in run["tasks"]:
        out(f"  {t['key']:<4} {paint(f'{t['status']:<10}', STATUS_COLOR.get(t['status'], 'd'))} {t['title'][:55]:<55} attempts {t['attempts']}" + (f"  ! {t['error'][:100]}" if t["error"] else ""))
    for a in ledger.for_run(run_id):
        out(f"  external {a['kind']} -> {a['target']}: {a['status']} {a['reference'] or ''} (id {a['id']})")


def show_tools() -> None:
    from .workflows.custom import TOOLS, tool_status

    for t in TOOLS.values():
        st = tool_status(t)
        flags = ("APPROVAL " if t.needs_approval else "") + ("" if t.idempotent else "NEVER-AUTO-REPEATED")
        out(f"{paint('available' if st['available'] else 'UNAVAILABLE', 'g' if st['available'] else 'r'):<20} {t.name:<22} {t.risk:<6} {flags}")
        if not st["available"]:
            out(paint(f"{'':>20}  {'; '.join(st['missing'])}", "d"))


async def repl() -> int:
    out(paint("JOSE terminal - type an objective, or: attach FILE | files | auto on|off | history | show RUN | graph RUN | continue RUN | tools | help | quit", "B"))
    attached: list[str] = []
    auto = False
    while True:
        line = await asyncio.to_thread(ask, paint(f"\njose{' [auto]' if auto else ''}{f' +{len(attached)} file(s)' if attached else ''}> ", "c"))
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        try:
            if cmd in ("quit", "exit", "q"):
                return 0
            elif cmd == "help":
                out("\n".join(["  <objective text>       plan and run it (attached files are used once)",
                               "  attach FILE | files | detach",
                               "  auto on|off            skip plan approval (risky actions still ask)",
                               "  history | tools",
                               "  show|graph|report|artifacts|continue|cancel RUN",
                               "  resolve ACTION performed|not_performed"]))
            elif cmd == "attach":
                path = rest.strip().strip('"')
                if not Path(path).expanduser().is_file():
                    out(paint(f"file not found: {path}", "r"))
                else:
                    attached.append(path)
                    out(f"attached: {path}")
            elif cmd == "files":
                out("\n".join(attached) or "no files attached")
            elif cmd == "detach":
                attached.clear()
            elif cmd == "auto":
                auto = rest.strip().lower() == "on"
                out(f"autonomous mode {'on' if auto else 'off'}")
            elif cmd == "history":
                show_history()
            elif cmd == "tools":
                show_tools()
            elif cmd == "show":
                show_run(rest.strip())
            elif cmd == "graph":
                render_graph(ex.manager.detail(rest.strip()))
            elif cmd == "report":
                render_report(rest.strip())
            elif cmd == "artifacts":
                show_artifacts(rest.strip())
            elif cmd == "continue":
                await continue_run(rest.strip(), False)
            elif cmd == "cancel":
                ex.manager.cancel(rest.strip())
                out("cancelled")
            elif cmd == "resolve":
                aid, _, outcome = rest.partition(" ")
                out(json.dumps(ledger.resolve_manually(aid, outcome.strip()), indent=2))
            else:
                files, attached = attached, []
                await run_objective(line, files, auto, False, False)
        except (Exception, SystemExit) as e:  # noqa: BLE001 - keep the prompt alive
            out(paint(f"error: {e}", "r"))


def show_artifacts(run_id: str) -> None:
    for a in artifact_service.list_for_run(run_id):
        out(f"{a['id']}  {a['filename']}  {a['size_bytes']} bytes  {artifact_service.run_dir(run_id) / a['filename']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="jose", description="JOSE terminal interface (same engine as the website)")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="plan and execute an objective")
    r.add_argument("objective")
    r.add_argument("-f", "--file", action="append", default=[], help="attach a file (repeatable)")
    r.add_argument("--auto", action="store_true", help="autonomous mode: skip plan approval (risky actions still ask)")
    r.add_argument("-v", "--verbose", action="store_true", help="also print every passed verification check")
    r.add_argument("--simulate", action="store_true", help="demo data; nothing real is fetched or sent")
    r.add_argument("-y", "--yes", action="store_true", help="answer yes to every prompt, including risky-action approvals. Use with care.")
    for name in ("show", "graph", "report", "artifacts", "cancel"):
        sub.add_parser(name).add_argument("run_id")
    c = sub.add_parser("continue")
    c.add_argument("run_id")
    c.add_argument("-y", "--yes", action="store_true")
    res = sub.add_parser("resolve")
    res.add_argument("action_id")
    res.add_argument("outcome", choices=["performed", "not_performed"])
    v = sub.add_parser("voice", help="voice interface: wake phrase, spoken plan, Accept / Relisten / Change / Cancel")
    v.add_argument("--wake", default=None, help="wake phrase (default 'hey jose', or JOSE_WAKE_PHRASE)")
    v.add_argument("--no-wake", action="store_true", help="do not wait for the wake phrase")
    v.add_argument("--simulate", action="store_true", help="demo data; nothing real is fetched or sent")
    v.add_argument("-f", "--file", action="append", default=[], help="attach a file to the first objective (repeatable); you can also say 'attach a file'")
    v.add_argument("--once", action="store_true", help="handle one objective and exit")
    v.add_argument("--text", action="store_true", help="no audio: type what you would say, read what JOSE would say")
    sub.add_parser("repl", help="typed interactive prompt")
    sub.add_parser("history")
    sub.add_parser("tools")
    args = ap.parse_args(argv)

    global VERBOSE
    VERBOSE = bool(getattr(args, "verbose", False))
    init_engine()
    runtime_settings.update({})
    if args.cmd == "repl":
        return asyncio.run(repl())
    if args.cmd is None:
        args = ap.parse_args(["voice"])
    if args.cmd == "voice":
        from .voice import VoiceUnavailable, build_session

        try:
            asyncio.run(build_session(args.text, args.wake, args.simulate, args.file).serve(once=args.once, wake=not args.no_wake))
        except VoiceUnavailable as e:
            if argv is None and len(sys.argv) == 1:  # plain `python -m app.cli`: don't dead-end, fall back to typing
                out(paint(f"voice is not available: {e}\nFalling back to the typed prompt.", "r"))
                return asyncio.run(repl())
            out(paint(f"voice is not available: {e}\nUse --text to type instead.", "r"))
            return 3
        except KeyboardInterrupt:
            pass
        return 0
    if args.cmd == "run":
        return asyncio.run(run_objective(args.objective, args.file, args.auto, args.simulate, args.yes))
    if args.cmd == "continue":
        return asyncio.run(continue_run(args.run_id, args.yes))
    if args.cmd == "history":
        show_history()
    elif args.cmd == "tools":
        show_tools()
    elif args.cmd == "show":
        show_run(args.run_id)
    elif args.cmd == "graph":
        render_graph(ex.manager.detail(args.run_id))
    elif args.cmd == "report":
        render_report(args.run_id)
    elif args.cmd == "artifacts":
        show_artifacts(args.run_id)
    elif args.cmd == "cancel":
        ex.manager.cancel(args.run_id)
        out("cancelled")
    elif args.cmd == "resolve":
        out(json.dumps(ledger.resolve_manually(args.action_id, args.outcome), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
