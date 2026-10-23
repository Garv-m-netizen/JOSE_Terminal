"""dev.run: allow-listed developer commands, no shell, approved folders only, scrubbed environment, hard timeout."""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import time

from ..config import get_settings
from ..tools import fs_sandbox as sbx
from .base import AgentContext, TaskError
from .verification_agent import Check

# arguments that can load code, redirect output to files or point git/pytest at other repositories
FORBIDDEN_ARGS = {"-c", "-p", "--exec", "--ext-diff", "--config", "--git-dir", "--work-tree", "--upload-pack", "--output", "-o", "--rootdir", "--confcutdir",
                  "--pyargs", "--import-mode", "--textconv", "--no-index", "-C", "--prefix", "--exec-path"}
SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)", re.I)


def allowed_prefixes() -> list[list[str]]:
    return [e.split() for e in get_settings().dev_allowed_commands.split(";") if e.strip()]


def check_command(cmd: list[str]) -> list[str]:
    """Return the allow-list entry matched, or raise TaskError. Pure validation, no execution."""
    if any(not a or "\x00" in a or "\n" in a for a in cmd):
        raise TaskError("command arguments must be non-empty single-line strings", "validation", retryable=False)
    match = next((p for p in allowed_prefixes() if cmd[: len(p)] == p), None)
    if match is None:
        raise TaskError(f"'{' '.join(cmd[:3])}' is not an allow-listed command. Allowed: {'; '.join(' '.join(p) for p in allowed_prefixes())}", "validation", retryable=False)
    prefixes = ("--output=", "--exec=", "--config=", "--git-dir=", "--rootdir=", "--work-tree=", "--confcutdir=", "--import-mode=")
    bad = [a for a in cmd[len(match):] if a in FORBIDDEN_ARGS or a.startswith(prefixes) or (re.match(r"^-[pcCo].+", a) and not a.startswith("--"))]
    if bad:
        raise TaskError(f"argument(s) {bad} are not permitted (they can load code or write outside the project)", "validation", retryable=False)
    return match


def clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)}


def run_sync(cmd: list[str], cwd, timeout: int, limit: int) -> dict:
    exe = shutil.which(cmd[0])
    if exe is None:
        raise TaskError(f"'{cmd[0]}' is not installed or not on PATH", "config", retryable=False)
    t0 = time.perf_counter()
    try:
        p = subprocess.run([exe, *cmd[1:]], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=clean_env(), stdin=subprocess.DEVNULL)
        code, so, se, timed_out = p.returncode, p.stdout, p.stderr, False
    except subprocess.TimeoutExpired as e:
        code, timed_out = -1, True
        so, se = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), (e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
    return {"command": cmd, "exit_code": code, "stdout": so[-limit:], "stderr": se[-limit:], "timed_out": timed_out, "duration_ms": int((time.perf_counter() - t0) * 1000),
            "stdout_truncated": len(so) > limit, "stderr_truncated": len(se) > limit}


async def t_dev_run(ctx: AgentContext, a) -> dict:
    match = check_command(a.command)
    try:
        cwd = sbx.resolve(a.cwd)
    except sbx.SandboxError as e:
        raise TaskError(str(e), "validation", retryable=False) from e
    if not cwd.is_dir():
        raise TaskError(f"'{a.cwd}' is not a directory", "validation", retryable=False)
    limit = get_settings().dev_output_limit
    async with ctx.activity(f"running {' '.join(a.command[:4])}"):
        r = await ctx.tool("dev.run", {"command": a.command, "cwd": sbx.rel(cwd)}, lambda: asyncio.to_thread(run_sync, a.command, cwd, a.timeout_s, limit),
                           lambda x: f"exit {x['exit_code']} in {x['duration_ms']} ms", simulated=False)
    r.update({"cwd": sbx.rel(cwd), "allow_listed_as": " ".join(match)})
    if r["timed_out"]:
        raise TaskError(f"the command did not finish within {a.timeout_s}s and was stopped", "timeout", retryable=False)
    if r["exit_code"] != 0 and not a.allow_nonzero:
        tail = (r["stderr"] or r["stdout"]).strip()[-700:]
        raise TaskError(f"the command exited with code {r['exit_code']}. Output (tail):\n{tail}", "command_failed", retryable=False)
    return r


def summary_counts(text: str) -> dict:
    """pytest's own final line, e.g. '3 passed, 1 failed in 0.4s'."""
    out = {}
    for n, word in re.findall(r"(\d+) (passed|failed|error|errors|skipped)", text):
        out[word.rstrip("s") if word == "errors" else word] = int(n)
    return out


def v_dev_run(a, out, ctx) -> list[Check]:
    ok_cmd = True
    try:
        check_command(out["command"])
    except TaskError:
        ok_cmd = False
    ck = [Check("dev.allow_listed", ok_cmd and out["command"] == a.command, "the command that ran is exactly the planned, allow-listed command"),
          Check("dev.finished", not out["timed_out"], f"finished in {out['duration_ms']} ms"),
          Check("dev.exit_recorded", isinstance(out["exit_code"], int), f"exit code {out['exit_code']} recorded"),
          Check("dev.exit_as_expected", a.allow_nonzero or out["exit_code"] == 0, "exit code 0" if out["exit_code"] == 0 else f"non-zero exit {out['exit_code']} (allowed by the plan)")]
    if "pytest" in " ".join(a.command[:3]):
        c = summary_counts(out["stdout"] + "\n" + out["stderr"])
        bad = c.get("failed", 0) + c.get("error", 0)
        ck.append(Check("dev.output_consistent_with_exit", (bad == 0) == (out["exit_code"] == 0) or not c, f"pytest summary {c or 'not found'} vs exit {out['exit_code']}"))
    return ck


IMPLS = {"dev.run": t_dev_run}
VERIFIERS = {"dev.run": v_dev_run}
