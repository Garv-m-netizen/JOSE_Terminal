"""dev.run: allow-list, argument filter, sandboxed cwd, scrubbed environment, honest exit codes."""
from __future__ import annotations

import json
import os
import sys

import pytest

from app.agents import tools_dev
from app.agents.base import TaskError
from app.config import get_settings

from .test_custom_workflow import finish, go, planner_only


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    proj = root / "proj"
    proj.mkdir(parents=True)
    (proj / "test_ok.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8")
    (proj / "test_bad.py").write_text("def test_bad():\n    assert 1 + 1 == 3\n", encoding="utf-8")
    monkeypatch.setenv("WORKSPACE_DIRS", str(root))
    get_settings.cache_clear()
    return proj


@pytest.mark.parametrize("cmd", [["rm", "-rf", "/"], ["python", "-c", "print(1)"], ["python", "evil.py"], ["git", "push"], ["npm", "install"], ["curl", "http://x"], ["bash", "-c", "ls"]])
def test_commands_outside_the_allow_list_are_refused(cmd):
    with pytest.raises(TaskError, match="not an allow-listed"):
        tools_dev.check_command(cmd)


@pytest.mark.parametrize("extra", [["-p", "evilplugin"], ["-pevil"], ["--rootdir=/"], ["-c", "x.ini"], ["--output=x"]])
def test_dangerous_arguments_are_refused(extra):
    with pytest.raises(TaskError, match="not permitted"):
        tools_dev.check_command(["python", "-m", "pytest", *extra])


def test_allowed_commands_pass_and_environment_is_scrubbed(monkeypatch):
    assert tools_dev.check_command(["python", "-m", "pytest", "-q", "-x"]) == ["python", "-m", "pytest"]
    monkeypatch.setenv("MY_API_KEY", "s3cret")
    monkeypatch.setenv("SMTP_PASSWORD", "hunter2")
    env = tools_dev.clean_env()
    assert "MY_API_KEY" not in env and "SMTP_PASSWORD" not in env and "PATH" in env or "Path" in env


def plan(cmd, allow_nonzero=False, cwd="proj"):
    return {"feasible": True, "assumptions": [], "tasks": [{"key": "T1", "title": "Run the tests", "tool": "dev.run",
            "args_json": json.dumps({"command": cmd, "cwd": cwd, "allow_nonzero": allow_nonzero})}]}


async def test_passing_tests_run_after_approval_and_are_verified(client, monkeypatch, project):
    r, _ = await go(client, monkeypatch, planner_only(plan(["python", "-m", "pytest", "-q", "test_ok.py"])), [], objective="run the tests in proj")
    assert r["tasks"][0]["risk_level"] == "MEDIUM"
    final = await finish(client, r["id"])
    assert final["status"] == "completed", final["error"]
    out = (await client.get(f"/api/runs/{r['id']}/tasks/T1")).json()["output"]
    assert out["exit_code"] == 0 and "1 passed" in out["stdout"] and out["allow_listed_as"] == "python -m pytest"
    checks = {c["check"] for c in (await client.get(f"/api/runs/{r['id']}/verification")).json()["current"]}
    assert {"dev.allow_listed", "dev.exit_as_expected", "dev.output_consistent_with_exit"} <= checks


async def test_failing_tests_fail_the_task_with_the_real_output_and_are_not_retried(client, monkeypatch, project):
    r, _ = await go(client, monkeypatch, planner_only(plan(["python", "-m", "pytest", "-q", "test_bad.py"])), [], objective="run the tests in proj")
    final = await finish(client, r["id"])
    t = final["tasks"][0]
    assert final["status"] in ("halted", "failed") and t["status"] == "failed" and t["attempts"] == 1
    assert "exited with code 1" in t["error"] and "assert 1 + 1 == 3" in t["error"]
    rep = (await client.get(f"/api/runs/{r['id']}/report")).json()
    assert not rep["workflow_completed"]


async def test_failing_exit_code_can_be_an_expected_result(client, monkeypatch, project):
    r, _ = await go(client, monkeypatch, planner_only(plan(["python", "-m", "pytest", "-q", "test_bad.py"], allow_nonzero=True)), [], objective="find out which tests fail in proj")
    final = await finish(client, r["id"])
    out = (await client.get(f"/api/runs/{r['id']}/tasks/T1")).json()["output"]
    assert final["status"] == "completed" and out["exit_code"] == 1 and "1 failed" in out["stdout"]


async def test_cwd_outside_the_approved_folders_is_refused(client, monkeypatch, project, tmp_path):
    r, _ = await go(client, monkeypatch, planner_only(plan(["git", "status"], cwd=str(tmp_path))), [], objective="show the git status of my project")
    final = await finish(client, r["id"])
    assert final["tasks"][0]["status"] == "failed" and "outside the approved" in final["tasks"][0]["error"]


async def test_denied_command_never_runs(client, monkeypatch, project):
    marker = project / "ran.txt"
    (project / "test_marker.py").write_text(f"def test_m():\n    open(r'{marker}', 'w').write('x')\n", encoding="utf-8")
    r, _ = await go(client, monkeypatch, planner_only(plan(["python", "-m", "pytest", "-q", "test_marker.py"])), [], objective="run the marker test")
    final = await finish(client, r["id"], decision="denied")
    assert final["status"] == "halted" and not marker.exists()
