"""Terminal interface: it drives the same engine, so these tests check the prompts and that nothing bypasses approvals."""
from __future__ import annotations

import json

from app import cli
from app.config import get_settings
from app.schemas.domain import ObjectiveCreate
from app.services import execution_service as ex
from app.services.llm import Providers, set_providers_override
from app.tools.email_provider import set_email_provider_override

from .helpers import FakeLLM
from .test_email_tools import FakeEmail, email_plan, handler

RESEARCH = ("Research three competitors in the AI automation market, compare their features, use publicly available pricing information, "
            "create an Excel comparison and a PowerPoint summary with source references.")


def test_simulated_run_from_the_terminal(capsys, monkeypatch):
    answers = iter(["a"])
    monkeypatch.setattr(cli, "ask", lambda prompt: next(answers))
    code = cli.main(["run", RESEARCH, "--simulate"])
    text = capsys.readouterr().out
    assert code == 0 and "Plan v1" in text and "Completed: all 11 steps" in text and "SIMULATED" in text
    assert "competitors.xlsx" in text and "pass xlsx" not in text  # quiet by default


def test_history_show_and_tools(capsys, monkeypatch):
    monkeypatch.setattr(cli, "ask", lambda prompt: "a")
    cli.main(["run", RESEARCH, "--simulate"])
    capsys.readouterr()
    cli.main(["history"])
    hist = capsys.readouterr().out
    run_id = hist.split()[0]
    assert run_id.startswith("run_") and "completed" in hist
    cli.main(["show", run_id])
    assert "T11" in capsys.readouterr().out
    cli.main(["tools"])
    tools = capsys.readouterr().out
    assert "email.send" in tools and "UNAVAILABLE" in tools and "NEVER-AUTO-REPEATED" in tools


def test_cancel_at_plan_prompt_runs_nothing(capsys, monkeypatch):
    monkeypatch.setattr(cli, "ask", lambda prompt: "c")
    assert cli.main(["run", RESEARCH, "--simulate"]) == 1
    out = capsys.readouterr().out
    assert "cancelled" in out and "tool " not in out


def test_unclear_objective_is_refused_with_the_reason(capsys):
    code = cli.main(["run", "make it better", "--simulate"])
    assert code == 2 and "JOSE did not start" in capsys.readouterr().out


def test_email_prompt_shows_exact_email_and_denial_sends_nothing(capsys, monkeypatch):
    monkeypatch.setenv("JOSE_MODE", "real")
    monkeypatch.setenv("GEMINI_API_KEY", "x" * 12)
    get_settings.cache_clear()
    fake = FakeEmail()
    set_email_provider_override(fake)
    prov = Providers(llm=FakeLLM(lambda n, p, pr: handler(email_plan())(n, p, pr)), search=None, fetcher=None, browser=None, simulated=False)
    set_providers_override(lambda s, i: prov)
    try:
        answers = iter(["a", "n"])  # approve the plan, then DENY the email
        monkeypatch.setattr(cli, "ask", lambda prompt: next(answers))
        code = cli.main(["run", "Write a short note and email it to anita@example.com with the subject Hello and the note attached"])
        out = capsys.readouterr().out
        assert code == 1 and fake.sent == []
        assert "APPROVAL REQUIRED" in out and "to     : anita@example.com" in out and "subject: Hello" in out and "the note is attached" in out
        assert "denied - the action will not run" in out and "Not completed (halted)" in out
        # continue from the terminal: the approval is asked again; this time approve
        run_id = ex.list_runs(1)[0]["id"]
        answers2 = iter(["y"])
        monkeypatch.setattr(cli, "ask", lambda prompt: next(answers2))
        assert cli.main(["continue", run_id]) == 0
        out2 = capsys.readouterr().out
        assert len(fake.sent) == 1 and "external email.send -> accepted" in out2 and "accepted by external services 1" in out2
    finally:
        set_email_provider_override(None)
