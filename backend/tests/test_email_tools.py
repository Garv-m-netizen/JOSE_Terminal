"""Email: provider error handling, recipient provenance, exact-email approval, ledger-guarded sending, ambiguity and resume.

All providers here are fakes. Nothing in this file contacts a mail server, and nothing here proves a real email was sent.
"""
from __future__ import annotations

import asyncio
import json
import smtplib

import pytest

from app.agents.custom_planner import CustomPlan
from app.config import get_settings
from app.services import ledger
from app.services.llm import set_providers_override
from app.tools import email_provider as ep
from app.tools.email_provider import EmailAmbiguous, EmailNotSent, OutgoingEmail, SmtpProvider, set_email_provider_override

from .test_custom_workflow import finish, go

OBJ = "Write a short note and email it to anita@example.com with the subject Hello and the note attached"


class FakeEmail:
    simulated = False
    name = "fake"
    sender = "jose@example.org"

    def __init__(self, fail: Exception | None = None):
        self.sent, self.fail = [], fail

    def send_sync(self, mail: OutgoingEmail) -> dict:
        if self.fail:
            raise self.fail
        self.sent.append(mail)
        return {"accepted": [*mail.to, *mail.cc], "refused": {}, "message_id": "<fake-1@example.org>", "server": "fake:25", "simulated": False}


@pytest.fixture(autouse=True)
def _reset_email():
    yield
    set_email_provider_override(None)


def email_plan(to=("anita@example.com",), subject="Hello", extra=None, attachments="@T1.artifact_id"):
    tasks = [
        {"key": "T1", "title": "Write the note", "tool": "fs.write_text", "args_json": json.dumps({"filename": "note.txt", "content": "Dear Anita, here is the note."})},
        {"key": "T2", "title": "Draft the email", "tool": "email.draft",
         "args_json": json.dumps({"to": list(to), "subject": subject, "body": "Hi Anita,\nthe note is attached.", "attachments": attachments})},
        {"key": "T3", "title": "Send the email", "tool": "email.send", "args_json": json.dumps({"draft": "@T2"})},
    ]
    return {"feasible": True, "assumptions": [], "tasks": tasks, **(extra or {})}


def handler(plan):
    def h(name, payload, prompt):
        if name == "plan_workflow":
            return plan
        raise AssertionError(name)
    return h


async def wait_for(client, run_id, pred, timeout=40):
    end = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < end:
        run = (await client.get(f"/api/runs/{run_id}")).json()
        if pred(run):
            return run
        await asyncio.sleep(0.05)
    raise AssertionError(f"timeout; last status {run['status']}")


# ------------------------------------------------------------------ provider behaviour (fake smtplib)

class FakeSMTP:
    mode = "ok"
    instances: list = []

    def __init__(self, host, port, timeout=None, **kw):
        if FakeSMTP.mode == "connect_fail":
            raise ConnectionRefusedError("refused")
        self.sent = None
        FakeSMTP.instances.append(self)

    def ehlo(self): pass
    def starttls(self, context=None): pass

    def login(self, user, pw):
        if FakeSMTP.mode == "auth_fail":
            raise smtplib.SMTPAuthenticationError(535, b"bad credentials hunter2")

    def send_message(self, msg, from_addr=None, to_addrs=None):
        if FakeSMTP.mode == "recipients_refused":
            raise smtplib.SMTPRecipientsRefused({a: (550, b"no such user") for a in to_addrs})
        if FakeSMTP.mode == "disconnect":
            raise smtplib.SMTPServerDisconnected("connection lost")
        if FakeSMTP.mode == "partial":
            self.sent = msg
            return {to_addrs[-1]: (550, b"mailbox full")}
        self.sent = msg
        return {}

    def quit(self): pass


@pytest.fixture
def smtp(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.org")
    monkeypatch.setenv("SMTP_USERNAME", "jose")
    monkeypatch.setenv("SMTP_PASSWORD", "hunter2")
    monkeypatch.setenv("EMAIL_FROM", "jose@example.org")
    get_settings.cache_clear()
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    FakeSMTP.mode, FakeSMTP.instances = "ok", []
    return FakeSMTP


def mail(**kw):
    return OutgoingEmail(sender="jose@example.org", to=kw.get("to", ["a@example.com"]), cc=kw.get("cc", []), subject="s", body="b")


def test_smtp_success_reports_acceptance_and_message_id(smtp):
    r = SmtpProvider().send_sync(mail(to=["a@example.com", "b@example.com"]))
    assert r["accepted"] == ["a@example.com", "b@example.com"] and r["refused"] == {} and r["message_id"].startswith("<") and not r["simulated"]
    assert smtp.instances[0].sent["Message-ID"] == r["message_id"]


@pytest.mark.parametrize("mode,exc", [("auth_fail", EmailNotSent), ("connect_fail", EmailNotSent), ("recipients_refused", EmailNotSent), ("disconnect", EmailAmbiguous)])
def test_smtp_errors_are_classified_by_whether_the_message_may_have_left(smtp, mode, exc):
    smtp.mode = mode
    with pytest.raises(exc) as e:
        SmtpProvider().send_sync(mail())
    assert "hunter2" not in str(e.value)  # the password never appears in an error


def test_partial_refusal_is_reported_not_hidden(smtp):
    smtp.mode = "partial"
    r = SmtpProvider().send_sync(mail(to=["a@example.com", "b@example.com"]))
    assert r["accepted"] == ["a@example.com"] and list(r["refused"]) == ["b@example.com"]


def test_unconfigured_provider_refuses_with_instructions(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "")
    get_settings.cache_clear()
    with pytest.raises(ep.EmailError, match="SMTP_HOST"):
        SmtpProvider()


def test_address_validation_blocks_header_injection():
    assert ep.normalize_address("Anita <Anita@Example.com>") == "anita@example.com"
    for bad in ["no-at-sign", "a@b", "a@b.c\nBcc: x@y.z", "a b@example.com", ""]:
        with pytest.raises(ValueError):
            ep.normalize_address(bad)


# ------------------------------------------------------------------ planning-time honesty

def test_unconfigured_email_is_marked_unavailable_to_the_planner(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "")
    get_settings.cache_clear()
    from app.workflows.custom import catalog_text

    assert "email.send [UNAVAILABLE" in catalog_text()
    set_email_provider_override(FakeEmail())
    assert "email.send [UNAVAILABLE" not in catalog_text()


def test_plan_using_unavailable_email_is_rejected_unless_simulated(monkeypatch):
    from app.agents.custom_planner import to_specs
    from app.workflows.custom import validate_custom_plan
    from app.workflows.research_report import PlanError

    monkeypatch.setenv("SMTP_HOST", "")
    get_settings.cache_clear()
    specs = to_specs(CustomPlan.model_validate(email_plan()))
    with pytest.raises(PlanError, match="not available"):
        validate_custom_plan(specs, set())
    validate_custom_plan(specs, set(), simulated=True)  # simulated runs never send anything


def test_email_send_must_reference_a_draft():
    from app.agents.custom_planner import to_specs
    from app.workflows.custom import validate_custom_plan
    from app.workflows.research_report import PlanError

    set_email_provider_override(FakeEmail())
    plan = email_plan()
    plan["tasks"][2]["args_json"] = json.dumps({"draft": "@T1"})
    with pytest.raises(PlanError, match="email.draft"):
        validate_custom_plan(to_specs(CustomPlan.model_validate(plan)), set())


# ------------------------------------------------------------------ end to end (fake provider)

async def test_email_flow_shows_exact_email_and_records_the_real_provider_response(client, monkeypatch):
    fake = FakeEmail()
    set_email_provider_override(fake)
    r, _ = await go(client, monkeypatch, handler(email_plan()), [], objective=OBJ)
    assert r["status"] != "rejected", r["error"]
    assert [t["action"] for t in r["tasks"]] == ["tool.fs.write_text", "tool.email.draft", "tool.email.send"]
    assert r["tasks"][2]["risk_level"] == "HIGH"
    run = await wait_for(client, r["id"], lambda x: x["pending_approvals"] or x["status"] in ("failed", "halted"))
    assert run["status"] == "awaiting_approval", run["error"]
    card = run["pending_approvals"][0]
    assert card["risk_level"] == "HIGH" and fake.sent == []  # nothing is sent before approval
    d = card["details"]
    assert d["to"] == "anita@example.com" and d["subject"] == "Hello" and "the note is attached" in d["body"] and d["attachments"][0].startswith("note.txt")
    assert d["recipient_basis"]["anita@example.com"] == "written in the objective" and d["plan_hash"] == run["plan_hash"]
    await client.post(f"/api/approvals/{card['id']}/resolve", json={"decision": "approved"})
    final = await finish(client, r["id"])
    assert final["status"] == "completed", final["error"]
    assert len(fake.sent) == 1 and fake.sent[0].to == ["anita@example.com"] and fake.sent[0].attachments[0][0] == "note.txt"
    send_out = next(t for t in final["tasks"] if t["key"] == "T3")["output_summary"]
    assert send_out["status"] == "accepted_by_mail_server" and send_out["message_id"] == "<fake-1@example.org>"
    # the ledger and the report say what actually happened, and no more
    rep = (await client.get(f"/api/runs/{r['id']}/report")).json()
    assert rep["workflow_completed"] and rep["levels"]["external_actions_accepted"] == 1 and rep["levels"]["outputs_verified"] == 3
    assert "<fake-1@example.org>" in rep["summary"] and any("not proof the message reached an inbox" in c for c in rep["caveats"])
    acts = (await client.get(f"/api/runs/{r['id']}/external-actions")).json()
    assert len(acts) == 1 and acts[0]["status"] == "accepted" and acts[0]["reference"] == "<fake-1@example.org>"


async def test_denied_email_is_never_sent_and_report_says_so(client, monkeypatch):
    fake = FakeEmail()
    set_email_provider_override(fake)
    r, _ = await go(client, monkeypatch, handler(email_plan()), [], objective=OBJ)
    final = await finish(client, r["id"], decision="denied")
    assert final["status"] == "halted" and fake.sent == []
    rep = (await client.get(f"/api/runs/{r['id']}/report")).json()
    assert not rep["workflow_completed"] and rep["levels"]["external_actions_accepted"] == 0 and rep["headline"].startswith("Not completed")


async def test_recipient_not_given_by_the_user_is_never_used(client, monkeypatch):
    fake = FakeEmail()
    set_email_provider_override(fake)
    r, _ = await go(client, monkeypatch, handler(email_plan(to=("boss@corp.com",))), [], objective=OBJ)  # the model invented an address
    final = await finish(client, r["id"])
    assert final["status"] in ("halted", "failed", "partial") and fake.sent == []
    t2 = next(t for t in final["tasks"] if t["key"] == "T2")
    assert t2["status"] == "failed" and "not given by you" in t2["error"]
    assert next(t for t in final["tasks"] if t["key"] == "T3")["status"] != "completed"


async def test_configured_recipient_resolves_only_when_set(client, monkeypatch):
    fake = FakeEmail()
    set_email_provider_override(fake)
    obj = "Email the note to the configured recipient with subject Hello"
    r, _ = await go(client, monkeypatch, handler(email_plan(to=("configured",))), [], objective=obj)
    final = await finish(client, r["id"])
    assert final["status"] == "halted" and fake.sent == []
    assert "EMAIL_DEFAULT_RECIPIENT" in next(t for t in final["tasks"] if t["key"] == "T2")["error"]
    monkeypatch.setenv("EMAIL_DEFAULT_RECIPIENT", "Team@Example.com")
    get_settings.cache_clear()
    r2, _ = await go(client, monkeypatch, handler(email_plan(to=("configured",))), [], objective=obj)
    final2 = await finish(client, r2["id"])
    assert final2["status"] == "completed" and fake.sent[0].to == ["team@example.com"]


async def test_missing_recipient_becomes_a_question_not_a_guess(client, monkeypatch):
    set_email_provider_override(FakeEmail())
    plan = {"feasible": False, "reason": "", "questions": ["Which address should the email go to?"], "assumptions": [], "tasks": []}
    r, _ = await go(client, monkeypatch, handler(plan), [], objective="Email the report to my manager")
    assert r["status"] == "rejected" and "Which address should the email go to?" in r["error"]
    ev = (await client.get(f"/api/runs/{r['id']}/events")).json()
    assert any(e["type"] == "run.needs_input" and e["details"]["questions"] for e in ev)


async def test_provider_login_failure_is_reported_and_not_retried(client, monkeypatch):
    fake = FakeEmail(fail=EmailNotSent("the mail server rejected the login (check SMTP_USERNAME / SMTP_PASSWORD): 535"))
    set_email_provider_override(fake)
    r, _ = await go(client, monkeypatch, handler(email_plan()), [], objective=OBJ)
    final = await finish(client, r["id"])
    assert final["status"] in ("halted", "partial", "failed") and fake.sent == []
    t3 = next(t for t in final["tasks"] if t["key"] == "T3")
    assert t3["status"] == "failed" and "login" in t3["error"] and t3["attempts"] == 1
    acts = (await client.get(f"/api/runs/{r['id']}/external-actions")).json()
    assert acts[0]["status"] == "failed"  # definitely not sent
    rep = (await client.get(f"/api/runs/{r['id']}/report")).json()
    assert not rep["workflow_completed"] and rep["levels"]["external_actions_accepted"] == 0


async def test_ambiguous_send_is_never_repeated_until_a_person_settles_it(client, monkeypatch):
    fake = FakeEmail(fail=EmailAmbiguous("the connection failed while sending (ServerDisconnected)"))
    set_email_provider_override(fake)
    r, _ = await go(client, monkeypatch, handler(email_plan()), [], objective=OBJ)
    final = await finish(client, r["id"])
    assert final["status"] in ("halted", "partial", "failed")
    acts = (await client.get(f"/api/runs/{r['id']}/external-actions")).json()
    assert acts[0]["status"] == "ambiguous"
    # continuing does NOT resend: the outcome is unknown, so a person has to decide
    fake.fail = None
    c = await client.post(f"/api/runs/{r['id']}/continue")
    assert c.status_code == 200, c.text
    again = await finish(client, r["id"])
    assert fake.sent == [], "an ambiguous email must not be resent automatically"
    ev = (await client.get(f"/api/runs/{r['id']}/events")).json()
    cont = next(e for e in ev if e["type"] == "run.continued")
    assert cont["details"]["audit"]["needs_human"][0]["task"] == "T3"
    # the person checked their Sent folder: it did go out -> JOSE records that and never sends it again
    res = await client.post(f"/api/external-actions/{acts[0]['id']}/resolve", json={"outcome": "performed", "note": "seen in Sent folder"})
    assert res.status_code == 200 and res.json()["status"] == "accepted"
    c2 = await client.post(f"/api/runs/{r['id']}/continue")
    assert c2.status_code == 200, c2.text
    done = await finish(client, r["id"])
    assert fake.sent == [] and done["status"] == "completed", done["error"]
    t3 = next(t for t in done["tasks"] if t["key"] == "T3")
    assert t3["status"] == "completed"
    rep = (await client.get(f"/api/runs/{r['id']}/report")).json()
    assert rep["levels"]["external_actions_accepted"] == 1


async def test_ledger_blocks_a_second_send_of_the_same_draft(client, monkeypatch):
    fake = FakeEmail()
    set_email_provider_override(fake)
    r, _ = await go(client, monkeypatch, handler(email_plan()), [], objective=OBJ)
    done = await finish(client, r["id"])
    assert done["status"] == "completed" and len(fake.sent) == 1
    # force the send step to run again (as a retry / resume would): the ledger must answer with the earlier confirmed send
    from app.agents import custom_agent
    from app.agents.orchestrator import Orchestrator, RunHandle
    from app.services import store

    tasks = store.get_tasks(r["id"])
    orch = Orchestrator(r["id"], RunHandle(r["id"]))
    outputs = orch.outputs()
    t3 = next(t for t in tasks if t.key == "T3")
    out = await custom_agent.run_tool(orch.make_ctx(t3, tasks, outputs))
    assert len(fake.sent) == 1, "the email was sent twice"
    assert out["duplicate_prevented"] is True and out["message_id"] == "<fake-1@example.org>"


def test_ledger_claim_semantics():
    from app.database import session_scope
    from app.models import Run

    with session_scope() as s:
        run = Run(objective="x")
        s.add(run)
        s.flush()
        rid = run.id
    key = ledger.make_key("email.send", rid, "h1")
    c1 = ledger.claim(rid, None, "email.send", key, "a@example.com", "s")
    assert c1.proceed
    assert not ledger.claim(rid, None, "email.send", key, "a@example.com", "s").proceed  # in flight
    ledger.failed(c1.action["id"], "login rejected")
    c3 = ledger.claim(rid, None, "email.send", key, "a@example.com", "s")
    assert c3.proceed and c3.action["attempts"] == 2  # a definite failure may be attempted again
    ledger.ambiguous(c3.action["id"], "timeout")
    assert not ledger.claim(rid, None, "email.send", key, "a@example.com", "s").proceed
    ledger.resolve_manually(c3.action["id"], "performed")
    blocked = ledger.claim(rid, None, "email.send", key, "a@example.com", "s")
    assert not blocked.proceed and blocked.action["status"] == "accepted"
    # a server restart turns anything still in flight into "ambiguous", never into "retry"
    k2 = ledger.make_key("email.send", rid, "h2")
    c4 = ledger.claim(rid, None, "email.send", k2, "b@example.com", "s")
    assert ledger.interrupt_in_flight() == 1
    assert [a for a in ledger.for_run(rid) if a["id"] == c4.action["id"]][0]["status"] == "ambiguous"


async def test_secrets_never_reach_events(client, monkeypatch, smtp):
    r, _ = await go(client, monkeypatch, handler(email_plan()), [], objective=OBJ)
    await finish(client, r["id"])
    blob = json.dumps((await client.get(f"/api/runs/{r['id']}/trace")).json()) + json.dumps((await client.get("/api/status")).json())
    assert "hunter2" not in blob
