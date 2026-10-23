from __future__ import annotations

import asyncio

import pymupdf
import pytest

from app.agents.intent_engine import parse_rules
from app.schemas.domain import SolvedAnswer, SolvedSet
from app.services.llm import Providers, set_providers_override
from app.tools import demo_math
from app.tools.telegram import SimulatedTelegram, TelegramClient, TelegramError

from .helpers import FakeLLM

TEACHER = "make a test of subject maths and class 10 and post it into the telegram group"


def test_teacher_objective_is_understood():
    i = parse_rules(TEACHER)
    assert i.feasible and i.workflow_type == "test_paper" and (i.subject, i.grade, i.post_to_telegram) == ("Mathematics", 10, True)
    assert i.risk_level == "HIGH" and any("answer key" in a for a in i.assumptions) and any("defaulting to 20" in a for a in i.assumptions)
    j = parse_rules("Create a 25 questions Science quiz for grade 8 on topics light and electricity, 40 minutes, then post it in Telegram")
    assert (j.num_questions, j.duration_minutes, j.topics) == (25, 40, ["Light", "Electricity"])


@pytest.mark.parametrize("text,frag", [
    ("make a maths test", "Which class"),
    ("make a test for class 10 and post it to telegram", "Which subject"),
    ("make a maths test for class 10 and email it to parents", "only post to Telegram"),
    ("make a maths test for class 10 and post it to the group", "Where should"),
    ("make a 3 questions maths test for class 10", "outside the supported range"),
    ("make a maths test for class 14", "outside"),
])
def test_teacher_objective_rejections(text, frag):
    i = parse_rules(text)
    assert not i.feasible and frag in i.rejection_reason


def test_demo_generator_and_independent_solver_agree():
    for n in (5, 20, 50):
        paper = demo_math.generate("Mathematics", 10, n)
        assert len({q.text for q in paper.questions}) == n
        for q in paper.questions:
            got = demo_math.solve(q.text, q.options)
            assert (got == q.answer) if q.type == "mcq" else demo_math.answers_match(got, q.final_answer) is True


async def _approve_all(client, run_id, decision="approved", timeout=30):
    end = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < end:
        run = (await client.get(f"/api/runs/{run_id}")).json()
        if run["status"] in ("completed", "failed", "partial", "halted", "cancelled"):
            return run
        for a in run["pending_approvals"]:
            await client.post(f"/api/approvals/{a['id']}/resolve", json={"decision": decision})
        await asyncio.sleep(0.05)
    raise AssertionError("run did not finish")


def pdf_text(content: bytes) -> str:
    return "\n".join(p.get_text() for p in pymupdf.open(stream=content, filetype="pdf"))


async def test_demo_teacher_flow_with_approval(client):
    r = (await client.post("/api/objectives", json={"objective": TEACHER, "mode": "autonomous"})).json()
    assert r["status"] != "rejected" and r["workflow_type"] == "test_paper" and r["simulated"] is True
    assert [t["action"] for t in r["tasks"]] == ["test.generate", "test.solve_check", "test.build_documents", "test.verify_documents", "telegram.send"]
    assert r["tasks"][-1]["risk_level"] == "HIGH"
    final = await _approve_all(client, r["id"])
    assert final["status"] == "completed", final["error"]
    arts = {a["kind"]: a for a in final["artifacts"]}
    paper = pdf_text((await client.get(arts["pdf"]["download_url"])).content)
    key = pdf_text((await client.get(arts["answer_key"]["download_url"])).content)
    assert "Maximum Marks: 38" in paper and "Answer:" not in paper and "Section A" in paper and "SIMULATED" in paper
    assert "TEACHER COPY" in key and key.count("Answer:") == 20
    trace = (await client.get(f"/api/runs/{r['id']}/trace")).json()
    types = [e["type"] for e in trace["events"]]
    assert types.index("approval.required") < types.index("approval.resolved") < types.index("run.completed")
    send = next(t for t in trace["tool_calls"] if t["tool"] == "telegram.sendDocument")
    assert send["simulated"] is True and send["args"]["dry_run"] is True


class FakeTelegram:
    simulated = False
    chat_id = "-1001234567890"

    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    async def send_document(self, path, caption):
        self.sent.append((path.name, caption))
        if self.fail:
            raise TelegramError("Telegram rejected the request: Forbidden: bot is not a member of the group chat")
        return {"message_id": 77, "chat": {"id": int(self.chat_id)}, "document": {"file_name": path.name}}


def exam_handler(corrupt_first=False, always_corrupt=False):
    state = {"gen": 0, "prompts": []}

    def handle(name, payload, prompt):
        if name == "generate_exam":
            state["gen"] += 1
            state["prompts"].append(prompt)
            d = demo_math.generate(payload["subject"], payload["grade"], payload["n"]).model_dump()
            d["title"] = "Mathematics - Class 10 Test"  # a real model would not carry the demo label
            if always_corrupt or (corrupt_first and state["gen"] == 1):
                q = next(x for x in d["questions"] if x["type"] == "mcq")
                q["answer"] = "ABCD"[("ABCD".index(q["answer"]) + 1) % 4]  # wrong answer key
            return d
        if name == "solve_exam":
            return SolvedSet(answers=[SolvedAnswer(id=q["id"], answer=demo_math.solve(q["text"], q["options"])) for q in payload["questions"]]).model_dump()
        raise AssertionError(name)
    return handle, state


async def _real(client, monkeypatch, handler, tg):
    monkeypatch.setenv("JOSE_MODE", "real")
    monkeypatch.setenv("GEMINI_API_KEY", "x" * 12)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-1001234567890")
    from app.config import get_settings

    get_settings.cache_clear()
    prov = Providers(llm=FakeLLM(handler), search=None, fetcher=None, browser=None, simulated=False, messenger=tg)
    set_providers_override(lambda s, i: prov)
    return (await client.post("/api/objectives", json={"objective": TEACHER, "mode": "autonomous"})).json()


async def test_real_send_requires_approval_and_posts_only_student_paper(client, monkeypatch):
    handler, _ = exam_handler()
    tg = FakeTelegram()
    r = await _real(client, monkeypatch, handler, tg)
    assert r["simulated"] is False
    run = {}
    for _ in range(200):  # wait for the approval; nothing may be sent before it
        run = (await client.get(f"/api/runs/{r['id']}")).json()
        if run["pending_approvals"]:
            break
        await asyncio.sleep(0.05)
    ap = run["pending_approvals"][0]
    assert run["status"] == "awaiting_approval" and tg.sent == []
    assert ap["risk_level"] == "HIGH" and "-10" in ap["target"] and "cannot delete" in ap["consequences"]
    assert ap["details"]["excluded"].endswith("TEACHER_ONLY.pdf") and "Maximum" not in ap["details"]["caption"]
    await client.post(f"/api/approvals/{ap['id']}/resolve", json={"decision": "approved"})
    final = await _approve_all(client, r["id"])
    assert final["status"] == "completed"
    assert len(tg.sent) == 1 and "Question_Paper" in tg.sent[0][0] and "Answer_Key" not in tg.sent[0][0]
    v = (await client.get(f"/api/runs/{r['id']}/verification")).json()
    assert {"tg.delivered", "tg.file_matches", "tg.chat_matches", "tg.answer_key_not_sent"} <= {c["check"] for c in v["current"] if c["passed"]}


async def test_denied_post_sends_nothing(client, monkeypatch):
    handler, _ = exam_handler()
    tg = FakeTelegram()
    r = await _real(client, monkeypatch, handler, tg)
    final = await _approve_all(client, r["id"], "denied")
    assert final["status"] == "halted" and final["halt_reason"] == "approval_denied" and tg.sent == []
    assert {a["kind"] for a in final["artifacts"]} == {"pdf", "answer_key"}  # the documents still exist for the teacher


async def test_wrong_answer_key_is_caught_and_redrafted_with_feedback(client, monkeypatch):
    handler, state = exam_handler(corrupt_first=True)
    tg = FakeTelegram()
    r = await _real(client, monkeypatch, handler, tg)
    final = await _approve_all(client, r["id"])
    assert final["status"] == "completed" and final["recovery_count"] >= 1 and state["gen"] == 2
    assert "Rejected questions" in state["prompts"][1] and "ONLY replacement questions" in state["prompts"][1]  # targeted repair, not a redraft
    v = (await client.get(f"/api/runs/{r['id']}/verification")).json()
    assert any(not c["passed"] and c["check"] == "solve.answers_agree" for c in v["checks"])  # the failure stays in the history
    assert len(tg.sent) == 1


async def test_persistently_wrong_paper_is_never_posted(client, monkeypatch):
    handler, state = exam_handler(always_corrupt=True)
    tg = FakeTelegram()
    r = await _real(client, monkeypatch, handler, tg)
    final = await _approve_all(client, r["id"])
    assert final["status"] == "failed" and tg.sent == [] and state["gen"] <= 3
    assert not final["artifacts"] and final["tasks"][-1]["status"] == "blocked"


async def test_telegram_failure_is_not_retried_and_needs_a_human(client, monkeypatch):
    handler, _ = exam_handler()
    tg = FakeTelegram(fail=True)
    r = await _real(client, monkeypatch, handler, tg)
    final = await _approve_all(client, r["id"])
    assert final["status"] == "halted" and len(tg.sent) == 1  # exactly one attempt - no blind repeat of an outward action
    assert "not a member" in final["error"]


async def test_real_mode_without_telegram_config_halts(client, monkeypatch):
    handler, _ = exam_handler()
    r = await _real(client, monkeypatch, handler, None)
    final = await _approve_all(client, r["id"])
    assert final["status"] == "halted" and "TELEGRAM_" in final["error"]


async def test_demo_rejects_non_maths_subject_for_human(client):
    r = (await client.post("/api/objectives", json={"objective": "make a 10 questions english test for class 8 and post it to telegram", "mode": "autonomous"})).json()
    final = await _approve_all(client, r["id"])
    assert final["status"] == "halted" and "GEMINI_API_KEY" in final["error"]


async def test_exam_workflow_can_be_saved_and_rerun(client):
    r = (await client.post("/api/objectives", json={"objective": TEACHER, "mode": "autonomous"})).json()
    assert (await _approve_all(client, r["id"]))["status"] == "completed"
    wf = (await client.post("/api/workflows", json={"run_id": r["id"], "name": "Weekly maths test"})).json()
    assert wf["workflow_type"] == "test_paper" and wf["params"]["grade"] == 10
    r2 = (await client.post(f"/api/workflows/{wf['id']}/run", json={"params": {"num_questions": 10}, "mode": "autonomous"})).json()
    final = await _approve_all(client, r2["id"])
    assert final["status"] == "completed" and final["params"]["num_questions"] == 10
    assert (await client.post(f"/api/workflows/{wf['id']}/run", json={"params": {"num_questions": 500}})).status_code == 422


def test_telegram_client_requires_config_and_scrubs_token(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    from app.config import get_settings

    get_settings.cache_clear()
    with pytest.raises(TelegramError):
        TelegramClient()
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:SECRET-TOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100123")
    get_settings.cache_clear()
    assert "SECRET-TOKEN" not in TelegramClient()._scrub("failed https://api.telegram.org/bot123456:SECRET-TOKEN/sendDocument")
    assert SimulatedTelegram().simulated


async def test_only_disputed_questions_are_replaced(client, monkeypatch):
    """A single wrong key must trigger a targeted repair, not a full redraft; verified questions survive unchanged."""
    base = exam_handler(corrupt_first=True)[0]
    state = {"full": 0, "repair": 0, "repair_prompt": "", "first_texts": None}

    def handler(name, payload, prompt):
        if name == "generate_exam":
            if "ONLY replacement questions" in prompt:
                state["repair"] += 1
                state["repair_prompt"] = prompt
                n = payload["n"]
                d = demo_math.generate("Mathematics", 10, 30).model_dump()  # a different set: fresh, correct questions
                d["title"] = "x" * 5
                return {"title": "Replacement questions", "instructions": [], "questions": [q for q in d["questions"] if q["type"] == "mcq"][-n:]}
            state["full"] += 1
        return base(name, payload, prompt)

    tg = FakeTelegram()
    r = await _real(client, monkeypatch, handler, tg)
    final = await _approve_all(client, r["id"])
    assert final["status"] == "completed", final["error"]
    assert state["full"] == 1 and state["repair"] == 1  # one full draft, then a targeted repair only
    assert "Rejected questions" in state["repair_prompt"]
    t1 = (await client.get(f"/api/runs/{r['id']}/tasks/T1")).json()["output"]
    assert any("replaced disputed questions" in x for x in t1["repairs"]) and len(t1["paper"]["questions"]) == 20
