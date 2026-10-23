from __future__ import annotations

import asyncio
import io
import zipfile

import pymupdf
import pytest

from app.agents.intent_engine import parse_rules
from app.services.llm import Providers, set_providers_override
from app.tools import demo_math
from app.workflows.test_paper import blueprint, default_duration, total_marks

from .helpers import FakeLLM
from .test_exam_workflow import FakeTelegram, _approve_all, exam_handler

OBJECTIVE = ("Create three different Class 10 Mathematics tests on Trigonometry, each containing 20 questions for 40 marks and 45 minutes. Keep the difficulty "
             "similar but vary the questions. Generate a separate answer key for each test, verify all answers and marks, and save the six PDFs in an organized folder.")


def pdf_text(b: bytes) -> str:
    return "\n".join(p.get_text() for p in pymupdf.open(stream=b, filetype="pdf"))


def test_objective_is_understood_in_full():
    i = parse_rules(OBJECTIVE)
    assert i.feasible and i.workflow_type == "test_paper"
    assert (i.subject, i.grade, i.topics, i.num_questions, i.total_marks, i.num_papers, i.duration_minutes) == ("Mathematics", 10, ["Trigonometry"], 20, 40, 3, 45)
    assert not i.post_to_telegram and any("Similar difficulty" in a for a in i.assumptions) and any("organised folder" in a for a in i.assumptions)
    assert "3 different test(s)" in i.success_criteria[0] and "exactly 40 marks" in i.success_criteria[0]


@pytest.mark.parametrize("n,marks,expect", [(20, 40, [("mcq", 10, 1), ("short", 5, 2), ("long", 5, 4)]), (10, 20, [("mcq", 4, 1), ("short", 4, 2), ("long", 2, 4)])])
def test_blueprint_hits_the_exact_marks_target(n, marks, expect):
    bp = blueprint(n, marks)
    assert [(s.type, s.count, s.marks) for s in bp] == expect and sum(s.count * s.marks for s in bp) == marks == total_marks(n, marks)
    assert [(s.type, s.count, s.marks) for s in blueprint(20)] == [("mcq", 10, 1), ("short", 6, 2), ("long", 4, 4)]  # default layout unchanged


@pytest.mark.parametrize("text,frag", [
    ("make 3 maths tests for class 10 of 5 questions for 100 marks", "cannot add up"),
    ("make 12 maths tests for class 10", "outside the supported range"),
    ("make a maths test for class 10 worth 500 marks", "outside the supported range"),
])
def test_impossible_requests_are_refused(text, frag):
    i = parse_rules(text)
    assert not i.feasible and frag in i.rejection_reason


async def test_three_tests_six_pdfs_in_an_organised_folder(client):
    r = (await client.post("/api/objectives", json={"objective": OBJECTIVE, "mode": "autonomous"})).json()
    assert r["status"] != "rejected", r["error"]
    acts = [t["action"] for t in r["tasks"]]
    assert acts.count("test.generate") == 3 and acts.count("test.build_documents") == 3 and acts[-1] == "test.verify_set" and len(acts) == 13
    assert [t["depends_on"] for t in r["tasks"] if t["action"] == "test.generate"][1:] == [["T1"], ["T5"]]  # each draft sees the earlier ones
    final = await _approve_all(client, r["id"])
    assert final["status"] == "completed", final["error"]
    files = [a for a in final["artifacts"] if a["kind"] in ("pdf", "answer_key")]
    assert len(files) == 6
    folders = {a["folder"] for a in files}
    assert folders == {"Class10_Mathematics_Trigonometry_Tests/Question Papers (Students)", "Class10_Mathematics_Trigonometry_Tests/Answer Keys (Teacher Only)"}
    texts = {}
    for a in files:
        texts[a["filename"]] = pdf_text((await client.get(a["download_url"])).content)
    papers = {k: v for k, v in texts.items() if "Question_Paper" in k}
    keys = {k: v for k, v in texts.items() if "Answer_Key" in k}
    assert len(papers) == 3 and len(keys) == 3
    for t in papers.values():
        assert "Maximum Marks: 40" in t and "Time: 45 minutes" in t and "Answer:" not in t
    for t in keys.values():
        assert "TEACHER COPY" in t and t.count("Answer:") == 20 and "Maximum Marks: 40" in t
    # the three tests really differ
    qs = [set(t.split("[1]")[0:0] or [ln for ln in t.splitlines() if ln[:2].strip().rstrip(".").isdigit()]) for t in papers.values()]
    assert len(set(map(frozenset, qs))) == 3
    # one ZIP keeps the folder structure
    z = zipfile.ZipFile(io.BytesIO((await client.get(f"/api/runs/{r['id']}/artifacts.zip")).content))
    assert len(z.namelist()) == 6 and all(n.startswith("Class10_Mathematics_Trigonometry_Tests/") for n in z.namelist())
    v = (await client.get(f"/api/runs/{r['id']}/verification")).json()
    passed = {c["check"] for c in v["current"] if c["passed"]}
    assert {"set.same_structure", "set.marks", "set.no_repeated_questions", "set.folder_layout", "set.duration", "pdf.duration_printed"} <= passed
    assert (await client.get("/api/runs/run_nope/artifacts.zip")).status_code == 404


def multi_handler(repeat_first_try: bool):
    """Real-mode fake model: for paper 2 the first draft re-uses paper 1's questions (it must be caught and redrafted)."""
    base = exam_handler()[0]
    state = {"p1": None, "calls": {}, "prompts": []}

    def h(name, payload, prompt):
        if name == "generate_exam" and "ONLY replacement" not in prompt:
            i = payload["paper"]
            state["calls"][i] = state["calls"].get(i, 0) + 1
            state["prompts"].append((i, prompt))
            d = demo_math.generate(payload["subject"], payload["grade"], payload["n"], marks=payload["marks"], paper=i, avoid=payload["avoid"]).model_dump()
            d["title"] = "Mathematics Test"
            if i == 1:
                state["p1"] = d
            if i == 2 and repeat_first_try and state["calls"][i] == 1 and state["p1"]:
                d = {**state["p1"]}  # a lazy model repeating the first test
            return d
        return base(name, payload, prompt)

    return h, state


async def test_repeated_questions_across_tests_are_caught_and_fixed(client, monkeypatch):
    h, state = multi_handler(True)
    monkeypatch.setenv("JOSE_MODE", "real")
    monkeypatch.setenv("GEMINI_API_KEY", "x" * 12)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-1001234567890")
    monkeypatch.setenv("MAX_RETRIES", "3")
    from app.config import get_settings

    get_settings.cache_clear()
    set_providers_override(lambda s, i: Providers(llm=FakeLLM(h), search=None, fetcher=None, browser=None, simulated=False, messenger=None))
    r = (await client.post("/api/objectives", json={"objective": OBJECTIVE, "mode": "autonomous"})).json()
    final = await _approve_all(client, r["id"])
    assert final["status"] == "completed", final["error"]
    assert state["calls"][2] >= 2 and final["recovery_count"] >= 1
    v = (await client.get(f"/api/runs/{r['id']}/verification")).json()
    assert any(c["check"] == "set.no_repeated_questions" and not c["passed"] for c in v["checks"])  # the failure stays in the history
    assert any(c["check"] == "set.no_repeated_questions" and c["passed"] for c in v["current"])
    assert any(i == 2 and "Test 2 repeats a question from Test 1" in p for i, p in state["prompts"][2:])  # the redraft was told what was wrong


async def test_posting_a_set_sends_only_student_papers(client, monkeypatch):
    h, _ = multi_handler(False)
    monkeypatch.setenv("JOSE_MODE", "real")
    monkeypatch.setenv("GEMINI_API_KEY", "x" * 12)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-1001234567890")
    from app.config import get_settings

    get_settings.cache_clear()
    tg = FakeTelegram()
    set_providers_override(lambda s, i: Providers(llm=FakeLLM(h), search=None, fetcher=None, browser=None, simulated=False, messenger=tg))
    r = (await client.post("/api/objectives", json={"objective": OBJECTIVE.replace("save the six", "post them to the telegram group and save the six"), "mode": "autonomous"})).json()
    assert r["tasks"][-1]["action"] == "telegram.send"
    final = await _approve_all(client, r["id"])
    assert final["status"] == "completed", final["error"]
    assert len(tg.sent) == 3 and all("Question_Paper" in n for n, _ in tg.sent) and not any("Answer_Key" in n for n, _ in tg.sent)
    assert all("Test" in c for _, c in tg.sent)


async def test_saved_multi_paper_workflow_keeps_its_parameters(client):
    r = (await client.post("/api/objectives", json={"objective": OBJECTIVE, "mode": "autonomous"})).json()
    assert (await _approve_all(client, r["id"]))["status"] == "completed"
    wf = (await client.post("/api/workflows", json={"run_id": r["id"], "name": "Trig sets"})).json()
    assert wf["params"]["num_papers"] == 3 and wf["params"]["total_marks"] == 40 and wf["params"]["topics"] == ["Trigonometry"]
    r2 = (await client.post(f"/api/workflows/{wf['id']}/run", json={"params": {"num_papers": 2}, "mode": "autonomous"})).json()
    f2 = await _approve_all(client, r2["id"])
    assert f2["status"] == "completed" and len([a for a in f2["artifacts"] if a["kind"] in ("pdf", "answer_key")]) == 4
