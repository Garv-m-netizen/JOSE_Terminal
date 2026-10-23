"""Test-paper workflow: for each requested paper, generate -> independently re-solve -> build PDFs -> verify;
then cross-check the whole set and (after approval) post the student papers to Telegram."""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..schemas.domain import ExamParams, TaskSpec
from .research_report import validate_plan

WORKFLOW_TYPE = "test_paper"
MARK_SETS = [(1, 2, 4), (1, 2, 3), (1, 3, 5), (1, 3, 4), (1, 2, 5), (2, 3, 5)]  # marks for (MCQ, short, long), tried in this order


@dataclass(frozen=True)
class Section:
    key: str
    type: str
    count: int
    marks: int
    label: str


def _sections(a: int, b: int, c: int, ms: tuple[int, int, int]) -> list[Section]:
    def lab(name: str, m: int) -> str:
        return f"{name} ({m} mark{'s' if m != 1 else ''} each)"

    out = [Section("A", "mcq", a, ms[0], lab("Multiple choice", ms[0]))]
    if b:
        out.append(Section("B", "short", b, ms[1], lab("Short answer", ms[1])))
    if c:
        out.append(Section("C", "long", c, ms[2], lab("Long answer", ms[2])))
    return out


def blueprint(n: int, marks: int | None = None) -> list[Section]:
    """Deterministic paper structure. Without a marks target: ~50% MCQ (1 mark), ~30% short (2), the rest long (4).
    With a target, find the question counts and mark values that add up to exactly that many marks (closest to that mix)."""
    if marks is None:
        mcq = max(1, round(n * 0.5))
        short = max(1, round(n * 0.3)) if n >= 3 else 0
        return _sections(mcq, short, max(0, n - mcq - short), MARK_SETS[0])
    for ms in MARK_SETS:
        best = None
        for a in range(1, n):
            for b in range(1, n - a):
                c = n - a - b
                if c >= 1 and a * ms[0] + b * ms[1] + c * ms[2] == marks:
                    cost = abs(a / n - 0.5) + abs(b / n - 0.3) + abs(c / n - 0.2)
                    if best is None or cost < best[0]:
                        best = (cost, a, b, c)
        if best:
            return _sections(best[1], best[2], best[3], ms)
    raise ValueError(f"{n} questions cannot add up to exactly {marks} marks using question values of 1-5 marks")


def total_marks(n: int, marks: int | None = None) -> int:
    return marks if marks is not None else sum(s.count * s.marks for s in blueprint(n))


def default_duration(n: int, marks: int | None = None) -> int:
    return max(10, int(round(total_marks(n, marks) * 1.5 / 5.0)) * 5)


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", s.title()) or "X"


def folder_name(p: ExamParams) -> str:
    topic = "_".join(slug(t) for t in p.topics[:2]) if p.topics else ""
    return "_".join(x for x in (f"Class{p.grade}", slug(p.subject), topic, "Tests" if p.num_papers > 1 else "Test") if x)


def build_plan(p: ExamParams) -> list[TaskSpec]:
    n, P = p.num_questions, p.num_papers
    folder = p.folder or folder_name(p)
    topics = ", ".join(p.topics) if p.topics else "the full syllabus"
    marks = total_marks(n, p.total_marks)
    tasks: list[TaskSpec] = []
    counter = iter(range(1, 200))

    def add(**kw) -> TaskSpec:
        t = TaskSpec(key=f"T{next(counter)}", **kw)
        tasks.append(t)
        return t

    verifies: list[TaskSpec] = []
    prev_gen: TaskSpec | None = None
    for i in range(1, P + 1):
        label = f"Test {i}" if P > 1 else "the test"
        gen = add(title=f"Draft {label}: {n} questions on {topics}", agent="research", action="test.generate",
                  params={**p.model_dump(), "paper": i, "folder": folder, "total_marks": p.total_marks},
                  depends_on=[prev_gen.key] if prev_gen else [],
                  description=f"Write a {marks}-mark paper ({p.subject}, class {p.grade}, {topics}) following a fixed section blueprint, with an answer and worked solution "
                              f"for every question." + (" Every question must differ from the earlier papers of this set." if i > 1 else ""),
                  expected_output=f"{n} questions with answers and solutions.",
                  verification_criteria=[f"exactly {n} questions matching the blueprint and adding up to {marks} marks", "MCQs have 4 options and a valid answer letter",
                                         "no duplicate questions", "every question has an answer and a solution"])
        chk = add(title=f"Re-solve every question of {label} independently", agent="verification", action="test.solve_check",
                  params={"paper": i}, depends_on=[gen.key],
                  description="A second, blind pass solves each question without seeing the answer key; answers are compared with the key.",
                  expected_output="Per-question match / mismatch / needs-teacher-review.",
                  verification_criteria=["no machine-checkable answer disagrees with the key"])
        build = add(title=f"Create question paper and answer key for {label} (PDF)", agent="workspace", action="test.build_documents",
                    params={"paper": i, "folder": folder, "num_papers": P}, depends_on=[chk.key],
                    resources=[f"file:{folder}/test{i}"],
                    description=f"Render a student-facing question paper (no answers) and a separate teacher-only answer key into the folder '{folder}'.",
                    expected_output=f"Two PDFs in {folder}/",
                    verification_criteria=["both files exist and reopen", "every question is in the paper", "the paper contains no answers"])
        ver = add(title=f"Verify the documents of {label}", agent="verification", action="test.verify_documents",
                  params={"paper": i}, depends_on=[build.key],
                  description="Reopen both PDFs and check content, marks and labels against the question set.",
                  expected_output="Pass/fail for each document check.", verification_criteria=["all document checks pass"])
        verifies.append(ver)
        prev_gen = gen
    last = verifies[-1]
    if P > 1:
        last = add(title=f"Cross-check all {P} tests together", agent="verification", action="test.verify_set",
                   params={"papers": P, "folder": folder, "num_questions": n, "total_marks": marks, "duration": p.duration_minutes},
                   depends_on=[v.key for v in verifies],
                   description="Check that every test has the same structure, marks and duration, that no question is repeated across tests, and that all PDFs are in the organised folder.",
                   expected_output="Pass/fail for each set-level check.",
                   verification_criteria=["same blueprint, marks and duration in every test", "no question appears in two tests", "all PDFs present in the folder"])
    if p.post_to_telegram:
        add(title="Post the question paper" + ("s" if P > 1 else "") + " to the Telegram group", agent="messenger", action="telegram.send",
            params={"dry_run": p.dry_run, "papers": P}, depends_on=[last.key] if P == 1 else [last.key] + [v.key for v in verifies],
            description="Send ONLY the student question paper PDF" + ("s" if P > 1 else "") + " with a short caption to the configured group. Requires explicit approval.",
            expected_output="Delivered Telegram message(s) with message ids.",
            verification_criteria=["Telegram confirms delivery with message ids", "the file names returned match the files sent", "no answer key was sent"])
    validate_plan(tasks)
    return tasks
