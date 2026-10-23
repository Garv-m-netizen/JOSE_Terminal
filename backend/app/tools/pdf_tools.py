"""PDF generation (PyMuPDF HTML story) for the student paper and the teacher answer key, plus read-back."""
from __future__ import annotations

import html
from pathlib import Path

import pymupdf

from ..schemas.domain import ExamPaper

CSS = """
body { font-family: sans-serif; font-size: 11pt; color: #111; }
h1 { font-size: 17pt; text-align: center; margin-bottom: 2pt; }
.meta { text-align: center; font-size: 10.5pt; margin-bottom: 8pt; }
h2 { font-size: 12.5pt; margin-top: 12pt; border-bottom: 1px solid #444; }
.q { margin-top: 7pt; }
.opt { margin-left: 18pt; margin-top: 0pt; margin-bottom: 0pt; }
.sol { margin-left: 14pt; color: #1a3d7c; font-size: 10pt; }
.note { font-size: 9.5pt; color: #444; }
.banner { text-align: center; color: #b42318; font-weight: bold; font-size: 10pt; }
"""


def _e(s: str) -> str:
    return html.escape(s).replace("\n", "<br/>")


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    story = pymupdf.Story(html=f"<body>{body}</body>", user_css=CSS)
    writer = pymupdf.DocumentWriter(str(path))
    media = pymupdf.paper_rect("a4")
    where = media + (50, 50, -50, -60)
    more = 1
    while more:
        dev = writer.begin_page(media)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()


def _header(paper: ExamPaper, subject: str, grade: int, minutes: int, marks: int, kind: str, simulated: bool) -> str:
    banner = '<p class="banner">SIMULATED SAMPLE PAPER - generated in demo mode</p>' if simulated else ""
    return (f"{banner}<h1>{_e(paper.title)}</h1><p class='meta'>{_e(subject)} | Class {grade} | {kind}<br/>"
            f"Time: {minutes} minutes &nbsp;&nbsp; Maximum Marks: {marks}</p>")


def _sections(paper: ExamPaper) -> list[tuple[str, str, list]]:
    """(key, heading, questions) derived from the paper itself, so the PDF always matches the real marks."""
    names = {"mcq": "Multiple choice", "short": "Short answer", "long": "Long answer"}
    out = []
    for key in ("A", "B", "C"):
        qs = [q for q in paper.questions if q.section == key]
        if qs:
            m = qs[0].marks
            out.append((key, f"{names[qs[0].type]} ({m} mark{'s' if m != 1 else ''} each)", qs))
    return out


def build_question_paper(path: Path, paper: ExamPaper, subject: str, grade: int, minutes: int, simulated: bool) -> None:
    marks = sum(q.marks for q in paper.questions)
    body = _header(paper, subject, grade, minutes, marks, "Question Paper", simulated)
    if paper.instructions:
        body += "<p><b>General Instructions</b></p>" + "".join(f"<p class='note'>{i + 1}. {_e(t)}</p>" for i, t in enumerate(paper.instructions))
    for key, label, qs in _sections(paper):
        body += f"<h2>Section {key} - {label}</h2>"
        for q in qs:
            n = q.id.removeprefix("Q")
            body += f"<p class='q'><b>{n}.</b> {_e(q.text)} <b>[{q.marks}]</b></p>"
            for letter, o in zip("ABCD", q.options):
                body += f"<p class='opt'>({letter}) {_e(o)}</p>"
    body += "<p class='note' style='text-align:center;margin-top:14pt'>*** End of paper ***</p>"
    _write(path, body)


def build_answer_key(path: Path, paper: ExamPaper, subject: str, grade: int, minutes: int, simulated: bool, review_ids: list[str]) -> None:
    marks = sum(q.marks for q in paper.questions)
    body = _header(paper, subject, grade, minutes, marks, "ANSWER KEY - TEACHER COPY (do not share)", simulated)
    for q in paper.questions:
        n = q.id.removeprefix("Q")
        ans = f"Option {q.answer}: {_e(q.options['ABCD'.index(q.answer)])}" if q.type == "mcq" and q.answer in "ABCD" and len(q.options) == 4 else _e(q.answer)
        flag = " <b style='color:#b42318'>[teacher review: not machine-checked]</b>" if q.id in review_ids else ""
        body += f"<p class='q'><b>Q{n}</b> ({q.marks} marks, {_e(q.topic)}) - <b>Answer:</b> {ans}{flag}</p><p class='sol'>{_e(q.solution)}</p>"
    _write(path, body)


def read_pdf(path: Path) -> dict:
    with pymupdf.open(path) as doc:
        text = "\n".join(p.get_text() for p in doc)
        return {"pages": doc.page_count, "text": text}
