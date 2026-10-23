"""Intent parsing for teacher-style objectives: 'make a maths test for class 10 and post it to the Telegram group'."""
from __future__ import annotations

import re

from ..schemas.domain import Intent
from ..workflows.test_paper import blueprint, default_duration, total_marks

SUBJECTS = [("social science", "Social Science"), ("computer science", "Computer Science"), ("mathematics", "Mathematics"),
            ("maths", "Mathematics"), ("math", "Mathematics"), ("physics", "Physics"), ("chemistry", "Chemistry"),
            ("biology", "Biology"), ("science", "Science"), ("english", "English"), ("hindi", "Hindi"), ("history", "History"),
            ("geography", "Geography"), ("civics", "Civics"), ("economics", "Economics"), ("accountancy", "Accountancy"), ("evs", "EVS")]
TEST_RE = re.compile(r"\b(tests?|quiz(?:zes)?|exams?|question papers?|papers|worksheets?|assessments?|mcqs?)\b", re.I)
MAKE_RE = re.compile(r"\b(make|create|prepare|generate|set|draft|write|build|design|need|want|give)\b", re.I)
OTHER_CHANNELS = re.compile(r"\b(whatsapp|e-?mail|slack|discord|sms|facebook|instagram|teams|google classroom|classroom)\b", re.I)
NUMWORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
ORDINALS = {"tenth": 10, "ninth": 9, "eighth": 8, "seventh": 7, "sixth": 6, "fifth": 5, "eleventh": 11, "twelfth": 12}


def parse_test_rules(text: str) -> Intent | None:
    """None means 'this is not a test objective'. Otherwise a feasible Intent or an explained rejection."""
    low = text.lower()
    if not (TEST_RE.search(low) and MAKE_RE.search(low)) or re.search(r"competitor|market share|pricing|vendors", low):
        return None

    def no(reason: str, risk: str = "LOW") -> Intent:
        return Intent(feasible=False, rejection_reason=reason, goal=text, workflow_type="test_paper", risk_level=risk)  # type: ignore[arg-type]

    if re.search(r"\b(delete|erase|wipe|overwrite|purchase|buy)\b", low):
        return no("This objective asks JOSE to delete, overwrite or buy something. JOSE has no allow-listed tool for that action, so it will not attempt it.", "HIGH")
    ch = OTHER_CHANNELS.search(low)
    if ch and "telegram" not in low:
        return no(f"JOSE can only post to Telegram. '{ch.group(1)}' is not a supported destination. Ask for Telegram, or ask only for the test and download it.", "HIGH")
    subject = next((label for key, label in SUBJECTS if re.search(rf"\b{re.escape(key)}\b", low)), "")
    if not subject:
        return no("Which subject is the test for (for example Mathematics, Science, English)? I will not guess it.")
    gm = re.search(r"\b(?:class|grade|std|standard)\s*-?\s*(\d{1,2})\b", low) or re.search(r"\b(\d{1,2})(?:th|st|nd|rd)\s*(?:class|grade|std|standard)\b", low)
    grade = int(gm.group(1)) if gm else next((v for k, v in ORDINALS.items() if k in low), None)
    if grade is None:
        return no("Which class or grade is the test for (for example 'class 10')? I will not guess it.")
    if not 1 <= grade <= 12:
        return no(f"Class {grade} is outside the supported range (1-12).")
    assumptions: list[str] = []
    qm = re.search(r"(\d{1,3})\s*(?:questions|qs|mcqs?)", low)
    n = int(qm.group(1)) if qm else None
    if n is not None and not 5 <= n <= 50:
        return no(f"{n} questions is outside the supported range (5-50).")
    if n is None:
        n = 20
        assumptions.append("No question count was given; defaulting to 20 questions.")
    papers = 1
    pm = re.search(r"\b(\d{1,2}|" + "|".join(NUMWORDS) + r")\s+(?:[\w\-]+\s+){0,4}?(?:tests|papers|quizzes|exams|worksheets|assessments)\b", low)
    if pm:
        papers = int(pm.group(1)) if pm.group(1).isdigit() else NUMWORDS[pm.group(1)]
        if not 1 <= papers <= 10:
            return no(f"{papers} separate tests is outside the supported range (1-10).")
    marks = None
    mm = re.search(r"(\d{1,3})\s*marks\b", low)
    if mm:
        marks = int(mm.group(1))
        if not 5 <= marks <= 200:
            return no(f"{marks} marks is outside the supported range (5-200).")
        try:
            blueprint(n, marks)
        except ValueError as e:
            return no(f"{e}. Change the number of questions or the total marks.")
    topics: list[str] = []
    tm = re.search(r"(?:chapters?|topics?|covering)\s*(?:of|:)?\s*([a-z][a-z ,&\-]{2,80}?)(?=\s+(?:for|then|in|to|post|send|class|grade)\b|\s+and\s+(?:post|send|share)\b|[.;]|\d|$)", low)
    if tm:
        topics = [t.strip(" ,").title() for t in re.split(r",| and ", tm.group(1)) if t.strip(" ,")][:6]
    if not topics:
        om = re.search(r"\bon\s+([a-z][a-z ,&\-]{2,60}?)(?=\s*(?:,|\.|;|$)|\s+(?:each|with|for|containing|having|keep|of|in)\b|\s+and\s+(?:post|send|share|save|make|create|generate)\b)", low)
        if om and not re.search(r"\b(the|a|an|one|each|all)\b", om.group(1)) and len(om.group(1).split()) <= 5:
            topics = [x.strip(" ,").title() for x in re.split(r",| and ", om.group(1)) if x.strip(" ,")][:6]
    if not topics:
        assumptions.append("No topics were named; the paper covers the whole grade-level syllabus.")
    diff = next((d for d in ("easy", "medium", "hard") if re.search(rf"\b{d}\b", low)), "mixed")
    if diff == "mixed":
        assumptions.append("No difficulty was named; using a mix of easy, medium and hard.")
    dm = re.search(r"(\d{2,3})\s*(?:minutes|mins|min)\b", low)
    dur = int(dm.group(1)) if dm else default_duration(n, marks)
    if not dm:
        assumptions.append(f"No duration was given; using {dur} minutes.")
    post = "telegram" in low
    if not post and re.search(r"\b(post|send|share|publish)\b", low):
        return no("Where should the test be posted? JOSE can post to a Telegram group - say 'post it to the Telegram group'.", "HIGH")
    if papers > 1:
        assumptions.append("'Similar difficulty' is enforced structurally: every test uses the same sections, question types, marks and duration; only the questions differ, and no question is repeated across tests.")
    if marks is None:
        assumptions.append(f"No total marks were given; using the standard layout ({total_marks(n)} marks).")
    assumptions.append("A separate answer key is created for the teacher only; it is never posted. All PDFs are saved in one organised folder (question papers and answer keys in separate sub-folders).")
    criteria = [f"{papers} different test(s), each with {n} questions adding up to exactly {total_marks(n, marks)} marks, with answers and worked solutions",
                "every answer independently re-solved and agreeing with the key (questions that cannot be machine-checked are flagged for teacher review)",
                "the student paper PDF contains no answers; the teacher answer key is a separate file"]
    if post:
        criteria.append("posted to Telegram only after explicit teacher approval, with delivery confirmed")
    return Intent(feasible=True, workflow_type="test_paper", goal=text, subject=subject, grade=grade, topics=topics, num_questions=n,
                  difficulty=diff, duration_minutes=dur, post_to_telegram=post, assumptions=assumptions, num_papers=papers, total_marks=marks,
                  risk_level="HIGH" if post else "LOW", outputs=["xlsx"], success_criteria=criteria)
