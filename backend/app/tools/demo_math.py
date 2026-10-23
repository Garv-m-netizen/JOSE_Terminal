"""Deterministic Class-10-style maths question generator and an independent text-based solver (demo mode only).

The questions are real, correct maths built from templates. The solver re-derives each answer by *parsing the
question text*, never by reading the answer key, so the solve-check in demo mode is a genuine consistency test.
"""
from __future__ import annotations

import math
import random
import re
from fractions import Fraction

from ..schemas.domain import ExamPaper, Question
from ..workflows.test_paper import blueprint

SIM = "[SIMULATED SAMPLE PAPER]"


def fmt(x) -> str:
    if isinstance(x, Fraction):
        return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
    if isinstance(x, float):
        return fmt(Fraction(x).limit_denominator(1000))
    return str(x)


def _opts(rng: random.Random, correct, cands: list) -> tuple[list[str], str]:
    seen = {fmt(correct)}
    options = [fmt(correct)]
    for c in cands + [correct + k for k in (1, 2, 3, -1, -2, 4, 5, 6)]:
        s = fmt(c)
        if s not in seen:
            seen.add(s)
            options.append(s)
        if len(options) == 4:
            break
    rng.shuffle(options)
    return options, "ABCD"[options.index(fmt(correct))]


TRIG = {"sin 30°": 0.5, "cos 60°": 0.5, "tan 45°": 1.0, "sin 90°": 1.0, "cos 0°": 1.0, "sin 0°": 0.0, "cos 90°": 0.0, "tan 60°": math.sqrt(3), "tan 30°": 1 / math.sqrt(3)}
TRIG_EXPRS = ["sin 30° + cos 60°", "tan 45° + sin 90°", "2 sin 30° + cos 0°", "4 cos 60° − sin 30°", "tan 60° × tan 30°", "sin 90° + cos 90° + tan 45°", "6 sin 30° − 2 cos 0°"]


def _mcq(i: int, rng: random.Random):
    k = i % 8
    if k == 0:
        a, b = rng.sample(range(1, 10), 2)
        S, P = a + b, a * b
        return (f"The sum of the roots of the equation x² − {S}x + {P} = 0 is", S, [P, -S, S + 1], f"For x² − Sx + P = 0 the sum of roots is S = {S}.", "Quadratic Equations")
    if k == 1:
        a, d, n = rng.randint(2, 9), rng.randint(2, 7), rng.randint(8, 15)
        return (f"The {n}th term of the AP {a}, {a + d}, {a + 2 * d}, ... is", a + (n - 1) * d, [a + n * d, a + d * (n - 2), n * d], f"a_n = a + (n − 1)d = {a} + {n - 1}×{d} = {a + (n - 1) * d}.", "Arithmetic Progressions")
    if k == 2:
        m = rng.randint(1, 6)
        return (f"The distance between the points (0, 0) and ({3 * m}, {4 * m}) is", 5 * m, [7 * m, 12 * m, 25 * m], f"d = √({3 * m}² + {4 * m}²) = {5 * m}.", "Coordinate Geometry")
    if k == 3:
        r, b = rng.randint(2, 9), rng.randint(2, 9)
        c = Fraction(r, r + b)
        return (f"A bag contains {r} red balls and {b} blue balls. One ball is drawn at random. The probability that it is red is", c,
                [Fraction(b, r + b), Fraction(r, b), Fraction(1, r + b)], f"P(red) = {r}/({r}+{b}) = {fmt(c)}.", "Probability")
    if k == 4:
        e = TRIG_EXPRS[(i // 8) % len(TRIG_EXPRS)]
        v = _eval_trig(e)
        return (f"The value of {e} is", v, [v + Fraction(1, 2), v + 1, v * 2], f"Using standard trigonometric values, {e} = {fmt(v)}.", "Trigonometry")
    if k == 5:
        g, m, n = rng.randint(2, 9), *rng.sample([2, 3, 5, 7, 11], 2)
        return (f"The HCF of {g * m} and {g * n} is", g, [g * m, g * n, g * m * n], f"{g * m} = {g}×{m} and {g * n} = {g}×{n} with {m},{n} coprime, so HCF = {g}.", "Real Numbers")
    if k == 6:
        mm, h = rng.randint(1, 3), rng.randint(2, 9)
        r = 7 * mm
        V = 22 * 7 * mm * mm * h
        return (f"The volume of a cylinder of radius {r} cm and height {h} cm (use π = 22/7) is, in cm³,", V, [V // 2, V * 2, V + 22], f"V = πr²h = (22/7)×{r}²×{h} = {V}.", "Surface Areas and Volumes")
    n = 5
    base = rng.randint(4, 12)
    nums = [base + rng.randint(0, 8) for _ in range(n - 1)]
    nums.append(((-sum(nums)) % n) + n * (base // n + 2))
    mean = Fraction(sum(nums), n)
    return (f"The mean of the numbers {', '.join(map(str, nums))} is", mean, [mean + 1, mean - 1, mean + 2], f"Mean = {sum(nums)}/{n} = {fmt(mean)}.", "Statistics")


def _eval_trig(expr: str) -> Fraction:
    e = expr
    for name, val in TRIG.items():
        e = e.replace(name, f"({val!r})")
    e = e.replace("×", "*").replace("÷", "/").replace("−", "-")
    e = re.sub(r"(\d)\s*\(", r"\1*(", e)
    if not re.fullmatch(r"[0-9.()+\-*/ eE]+", e):
        raise ValueError(f"unsafe expression {expr}")
    return Fraction(eval(e, {"__builtins__": {}}, {})).limit_denominator(1000)  # noqa: S307 - sanitised numeric expression


def _short(i: int, rng: random.Random):
    k = i % 5
    if k == 0:
        a, b, c = rng.randint(1, 4), rng.randint(-9, 9), rng.randint(-5, 5)
        D = b * b - 4 * a * c
        nat = "real and distinct" if D > 0 else "real and equal" if D == 0 else "not real"
        return (f"Find the discriminant of the equation {a}x² + ({b})x + ({c}) = 0 and state whether its roots are real.", str(D), f"D = b² − 4ac = {b * b} − {4 * a * c} = {D}. The roots are {nat}.", "Quadratic Equations")
    if k == 1:
        a, d, n = rng.randint(1, 9), rng.randint(2, 6), rng.randint(10, 20)
        S = n * (2 * a + (n - 1) * d) // 2
        return (f"Find the sum of the first {n} terms of the AP {a}, {a + d}, {a + 2 * d}, ...", str(S), f"S_n = n/2 [2a + (n − 1)d] = {n}/2 × [{2 * a} + {(n - 1) * d}] = {S}.", "Arithmetic Progressions")
    if k == 2:
        g, m, n = rng.randint(2, 6), *rng.sample([2, 3, 5, 7], 2)
        a, b = g * m, g * n
        L = a * b // math.gcd(a, b)
        return (f"Find the LCM of {a} and {b}.", str(L), f"HCF = {math.gcd(a, b)}; LCM = {a}×{b}/HCF = {L}.", "Real Numbers")
    if k == 3:
        x1, y1 = rng.randint(-8, 8) * 2, rng.randint(-8, 8) * 2
        x2, y2 = rng.randint(-8, 8) * 2, rng.randint(-8, 8) * 2
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        return (f"Find the coordinates of the mid-point of the line segment joining A({x1}, {y1}) and B({x2}, {y2}).", f"({mx}, {my})", f"Mid-point = ((x₁+x₂)/2, (y₁+y₂)/2) = ({mx}, {my}).", "Coordinate Geometry")
    x, y = rng.randint(2, 12), rng.randint(1, 9)
    return (f"Solve the pair of equations: x + y = {x + y} and x − y = {x - y}.", f"x = {x}, y = {y}", f"Adding gives 2x = {2 * x}, so x = {x}; then y = {y}.", "Pair of Linear Equations")


def _long(i: int, rng: random.Random):
    k = i % 3
    if k == 0:
        a, b = rng.sample(range(1, 12), 2)
        lo, hi = sorted((a, b))
        return (f"Solve the equation x² − {a + b}x + {a * b} = 0 by factorisation.", f"x = {lo} or x = {hi}",
                f"Split the middle term: x² − {lo}x − {hi}x + {a * b} = 0 ⇒ (x − {lo})(x − {hi}) = 0 ⇒ x = {lo} or x = {hi}.", "Quadratic Equations")
    if k == 1:
        m = rng.randint(2, 8)
        return (f"A ladder {5 * m} m long reaches a window {4 * m} m above the ground. Find the distance of the foot of the ladder from the wall.", f"{3 * m} m",
                f"By Pythagoras, d² = {5 * m}² − {4 * m}² = {25 * m * m - 16 * m * m} ⇒ d = {3 * m} m.", "Triangles")
    a, d, n = rng.randint(3, 9), rng.randint(2, 6), rng.randint(9, 16)
    an, S = a + (n - 1) * d, n * (2 * a + (n - 1) * d) // 2
    return (f"In an AP the first term is {a} and the common difference is {d}. Find the {n}th term and the sum of the first {n} terms.", f"a_n = {an}, S_n = {S}",
            f"a_n = a + (n − 1)d = {an}. S_n = n/2 (a + a_n) = {n}/2 × ({a} + {an}) = {S}.", "Arithmetic Progressions")


def generate(subject: str, grade: int, n: int, difficulty: str = "mixed", marks: int | None = None, paper: int = 1, avoid: list[str] | None = None) -> ExamPaper:
    if "math" not in subject.lower():
        raise ValueError("demo mode can only generate Mathematics papers")
    rng = random.Random(f"{subject}|{grade}|{n}|{paper}")
    questions: list[Question] = []
    seen: set[str] = set(avoid or [])  # questions already used in other tests of the set
    qn = 0
    for sec in blueprint(n, marks):
        for j in range(sec.count):
            for attempt in range(40):
                r = random.Random(f"{rng.random()}|{qn}|{attempt}")
                idx = j + (paper - 1) * 3 + attempt  # template rotates with the position, the paper number and, on a repeat, the attempt
                if sec.type == "mcq":
                    text, correct, cands, sol, topic = _mcq(idx, r)
                    options, letter = _opts(r, correct, cands)
                    q = dict(options=options, answer=letter, final_answer=fmt(correct))
                elif sec.type == "short":
                    text, fa, sol, topic = _short(idx, r)
                    q = dict(options=[], answer=fa, final_answer=fa)
                else:
                    text, fa, sol, topic = _long(idx, r)
                    q = dict(options=[], answer=fa, final_answer=fa)
                if text not in seen:
                    seen.add(text)
                    break
            qn += 1
            questions.append(Question(id=f"Q{qn}", section=sec.key, type=sec.type, text=text, solution=sol, marks=sec.marks, topic=topic, **q))  # type: ignore[arg-type]
    return ExamPaper(title=f"Mathematics - Class {grade} Test {SIM}",
                     instructions=["All questions are compulsory.", "Show all steps for Sections B and C.", "Calculators are not allowed."], questions=questions)


# ------------------------------------------------------------------ independent solver (parses text only)

def solve(text: str, options: list[str]) -> str:
    """Return an option letter for MCQs, otherwise the final value. Never sees the answer key."""
    val = _solve_value(text)
    if options:
        for letter, o in zip("ABCD", options):
            if _num(o) is not None and val is not None and abs(_num(o) - val) < 1e-9:
                return letter
        return "?"
    return val if isinstance(val, str) else fmt(val)


def _num(s: str):
    try:
        return float(Fraction(s.strip()))
    except (ValueError, ZeroDivisionError):
        return None


def _solve_value(t: str):  # noqa: C901
    m = re.search(r"sum of the roots of the equation x² − (\d+)x", t)
    if m:
        return float(m.group(1))
    m = re.search(r"The (\d+)th term of the AP (\d+), (\d+),", t)
    if m:
        n, a, b = map(int, m.groups())
        return float(a + (n - 1) * (b - a))
    m = re.search(r"points \(0, 0\) and \((\d+), (\d+)\)", t)
    if m:
        return math.hypot(int(m.group(1)), int(m.group(2)))
    m = re.search(r"contains (\d+) red balls and (\d+) blue", t)
    if m:
        r, b = map(int, m.groups())
        return r / (r + b)
    m = re.search(r"The value of (.+) is", t)
    if m:
        return float(_eval_trig(m.group(1)))
    m = re.search(r"HCF of (\d+) and (\d+)", t)
    if m and t.startswith("The"):
        return float(math.gcd(int(m.group(1)), int(m.group(2))))
    m = re.search(r"radius (\d+) cm and height (\d+) cm", t)
    if m:
        r, h = map(int, m.groups())
        return 22 * r * r * h / 7
    m = re.search(r"The mean of the numbers ([\d, ]+) is", t)
    if m:
        xs = [int(x) for x in m.group(1).split(",")]
        return sum(xs) / len(xs)
    m = re.search(r"equation (-?\d+)x² \+ \((-?\d+)\)x \+ \((-?\d+)\) = 0", t)
    if m:
        a, b, c = map(int, m.groups())
        return str(b * b - 4 * a * c)
    m = re.search(r"sum of the first (\d+) terms of the AP (\d+), (\d+),", t)
    if m:
        n, a, b = map(int, m.groups())
        return str(n * (2 * a + (n - 1) * (b - a)) // 2)
    m = re.search(r"LCM of (\d+) and (\d+)", t)
    if m:
        a, b = map(int, m.groups())
        return str(a * b // math.gcd(a, b))
    m = re.search(r"A\((-?\d+), (-?\d+)\) and B\((-?\d+), (-?\d+)\)", t)
    if m:
        x1, y1, x2, y2 = map(int, m.groups())
        return f"({(x1 + x2) // 2}, {(y1 + y2) // 2})"
    m = re.search(r"x \+ y = (-?\d+) and x − y = (-?\d+)", t)
    if m:
        s, d = map(int, m.groups())
        return f"x = {(s + d) // 2}, y = {(s - d) // 2}"
    m = re.search(r"x² − (\d+)x \+ (\d+) = 0 by factorisation", t)
    if m:
        S, P = map(int, m.groups())
        disc = math.isqrt(S * S - 4 * P)
        lo, hi = (S - disc) // 2, (S + disc) // 2
        return f"x = {lo} or x = {hi}"
    m = re.search(r"ladder (\d+) m long reaches a window (\d+) m", t)
    if m:
        L, h = map(int, m.groups())
        return f"{math.isqrt(L * L - h * h)} m"
    m = re.search(r"first term is (\d+) and the common difference is (\d+)\. Find the (\d+)th", t)
    if m:
        a, d, n = map(int, m.groups())
        return f"a_n = {a + (n - 1) * d}, S_n = {n * (2 * a + (n - 1) * d) // 2}"
    return None


NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def answers_match(a: str, b: str) -> bool | None:
    """True/False when both contain numbers; None when the answer cannot be compared mechanically."""
    na, nb = NUM_RE.findall(a or ""), NUM_RE.findall(b or "")
    if not na or not nb:
        return None
    try:
        return sorted(float(Fraction(x)) for x in na) == sorted(float(Fraction(x)) for x in nb)
    except (ValueError, ZeroDivisionError):
        return None
