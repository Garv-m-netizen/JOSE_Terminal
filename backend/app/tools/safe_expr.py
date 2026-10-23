"""A tiny, safe expression evaluator for computed table columns. No eval(), no attribute access, no imports.

Column references use brackets:  ([Maths] + [Science]) / 200 * 100        text:  str(round([Total] / 4, 2)) + " %"
Conditions use Python's ternary:  "PASS" if [Percentage] >= 33 else "NEEDS IMPROVEMENT"
Functions: sum avg min max round abs int float str upper lower len
"""
from __future__ import annotations

import ast
import math
import re
from typing import Any, Callable

MAX_LEN, MAX_NODES = 400, 120
COL_RE = re.compile(r"\[([^\[\]]+)\]")


class ExprError(ValueError):
    pass


def is_num(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    try:
        float(str(v).strip().replace(",", ""))
        return str(v).strip() != ""
    except ValueError:
        return False


def num(v: Any) -> float | int:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v
    if v is None or str(v).strip() == "":
        raise ExprError("a number was expected but the value is empty")
    try:
        f = float(str(v).strip().replace(",", ""))
    except ValueError:
        raise ExprError(f"'{v}' is not a number") from None
    return int(f) if f.is_integer() else f


def _flat(args) -> list:
    out = []
    for a in args:
        out += _flat(a) if isinstance(a, (list, tuple)) else [a]
    return out


FUNCS: dict[str, Callable[..., Any]] = {
    "sum": lambda *a: sum(num(x) for x in _flat(a)),
    "avg": lambda *a: sum(num(x) for x in _flat(a)) / len(_flat(a)),
    "min": lambda *a: min(num(x) for x in _flat(a)),
    "max": lambda *a: max(num(x) for x in _flat(a)),
    "round": lambda x, n=0: round(num(x), int(n)) if int(n) > 0 else int(round(num(x))),
    "abs": lambda x: abs(num(x)),
    "int": lambda x: int(num(x)),
    "float": lambda x: float(num(x)),
    "str": lambda x: _fmt(x),
    "upper": lambda x: str(x).upper(),
    "lower": lambda x: str(x).lower(),
    "len": lambda x: len(str(x)),
}


def _fmt(x: Any) -> str:
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return "" if x is None else str(x)


def compile_expr(expr: str) -> tuple[ast.AST, dict[str, str]]:
    if len(expr) > MAX_LEN:
        raise ExprError(f"expression is longer than {MAX_LEN} characters")
    cols: dict[str, str] = {}

    def repl(m: re.Match) -> str:
        name = f"__c{len(cols)}__"
        cols[name] = m.group(1).strip()
        return name

    src = COL_RE.sub(repl, expr)
    try:
        tree = ast.parse(src.strip(), mode="eval")
    except SyntaxError as e:
        raise ExprError(f"cannot parse '{expr}': {e.msg}") from None
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        raise ExprError("expression is too complex")
    ok = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call, ast.Constant, ast.Name, ast.Load,
          ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
          ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
    for n in nodes:
        if not isinstance(n, ok):
            raise ExprError(f"'{type(n).__name__}' is not allowed in expressions")
        if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name) and n.func.id in FUNCS and not n.keywords):
            raise ExprError("only these functions may be called: " + ", ".join(FUNCS))
        if isinstance(n, ast.Name) and n.id not in cols and n.id not in FUNCS and n.id not in ("True", "False", "None"):
            raise ExprError(f"unknown name '{n.id}' - write column names in [brackets]")
        if isinstance(n, ast.Constant) and not isinstance(n.value, (str, int, float, bool, type(None))):
            raise ExprError("unsupported constant")
    return tree, cols


def referenced_columns(expr: str) -> list[str]:
    return list(compile_expr(expr)[1].values())


def evaluate(tree: ast.AST, cols: dict[str, str], lookup: Callable[[str], Any]) -> Any:
    def ev(n: ast.AST) -> Any:  # noqa: C901
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in cols:
                return lookup(cols[n.id])
            return {"True": True, "False": False, "None": None}[n.id]
        if isinstance(n, ast.UnaryOp):
            v = ev(n.operand)
            return (not v) if isinstance(n.op, ast.Not) else (-num(v) if isinstance(n.op, ast.USub) else num(v))
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add) and ((isinstance(a, str) and not is_num(a)) or (isinstance(b, str) and not is_num(b))):
                return _fmt(a) + _fmt(b)
            x, y = num(a), num(b)
            if isinstance(n.op, ast.Add):
                return x + y
            if isinstance(n.op, ast.Sub):
                return x - y
            if isinstance(n.op, ast.Mult):
                return x * y
            if isinstance(n.op, (ast.Div, ast.FloorDiv, ast.Mod)) and y == 0:
                raise ExprError("division by zero")
            if isinstance(n.op, ast.Div):
                return x / y
            if isinstance(n.op, ast.FloorDiv):
                return x // y
            if isinstance(n.op, ast.Mod):
                return x % y
            if isinstance(n.op, ast.Pow):
                if abs(y) > 10 or abs(x) > 1e6:
                    raise ExprError("exponent too large")
                return x ** y
        if isinstance(n, ast.BoolOp):
            vals = [ev(v) for v in n.values]
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        if isinstance(n, ast.Compare):
            left = ev(n.left)
            for op, comp in zip(n.ops, n.comparators):
                right = ev(comp)
                a, b = (num(left), num(right)) if is_num(left) and is_num(right) else (left, right)
                ok = {ast.Eq: a == b, ast.NotEq: a != b}.get(type(op))
                if ok is None:
                    if not (is_num(left) and is_num(right)):
                        raise ExprError("ordering comparisons need numbers")
                    ok = {ast.Lt: a < b, ast.LtE: a <= b, ast.Gt: a > b, ast.GtE: a >= b}[type(op)]
                if not ok:
                    return False
                left = right
            return True
        if isinstance(n, ast.IfExp):
            return ev(n.body) if ev(n.test) else ev(n.orelse)
        if isinstance(n, ast.Call):
            return FUNCS[n.func.id](*[ev(a) for a in n.args])  # type: ignore[attr-defined]
        raise ExprError(f"unsupported expression element {type(n).__name__}")

    out = ev(tree)
    if isinstance(out, float):
        if math.isnan(out) or math.isinf(out):
            raise ExprError("the result is not a finite number")
        out = round(out, 10)
        if out.is_integer():
            out = int(out)
    return out
