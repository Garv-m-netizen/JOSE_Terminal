"""The Research -> Excel -> PowerPoint workflow: action registry, plan template and plan validation."""
from __future__ import annotations

from dataclasses import dataclass

from ..schemas.domain import TaskSpec, WorkflowParams

WORKFLOW_TYPE = "research_report"
EXCEL_NAME = "competitors.xlsx"
PPTX_NAME = "competitors.pptx"


@dataclass(frozen=True)
class ActionSpec:
    agent: str
    requires: tuple[str, ...] = ()  # actions that must appear among the task's transitive dependencies


ACTIONS: dict[str, ActionSpec] = {
    "file.read_inputs": ActionSpec("workspace"),
    "research.discover": ActionSpec("research"),
    "browser.collect": ActionSpec("browser", ("research.discover",)),
    "research.extract": ActionSpec("research", ("browser.collect",)),
    "workspace.normalize": ActionSpec("workspace", ("research.extract",)),
    "workspace.build_excel": ActionSpec("workspace", ("workspace.normalize",)),
    "workspace.build_pptx": ActionSpec("workspace", ("workspace.normalize",)),
    "verify.deliverables": ActionSpec("verification", ("workspace.normalize",)),
    # test-paper workflow
    "test.generate": ActionSpec("research"),
    "test.solve_check": ActionSpec("verification", ("test.generate",)),
    "test.build_documents": ActionSpec("workspace", ("test.solve_check",)),
    "test.verify_documents": ActionSpec("verification", ("test.build_documents",)),
    "test.verify_set": ActionSpec("verification", ("test.verify_documents",)),
    "telegram.send": ActionSpec("messenger", ("test.verify_documents",)),
}


class PlanError(ValueError):
    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def find_cycle(tasks: list[TaskSpec]) -> list[str] | None:
    """Return a list of keys forming a cycle, or None (Kahn's algorithm)."""
    deps = {t.key: set(t.depends_on) & {x.key for x in tasks} for t in tasks}
    indeg = {k: len(v) for k, v in deps.items()}
    queue = [k for k, d in indeg.items() if d == 0]
    seen = 0
    while queue:
        k = queue.pop()
        seen += 1
        for other, d in deps.items():
            if k in d:
                indeg[other] -= 1
                if indeg[other] == 0:
                    queue.append(other)
    if seen == len(tasks):
        return None
    return sorted(k for k, d in indeg.items() if d > 0)


def ancestors(tasks: list[TaskSpec], key: str) -> set[str]:
    by_key = {t.key: t for t in tasks}
    out: set[str] = set()
    stack = list(by_key[key].depends_on)
    while stack:
        k = stack.pop()
        if k in out or k not in by_key:
            continue
        out.add(k)
        stack.extend(by_key[k].depends_on)
    return out


def get_spec(action: str) -> ActionSpec | None:
    if action.startswith("tool."):
        from .custom import tool_of

        t = tool_of(action)
        return ActionSpec(t.agent) if t else None
    return ACTIONS.get(action)


def validate_plan(tasks: list[TaskSpec]) -> None:
    """Raise PlanError listing every structural problem. Pure and deterministic."""
    problems: list[str] = []
    if not tasks:
        raise PlanError(["plan has no tasks"])
    if any(t.action.startswith("tool.") for t in tasks):
        from .custom import validate_custom_plan

        if not all(t.action.startswith("tool.") for t in tasks):
            raise PlanError(["a plan cannot mix general tools with built-in workflow actions"])
        return validate_custom_plan(tasks)
    keys = [t.key for t in tasks]
    dup = {k for k in keys if keys.count(k) > 1}
    if dup:
        problems.append(f"duplicate task keys: {sorted(dup)}")
    keyset = set(keys)
    for t in tasks:
        if t.key in t.depends_on:
            problems.append(f"{t.key} depends on itself")
        missing = [d for d in t.depends_on if d not in keyset]
        if missing:
            problems.append(f"{t.key} depends on unknown tasks {missing}")
        spec = ACTIONS.get(t.action)
        if spec is None:
            problems.append(f"{t.key}: action '{t.action}' is not in the allowlist")
        elif spec.agent != t.agent:
            problems.append(f"{t.key}: action '{t.action}' must be run by the {spec.agent} agent, not {t.agent}")
    if problems:
        raise PlanError(problems)
    cycle = find_cycle(tasks)
    if cycle:
        raise PlanError([f"circular dependency among tasks: {', '.join(cycle)}"])
    by_key = {t.key: t for t in tasks}
    for t in tasks:
        anc_actions = {by_key[a].action for a in ancestors(tasks, t.key)}
        for req in ACTIONS[t.action].requires:
            if req not in anc_actions:
                problems.append(f"{t.key} ({t.action}) must depend on a '{req}' task")
    if any(t.action.startswith("test.") for t in tasks):
        gens = sum(t.action == "test.generate" for t in tasks)
        if gens < 1:
            problems.append("a test plan needs at least one test.generate task")
        for act in ("test.solve_check", "test.build_documents", "test.verify_documents"):
            if sum(t.action == act for t in tasks) != gens:
                problems.append(f"every drafted test needs its own {act} task ({gens} expected)")
        if any(t.action == "telegram.send" for t in tasks) and not any(t.action == "test.verify_documents" for t in tasks):
            problems.append("posting requires document verification first")
        if problems:
            raise PlanError(problems)
        return
    extracts = [t for t in tasks if t.action == "research.extract"]
    collects = [t for t in tasks if t.action == "browser.collect"]
    for e in extracts:
        slot = e.params.get("slot")
        if not any(by_key[a].action == "browser.collect" and by_key[a].params.get("slot") == slot
                   for a in ancestors(tasks, e.key)):
            problems.append(f"{e.key}: extraction for slot {slot} must depend on its collection task")
    for n in (t for t in tasks if t.action == "workspace.normalize"):
        anc = ancestors(tasks, n.key)
        missing = [e.key for e in extracts if e.key not in anc]
        if missing:
            problems.append(f"{n.key}: normalisation must depend on all extraction tasks (missing {missing})")
    builds = [t for t in tasks if t.action in ("workspace.build_excel", "workspace.build_pptx")]
    if not builds:
        problems.append("plan produces no deliverable (needs an Excel or PowerPoint build task)")
    for v in (t for t in tasks if t.action == "verify.deliverables"):
        anc = ancestors(tasks, v.key)
        missing = [b.key for b in builds if b.key not in anc]
        if missing:
            problems.append(f"{v.key}: deliverable verification must depend on {missing}")
    if not any(t.action == "verify.deliverables" for t in tasks):
        problems.append("plan has no final deliverable verification task")
    if not extracts or not collects:
        problems.append("plan has no research tasks")
    if problems:
        raise PlanError(problems)


def build_plan(p: WorkflowParams) -> list[TaskSpec]:
    """Deterministic, dependency-aware plan for the validated parameters."""
    tasks: list[TaskSpec] = []
    n, ind = p.competitor_count, p.industry
    counter = iter(range(1, 1000))

    def add(**kw) -> TaskSpec:
        t = TaskSpec(key=f"T{next(counter)}", **kw)
        tasks.append(t)
        return t

    inputs = None
    if p.has_inputs:
        inputs = add(title="Read user-provided input documents", agent="workspace", action="file.read_inputs",
                     description="Extract text from the uploaded files (PDF/text/CSV/XLSX) so later steps can use them as context.",
                     expected_output="Extracted text excerpts per uploaded file.",
                     verification_criteria=["every uploaded file was read", "each file yielded readable text"])
    disc = add(title=f"Identify {n} competitors in {ind}", agent="research", action="research.discover",
               params={"industry": ind, "count": n},
               description=f"Search the web and select {n} real competitors in '{ind}' (plus backups for recovery).",
               depends_on=[inputs.key] if inputs else [],
               expected_output=f"At least {n} candidate competitors with name and official website.",
               verification_criteria=[f"at least {n} candidates", "each candidate has a valid public https URL",
                                      "candidate domains are unique"])
    collects, extracts = [], []
    for i in range(n):
        c = add(title=f"Collect pages for competitor #{i + 1}", agent="browser", action="browser.collect",
                params={"slot": i, "industry": ind}, depends_on=[disc.key],
                description="Fetch the homepage, locate and fetch the pricing page, and fetch one independent source for cross-checking.",
                expected_output="Page text, title, URL and retrieval timestamp for 2-3 pages.",
                verification_criteria=["homepage was retrieved with readable text", "every page has a URL and timestamp"])
        e = add(title=f"Extract facts for competitor #{i + 1}", agent="research", action="research.extract",
                params={"slot": i, "industry": ind}, depends_on=[c.key],
                description="Extract product, features and pricing; keep only claims whose evidence quote appears verbatim in the fetched pages.",
                expected_output="Evidence-backed features and pricing status (paid/free/contact sales/not found).",
                verification_criteria=["at least one feature with verbatim evidence", "pricing status is determined",
                                       "paid plans have a price present in the evidence", "unavailable pricing is not recorded as 0"])
        collects.append(c)
        extracts.append(e)
    norm = add(title="Normalise findings into a consistent schema", agent="workspace", action="workspace.normalize",
               params={"count": n}, depends_on=[e.key for e in extracts],
               description="Merge all competitor facts into one validated schema and compute summary statistics.",
               expected_output=f"{n} validated competitor records.",
               verification_criteria=[f"exactly {n} unique records", "every record validates against the schema",
                                      "every feature and price links to a stored evidence record"])
    builds = []
    if "xlsx" in p.outputs:
        builds.append(add(title="Generate Excel comparison workbook", agent="workspace", action="workspace.build_excel",
                          params={"filename": EXCEL_NAME, "export_name": p.export_name}, depends_on=[norm.key],
                          resources=[f"file:{EXCEL_NAME}"],
                          description="Create competitors.xlsx with Summary, Comparison and Sources sheets.",
                          expected_output=EXCEL_NAME,
                          verification_criteria=["file exists and reopens", "required sheets present", f"{n} comparison rows",
                                                 "source URLs match stored evidence"]))
    if "pptx" in p.outputs:
        builds.append(add(title="Generate PowerPoint summary", agent="workspace", action="workspace.build_pptx",
                          params={"filename": PPTX_NAME, "export_name": p.export_name}, depends_on=[norm.key],
                          resources=[f"file:{PPTX_NAME}"],
                          description="Create competitors.pptx: title, methodology, comparison, key findings, limitations, references.",
                          expected_output=PPTX_NAME,
                          verification_criteria=["file exists and reopens", "required slides present", "references list every source"]))
    add(title="Independently verify deliverables against evidence", agent="verification", action="verify.deliverables",
        params={"count": n}, depends_on=[b.key for b in builds],
        description="Reopen each generated file and check structure, completeness and agreement with stored evidence.",
        expected_output="Pass/fail result for each deliverable check.",
        verification_criteria=["all deliverable checks pass"])
    validate_plan(tasks)
    return tasks
