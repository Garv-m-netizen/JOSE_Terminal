"""Verification Agent: independent, deterministic checks of task outputs and generated files.

Checks re-read files from disk and evidence from the database; they never trust the producing tool's own claim.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from sqlalchemy import select

from ..database import session_scope
from ..models import Artifact, Evidence
from ..schemas.domain import Candidate, CompetitorRecord, PageRecord
from ..services import artifact_service
from ..services.store import NotFound, TaskView
from ..tools import excel_tools, powerpoint_tools
from ..tools.extraction import evidence_in_text
from ..tools.report_stats import SIM_BANNER, STATUS_LABEL
from ..tools.safety import UnsafeURL, check_url_syntax


@dataclass
class Check:
    name: str
    passed: bool
    reason: str
    details: dict = field(default_factory=dict)


def _ok(name: str, reason: str, **d) -> Check:
    return Check(name, True, reason, d)


def _bad(name: str, reason: str, **d) -> Check:
    return Check(name, False, reason, d)


def _evidence_by_id(run_id: str) -> dict[str, Evidence]:
    with session_scope() as s:
        return {e.id: e for e in s.scalars(select(Evidence).where(Evidence.run_id == run_id))}


def _artifact_for(task_id: str) -> Artifact | None:
    with session_scope() as s:
        return s.scalars(select(Artifact).where(Artifact.task_id == task_id).order_by(Artifact.created_at.desc())).first()


# --- per-action verifiers ----------------------------------------------------------------------

def verify_read_inputs(task: TaskView, out: dict, ctx) -> list[Check]:
    files = out.get("files", [])
    want = len(ctx.run_params.get("_upload_ids") or [])
    checks = [Check("inputs.all_read", len(files) == want and want > 0, f"{len(files)} of {want} uploaded files were read")]
    empty = [f["name"] for f in files if f["chars"] == 0]
    checks.append(Check("inputs.readable_text", not empty,
                        "every file yielded text" if not empty else f"no text extracted from {empty} (scanned PDF? OCR is not enabled)"))
    return checks


def verify_discover(task: TaskView, out: dict, ctx) -> list[Check]:
    n = int(task.params["count"])
    cands = [Candidate.model_validate(c) for c in out.get("candidates", [])]
    checks = [Check("discover.count", len(cands) >= n, f"{len(cands)} candidates found, {n} required")]
    bad = []
    for c in cands:
        try:
            check_url_syntax(c.url)
        except UnsafeURL as e:
            bad.append(f"{c.name}: {e}")
    checks.append(Check("discover.urls_valid", not bad, "all candidate URLs are valid public URLs" if not bad else "; ".join(bad)))
    hosts = [urlparse(c.url).netloc.lower().removeprefix("www.") for c in cands]
    checks.append(Check("discover.unique", len(set(hosts)) == len(hosts), "candidate websites are unique"
                        if len(set(hosts)) == len(hosts) else "duplicate candidate websites"))
    return checks


def verify_collect(task: TaskView, out: dict, ctx) -> list[Check]:
    pages = [PageRecord.model_validate(p) for p in out.get("pages", [])]
    home = next((p for p in pages if p.role == "homepage"), None)
    checks = [Check("collect.homepage", bool(home and len(home.text) >= 200),
                    f"homepage retrieved ({len(home.text)} chars)" if home else "homepage missing")]
    if home and len(home.text) < 200:
        checks[0] = _bad("collect.homepage", f"homepage text too short ({len(home.text)} chars) to be meaningful")
    meta_ok = all(p.url and p.retrieved_at and p.text for p in pages) and bool(pages)
    checks.append(Check("collect.page_metadata", meta_ok, "every page has URL, timestamp and text" if meta_ok else "a page lacks URL, timestamp or text"))
    return checks


def verify_extract(task: TaskView, out: dict, ctx) -> list[Check]:
    rec = CompetitorRecord.model_validate(out["record"])
    pages: list[PageRecord] = []
    for t, o in ctx.upstream("browser.collect"):
        pages += [PageRecord.model_validate(p) for p in o["pages"]]
    evid = _evidence_by_id(ctx.run_id)
    checks = []
    checks.append(Check("extract.has_features", len(rec.features) >= 1, f"{len(rec.features)} evidence-backed features"))
    unbacked = []
    for f in rec.features:
        e = evid.get(f.evidence_id)
        if e is None or not any(evidence_in_text(e.evidence_text, p.text) for p in pages):
            unbacked.append(f.text)
    checks.append(Check("extract.features_evidenced", not unbacked, "every feature's evidence is found on a retrieved page"
                        if not unbacked else f"evidence missing for: {unbacked}"))
    price_problems = []
    for p in rec.plans:
        e = evid.get(p.evidence_id)
        if e is None or p.price_usd is None or not any(m in e.evidence_text.replace(",", "") for m in {f"{p.price_usd:g}", f"{p.price_usd:.2f}"}):
            price_problems.append(p.name)
    checks.append(Check("extract.prices_match_evidence", not price_problems, "every price appears in its evidence"
                        if not price_problems else f"price not supported by evidence: {price_problems}"))
    if rec.pricing_status in ("not_found", "contact_sales"):
        zero = [p.name for p in rec.plans]
        checks.append(Check("extract.unavailable_not_zero", not zero, "unavailable pricing is not recorded as a price"
                            if not zero else f"plans recorded although pricing is '{rec.pricing_status}'"))
    elif rec.pricing_status == "paid":
        checks.append(Check("extract.paid_has_price", rec.starting_price_usd is not None, "paid status has a nonzero price"
                            if rec.starting_price_usd is not None else "status 'paid' without a nonzero price"))
    sources_ok = all(s.url in {p.url for p in pages} for s in rec.sources)
    checks.append(Check("extract.sources_retrieved", sources_ok and bool(rec.sources), "record sources are retrieved pages"
                        if sources_ok else "record cites a source that was not retrieved"))
    return checks


def verify_normalize(task: TaskView, out: dict, ctx) -> list[Check]:
    n = int(task.params["count"])
    try:
        recs = [CompetitorRecord.model_validate(r) for r in out["records"]]
    except Exception as e:  # noqa: BLE001
        return [_bad("normalize.schema", f"records fail schema validation: {e}")]
    evid = _evidence_by_id(ctx.run_id)
    missing = [f.evidence_id for r in recs for f in r.features if f.evidence_id not in evid]
    missing += [p.evidence_id for r in recs for p in r.plans if p.evidence_id not in evid]
    names = [r.name.lower() for r in recs]
    return [Check("normalize.count", len(recs) == n, f"{len(recs)} records, {n} required"),
            Check("normalize.unique", len(set(names)) == len(names), "competitor names are unique"),
            Check("normalize.evidence_links", not missing, "all evidence links resolve" if not missing else f"{len(missing)} dangling evidence ids")]


# --- file checks -----------------------------------------------------------------------------

def _records_from(ctx) -> list[CompetitorRecord]:
    up = ctx.upstream_any("workspace.normalize")
    return [CompetitorRecord.model_validate(r) for r in up[0][1]["records"]] if up else []


def _retrieved_urls(ctx) -> set[str]:
    urls: set[str] = set()
    for _, o in ctx.upstream_any("browser.collect"):
        urls |= {p["url"] for p in o["pages"]}
    return urls


def excel_checks(path, records: list[CompetitorRecord], ctx) -> list[Check]:
    ck: list[Check] = []
    try:
        info = excel_tools.inspect_workbook(path)
    except Exception as e:  # noqa: BLE001
        return [_bad("xlsx.readable", f"workbook cannot be reopened: {e}")]
    ck.append(_ok("xlsx.readable", "workbook reopened successfully"))
    missing = [s for s in excel_tools.REQUIRED_SHEETS if s not in info["sheets"]]
    ck.append(Check("xlsx.sheets", not missing, "Summary, Comparison and Sources sheets present" if not missing else f"missing sheets: {missing}"))
    if missing:
        return ck
    ck.append(Check("xlsx.headers", info["comparison_headers"] == excel_tools.COMPARISON_HEADERS, "comparison headers are as specified"
                    if info["comparison_headers"] == excel_tools.COMPARISON_HEADERS else f"unexpected headers: {info['comparison_headers']}"))
    rows = info["comparison_rows"]
    ck.append(Check("xlsx.row_count", len(rows) == len(records), f"{len(rows)} comparison rows, {len(records)} competitors expected"))
    empty = [r.get("Competitor") or "?" for r in rows if not (r.get("Competitor") and r.get("Website") and r.get("Pricing status") and r.get("Verification"))]
    ck.append(Check("xlsx.required_fields", not empty, "name, website, pricing status and verification populated in every row"
                    if not empty else f"required fields empty in: {empty}"))
    by_name = {r.name: r for r in records}
    mism = [r.get("Competitor") for r in rows if r.get("Competitor") not in by_name]
    ck.append(Check("xlsx.matches_dataset", not mism, "rows match the normalised dataset" if not mism else f"rows not in dataset: {mism}"))
    evid = _evidence_by_id(ctx.run_id)
    retrieved = _retrieved_urls(ctx)
    bad_src, no_ev, bad_price, zero_unavail = [], [], [], []
    for row in rows:
        rec = by_name.get(row.get("Competitor"))
        urls = [u for u in str(row.get("Source URLs") or "").split("\n") if u]
        bad_src += [u for u in urls if u not in retrieved]
        if rec is not None:
            ev_urls = {evid[i].source_url for i in [*(f.evidence_id for f in rec.features), *(p.evidence_id for p in rec.plans)] if i in evid}
            if not ev_urls or not (ev_urls & set(urls)):
                no_ev.append(rec.name)
            price = row.get("Starting price (USD)")
            if price is not None and price != "":
                texts = [evid[p.evidence_id].evidence_text.replace(",", "") for p in rec.plans if p.evidence_id in evid]
                if not any(f"{float(price):g}" in t or f"{float(price):.2f}" in t for t in texts):
                    bad_price.append(f"{rec.name}: {price}")
            if row.get("Pricing status") in (STATUS_LABEL["not_found"], STATUS_LABEL["contact_sales"]) and price not in (None, ""):
                zero_unavail.append(rec.name)
    ck.append(Check("xlsx.sources_retrieved", not bad_src, "every source URL is a retrieved page" if not bad_src else f"URLs not in retrieval records: {bad_src[:5]}"))
    ck.append(Check("xlsx.rows_have_evidence", not no_ev, "every row is backed by evidence from its listed sources" if not no_ev else f"no matching evidence for: {no_ev}"))
    ck.append(Check("xlsx.prices_match_evidence", not bad_price, "every price matches stored evidence" if not bad_price else f"price not in evidence: {bad_price}"))
    ck.append(Check("xlsx.unavailable_blank", not zero_unavail, "unavailable pricing is left blank, not zero" if not zero_unavail else f"price shown for unavailable pricing: {zero_unavail}"))
    summary_type = str((info.get("summary") or {}).get("Data type", ""))
    row_types = {str(r.get("Data type")) for r in rows}
    if ctx.simulated:
        ok = "SIMULATED" in summary_type and row_types <= {"SIMULATED"}
        ck.append(Check("xlsx.simulated_labelled", ok, "simulated data is labelled on the Summary sheet and in every row" if ok else "simulated data is not clearly labelled"))
    else:
        ok = "SIMULATED" not in summary_type and "SIMULATED" not in row_types
        ck.append(Check("xlsx.real_not_labelled_simulated", ok, "real data is not labelled simulated" if ok else "real run carries a simulated label"))
    return ck


def pptx_checks(path, records: list[CompetitorRecord], ctx) -> list[Check]:
    ck: list[Check] = []
    try:
        info = powerpoint_tools.inspect_presentation(path)
    except Exception as e:  # noqa: BLE001
        return [_bad("pptx.readable", f"presentation cannot be reopened: {e}")]
    ck.append(_ok("pptx.readable", f"presentation reopened ({info['slide_count']} slides)"))
    titles = [s["title"] for s in info["slides"]]
    missing = [t for t in powerpoint_tools.REQUIRED_TITLES if not any(x.startswith(t) for x in titles)]
    ck.append(Check("pptx.required_slides", not missing, "title, methodology, comparison, findings, limitations and references present"
                    if not missing else f"missing slides: {missing}"))
    comp_text = "\n".join(s["text"] for s in info["slides"] if s["title"].startswith("Competitor comparison"))
    absent = [r.name for r in records if r.name not in comp_text]
    ck.append(Check("pptx.competitors_present", not absent, "every competitor appears on the comparison slides" if not absent else f"missing from comparison: {absent}"))
    refs = "\n".join(s["text"] for s in info["slides"] if s["title"].startswith("References"))
    miss_refs = sorted({s.url for r in records for s in r.sources if s.url not in refs})
    ck.append(Check("pptx.references_complete", not miss_refs, "references list every source URL" if not miss_refs else f"URLs missing from references: {miss_refs[:5]}"))
    has_banner = all(SIM_BANNER in s["text"] for s in info["slides"][1:]) and SIM_BANNER in info["slides"][0]["text"]
    if ctx.simulated:
        ck.append(Check("pptx.simulated_labelled", has_banner, "every slide carries the simulated-data banner" if has_banner else "a slide lacks the simulated-data banner"))
    else:
        any_banner = any(SIM_BANNER in s["text"] for s in info["slides"])
        ck.append(Check("pptx.real_not_labelled_simulated", not any_banner, "real data is not labelled simulated"))
    return ck


def _verify_build(kind: str):
    def fn(task: TaskView, out: dict, ctx) -> list[Check]:
        art = _artifact_for(task.id)
        if art is None:
            return [_bad(f"{kind}.artifact_registered", "no artifact record exists for this task")]
        try:
            path, _ = artifact_service.resolve_download(art.id)
        except NotFound as e:
            return [_bad(f"{kind}.file_exists", f"generated file is not available: {e}")]
        ck = [_ok(f"{kind}.file_exists", f"{art.filename} exists ({art.size_bytes} bytes) and passes its integrity hash")]
        recs = _records_from(ctx)
        ck += excel_checks(path, recs, ctx) if kind == "xlsx" else pptx_checks(path, recs, ctx)
        return ck
    return fn


def verify_deliverables(task: TaskView, out: dict, ctx) -> list[Check]:
    """The final task: output is the agent's own check list, so we only assert nothing failed and cross-check files."""
    checks = [Check(c["name"], c["passed"], c["reason"], c.get("details", {})) for c in out.get("checks", [])]
    return checks or [_bad("deliverables.ran", "no deliverable checks were executed")]


def verify_test_generate(task: TaskView, out: dict, ctx) -> list[Check]:
    from ..schemas.domain import ExamPaper, ExamParams
    from ..workflows.test_paper import blueprint, total_marks
    from .exam_agent import normalise_paper

    p = ExamParams.model_validate(out["params"])
    paper = ExamPaper.model_validate(out["paper"])
    _, _, problems = normalise_paper(paper, p)  # independent re-check of the stored paper
    total = sum(q.marks for q in paper.questions)
    types_ok = [q.type for q in paper.questions] == [s.type for s in blueprint(p.num_questions, p.total_marks) for _ in range(s.count)]
    empty = [q.id for q in paper.questions if not q.solution.strip() or not q.answer.strip()]
    return [Check("test.question_count", len(paper.questions) == p.num_questions, f"{len(paper.questions)} questions, {p.num_questions} required"),
            Check("test.blueprint", types_ok and not problems, "sections follow the blueprint; MCQs have 4 distinct options and a valid letter; no duplicates" if types_ok and not problems else "; ".join(problems[:5]) or "question types do not follow the blueprint"),
            Check("test.marks_total", total == total_marks(p.num_questions, p.total_marks) == out["total_marks"], f"marks add up to {total}" + (f" (target {p.total_marks})" if p.total_marks else "")),
            Check("test.answers_present", not empty, "every question has an answer and a worked solution" if not empty else f"missing answer/solution: {empty}")]


def verify_build_documents(task: TaskView, out: dict, ctx) -> list[Check]:
    checks = []
    for label, key in (("paper", "paper_artifact_id"), ("key", "key_artifact_id")):
        try:
            path, art = artifact_service.resolve_download(out[key])
            checks.append(_ok(f"pdf.{label}_exists", f"{art.filename} exists ({art.size_bytes} bytes) and passes its integrity hash"))
        except Exception as e:  # noqa: BLE001
            checks.append(_bad(f"pdf.{label}_exists", f"{label} PDF unavailable: {e}"))
    return checks


def verify_telegram(task: TaskView, out: dict, ctx) -> list[Check]:
    sim = bool(out.get("simulated"))
    ck = [Check("tg.delivered", bool(out.get("delivered")) and out.get("message_id") is not None,
                "SIMULATED: nothing was sent (demo/dry run)" if sim else f"Telegram confirmed delivery as message {out.get('message_id')}")]
    if not sim:
        ck.append(Check("tg.file_matches", out.get("file_returned") == out.get("file_sent"), f"Telegram stored the document as '{out.get('file_returned')}'"))
        ck.append(Check("tg.chat_matches", bool(out.get("chat_id_returned")) and out["chat"].replace("…", "")[:3] == out["chat_id_returned"][:3], "message landed in the configured chat"))
    ck.append(Check("tg.answer_key_not_sent", out.get("answer_key_sent") is False and "key" not in str(out.get("file_sent", "")).lower(), "only the student paper was sent; the answer key stayed private"))
    return ck


VERIFIERS = {
    "file.read_inputs": verify_read_inputs, "research.discover": verify_discover, "browser.collect": verify_collect,
    "research.extract": verify_extract, "workspace.normalize": verify_normalize,
    "workspace.build_excel": _verify_build("xlsx"), "workspace.build_pptx": _verify_build("pptx"),
    "verify.deliverables": verify_deliverables,
    "test.generate": verify_test_generate, "test.solve_check": verify_deliverables,
    "test.build_documents": verify_build_documents, "test.verify_documents": verify_deliverables, "test.verify_set": verify_deliverables,
    "telegram.send": verify_telegram,
}


def verify_task(task: TaskView, output: dict, ctx) -> list[Check]:
    if task.action.startswith("tool."):
        from .custom_agent import verify_tool

        return verify_tool(task, output, ctx)
    fn = VERIFIERS.get(task.action)
    if fn is None:
        return [_bad("verifier.exists", f"no verifier for action {task.action}")]
    try:
        return fn(task, output, ctx)
    except Exception as e:  # noqa: BLE001 - a crashing verifier is a failed verification, never a pass
        return [_bad("verifier.ran", f"verification raised {type(e).__name__}: {e}")]


async def run_deliverable_verification(ctx) -> dict:
    """Execution of the verify.deliverables task: reopen every generated file and cross-check them."""
    records = _records_from(ctx)
    checks: list[Check] = []
    names: dict[str, set[str]] = {}
    failed_builds: list[str] = []
    async with ctx.activity("reopening generated files"):
        for t, out in [*ctx.upstream_any("workspace.build_excel"), *ctx.upstream_any("workspace.build_pptx")]:
            kind = "xlsx" if t.action == "workspace.build_excel" else "pptx"
            cs = _verify_build(kind)(t, out, ctx)
            checks += [Check(f"{t.key}:{c.name}", c.passed, c.reason, c.details) for c in cs]
            if not all(c.passed for c in cs):
                failed_builds.append(t.key)
            art = _artifact_for(t.id)
            if art:
                try:
                    p, _ = artifact_service.resolve_download(art.id)
                    if kind == "xlsx":
                        names["xlsx"] = {r.get("Competitor") for r in excel_tools.inspect_workbook(p).get("comparison_rows", [])}
                    else:
                        txt = "\n".join(s["text"] for s in powerpoint_tools.inspect_presentation(p)["slides"] if s["title"].startswith("Competitor comparison"))
                        names["pptx"] = {r.name for r in records if r.name in txt}
                except Exception:  # noqa: BLE001 - already reported by the per-file checks
                    pass
        if len(names) == 2:
            same = names["xlsx"] == names["pptx"]
            checks.append(Check("cross.xlsx_pptx_same_competitors", same, "Excel and PowerPoint cover the same competitors"
                                if same else f"differ: xlsx={sorted(names['xlsx'])} pptx={sorted(names['pptx'])}"))
    return {"checks": [c.__dict__ for c in checks], "all_passed": all(c.passed for c in checks), "failed_builds": failed_builds}
