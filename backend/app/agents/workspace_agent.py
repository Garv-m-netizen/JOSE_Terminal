"""Workspace Agent: reads input documents, normalises research into a schema and writes Excel/PowerPoint files."""
from __future__ import annotations

import asyncio

from sqlalchemy import select

from ..database import session_scope
from ..models import Evidence, Upload
from ..schemas.domain import CompetitorRecord
from ..services import artifact_service
from ..tools import excel_tools, powerpoint_tools
from ..tools.file_tools import MAX_EXCERPT_CHARS, FileRejected, extract_text
from ..tools.report_stats import compute_stats
from ..tools.safety import UnsafePath, safe_join
from .base import AgentContext, TaskError


def evidence_rows(run_id: str) -> list[dict]:
    with session_scope() as s:
        rows = s.scalars(select(Evidence).where(Evidence.run_id == run_id).order_by(Evidence.subject, Evidence.claim_type)).all()
        return [{"id": e.id, "task_id": e.task_id, "subject": e.subject, "claim_type": e.claim_type, "claim": e.claim,
                 "source_url": e.source_url, "page_title": e.page_title,
                 "retrieved_at": e.retrieved_at.strftime("%Y-%m-%d %H:%M"), "evidence_text": e.evidence_text,
                 "status": e.status} for e in rows]


def _valid_evidence(run_id: str, records: list[CompetitorRecord]) -> list[dict]:
    """Only evidence belonging to the final (current) records - discarded retry attempts are excluded."""
    wanted = {f.evidence_id for r in records for f in r.features} | {p.evidence_id for r in records for p in r.plans}
    return [e for e in evidence_rows(run_id) if e["id"] in wanted or e["claim_type"] == "summary" and e["subject"] in {r.name for r in records}]


async def read_inputs(ctx: AgentContext) -> dict:
    from ..config import get_settings

    ids = ctx.run_params.get("_upload_ids") or []
    if not ids:
        raise TaskError("no uploaded files are attached to this run", "validation", retryable=False)
    files = []
    async with ctx.activity("reading uploaded documents"):
        for uid in ids:
            with session_scope() as s:
                up = s.get(Upload, uid)
                if not up:
                    raise TaskError(f"uploaded file {uid} not found", "file", retryable=False)
                stored, name = up.stored_name, up.filename
            try:
                path = safe_join(get_settings().upload_path, stored)
                data = await ctx.tool("file.read", {"file": name}, lambda p=path: asyncio.to_thread(extract_text, p),
                                      lambda r: f"{r['kind']}, {len(r['text'])} chars", simulated=False)
            except (FileRejected, UnsafePath) as e:
                raise TaskError(str(e), "file", retryable=False) from e
            files.append({"name": name, "kind": data["kind"], "chars": len(data["text"]),
                          "excerpt": data["text"][:MAX_EXCERPT_CHARS], "needs_ocr": bool(data.get("needs_ocr"))})
    return {"files": files}


async def normalize(ctx: AgentContext) -> dict:
    n = int(ctx.task.params["count"])
    records: list[CompetitorRecord] = []
    for t, out in ctx.upstream("research.extract"):
        records.append(CompetitorRecord.model_validate(out["record"]))
    if len(records) != n:
        raise TaskError(f"expected {n} competitor records, found {len(records)}", "validation")
    seen: dict[str, str] = {}
    for r in records:
        host = r.website.split("//")[-1].removeprefix("www.").lower()
        if host in seen:
            raise TaskError(f"duplicate competitor: {r.name} and {seen[host]} share {host}", "duplicate")
        seen[host] = r.name
    records.sort(key=lambda r: r.name.lower())
    async with ctx.activity("normalising competitor records"):
        return {"records": [r.model_dump(mode="json") for r in records], "stats": compute_stats(records),
                "simulated": ctx.simulated}


def _records(ctx: AgentContext) -> list[CompetitorRecord]:
    up = ctx.upstream("workspace.normalize")
    if not up:
        raise TaskError("normalised data is missing", "validation", retryable=False)
    return [CompetitorRecord.model_validate(r) for r in up[0][1]["records"]]


def _meta(ctx: AgentContext) -> dict:
    inputs = [f["name"] for _, o in ctx.upstream_any("file.read_inputs") for f in o.get("files", [])]
    return {"simulated": ctx.simulated, "industry": ctx.run_params.get("industry", ""), "input_files": inputs}


async def _build(ctx: AgentContext, kind: str) -> dict:
    records = _records(ctx)
    filename = ctx.task.params["filename"]
    path = artifact_service.run_dir(ctx.run_id) / filename
    meta = _meta(ctx)
    async with ctx.activity(f"writing {filename}"):
        if kind == "xlsx":
            ev = _valid_evidence(ctx.run_id, records)
            await ctx.tool("excel.write", {"file": filename, "rows": len(records)},
                           lambda: asyncio.to_thread(excel_tools.build_workbook, path, records, ev, meta),
                           lambda _: f"{filename} written", simulated=False)
        else:
            await ctx.tool("pptx.write", {"file": filename, "competitors": len(records)},
                           lambda: asyncio.to_thread(powerpoint_tools.build_presentation, path, records, meta),
                           lambda _: f"{filename} written", simulated=False)
        if not path.is_file():
            raise TaskError(f"{filename} was not created", "file")
        art = artifact_service.register(ctx.run_id, ctx.task.id, path, kind, ctx.simulated)
        exported = None
        if ctx.task.params.get("export_name"):
            dst = artifact_service.export_copy(path, ctx.task.params["export_name"], filename)
            exported = str(dst.relative_to(ctx.settings.output_path))
    return {"artifact_id": art["id"], "filename": filename, "size_bytes": art["size_bytes"], "exported_to": exported}


async def build_excel(ctx: AgentContext) -> dict:
    return await _build(ctx, "xlsx")


async def build_pptx(ctx: AgentContext) -> dict:
    return await _build(ctx, "pptx")
