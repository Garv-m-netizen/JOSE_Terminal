"""Messenger Agent: posts the student question paper(s) to Telegram. Only ever runs after explicit human approval."""
from __future__ import annotations

from ..schemas.domain import ExamPaper, ExamParams
from ..services import artifact_service
from ..tools.telegram import TelegramError, mask_chat
from .base import AgentContext, TaskError


def _papers(ctx: AgentContext) -> list[tuple[int, ExamPaper, ExamParams, dict, dict]]:
    gens = {o.get("paper_index", 1): o for _, o in ctx.upstream_any("test.generate")}
    builds = {o.get("paper_index", 1): o for _, o in ctx.upstream_any("test.build_documents")}
    out = []
    for i in sorted(set(gens) & set(builds)):
        out.append((i, ExamPaper.model_validate(gens[i]["paper"]), ExamParams.model_validate(gens[i]["params"]), gens[i], builds[i]))
    return out


def _caption(ctx: AgentContext, i: int, paper: ExamPaper, p: ExamParams, g: dict) -> str:
    return (f"{paper.title}\n{p.subject.title()} | Class {p.grade}" + (f" | {', '.join(p.topics)}" if p.topics else "")
            + f"\n{len(paper.questions)} questions | {g['total_marks']} marks | {p.duration_minutes} minutes"
            + ("\n\n[SIMULATED SAMPLE - demo mode]" if ctx.simulated else ""))


def preview(ctx: AgentContext) -> dict:
    """What would be sent. Used both for the approval card and for the actual send (single source of truth)."""
    papers = _papers(ctx)
    if not papers:
        return {}
    first = papers[0]
    cap = _caption(ctx, first[0], first[1], first[2], first[3])
    if len(papers) > 1:
        cap = f"{len(papers)} tests, each: " + cap.split("\n", 1)[1].replace("\n\n[SIMULATED", "\n[SIMULATED")
    return {"file": ", ".join(b["paper_file"] for *_, b in papers), "artifact_ids": [b["paper_artifact_id"] for *_, b in papers], "caption": cap,
            "answer_key_excluded": ", ".join(b["key_file"] for *_, b in papers)}


async def send(ctx: AgentContext) -> dict:
    papers = _papers(ctx)
    if not papers:
        raise TaskError("nothing to send: the question papers have not been built", "validation", retryable=False)
    prov = ctx.providers
    dry = bool(ctx.task.params.get("dry_run")) or ctx.simulated
    messenger = prov.messenger
    if messenger is None or (not dry and getattr(messenger, "simulated", False)):
        raise TaskError(prov.extra.get("messenger_error") or "Telegram is not configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)", "config", retryable=False)
    sent, returned, first = [], [], None
    async with ctx.activity("posting the question paper(s) to Telegram" if not dry else "simulating the Telegram post (nothing is sent)"):
        for i, paper, p, g, b in papers:
            path, art = artifact_service.resolve_download(b["paper_artifact_id"])
            if art.kind != "pdf" or "key" in art.filename.lower():
                raise TaskError("refusing to send: the selected file is not a student question paper", "validation", retryable=False)
            try:
                res = await ctx.tool("telegram.sendDocument", {"chat": mask_chat(messenger.chat_id), "file": art.filename, "dry_run": dry},
                                     lambda path=path, i=i, paper=paper, p=p, g=g: messenger.send_document(path, _caption(ctx, i, paper, p, g)),
                                     lambda r: f"message {r.get('message_id')} delivered" if not dry else "simulated - nothing sent", simulated=dry)
            except TelegramError as e:
                raise TaskError(str(e), "config" if "TELEGRAM_" in str(e) else "network", retryable=False) from e
            first = first or res
            sent.append(art.filename)
            returned.append((res.get("document") or {}).get("file_name"))
    return {"delivered": True, "simulated": dry, "chat": mask_chat(messenger.chat_id), "message_id": (first or {}).get("message_id"),
            "chat_id_returned": str(((first or {}).get("chat") or {}).get("id", "")), "file_sent": ", ".join(sent), "file_returned": ", ".join(str(x) for x in returned),
            "answer_key_sent": False, "caption": preview(ctx).get("caption", ""), "count": len(sent)}
