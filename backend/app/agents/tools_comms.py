"""Email tools: email.draft (no side effects) and email.send (HIGH risk, approved exact draft, ledger-guarded).

The result of email.send is "accepted by the mail server" at best. It never claims inbox delivery.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from ..config import get_settings
from ..services import artifact_service, ledger, store
from ..services.capabilities import missing_permissions
from ..tools.email_provider import (MAX_ATTACH_BYTES, EmailAmbiguous, EmailError, EmailNotSent, OutgoingEmail, get_email_provider,
                                    normalize_address)
from ..tools.file_tools import sha256_file
from ..workflows.custom import resolve_args
from .base import AgentContext, TaskError
from .verification_agent import Check

ART_RE = re.compile(r"^art_[0-9a-f]{12}$")
CTRL_RE = re.compile(r"[\r\n\x00]")


def _ck(name: str, ok: bool, reason: str) -> Check:
    return Check(name, bool(ok), reason)


def _text(v) -> str:
    if isinstance(v, list):
        return "\n\n".join(map(str, v))
    if isinstance(v, dict):
        return str(v.get("text", v))
    return str(v)


def draft_hash(d: dict) -> str:
    blob = json.dumps({k: d[k] for k in ("from", "to", "cc", "subject", "body")} | {"attachments": [(x["filename"], x["sha256"]) for x in d["attachments"]]},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _flatten_ids(v) -> list[str]:
    if v is None or v == "":
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, dict):
        return [v["artifact_id"]] if "artifact_id" in v else _flatten_ids(v.get("files", []))
    out: list[str] = []
    for x in v if isinstance(v, list) else []:
        out += _flatten_ids(x)
    return out


def _allowed_extra() -> list[str]:
    return [x.strip().lower() for x in get_settings().email_allowed_recipients.split(",") if x.strip()]


def _provenance(addr: str, objective: str) -> str | None:
    """Why this address may be used: it was written by the user, is the configured default, or is explicitly allow-listed."""
    s = get_settings()
    if addr in objective.lower():
        return "written in the objective"
    if s.email_default_recipient.strip() and addr == s.email_default_recipient.strip().lower():
        return "the configured default recipient"
    for entry in _allowed_extra():
        if entry == addr or (entry.startswith("@") and addr.endswith(entry)):
            return "allow-listed in EMAIL_ALLOWED_RECIPIENTS"
    return None


def _resolve_recipient(raw: str, objective: str) -> tuple[str, str]:
    s = get_settings()
    if raw.strip().lower() == "configured":
        default = s.email_default_recipient.strip()
        if not default:
            raise TaskError("the objective refers to the configured recipient, but EMAIL_DEFAULT_RECIPIENT is not set. Tell me the address to use.", "needs_input", retryable=False)
        return normalize_address(default), "the configured default recipient"
    try:
        addr = normalize_address(raw)
    except ValueError as e:
        raise TaskError(f"{e}. Which address should I use?", "needs_input", retryable=False) from e
    why = _provenance(addr, objective)
    if why is None:
        raise TaskError(f"the recipient {addr} was not given by you (it is not in the objective, EMAIL_DEFAULT_RECIPIENT or EMAIL_ALLOWED_RECIPIENTS). "
                        "JOSE never invents addresses. Tell me which address to use.", "needs_input", retryable=False)
    return addr, why


def _attachment(ctx: AgentContext, ref: str) -> tuple[dict, Path]:
    from .custom_agent import find_upload

    if ART_RE.match(ref):
        try:
            path, art = artifact_service.resolve_download(ref)
        except Exception as e:  # noqa: BLE001
            raise TaskError(f"attachment {ref} is not available: {e}", "validation", retryable=False) from e
        if art.run_id != ctx.run_id or art.kind == "screenshot":
            raise TaskError(f"refusing to attach {art.filename}: it is not a deliverable of this run (use fs.import to bring in an approved file)", "validation", retryable=False)
        return {"kind": "artifact", "artifact_id": art.id, "filename": art.filename, "size": path.stat().st_size, "sha256": art.sha256}, path
    path, shown = find_upload(ctx, ref)
    return {"kind": "upload", "artifact_id": None, "filename": shown, "size": path.stat().st_size, "sha256": sha256_file(path)}, path


def _load_attachments(ctx: AgentContext, refs: list[str]) -> tuple[list[dict], list[Path]]:
    infos, paths, total = [], [], 0
    for ref in refs:
        info, path = _attachment(ctx, ref)
        total += info["size"]
        infos.append(info)
        paths.append(path)
    if total > MAX_ATTACH_BYTES:
        raise TaskError(f"attachments total {total // (1024 * 1024)} MB, over the {MAX_ATTACH_BYTES // (1024 * 1024)} MB limit", "validation", retryable=False)
    if len({i["filename"].lower() for i in infos}) != len(infos):
        raise TaskError("two attachments have the same file name", "validation", retryable=False)
    return infos, paths


# ------------------------------------------------------------------ email.draft

async def t_email_draft(ctx: AgentContext, a) -> dict:
    miss = missing_permissions(("email.send",)) if not ctx.simulated else []
    if miss:
        raise TaskError("; ".join(miss), "permission", retryable=False)
    objective = store.get_run(ctx.run_id).objective
    sender = get_email_provider(ctx.simulated).sender if not ctx.simulated else "jose@simulated.invalid"
    why: dict[str, str] = {}
    to, cc = [], []
    for raw in a.to:
        addr, w = _resolve_recipient(raw, objective)
        why[addr] = w
        if addr not in to:
            to.append(addr)
    for raw in a.cc:
        addr, w = _resolve_recipient(raw, objective)
        why[addr] = w
        if addr not in cc and addr not in to:
            cc.append(addr)
    subject = a.subject.strip()
    body = _text(resolve_args(a.body, ctx.outputs)).strip()
    if CTRL_RE.search(subject):
        raise TaskError("the subject must be a single line", "validation", retryable=False)
    if not body:
        raise TaskError("the email body is empty", "validation")
    if len(body) > 100_000:
        raise TaskError("the email body is too long", "validation", retryable=False)
    refs = _flatten_ids(resolve_args(a.attachments, ctx.outputs))
    infos, _ = await asyncio.to_thread(_load_attachments, ctx, refs)
    d = {"from": sender, "to": to, "cc": cc, "subject": subject, "body": body, "attachments": infos}
    h = draft_hash(d)
    async with ctx.activity("preparing the email draft"):
        await ctx.tool("email.draft", {"to": to, "subject": subject, "attachments": [i["filename"] for i in infos]}, _noop, lambda _: "draft prepared (nothing sent)", simulated=False)
    return {"draft": d, "draft_hash": h, "recipient_basis": why, "sent": False,
            "warnings": (["SIMULATED run: the email will not actually be sent"] if ctx.simulated else [])}


async def _noop():
    return None


def v_email_draft(a, out, ctx) -> list[Check]:
    d = out["draft"]
    objective = store.get_run(ctx.run_id).objective
    bad_addr, no_basis = [], []
    for addr in [*d["to"], *d["cc"]]:
        try:
            normalize_address(addr)
        except ValueError:
            bad_addr.append(addr)
        if _provenance(addr, objective) is None:
            no_basis.append(addr)
    att_bad = []
    for x in d["attachments"]:
        try:
            if x["kind"] == "artifact":
                path, art = artifact_service.resolve_download(x["artifact_id"])
                if art.sha256 != x["sha256"] or path.stat().st_size != x["size"]:
                    att_bad.append(x["filename"])
            else:
                from .custom_agent import find_upload

                path, _ = find_upload(ctx, x["filename"])
                if sha256_file(path) != x["sha256"]:
                    att_bad.append(x["filename"])
        except Exception:  # noqa: BLE001
            att_bad.append(x["filename"])
    return [_ck("draft.recipients_valid", bool(d["to"]) and not bad_addr, f"{len(d['to'])} recipient(s) are syntactically valid" if not bad_addr else f"invalid: {bad_addr}"),
            _ck("draft.recipients_provenance", not no_basis, "every recipient comes from the objective, the configured default or the allow-list" if not no_basis else f"no basis for: {no_basis}"),
            _ck("draft.single_line_headers", not CTRL_RE.search(d["subject"]) and bool(d["subject"].strip()), "subject is a single non-empty line"),
            _ck("draft.body", bool(d["body"].strip()), f"{len(d['body'])} characters"),
            _ck("draft.attachments", not att_bad, f"{len(d['attachments'])} attachment(s) exist with the recorded hashes" if not att_bad else f"changed or missing: {att_bad}"),
            _ck("draft.hash", draft_hash(d) == out["draft_hash"], "draft content matches its hash"),
            _ck("draft.not_sent", out.get("sent") is False, "no message was sent by drafting")]


# ------------------------------------------------------------------ email.send

def _draft_from(ctx: AgentContext, ref) -> dict:
    out = resolve_args(ref, ctx.outputs)
    if not isinstance(out, dict) or "draft" not in out or "draft_hash" not in out:
        raise TaskError("email.send needs the output of an email.draft task", "validation")
    return out


def preview_email(ctx: AgentContext) -> dict:
    """The exact email, for the approval card. Same data the send step uses."""
    try:
        args = ctx.task.params["args"]
        out = _draft_from(ctx, args["draft"])
        d = out["draft"]
        return {"from": d["from"], "to": ", ".join(d["to"]), "cc": ", ".join(d["cc"]) or None, "subject": d["subject"], "body": d["body"],
                "attachments": [f"{x['filename']} ({x['size']} bytes)" for x in d["attachments"]], "draft_hash": out["draft_hash"],
                "recipient_basis": out.get("recipient_basis", {})}
    except Exception:  # noqa: BLE001 - best effort for the card; the send step re-validates everything
        return {}


async def t_email_send(ctx: AgentContext, a) -> dict:
    out = _draft_from(ctx, a.draft)
    d = out["draft"]
    if draft_hash(d) != out["draft_hash"]:
        raise TaskError("the draft was modified after it was prepared; refusing to send", "validation", retryable=False)
    if not ctx.simulated:
        miss = missing_permissions(("email.send",))
        if miss:
            raise TaskError("; ".join(miss), "permission", retryable=False)
    infos, paths = await asyncio.to_thread(_load_attachments, ctx, [x["artifact_id"] or x["filename"] for x in d["attachments"]])
    for want, got in zip(d["attachments"], infos):
        if want["sha256"] != got["sha256"]:
            raise TaskError(f"attachment {want['filename']} changed after the draft was approved; refusing to send", "validation", retryable=False)
    try:
        provider = get_email_provider(ctx.simulated)
    except EmailError as e:
        raise TaskError(str(e), "permission", retryable=False) from e
    simulated = bool(getattr(provider, "simulated", False))
    key = ledger.make_key("email.send", ctx.run_id, out["draft_hash"])
    claim = ledger.claim(ctx.run_id, ctx.task.id, "email.send", key, ", ".join(d["to"]), f"{d['subject']} -> {', '.join(d['to'])}", simulated)
    if not claim.proceed:
        act = claim.action
        if act["status"] == "accepted":  # idempotent: report the earlier, confirmed send; do not send again
            r = act["response"]
            manual = bool(r.get("manual"))  # a person confirmed it happened; the mail server never told us
            return {"status": "confirmed_by_user" if manual else r.get("status", "accepted_by_mail_server"), "message_id": act["reference"],
                    "accepted": [*d["to"], *d["cc"]] if manual else r.get("accepted", []), "refused": r.get("refused", {}),
                    "server": r.get("server"), "simulated": act["simulated"], "draft_hash": out["draft_hash"], "to": d["to"], "subject": d["subject"],
                    "attachments": [x["filename"] for x in d["attachments"]], "ledger_id": act["id"], "duplicate_prevented": True, "sent_at": r.get("sent_at"),
                    "manual_confirmation": manual}
        raise TaskError(f"not sending: {claim.reason}", "ambiguous", retryable=False)
    action_id = claim.action["id"]
    mail = OutgoingEmail(sender=d["from"] if not simulated else provider.sender, to=d["to"], cc=d["cc"], subject=d["subject"], body=d["body"],
                         attachments=[(x["filename"], p) for x, p in zip(d["attachments"], paths)])
    try:
        async with ctx.activity("sending the email" if not simulated else "simulating the email (nothing is sent)"):
            res = await ctx.tool("email.send", {"to": d["to"], "subject": d["subject"], "attachments": [x["filename"] for x in d["attachments"]], "simulated": simulated},
                                 lambda: asyncio.to_thread(provider.send_sync, mail),
                                 lambda r: ("simulated - nothing sent" if r["simulated"] else f"accepted by {r['server']} for {len(r['accepted'])} recipient(s)"), simulated=simulated)
    except EmailNotSent as e:
        ledger.failed(action_id, str(e))
        raise TaskError(str(e), "config" if "login" in str(e) or "connect" in str(e) else "network", retryable=False) from e
    except EmailAmbiguous as e:
        ledger.ambiguous(action_id, str(e))
        raise TaskError(f"{e}. Check the recipient's mailbox / your Sent folder before trying again.", "ambiguous", retryable=False) from e
    except BaseException as e:  # cancelled or timed out while the request may be in flight: the outcome is unknown
        ledger.ambiguous(action_id, f"interrupted ({type(e).__name__}) while sending; the outcome is unknown")
        raise
    sent_at = datetime.now(timezone.utc).isoformat()
    status = "simulated_not_sent" if simulated else "accepted_by_mail_server"
    ledger.accepted(action_id, res["message_id"], {**res, "status": status, "sent_at": sent_at})
    return {"status": status, "message_id": res["message_id"], "accepted": res["accepted"], "refused": res["refused"], "server": res["server"], "simulated": simulated,
            "draft_hash": out["draft_hash"], "to": d["to"], "subject": d["subject"], "attachments": [x["filename"] for x in d["attachments"]],
            "ledger_id": action_id, "duplicate_prevented": False, "sent_at": sent_at}


def v_email_send(a, out, ctx) -> list[Check]:
    draft_out = _draft_from(ctx, a.draft)
    d = draft_out["draft"]
    expected = set([*d["to"], *d["cc"]])
    acts = [x for x in ledger.for_run(ctx.run_id) if x["id"] == out["ledger_id"]]
    act = acts[0] if acts else None
    ck = [_ck("email.sent_draft_is_approved_draft", out["draft_hash"] == draft_out["draft_hash"] == draft_hash(d), "the email sent is exactly the draft that was shown for approval"),
          _ck("email.recipients_accepted", (set(out["accepted"]) == expected and not out["refused"]),
              f"the mail server accepted all {len(expected)} recipient(s)" if set(out["accepted"]) == expected and not out["refused"] else f"accepted={out['accepted']} refused={out['refused']}"),
          _ck("email.message_id", bool(out["message_id"]), f"message reference recorded: {out['message_id']}"),
          _ck("email.ledger_accepted", bool(act) and act["status"] == "accepted" and act["reference"] == out["message_id"],
              "the action ledger records this send as accepted with the same reference" if act else "no ledger record exists for this send")]
    if out.get("manual_confirmation"):
        ck.append(_ck("email.confirmed_by_person", out["status"] == "confirmed_by_user", "a person confirmed this email was sent; the mail server did not report it to JOSE"))
    elif out["simulated"]:
        ck.append(_ck("email.simulated_notice", out["status"] == "simulated_not_sent", "SIMULATED: nothing was sent"))
    else:
        ck.append(_ck("email.status_honest", out["status"] == "accepted_by_mail_server", "accepted by the mail server (this does not prove inbox delivery)"))
    return ck


IMPLS = {"email.draft": t_email_draft, "email.send": t_email_send}
VERIFIERS = {"email.draft": v_email_draft, "email.send": v_email_send}
