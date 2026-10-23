"""Risk classification and approval policy. Pure functions; the LLM has no say in these decisions."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import get_settings
from ..schemas.domain import RiskLevel

# action -> baseline risk. Anything not listed is not executable at all.
BASE_RISK: dict[str, RiskLevel] = {
    "file.read_inputs": "LOW", "research.discover": "LOW", "browser.collect": "LOW", "research.extract": "LOW",
    "workspace.normalize": "LOW", "workspace.build_excel": "LOW", "workspace.build_pptx": "LOW",
    "verify.deliverables": "LOW", "test.generate": "LOW", "test.solve_check": "LOW", "test.build_documents": "LOW",
    "test.verify_documents": "LOW", "test.verify_set": "LOW", "telegram.send": "HIGH",  # posting to a group is outward-facing and irreversible
}
ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


@dataclass
class RiskAssessment:
    level: RiskLevel = "LOW"
    reasons: list[str] = field(default_factory=list)
    target: str = ""
    consequences: str = ""
    details: dict = field(default_factory=dict)

    def raise_to(self, level: RiskLevel, reason: str) -> None:
        if ORDER[level] > ORDER[self.level]:
            self.level = level
        self.reasons.append(reason)


def tool_of_missing(action: str) -> bool:
    from ..workflows.custom import tool_of

    return tool_of(action) is None


def export_path(export_name: str, filename: str) -> Path:
    from .runtime_settings import load
    from ..tools.safety import safe_join

    return safe_join(load().output_path, "exports", export_name, filename)


def assess(action: str, params: dict, resolved: dict | None = None) -> RiskAssessment:
    """Classify one task. `resolved` carries runtime facts (e.g. the candidate URL chosen by discovery)."""
    if action.startswith("tool."):
        from ..workflows.custom import tool_of

        spec = tool_of(action)
        base = spec.risk if spec else "HIGH"
    else:
        base = BASE_RISK.get(action)
    if base is None or (action.startswith("tool.") and tool_of_missing(action)):
        a = RiskAssessment("HIGH", [f"action '{action}' is not allow-listed"], target=action,
                           consequences="The action would not be executed.")
        return a
    a = RiskAssessment(base, target=action)  # type: ignore[arg-type]
    resolved = resolved or {}
    if action == "browser.collect":
        url = resolved.get("url", "")
        a.target = url or "competitor website"
        if url.startswith("http://"):
            a.raise_to("MEDIUM", "page is served over unencrypted HTTP")
            a.consequences = "Page content could be altered in transit; treated as lower-trust evidence."
    if action in ("telegram.send", "tool.telegram.send_files"):
        from ..config import get_settings
        from ..tools.telegram import mask_chat

        chat = get_settings().telegram_chat_id
        dry = bool(resolved.get("dry_run", params.get("dry_run")))
        a.target = "Telegram chat (simulated)" if dry else f"Telegram chat {mask_chat(chat) if chat else '(not configured)'}"
        a.details = {"file": resolved.get("file"), "caption": resolved.get("caption"), "excluded": resolved.get("answer_key_excluded"), "dry_run": dry}
        a.reasons.append("posting to a group is visible to every member and cannot be unsent by JOSE")
        a.consequences = ("SIMULATED: nothing will actually be sent; approval is still required to show the flow."
                          if dry else "The question paper PDF and caption will be posted to the Telegram group immediately, visible to every member. JOSE cannot delete it afterwards.")
    if action == "tool.email.send":
        dry = bool(resolved.get("dry_run"))
        a.target = f"email to {resolved.get('to') or '(draft not resolved)'}"
        a.details = {k: resolved.get(k) for k in ("from", "to", "cc", "subject", "body", "attachments", "draft_hash", "recipient_basis")} | {"dry_run": dry}
        a.reasons.append("an email leaves your control the moment the mail server accepts it and cannot be recalled")
        a.consequences = ("SIMULATED: nothing will actually be sent; approval is still required to show the flow." if dry else
                          f"This exact email will be sent from {resolved.get('from')} to {resolved.get('to')} immediately"
                          + (f" with {len(resolved.get('attachments') or [])} attachment(s)" if resolved.get("attachments") else "")
                          + ". JOSE cannot unsend it, and will report it as 'accepted by the mail server', not as delivered.")
    if action == "tool.dev.run":
        args = params.get("args", {})
        a.target = " ".join(args.get("command", []))[:200]
        a.details = {"command": " ".join(args.get("command", [])), "cwd": args.get("cwd", ".")}
        a.reasons.append("runs a program on this computer (test suites and build scripts can change files)")
        a.consequences = f"Runs `{a.target}` in {args.get('cwd', '.')} without a shell, with secrets removed from its environment, and stops it after {args.get('timeout_s', 120)}s."
    if action in ("tool.fs.copy", "tool.fs.move"):
        args = params.get("args", {})
        a.target = f"{args.get('source')} -> {args.get('destination')}"
        a.details = {"source": args.get("source"), "destination": args.get("destination")}
        a.reasons.append("changes files inside the approved folders" + (" (the original is removed from its old location)" if action.endswith("move") else ""))
        a.consequences = ("The file is copied to the destination; nothing is overwritten." if action.endswith("copy")
                          else "The file is moved to the destination; the old location will no longer contain it. Nothing is overwritten.")
    if action in ("workspace.build_excel", "workspace.build_pptx") and params.get("export_name"):
        try:
            target = export_path(params["export_name"], params.get("filename", "output"))
        except Exception:  # noqa: BLE001
            a.raise_to("HIGH", "export path is invalid")
            return a
        a.target = str(target)
        a.details = {"export_name": params["export_name"], "exists": target.exists()}
        if target.exists():
            a.raise_to("HIGH", "an existing file in the exports folder would be overwritten")
            a.consequences = (f"The existing file '{target.name}' in exports/{params['export_name']} will be replaced and the "
                              "previous content cannot be restored.")
    return a


def needs_approval(level: RiskLevel, mode: str) -> bool:
    """HIGH always requires a human. MEDIUM follows the configured policy. LOW never does."""
    if level == "HIGH":
        return True
    if level == "MEDIUM":
        from .runtime_settings import load

        return not (mode == "autonomous" and load().auto_approve_medium)
    return False
