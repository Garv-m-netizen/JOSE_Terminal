"""What JOSE is actually able and permitted to do right now.

Permissions are facts about configuration (is an SMTP server configured? is a Telegram chat configured?), never something a
model can grant itself. The planner is shown the result so it can say exactly what is missing instead of pretending.
"""
from __future__ import annotations

from ..config import get_settings

# Capabilities that were asked for in the product vision but have NO working implementation yet. Listed so the planner
# refuses honestly ("not available") and the UI can show them as such.
UNAVAILABLE: dict[str, str] = {
    "calendar": "no calendar integration is implemented",
    "powerpoint.generate": "general PowerPoint generation is not available as a tool (only inside the competitor-research workflow)",
    "browser.interact": "interactive browser automation (clicking, forms, logins) is not implemented; web.fetch reads public pages only",
    "desktop.apps": "controlling desktop applications is not implemented",
    "shell.arbitrary": "arbitrary shell commands are not available; only allow-listed developer commands run, through dev.run",
}


def permission_status() -> dict[str, tuple[bool, str]]:
    """permission name -> (granted, reason when not granted)."""
    from ..tools.email_provider import _override as email_override

    s = get_settings()
    return {
        "uploads.read": (True, ""),
        "artifacts.write": (True, ""),
        "fs.read": (True, ""),
        "fs.write": (True, ""),
        "dev.run": (bool(s.dev_allowed_commands.strip()), "no developer commands are allow-listed (DEV_ALLOWED_COMMANDS)"),
        "network.http": (True, ""),
        "llm": (s.llm_available, "" if s.llm_available else "no GEMINI_API_KEY / GROQ_API_KEY configured"),
        "web.search": (s.search_provider.lower() == "duckduckgo" or bool({"tavily": s.tavily_api_key, "brave": s.brave_api_key}.get(s.search_provider.lower())),
                       f"search provider '{s.search_provider}' needs an API key"),
        "email.send": (s.email_configured or email_override is not None,
                       "email is not configured (set SMTP_HOST and EMAIL_FROM in backend/.env)"),
        "telegram.send": (s.telegram_configured, "Telegram is not configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)"),
    }


def missing_permissions(required: tuple[str, ...]) -> list[str]:
    st = permission_status()
    return [f"{p}: {st[p][1]}" for p in required if p in st and not st[p][0]]


def describe() -> dict:
    s = get_settings()
    return {"workspace_roots": [str(r) for r in s.workspace_roots()],
            "permissions": {k: {"granted": g, "reason": r or None} for k, (g, r) in permission_status().items()},
            "unavailable": [{"capability": k, "reason": v} for k, v in UNAVAILABLE.items()]}
