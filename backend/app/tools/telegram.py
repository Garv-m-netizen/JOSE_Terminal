"""Telegram Bot API client. The token never leaves this module: errors and logs are scrubbed of it."""
from __future__ import annotations

from pathlib import Path

import httpx

from ..config import get_settings


class TelegramError(RuntimeError):
    pass


def mask_chat(chat_id: str) -> str:
    c = str(chat_id)
    return c if len(c) <= 6 else f"{c[:3]}…{c[-3:]}"


class TelegramClient:
    simulated = False

    def __init__(self) -> None:
        s = get_settings()
        self._token, self.chat_id = s.telegram_bot_token.strip(), s.telegram_chat_id.strip()
        if not self._token or not self.chat_id:
            raise TelegramError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must both be set in backend/.env")

    def _scrub(self, text: str) -> str:
        return text.replace(self._token, "[token]")

    async def send_document(self, path: Path, caption: str) -> dict:
        url = f"https://api.telegram.org/bot{self._token}/sendDocument"
        try:
            async with httpx.AsyncClient(timeout=60) as c:
                with path.open("rb") as f:
                    r = await c.post(url, data={"chat_id": self.chat_id, "caption": caption[:1000]},
                                     files={"document": (path.name, f, "application/pdf")})
        except httpx.HTTPError as e:
            raise TelegramError(self._scrub(f"network error talking to Telegram: {type(e).__name__}")) from None
        try:
            data = r.json()
        except ValueError:
            raise TelegramError(f"Telegram returned a non-JSON response (HTTP {r.status_code})") from None
        if not data.get("ok"):
            raise TelegramError(self._scrub(f"Telegram rejected the request: {data.get('description', r.status_code)}"))
        return data["result"]


class SimulatedTelegram:
    """Demo/dry-run: nothing is sent. Returns a clearly fake receipt."""

    simulated = True
    chat_id = "(simulated)"

    async def send_document(self, path: Path, caption: str) -> dict:
        return {"message_id": 0, "chat": {"id": 0}, "document": {"file_name": path.name}, "simulated": True}
