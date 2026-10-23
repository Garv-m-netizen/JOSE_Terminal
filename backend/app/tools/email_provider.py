"""Email delivery through a configured SMTP server. Credentials are read from the environment here and nowhere else.

What a successful `send` proves: the SMTP server ACCEPTED the message for delivery (and which recipients it accepted/refused).
It does not prove the message reached an inbox. Callers must report it as "accepted by the mail server".

Errors are classified by whether the message could possibly have been handed over:
  * EmailNotSent   - rejected before the message body was transmitted (connect/TLS/auth/sender/recipients). Safe to try again later.
  * EmailAmbiguous - the connection broke while or after the body was transmitted. Do not resend automatically.
"""
from __future__ import annotations

import mimetypes
import re
import smtplib
import ssl
from dataclasses import dataclass, field
from email.message import EmailMessage
from email.utils import formatdate, make_msgid, parseaddr
from pathlib import Path

from ..config import get_settings

ADDR_RE = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$")
MAX_ATTACH_BYTES = 20 * 1024 * 1024
MAX_RECIPIENTS = 20


class EmailError(RuntimeError):
    pass


class EmailNotSent(EmailError):
    """Definitely not handed to the mail server."""


class EmailAmbiguous(EmailError):
    """May have been handed to the mail server."""


def valid_address(addr: str) -> bool:
    return bool(ADDR_RE.match(addr)) and len(addr) <= 254 and "\r" not in addr and "\n" not in addr


def normalize_address(raw: str) -> str:
    """Accepts 'Name <a@b.c>' or 'a@b.c'; returns the bare lower-cased address or raises ValueError."""
    _, addr = parseaddr(raw.strip())
    addr = addr.strip().lower()
    if not valid_address(addr):
        raise ValueError(f"'{raw}' is not a valid email address")
    return addr


@dataclass
class OutgoingEmail:
    sender: str
    to: list[str]
    subject: str
    body: str
    cc: list[str] = field(default_factory=list)
    attachments: list[tuple[str, Path]] = field(default_factory=list)  # (display filename, path)
    message_id: str = ""

    def build(self) -> EmailMessage:
        m = EmailMessage()
        m["From"], m["To"], m["Subject"] = self.sender, ", ".join(self.to), self.subject
        if self.cc:
            m["Cc"] = ", ".join(self.cc)
        m["Date"] = formatdate(localtime=True)
        self.message_id = self.message_id or make_msgid(domain=(self.sender.rsplit("@", 1)[-1] or "jose.local"))
        m["Message-ID"] = self.message_id
        m.set_content(self.body)
        for name, path in self.attachments:
            ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
            main, sub = ctype.split("/", 1)
            m.add_attachment(path.read_bytes(), maintype=main, subtype=sub, filename=name)
        return m


class SmtpProvider:
    simulated = False
    name = "smtp"

    def __init__(self) -> None:
        s = get_settings()
        if not s.email_configured:
            raise EmailError("Email is not configured: set SMTP_HOST and EMAIL_FROM (and SMTP_USERNAME/SMTP_PASSWORD if the server requires login) in backend/.env")
        self._s = s
        self.sender = s.email_from.strip() or s.smtp_username.strip()

    def _scrub(self, text: str) -> str:
        pw = self._s.smtp_password
        return text.replace(pw, "[password]") if pw else text

    def send_sync(self, mail: OutgoingEmail) -> dict:
        s = self._s
        msg = mail.build()
        recipients = [*mail.to, *mail.cc]
        server: smtplib.SMTP | None = None
        try:
            ctx = ssl.create_default_context()
            if s.smtp_security.lower() == "ssl":
                server = smtplib.SMTP_SSL(s.smtp_host, s.smtp_port, timeout=s.smtp_timeout_seconds, context=ctx)
            else:
                server = smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=s.smtp_timeout_seconds)
                server.ehlo()
                if s.smtp_security.lower() == "starttls":
                    server.starttls(context=ctx)
                    server.ehlo()
            if s.smtp_username:
                server.login(s.smtp_username, s.smtp_password)
        except smtplib.SMTPAuthenticationError as e:
            raise EmailNotSent(self._scrub(f"the mail server rejected the login (check SMTP_USERNAME / SMTP_PASSWORD): {e.smtp_code}")) from None
        except (smtplib.SMTPException, OSError, ssl.SSLError) as e:
            raise EmailNotSent(self._scrub(f"could not connect to the mail server {s.smtp_host}:{s.smtp_port}: {type(e).__name__}: {e}")) from None
        try:
            try:
                refused = server.send_message(msg, from_addr=self.sender, to_addrs=recipients)
            except smtplib.SMTPRecipientsRefused as e:
                raise EmailNotSent(self._scrub(f"the mail server refused every recipient: {self._fmt_refused(e.recipients)}")) from None
            except smtplib.SMTPSenderRefused as e:
                raise EmailNotSent(self._scrub(f"the mail server refused the sender address: {e.smtp_code} {e.smtp_error!r}")) from None
            except smtplib.SMTPDataError as e:
                raise EmailNotSent(self._scrub(f"the mail server rejected the message: {e.smtp_code} {e.smtp_error!r}")) from None
            except (smtplib.SMTPServerDisconnected, TimeoutError, OSError, ssl.SSLError) as e:
                raise EmailAmbiguous(self._scrub(f"the connection failed while sending ({type(e).__name__}); the message may or may not have been accepted")) from None
            except smtplib.SMTPException as e:
                raise EmailAmbiguous(self._scrub(f"unexpected mail server error while sending ({type(e).__name__}): {e}")) from None
        finally:
            try:
                server.quit()
            except Exception:  # noqa: BLE001 - the outcome is already decided
                pass
        accepted = [r for r in recipients if r not in refused]
        return {"accepted": accepted, "refused": self._fmt_refused(refused), "message_id": mail.message_id,
                "server": f"{s.smtp_host}:{s.smtp_port}", "simulated": False}

    @staticmethod
    def _fmt_refused(refused: dict) -> dict:
        return {k: f"{v[0]} {v[1].decode(errors='replace') if isinstance(v[1], bytes) else v[1]}" for k, v in refused.items()}


class SimulatedEmail:
    """Demo / dry-run: nothing leaves the machine. The receipt says so."""

    simulated = True
    name = "simulated"
    sender = "jose@simulated.invalid"

    def send_sync(self, mail: OutgoingEmail) -> dict:
        mail.build()  # still exercises address/attachment handling
        return {"accepted": [*mail.to, *mail.cc], "refused": {}, "message_id": mail.message_id or "<simulated@jose.invalid>", "server": "(simulated)", "simulated": True}


_override = None


def set_email_provider_override(provider) -> None:
    """Tests inject a fake provider here; None restores the real one."""
    global _override
    _override = provider


def get_email_provider(simulated: bool = False):
    if _override is not None:
        return _override
    if simulated:
        return SimulatedEmail()
    return SmtpProvider()


def email_status() -> dict:
    s = get_settings()
    configured = _override is not None or s.email_configured
    return {"configured": configured, "provider": "smtp" if configured else None, "sender": (s.email_from or s.smtp_username or None) if configured else None,
            "default_recipient_set": bool(s.email_default_recipient.strip()),
            "note": None if configured else "set SMTP_HOST and EMAIL_FROM (and SMTP_USERNAME / SMTP_PASSWORD) in backend/.env to send email for real"}
