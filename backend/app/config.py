"""Environment-based configuration. Secrets live here only and are never serialised to clients."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BACKEND_DIR / ".env"), extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    llm_retry_scale: float = Field(1.0, ge=0, le=5)  # multiplies the wait between provider retries (tests set 0)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    jose_mode: str = "auto"  # auto | real | demo

    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # --- email (SMTP). The password never leaves this object: it is not exposed by the API, prompts, events or logs.
    smtp_host: str = ""
    smtp_port: int = Field(587, ge=1, le=65535)
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_security: str = "starttls"  # starttls | ssl | none
    smtp_timeout_seconds: float = Field(30, ge=3, le=300)
    email_from: str = "bhatejagarv@gmail.com"
    email_default_recipient: str = ""  # what the literal recipient "configured" resolves to
    email_allowed_recipients: str = ""  # comma separated extra addresses/domains (@example.com) the planner may use

    # --- filesystem sandbox: tools may only touch these directories (os.pathsep separated). Default: backend/data/workspace
    workspace_dirs: str = ""

    # --- developer tools: only these command prefixes may run (no shell, no pipes), only inside WORKSPACE_DIRS. ';'-separated.
    dev_allowed_commands: str = "python -m pytest;pytest;git status;git diff;git log;git branch;npm test;npm run lint;npm run build"
    dev_output_limit: int = Field(20000, ge=1000, le=200000)

    search_provider: str = "duckduckgo"
    tavily_api_key: str = ""
    brave_api_key: str = ""

    max_retries: int = Field(3, ge=0, le=5)
    max_parallel_tasks: int = Field(3, ge=1, le=8)
    http_timeout_seconds: float = Field(20, ge=2, le=120)
    task_timeout_seconds: float = Field(240, ge=10, le=1800)
    retry_backoff_seconds: float = Field(1.0, ge=0, le=30)
    max_upload_mb: int = Field(10, ge=1, le=100)

    browser_enabled: bool = True
    browser_headless: bool = True
    auto_approve_medium: bool = False
    # Test hook / local dev: allow fetching loopback/private hosts. Never enable in production.
    allow_private_hosts: bool = False
    respect_robots: bool = True

    database_url: str = "sqlite:///./data/jose.db"
    output_dir: str = "./data/outputs"
    upload_dir: str = "./data/uploads"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    def resolve_path(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (BACKEND_DIR / p).resolve()

    @property
    def output_path(self) -> Path:
        return self.resolve_path(self.output_dir)

    @property
    def upload_path(self) -> Path:
        return self.resolve_path(self.upload_dir)

    @property
    def db_url(self) -> str:
        if self.database_url.startswith("sqlite:///./"):
            rel = self.database_url.removeprefix("sqlite:///./")
            return f"sqlite:///{(BACKEND_DIR / rel).as_posix()}"
        return self.database_url

    @property
    def email_configured(self) -> bool:
        return bool(self.smtp_host.strip() and (self.email_from.strip() or self.smtp_username.strip()))

    @property
    def telegram_configured(self) -> bool:
        return bool(self.telegram_bot_token.strip() and self.telegram_chat_id.strip())

    def workspace_roots(self) -> list[Path]:
        import os

        raw = [x for x in self.workspace_dirs.split(os.pathsep) if x.strip()] or ["./data/workspace"]
        roots = []
        for r in raw:
            p = self.resolve_path(r.strip())
            p.mkdir(parents=True, exist_ok=True)
            roots.append(p.resolve())
        return roots

    @property
    def llm_available(self) -> bool:
        return bool(self.gemini_api_key.strip() or self.groq_api_key.strip())

    @property
    def demo_forced(self) -> bool:
        """True when runs must be simulated (explicit demo mode, or real mode impossible without a key)."""
        mode = self.jose_mode.lower()
        if mode == "demo":
            return True
        if mode == "real":
            return False
        return not self.llm_available


@lru_cache
def get_settings() -> Settings:
    return Settings()
