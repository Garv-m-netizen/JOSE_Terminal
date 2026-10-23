"""Runtime-adjustable execution settings (persisted in SQLite) layered over environment config.

Secrets are never part of this object; the API exposes only whether a key is *configured*.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import get_settings
from ..database import session_scope
from ..models import SettingRow

EDITABLE = ("max_retries", "max_parallel_tasks", "http_timeout_seconds", "task_timeout_seconds",
            "browser_enabled", "browser_headless", "auto_approve_medium", "output_dir")


@dataclass
class RuntimeSettings:
    max_retries: int
    max_parallel_tasks: int
    http_timeout_seconds: float
    task_timeout_seconds: float
    browser_enabled: bool
    browser_headless: bool
    auto_approve_medium: bool
    output_dir: str

    @property
    def output_path(self) -> Path:
        return get_settings().resolve_path(self.output_dir)


def load() -> RuntimeSettings:
    base = get_settings()
    vals = {k: getattr(base, k) for k in EDITABLE}
    with session_scope() as s:
        for row in s.query(SettingRow).all():
            if row.key in EDITABLE:
                vals[row.key] = row.value.get("v")
    return RuntimeSettings(**vals)


def update(changes: dict) -> RuntimeSettings:
    base = get_settings()
    with session_scope() as s:
        for k, v in changes.items():
            if k not in EDITABLE or v is None:
                continue
            if k == "output_dir":
                p = base.resolve_path(str(v))
                p.mkdir(parents=True, exist_ok=True)
                probe = p / ".write_test"
                try:
                    probe.write_text("ok")
                    probe.unlink()
                except OSError as e:
                    raise ValueError(f"output directory is not writable: {e}") from e
            row = s.get(SettingRow, k)
            if row:
                row.value = {"v": v}
            else:
                s.add(SettingRow(key=k, value={"v": v}))
    # keep the singleton Settings in sync for modules reading it directly
    rt = load()
    base.http_timeout_seconds = rt.http_timeout_seconds
    base.task_timeout_seconds = rt.task_timeout_seconds
    base.browser_enabled = rt.browser_enabled
    base.browser_headless = rt.browser_headless
    return rt
