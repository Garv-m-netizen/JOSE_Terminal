from __future__ import annotations

import asyncio

import httpx
import pytest
import pytest_asyncio

from app import database
from app.config import get_settings
from app.services import execution_service as ex
from app.services.llm import set_providers_override


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "up"))
    monkeypatch.setenv("JOSE_MODE", "demo")
    for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "TAVILY_API_KEY", "BRAVE_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.setenv(k, "")  # tests must never see the developer's real .env credentials
    monkeypatch.setenv("RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setenv("LLM_RETRY_SCALE", "0")
    monkeypatch.setenv("MAX_RETRIES", "2")
    get_settings.cache_clear()
    database.init_engine()
    set_providers_override(None)
    ex.manager.handles.clear()
    yield
    set_providers_override(None)
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def client():
    from app.main import app

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def wait_status(client, run_id, targets, timeout=60.0):
    end = asyncio.get_running_loop().time() + timeout
    last = None
    while asyncio.get_running_loop().time() < end:
        last = (await client.get(f"/api/runs/{run_id}")).json()
        if last["status"] in targets:
            return last
        await asyncio.sleep(0.05)
    raise AssertionError(f"timeout waiting for {targets}; last status={last and last['status']} error={last and last.get('error')}")


OBJECTIVE = ("Research five competitors in the AI automation market, compare their features, use publicly available "
             "pricing information, create a structured Excel comparison, and generate a PowerPoint summary with source references.")
