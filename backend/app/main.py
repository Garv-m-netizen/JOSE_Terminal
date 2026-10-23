from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from .api.routes import router
from .config import BACKEND_DIR, get_settings
from .database import init_engine
from .models.state import InvalidTransition
from .services import execution_service as ex, runtime_settings
from .services.execution_service import Conflict
from .services.store import NotFound
from .workflows.research_report import PlanError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_engine()
    runtime_settings.update({})  # apply persisted overrides
    n = ex.manager.mark_interrupted()
    if n:
        logging.getLogger("jose").warning("marked %d interrupted run(s) as failed", n)
    yield
    from .tools.browser import get_browser_tool

    await get_browser_tool().close()


def create_app() -> FastAPI:
    app = FastAPI(title="JOSE - Autonomous Execution & Intelligence System", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in get_settings().cors_origins.split(",") if o.strip()],
                       allow_methods=["*"], allow_headers=["*"], expose_headers=["Content-Disposition"])

    @app.exception_handler(NotFound)
    async def _nf(_: Request, e: NotFound):
        return JSONResponse({"detail": str(e)}, status_code=404)

    @app.exception_handler(Conflict)
    async def _cf(_: Request, e: Conflict):
        return JSONResponse({"detail": str(e)}, status_code=409)

    @app.exception_handler(InvalidTransition)
    async def _it(_: Request, e: InvalidTransition):
        return JSONResponse({"detail": str(e)}, status_code=409)

    @app.exception_handler(PlanError)
    async def _pe(_: Request, e: PlanError):
        return JSONResponse({"detail": "invalid plan", "problems": e.problems}, status_code=422)

    @app.exception_handler(ValueError)
    async def _ve(_: Request, e: ValueError):
        return JSONResponse({"detail": str(e)}, status_code=422)

    app.include_router(router)

    dist = BACKEND_DIR.parent / "frontend" / "dist"
    if dist.is_dir():  # serve the built SPA when present
        from fastapi.staticfiles import StaticFiles

        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str):
            if path == "api" or path.startswith("api/"):
                return JSONResponse({"detail": "not found"}, status_code=404)
            f = (dist / path).resolve()
            if path and dist.resolve() in f.parents and f.is_file():
                return FileResponse(f)
            return FileResponse(dist / "index.html")

    return app


app = create_app()
