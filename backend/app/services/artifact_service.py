"""Artifact registration and authorised download resolution. Files are only ever reached via a DB id."""
from __future__ import annotations

import shutil
from pathlib import Path

from sqlalchemy import select

from ..database import session_scope
from ..models import Artifact
from ..tools.file_tools import sha256_file
from ..tools.safety import UnsafePath, safe_join
from .event_service import events
from .runtime_settings import load
from .store import NotFound, iso

MEDIA = {"xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
         "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
         "screenshot": "image/png", "pdf": "application/pdf", "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "csv": "text/csv", "answer_key": "application/pdf"}


def run_dir(run_id: str) -> Path:
    root = load().output_path
    root.mkdir(parents=True, exist_ok=True)
    d = safe_join(root, run_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def register(run_id: str, task_id: str | None, path: Path, kind: str, simulated: bool) -> dict:
    root = load().output_path.resolve()
    rel = path.resolve().relative_to(root).as_posix()  # raises if outside the output root
    with session_scope() as s:
        for old in s.scalars(select(Artifact).where(Artifact.run_id == run_id, Artifact.rel_path == rel)):
            s.delete(old)  # a rebuilt file replaces its earlier record instead of piling up duplicates
        a = Artifact(run_id=run_id, task_id=task_id, filename=path.name, kind=kind, root=str(root), rel_path=rel,
                     size_bytes=path.stat().st_size, sha256=sha256_file(path), simulated=simulated)
        s.add(a)
        s.flush()
        d = to_dict(a)
    events.emit(run_id, "artifact.created", task_id, artifact_id=d["id"], filename=d["filename"], kind=kind,
                size_bytes=d["size_bytes"], simulated=simulated)
    return d


def to_dict(a: Artifact) -> dict:
    return {"id": a.id, "run_id": a.run_id, "task_id": a.task_id, "filename": a.filename, "kind": a.kind,
            "size_bytes": a.size_bytes, "sha256": a.sha256, "simulated": a.simulated, "created_at": iso(a.created_at),
            "download_url": f"/api/artifacts/{a.id}/download",
            "path": a.rel_path, "folder": "/".join(a.rel_path.split("/")[1:-1])}


def list_for_run(run_id: str) -> list[dict]:
    with session_scope() as s:
        return [to_dict(a) for a in s.scalars(select(Artifact).where(Artifact.run_id == run_id).order_by(Artifact.created_at))]


def resolve_download(artifact_id: str) -> tuple[Path, Artifact]:
    """Return the on-disk path only if the artifact exists in the DB, lives under its recorded root, and is intact."""
    with session_scope() as s:
        a = s.get(Artifact, artifact_id)
        if not a:
            raise NotFound("artifact not found")
        root = Path(a.root)
        try:
            p = safe_join(root, *a.rel_path.split("/"))
        except UnsafePath as e:
            raise NotFound("artifact path is invalid") from e
        if not p.is_file():
            raise NotFound("artifact file is missing on disk")
        if sha256_file(p) != a.sha256:
            raise NotFound("artifact failed its integrity check (file was modified after creation)")
        return p, a


def export_copy(src: Path, export_name: str, filename: str) -> Path:
    """Copy a deliverable into exports/<name>/ (the only place an existing file may be overwritten, after approval)."""
    from .risk import export_path

    dst = export_path(export_name, filename)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst
