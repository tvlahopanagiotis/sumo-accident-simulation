from __future__ import annotations

from collections.abc import Mapping
import mimetypes
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field
import yaml

from ..app.config import CONFIGS_DIR, PROJECT_ROOT, load_config_raw, prepare_runtime_config, resolve_config_path, save_config, validate_config
from .jobs import job_manager
from .workflows import WORKFLOW_SPECS


class ConfigPayload(BaseModel):
    path: str | None = None
    config: dict[str, Any] | None = None
    raw_yaml: str | None = None


class ValidationPayload(BaseModel):
    config: dict[str, Any] | None = None
    raw_yaml: str | None = None


class JobCreatePayload(BaseModel):
    workflow_id: str = Field(..., description="Workflow identifier from /api/workflows")
    payload: dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="SAS GUI API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _relative_to_root(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _resolve_safe_path(path: str) -> Path:
    target = (PROJECT_ROOT / path).resolve()
    allowed_roots = [
        PROJECT_ROOT / "results",
        PROJECT_ROOT / "configs",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "frontend" / "public" / "branding",
    ]
    if not any(root.resolve() in target.parents or target == root.resolve() for root in allowed_roots):
        raise HTTPException(status_code=403, detail="Path is outside allowed roots")
    return target


def _build_tree(path: Path, depth: int = 2) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_dir():
        return []

    nodes: list[dict[str, Any]] = []
    for child in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower())):
        node = {
            "name": child.name,
            "path": _relative_to_root(child),
            "kind": "directory" if child.is_dir() else "file",
        }
        if child.is_dir() and depth > 0:
            node["children"] = _build_tree(child, depth=depth - 1)
        nodes.append(node)
    return nodes


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/workflows")
def list_workflows() -> dict[str, Any]:
    return {
        "workflows": [spec.to_dict() for spec in WORKFLOW_SPECS.values()],
    }


@app.get("/api/configs")
def list_configs() -> dict[str, Any]:
    configs = sorted(CONFIGS_DIR.rglob("*.yaml"))
    return {
        "configs": [
            {
                "path": _relative_to_root(path),
                "name": path.name,
            }
            for path in configs
        ]
    }


@app.get("/api/config")
def get_config(path: str = Query(...)) -> dict[str, Any]:
    resolved = resolve_config_path(path)
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Config not found")
    raw_yaml = resolved.read_text(encoding="utf-8")
    return {
        "path": _relative_to_root(resolved),
        "raw_yaml": raw_yaml,
        "config": load_config_raw(resolved),
    }


@app.post("/api/config/validate")
def validate_config_payload(payload: ValidationPayload) -> dict[str, Any]:
    config = payload.config
    if config is None and payload.raw_yaml is not None:
        config = yaml.safe_load(payload.raw_yaml)
    if not isinstance(config, Mapping):
        raise HTTPException(status_code=400, detail="Config payload must be a YAML mapping")
    try:
        validate_config(prepare_runtime_config(dict(config)))
    except SystemExit as exc:
        raise HTTPException(status_code=400, detail="Config validation failed") from exc
    return {"valid": True}


@app.post("/api/config/save")
def save_config_payload(payload: ConfigPayload) -> dict[str, Any]:
    if not payload.path:
        raise HTTPException(status_code=400, detail="path is required")
    config = payload.config
    if config is None and payload.raw_yaml is not None:
        config = yaml.safe_load(payload.raw_yaml)
    if not isinstance(config, Mapping):
        raise HTTPException(status_code=400, detail="Config payload must be a YAML mapping")
    target = save_config(dict(config), payload.path)
    return {
        "path": _relative_to_root(target),
        "saved": True,
    }


@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    return {"jobs": job_manager.list_jobs()}


@app.post("/api/jobs")
def create_job(payload: JobCreatePayload) -> dict[str, Any]:
    spec = WORKFLOW_SPECS.get(payload.workflow_id)
    if not spec:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return job_manager.create_job(spec.id, spec.title, payload.payload)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    if not job_manager.cancel_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
    return {"cancelled": True}


@app.get("/api/results")
def list_results(depth: int = Query(2, ge=0, le=5)) -> dict[str, Any]:
    results_dir = PROJECT_ROOT / "results"
    return {
        "root": "results",
        "entries": _build_tree(results_dir, depth=depth),
    }


@app.get("/api/fs/tree")
def list_tree(path: str, depth: int = Query(2, ge=0, le=5)) -> dict[str, Any]:
    resolved = _resolve_safe_path(path)
    return {
        "path": _relative_to_root(resolved),
        "entries": _build_tree(resolved, depth=depth),
    }


@app.get("/api/files/content")
def get_file_content(path: str):
    resolved = _resolve_safe_path(path)
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    media_type, _ = mimetypes.guess_type(resolved.name)
    return FileResponse(resolved, media_type=media_type)


@app.get("/api/files/text")
def get_file_text(path: str) -> PlainTextResponse:
    resolved = _resolve_safe_path(path)
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return PlainTextResponse(resolved.read_text(encoding="utf-8", errors="replace"))


@app.get("/api/branding")
def branding() -> dict[str, Any]:
    return {
        "name": "AntifragiCity SAS",
        "project": "Horizon Europe AntifragiCity",
        "colors": {
            "primary": "#f93262",
            "secondary": "#ffbea1",
            "ink": "#29131b",
            "surface": "#fff6f2",
            "surface_alt": "#ffe7de",
            "border": "#f2b9aa",
        },
        "logo_path": "frontend/public/branding/antifragicity-logo-main-h.svg",
        "favicon_path": "frontend/public/branding/antifragicity-favicon.svg",
        "font_note": "Official font files were not readable from this environment; the first version uses a clean web-safe fallback stack and can be updated once the font assets are accessible.",
    }


def main() -> None:
    import uvicorn

    uvicorn.run("sas.gui.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
