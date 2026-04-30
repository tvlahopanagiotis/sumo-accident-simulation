from __future__ import annotations

from collections.abc import Mapping
import io
import json
import mimetypes
from pathlib import Path
from typing import Any
import zipfile

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
import yaml

from ..app.config import CONFIGS_DIR, PROJECT_ROOT, load_config_raw, prepare_runtime_config, resolve_config_path, save_config, validate_config
from .cities import build_city_network_preview, delete_city_ways, discover_cities, update_city_speed_limits
from .generator_inputs import build_city_demand_preview
from .jobs import job_manager
from .locations import search_locations
from .results import build_run_summary, delete_run, find_run_root, list_run_registry
from .traffic_feeds import build_traffic_feed_preview, discover_traffic_feeds
from .workflows import WORKFLOW_SPECS


class ConfigPayload(BaseModel):
    path: str | None = None
    config: dict[str, Any] | None = None
    raw_yaml: str | None = None


class ValidationPayload(BaseModel):
    config: dict[str, Any] | None = None
    raw_yaml: str | None = None


class ConfigCreatePayload(BaseModel):
    path: str
    source_path: str | None = None


class ConfigDeletePayload(BaseModel):
    path: str


class JobCreatePayload(BaseModel):
    workflow_id: str = Field(..., description="Workflow identifier from /api/workflows")
    payload: dict[str, Any] = Field(default_factory=dict)


class CitySpeedLimitUpdatePayload(BaseModel):
    way_ids: list[str] = Field(default_factory=list)
    speed_kph: float


class CityWaySelectionPayload(BaseModel):
    way_ids: list[str] = Field(default_factory=list)


class ResultDeletePayload(BaseModel):
    path: str


app = FastAPI(title="AntifragiCity SUMA API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "https://sas.rhoe-api.gr",
        "https://suma.rhoe-api.gr",
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
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "frontend" / "public" / "branding",
    ]
    allowed_file = (PROJECT_ROOT / "README.md").resolve()
    if target == allowed_file:
        return target
    if not any(root.resolve() in target.parents or target == root.resolve() for root in allowed_roots):
        raise HTTPException(status_code=403, detail="Path is outside allowed roots")
    return target


def _normalize_config_target(path: str) -> str:
    raw = Path(path)
    if raw.is_absolute():
        raise HTTPException(status_code=400, detail="Config path must be repository-relative")
    target = raw if raw.parts[:1] == ("configs",) else Path("configs") / raw
    if target.suffix.lower() not in {".yaml", ".yml"}:
        target = target.with_suffix(".yaml")
    resolved = (PROJECT_ROOT / target).resolve()
    if CONFIGS_DIR.resolve() not in resolved.parents:
        raise HTTPException(status_code=400, detail="Config path must live under configs/")
    return target.as_posix()


def _build_clean_config() -> dict[str, Any]:
    config = load_config_raw()
    sumo_configs = sorted((PROJECT_ROOT / "data").rglob("*.sumocfg"))
    default_sumo_cfg = _relative_to_root(sumo_configs[0]) if sumo_configs else "data/cities/example/network/example.sumocfg"
    config.setdefault("sumo", {})["config_file"] = default_sumo_cfg
    config.setdefault("output", {})["output_folder"] = "results/custom/new_config"
    return config


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


def _cleanup_empty_config_dirs(path: Path) -> None:
    current = path.parent
    while current != CONFIGS_DIR and current.exists():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


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


@app.get("/api/sumo-configs")
def list_sumo_configs() -> dict[str, Any]:
    paths = sorted((PROJECT_ROOT / "data").rglob("*.sumocfg"))
    return {
        "sumo_configs": [_relative_to_root(path) for path in paths]
    }


@app.get("/api/output-folders")
def list_output_folders() -> dict[str, Any]:
    folders: set[str] = {"results/custom/new_config"}
    for config_path in CONFIGS_DIR.rglob("*.yaml"):
        try:
            config = load_config_raw(config_path)
        except Exception:
            continue
        output_cfg = config.get("output", {}) if isinstance(config, dict) else {}
        output_folder = output_cfg.get("output_folder") if isinstance(output_cfg, dict) else None
        if isinstance(output_folder, str) and output_folder.strip():
            folders.add(output_folder)
    return {"output_folders": sorted(folders)}


@app.get("/api/data-output-folders")
def list_data_output_folders() -> dict[str, Any]:
    folders: set[str] = set()
    for root in [
        PROJECT_ROOT / "data" / "cities",
        PROJECT_ROOT / "data" / "benchmarks",
        PROJECT_ROOT / "data" / "synthetic",
    ]:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_dir() and path.name in {"network", "outputs", "routes"}:
                folders.add(_relative_to_root(path))
        if root.exists():
            folders.add(_relative_to_root(root))
    return {"data_output_folders": sorted(folders)}


@app.get("/api/docs")
def list_docs() -> dict[str, Any]:
    docs_root = PROJECT_ROOT / "docs"
    paths = [PROJECT_ROOT / "README.md", *sorted(docs_root.rglob("*.md"))]
    return {
        "docs": [
            {
                "path": _relative_to_root(path),
                "name": path.name,
            }
            for path in paths
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


@app.post("/api/config/create")
def create_config_payload(payload: ConfigCreatePayload) -> dict[str, Any]:
    target_path = _normalize_config_target(payload.path)
    resolved_target = (PROJECT_ROOT / target_path).resolve()
    if resolved_target.exists():
        raise HTTPException(status_code=409, detail="Target config already exists")

    if payload.source_path:
        source = resolve_config_path(payload.source_path)
        if not source.exists():
            raise HTTPException(status_code=404, detail="Source config not found")
        config = load_config_raw(source)
    else:
        config = _build_clean_config()

    saved = save_config(config, target_path)
    return {
        "path": _relative_to_root(saved),
        "created": True,
        "mode": "clone" if payload.source_path else "clean",
    }


@app.post("/api/config/delete")
def delete_config_payload(payload: ConfigDeletePayload) -> dict[str, Any]:
    resolved = resolve_config_path(payload.path)
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="Config not found")
    if CONFIGS_DIR.resolve() not in resolved.resolve().parents:
        raise HTTPException(status_code=400, detail="Only files under configs/ can be deleted")
    resolved.unlink()
    _cleanup_empty_config_dirs(resolved)
    return {"deleted": True}


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


@app.delete("/api/jobs/{job_id}")
def forget_job(job_id: str) -> dict[str, Any]:
    if not job_manager.forget_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found or still running")
    return {"deleted": True}


@app.delete("/api/jobs")
def clear_finished_jobs() -> dict[str, Any]:
    return {"deleted_count": job_manager.clear_finished()}


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


@app.get("/api/results/runs")
def list_result_runs() -> dict[str, Any]:
    return {"runs": list_run_registry()}


@app.get("/api/results/summary")
def get_result_summary(path: str = Query(...)) -> dict[str, Any]:
    resolved = _resolve_safe_path(path)
    summary = build_run_summary(resolved)
    if summary is None:
        raise HTTPException(status_code=404, detail="No run summary found for the selected path")
    return summary


@app.get("/api/results/export")
def export_result(path: str = Query(...), format: str = Query("json", pattern="^(json|csv|zip)$")):
    resolved = _resolve_safe_path(path)
    run_root = find_run_root(resolved)
    if run_root is None:
        raise HTTPException(status_code=404, detail="No run summary found for the selected path")

    if format == "csv":
        csv_path = run_root / "network_metrics.csv"
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="network_metrics.csv not found")
        return FileResponse(
            csv_path,
            media_type="text/csv",
            filename=f"{run_root.name}_network_metrics.csv",
        )

    if format == "json":
        summary = build_run_summary(run_root)
        if summary is None:
            raise HTTPException(status_code=404, detail="No run summary found for the selected path")
        payload = json.dumps(summary, indent=2).encode("utf-8")
        return Response(
            payload,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{run_root.name}_summary.json"'},
        )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(run_root.rglob("*")):
            if item.is_file():
                archive.write(item, item.relative_to(run_root))
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{run_root.name}_artifacts.zip"'},
    )


@app.post("/api/results/delete")
def delete_result(payload: ResultDeletePayload) -> dict[str, Any]:
    try:
        return delete_run(_resolve_safe_path(payload.path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/locations/search")
def search_place(query: str = Query(..., min_length=2), limit: int = Query(6, ge=1, le=10)) -> dict[str, Any]:
    try:
        return {"results": search_locations(query, limit=limit)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Location search failed: {exc}") from exc


@app.get("/api/cities")
def list_cities() -> dict[str, Any]:
    return {"cities": discover_cities()}


@app.get("/api/cities/{city_slug}/network-preview")
def get_city_network_preview(city_slug: str) -> dict[str, Any]:
    try:
        return build_city_network_preview(city_slug)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/cities/{city_slug}/demand-preview")
def get_city_demand_preview(city_slug: str) -> dict[str, Any]:
    try:
        return build_city_demand_preview(city_slug)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/cities/{city_slug}/speed-limits")
def update_city_speed_limit_tags(city_slug: str, payload: CitySpeedLimitUpdatePayload) -> dict[str, Any]:
    try:
        return update_city_speed_limits(city_slug, payload.way_ids, payload.speed_kph)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/cities/{city_slug}/delete-ways")
def delete_city_way_tags(city_slug: str, payload: CityWaySelectionPayload) -> dict[str, Any]:
    try:
        return delete_city_ways(city_slug, payload.way_ids)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/traffic-feeds")
def list_traffic_feeds() -> dict[str, Any]:
    return {"feeds": discover_traffic_feeds()}


@app.get("/api/traffic-feeds/{city_slug}")
def get_traffic_feed_preview(
    city_slug: str,
    target_city_slug: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        return build_traffic_feed_preview(city_slug, target_city_slug=target_city_slug)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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
        "name": "AntifragiCity SUMA",
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
        "eu_logo_path": "frontend/public/branding/eu-funded-by-eu.png",
        "rhoe_logo_path": "frontend/public/branding/rhoe-logo-main-on-white.png",
        "project_url": "https://antifragicity.eu",
        "footer_disclaimer": (
            "This project has received funding from the European Union’s Horizon Europe research "
            "and innovation programme under grant agreement No. 101203052. Views and opinions "
            "expressed are however those of the author(s) only and do not necessarily reflect "
            "those of the European Union or the European Climate, Infrastructure and Environment "
            "Executive Agency (CINEA). Neither the European Union nor the granting authority can "
            "be held responsible for them."
        ),
        "copyright": "© AntifragiCity. All rights reserved.",
    }


def main() -> None:
    import uvicorn

    uvicorn.run("sas.gui.app:app", host="127.0.0.1", port=12000, reload=False)


if __name__ == "__main__":
    main()
