from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ..app.config import PROJECT_ROOT
from .cities import build_city_network_preview

_CITIES_ROOT = (PROJECT_ROOT / "data" / "cities").resolve()


def _provider_profile(provider: str, city_slug: str) -> dict[str, Any]:
    if provider == "govgr":
        return {
            "provider": provider,
            "provider_label": "Greek National / Regional Feed Pattern",
            "integration_stage": "city_specific",
            "coverage_note": (
                "The current implementation is operational for Thessaloniki and "
                "acts as the template for future city-specific traffic-feed adapters."
            ),
            "workflow_slots": [
                {
                    "id": "download_realtime_or_historical",
                    "title": "Download Realtime / Historical Feeds",
                    "status": "ready" if city_slug == "thessaloniki" else "planned",
                    "description": "Pull raw feed pages and historical archives into a structured local run folder.",
                },
                {
                    "id": "build_calibration_targets",
                    "title": "Build Calibration Targets",
                    "status": "ready" if city_slug == "thessaloniki" else "planned",
                    "description": "Convert downloaded history into network- and link-level target tables for calibration and validation.",
                },
                {
                    "id": "map_feed_links_to_osm",
                    "title": "Map Feed Links To OSM",
                    "status": "ready" if city_slug == "thessaloniki" else "planned",
                    "description": "Overlay the subset of feed link identifiers that currently match OSM way ids in the city extract.",
                },
                {
                    "id": "city_specific_adapter",
                    "title": "Additional City Adapter",
                    "status": "planned",
                    "description": "Reserve a slot for another city-specific feed source without changing the operator workflow shape.",
                },
            ],
        }
    return {
        "provider": provider,
        "provider_label": provider.upper(),
        "integration_stage": "planned",
        "coverage_note": "No provider profile is defined yet.",
        "workflow_slots": [],
    }


def _relative_to_root(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_metadata(city_root: Path) -> dict[str, Any]:
    metadata_path = city_root / "city_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return _read_json(metadata_path)
    except Exception:
        return {}


def _display_name(city_root: Path, metadata: dict[str, Any]) -> str:
    return (
        str(metadata.get("place") or metadata.get("display_name") or city_root.name)
        .replace("_", " ")
        .strip()
    )


def _sample_csv(path: Path, limit: int = 5) -> dict[str, Any] | None:
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if not lines:
        return None

    delimiter = ";" if ";" in lines[0] else ","
    reader = csv.DictReader(lines, delimiter=delimiter)
    rows: list[dict[str, str]] = []
    for index, row in enumerate(reader):
        if index >= limit:
            break
        rows.append({str(key): str(value) for key, value in row.items()})

    return {
        "path": _relative_to_root(path),
        "delimiter": delimiter,
        "columns": reader.fieldnames or [],
        "rows": rows,
    }


def _file_listing(root: Path, depth: int = 2) -> list[str]:
    items: list[str] = []
    if not root.exists():
        return items
    base_depth = len(root.parts)
    for path in sorted(root.rglob("*")):
        if path.name.startswith("."):
            continue
        rel_depth = len(path.parts) - base_depth
        if rel_depth > depth:
            continue
        items.append(_relative_to_root(path))
    return items


def _read_latest_link_rows(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if not lines:
        return {}
    delimiter = ";" if ";" in lines[0] else ","
    reader = csv.DictReader(lines, delimiter=delimiter)
    latest: dict[str, dict[str, dict[str, str]]] = {}
    for row in reader:
        link_id = str(row.get("Link_id", "")).strip().strip('"')
        direction = str(row.get("Link_Direction", "")).strip().strip('"') or "?"
        timestamp = str(row.get("Timestamp", "")).strip().strip('"')
        if not link_id:
            continue
        current = latest.setdefault(link_id, {}).get(direction)
        current_ts = "" if current is None else str(current.get("Timestamp", ""))
        if current is None or timestamp >= current_ts:
            latest.setdefault(link_id, {})[direction] = {
                str(key): str(value).strip().strip('"')
                for key, value in row.items()
                if key is not None
            }
    return latest


def _congestion_rank(value: str | None) -> tuple[int, str | None]:
    normalized = (value or "").strip().lower()
    ranking = {"low": 1, "medium": 2, "high": 3}
    return ranking.get(normalized, 0), value


def _build_link_alignment(city_slug: str, govgr_root: Path) -> dict[str, Any] | None:
    try:
        network_preview = build_city_network_preview(city_slug)
    except Exception:
        return None

    feature_by_id = {
        str(feature["id"]): feature for feature in network_preview.get("features", [])
    }
    speed_rows = _read_latest_link_rows(
        govgr_root / "thessaloniki_vehicle_speed" / "speed.csv"
    )
    congestion_rows = _read_latest_link_rows(
        govgr_root / "thessaloniki_congestion" / "congestions.csv"
    )

    matched_features: list[dict[str, Any]] = []
    speed_unique = set(speed_rows)
    congestion_unique = set(congestion_rows)
    matched_link_ids = set()

    for link_id in sorted(set(speed_rows) | set(congestion_rows)):
        feature = feature_by_id.get(link_id)
        if feature is None:
            continue
        matched_link_ids.add(link_id)
        speed_dirs = speed_rows.get(link_id, {})
        congestion_dirs = congestion_rows.get(link_id, {})

        speed_values = [
            float(payload["Speed"])
            for payload in speed_dirs.values()
            if payload.get("Speed") not in {None, ""}
        ]
        mean_speed = round(sum(speed_values) / len(speed_values), 2) if speed_values else None

        worst_congestion_rank = 0
        worst_congestion = None
        for payload in congestion_dirs.values():
            rank, label = _congestion_rank(payload.get("Congestion"))
            if rank >= worst_congestion_rank:
                worst_congestion_rank = rank
                worst_congestion = label

        latest_timestamp = max(
            [
                payload.get("Timestamp", "")
                for payload in [*speed_dirs.values(), *congestion_dirs.values()]
                if payload.get("Timestamp")
            ],
            default=None,
        )

        matched_features.append(
            {
                "id": link_id,
                "name": feature.get("name"),
                "road_type": feature.get("road_type"),
                "speed_limit_kph": feature.get("speed_kph"),
                "coords": feature.get("coords"),
                "oneway": feature.get("oneway"),
                "reverse_oneway": feature.get("reverse_oneway"),
                "speed_current_kph": mean_speed,
                "congestion_level": worst_congestion,
                "latest_timestamp": latest_timestamp,
                "direction_values": {
                    "speed": speed_dirs,
                    "congestion": congestion_dirs,
                },
            }
        )

    total_feed_links = len(set(speed_rows) | set(congestion_rows))
    return {
        "bbox": network_preview.get("bbox"),
        "stats": {
            "network_feature_count": network_preview.get("stats", {}).get("feature_count", 0),
            "feed_speed_link_count": len(speed_unique),
            "feed_congestion_link_count": len(congestion_unique),
            "matched_link_count": len(matched_link_ids),
            "match_ratio": (len(matched_link_ids) / total_feed_links) if total_feed_links else 0.0,
            "unmatched_link_count": total_feed_links - len(matched_link_ids),
        },
        "features": matched_features,
    }


def _discover_catalogs(govgr_root: Path) -> list[dict[str, Any]]:
    catalogs: list[dict[str, Any]] = []
    for directory in sorted(govgr_root.iterdir()):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        datapackage_path = directory / "datapackage.json"
        if not datapackage_path.exists():
            continue

        datapackage = _read_json(datapackage_path)
        resources = datapackage.get("resources", [])
        sample_csv_path = next(iter(sorted(directory.glob("*.csv"))), None)
        catalogs.append(
            {
                "id": directory.name,
                "title": datapackage.get("title") or directory.name,
                "description": datapackage.get("description") or "",
                "version": datapackage.get("version"),
                "keywords": datapackage.get("keywords") or [],
                "path": _relative_to_root(directory),
                "datapackage_path": _relative_to_root(datapackage_path),
                "resources": [
                    {
                        "title": resource.get("title") or resource.get("name"),
                        "format": resource.get("format"),
                        "path": resource.get("path"),
                        "description": resource.get("description") or "",
                        "source_urls": [
                            source.get("path")
                            for source in resource.get("sources", [])
                            if isinstance(source, dict) and source.get("path")
                        ],
                    }
                    for resource in resources
                    if isinstance(resource, dict)
                ],
                "sample_csv": _sample_csv(sample_csv_path) if sample_csv_path else None,
            }
        )
    return catalogs


def _discover_download_runs(downloads_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not downloads_root.exists():
        return runs

    for directory in sorted(downloads_root.iterdir(), reverse=True):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        quality_report_path = directory / "quality_report.json"
        report = _read_json(quality_report_path) if quality_report_path.exists() else {}
        datasets = report.get("datasets", {}) if isinstance(report, dict) else {}
        dataset_summaries = []
        for dataset_name, payload in datasets.items():
            if not isinstance(payload, dict):
                continue
            realtime = payload.get("realtime") if isinstance(payload.get("realtime"), dict) else {}
            historical = (
                payload.get("historical") if isinstance(payload.get("historical"), dict) else {}
            )
            dataset_summaries.append(
                {
                    "name": dataset_name,
                    "realtime_rows_clean": realtime.get("rows_clean"),
                    "realtime_pages_downloaded": realtime.get("pages_downloaded"),
                    "baseline_files": realtime.get("baseline_files") or [],
                    "clean_csv": realtime.get("clean_csv"),
                    "historical_files_downloaded": historical.get("files_downloaded"),
                    "historical_extracted_dir": historical.get("extracted_dir"),
                }
            )

        runs.append(
            {
                "name": directory.name,
                "path": _relative_to_root(directory),
                "quality_report_path": _relative_to_root(quality_report_path)
                if quality_report_path.exists()
                else None,
                "started_utc": report.get("started_utc"),
                "finished_utc": report.get("finished_utc"),
                "args": report.get("args") if isinstance(report, dict) else {},
                "datasets": dataset_summaries,
                "files": _file_listing(directory, depth=3),
            }
        )
    return runs


def _discover_target_exports(targets_root: Path) -> list[dict[str, Any]]:
    exports: list[dict[str, Any]] = []
    if not targets_root.exists():
        return exports

    for directory in sorted(targets_root.iterdir(), reverse=True):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        summary_path = directory / "targets_summary.json"
        summary = _read_json(summary_path) if summary_path.exists() else {}
        outputs = summary.get("outputs", {}) if isinstance(summary, dict) else {}
        exports.append(
            {
                "name": directory.name,
                "path": _relative_to_root(directory),
                "summary_path": _relative_to_root(summary_path) if summary_path.exists() else None,
                "calibration_year": summary.get("calibration_year"),
                "validation_year": summary.get("validation_year"),
                "sets": [
                    {
                        "name": set_name,
                        "files": set_payload.get("files") if isinstance(set_payload, dict) else [],
                        "speed_meta": set_payload.get("speed_meta")
                        if isinstance(set_payload, dict)
                        else {},
                        "travel_time_meta": set_payload.get("travel_time_meta")
                        if isinstance(set_payload, dict)
                        else {},
                    }
                    for set_name, set_payload in outputs.items()
                ],
                "files": _file_listing(directory, depth=2),
            }
        )
    return exports


def discover_traffic_feeds() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not _CITIES_ROOT.exists():
        return records

    for city_root in sorted(_CITIES_ROOT.iterdir()):
        if not city_root.is_dir() or city_root.name.startswith("."):
            continue
        govgr_root = city_root / "govgr"
        if not govgr_root.exists():
            continue
        metadata = _load_metadata(city_root)
        records.append(
            {
                "slug": city_root.name,
                "display_name": _display_name(city_root, metadata),
                "provider": "govgr",
                **_provider_profile("govgr", city_root.name),
                "city_root": _relative_to_root(city_root),
                "provider_root": _relative_to_root(govgr_root),
                "catalog_count": len(_discover_catalogs(govgr_root)),
                "download_run_count": len(_discover_download_runs(govgr_root / "downloads")),
                "target_export_count": len(_discover_target_exports(govgr_root / "targets")),
                "metadata": metadata,
            }
        )
    return records


def build_traffic_feed_preview(source_city_slug: str, target_city_slug: str | None = None) -> dict[str, Any]:
    target_slug = target_city_slug or source_city_slug
    source_city_root = _CITIES_ROOT / source_city_slug
    source_govgr_root = source_city_root / "govgr"
    if not source_govgr_root.exists():
        raise FileNotFoundError(f"No traffic-feed integration found for city '{source_city_slug}'")

    target_city_root = _CITIES_ROOT / target_slug
    target_govgr_root = target_city_root / "govgr"
    if not target_govgr_root.exists():
        raise FileNotFoundError(f"No traffic-feed integration found for city '{target_slug}'")

    source_metadata = _load_metadata(source_city_root)
    target_metadata = _load_metadata(target_city_root)
    return {
        "source": {
            "slug": source_city_slug,
            "display_name": _display_name(source_city_root, source_metadata),
            "provider": "govgr",
            **_provider_profile("govgr", source_city_slug),
            "city_root": _relative_to_root(source_city_root),
            "provider_root": _relative_to_root(source_govgr_root),
            "metadata": source_metadata,
        },
        "target_city": {
            "slug": target_slug,
            "display_name": _display_name(target_city_root, target_metadata),
            "city_root": _relative_to_root(target_city_root),
            "provider_root": _relative_to_root(target_govgr_root),
            "metadata": target_metadata,
        },
        "catalog_datasets": _discover_catalogs(source_govgr_root),
        "download_runs": _discover_download_runs(target_govgr_root / "downloads"),
        "target_exports": _discover_target_exports(target_govgr_root / "targets"),
        "linked_network": _build_link_alignment(target_slug, source_govgr_root),
    }
