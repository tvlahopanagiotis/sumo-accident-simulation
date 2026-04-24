from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from ..app.config import PROJECT_ROOT

_RESULTS_ROOT = (PROJECT_ROOT / "results").resolve()
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".webp"}


def _json_load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _to_number(value: str | None) -> float | int | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number.is_integer():
        return int(number)
    return round(number, 4)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def find_run_root(selected_path: Path) -> Path | None:
    current = selected_path if selected_path.is_dir() else selected_path.parent
    for candidate in [current, *current.parents]:
        if candidate == PROJECT_ROOT:
            break
        if candidate.resolve() == _RESULTS_ROOT:
            break
        if (candidate / "metadata.json").exists() and (candidate / "network_metrics.csv").exists():
            return candidate
        if candidate.resolve() == _RESULTS_ROOT:
            return None
        if _RESULTS_ROOT not in candidate.resolve().parents and candidate.resolve() != _RESULTS_ROOT:
            break
    return None


def _load_network_metrics(path: Path) -> dict[str, Any]:
    steps: list[int] = []
    timestamps_seconds: list[int] = []
    vehicle_count: list[float] = []
    mean_speed_kmh: list[float] = []
    throughput_per_hour: list[float] = []
    mean_delay_seconds: list[float] = []
    active_accidents: list[float] = []
    speed_ratio: list[float] = []

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            steps.append(int(row["step"]))
            timestamps_seconds.append(int(row["timestamp_seconds"]))
            vehicle_count.append(float(row["vehicle_count"]))
            mean_speed_kmh.append(float(row["mean_speed_kmh"]))
            throughput_per_hour.append(float(row["throughput_per_hour"]))
            mean_delay_seconds.append(float(row["mean_delay_seconds"]))
            active_accidents.append(float(row["active_accidents"]))
            speed_ratio.append(float(row["speed_ratio"]))

    stats = {
        "samples": len(steps),
        "peak_vehicle_count": max(vehicle_count) if vehicle_count else 0,
        "peak_throughput_per_hour": max(throughput_per_hour) if throughput_per_hour else 0,
        "peak_active_accidents": max(active_accidents) if active_accidents else 0,
        "max_mean_delay_seconds": max(mean_delay_seconds) if mean_delay_seconds else 0,
        "mean_vehicle_count": round(mean(vehicle_count), 2) if vehicle_count else 0,
        "mean_speed_kmh": round(mean(mean_speed_kmh), 2) if mean_speed_kmh else 0,
        "mean_throughput_per_hour": round(mean(throughput_per_hour), 2) if throughput_per_hour else 0,
        "mean_delay_seconds": round(mean(mean_delay_seconds), 2) if mean_delay_seconds else 0,
        "min_speed_ratio": round(min(speed_ratio), 4) if speed_ratio else 0,
    }

    return {
        "series": {
            "steps": steps,
            "timestamps_seconds": timestamps_seconds,
            "vehicle_count": vehicle_count,
            "mean_speed_kmh": mean_speed_kmh,
            "throughput_per_hour": throughput_per_hour,
            "mean_delay_seconds": mean_delay_seconds,
            "active_accidents": active_accidents,
            "speed_ratio": speed_ratio,
        },
        "stats": stats,
    }


def _load_accidents(path: Path) -> dict[str, Any]:
    items = _json_load(path)
    if not isinstance(items, list):
        items = []

    severity_counts: dict[str, int] = {}
    max_duration = 0
    max_queue = 0
    max_affected = 0
    normalized: list[dict[str, Any]] = []

    for item in items:
        severity = str(item.get("severity", "UNKNOWN")).upper()
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

        impact = item.get("impact", {}) if isinstance(item.get("impact"), dict) else {}
        duration_seconds = int(item.get("duration_seconds", 0) or 0)
        queue = int(impact.get("peak_queue_length_vehicles", 0) or 0)
        affected = int(impact.get("vehicles_affected", 0) or 0)

        max_duration = max(max_duration, duration_seconds)
        max_queue = max(max_queue, queue)
        max_affected = max(max_affected, affected)

        normalized.append(
            {
                "accident_id": item.get("accident_id"),
                "severity": severity,
                "trigger_step": item.get("trigger_step"),
                "resolved_step": item.get("resolved_step"),
                "duration_seconds": duration_seconds,
                "edge_id": item.get("location", {}).get("edge_id") if isinstance(item.get("location"), dict) else None,
                "x": _to_number(str(item.get("location", {}).get("x"))) if isinstance(item.get("location"), dict) else None,
                "y": _to_number(str(item.get("location", {}).get("y"))) if isinstance(item.get("location"), dict) else None,
                "position_on_edge_m": _to_number(str(item.get("location", {}).get("position_on_edge_m"))) if isinstance(item.get("location"), dict) else None,
                "peak_queue_length_vehicles": queue,
                "vehicles_affected": affected,
            }
        )

    return {
        "count": len(normalized),
        "by_severity": severity_counts,
        "max_duration_seconds": max_duration,
        "max_queue_length_vehicles": max_queue,
        "max_vehicles_affected": max_affected,
        "items": normalized,
    }


def _load_antifragility(path: Path) -> dict[str, Any]:
    payload = _json_load(path)
    if not isinstance(payload, dict):
        payload = {}
    return {
        "antifragility_index": payload.get("antifragility_index"),
        "n_events_measured": payload.get("n_events_measured"),
        "total_accidents": payload.get("total_accidents"),
        "std_dev": payload.get("std_dev"),
        "ci_95_low": payload.get("ci_95_low"),
        "ci_95_high": payload.get("ci_95_high"),
        "interpretation": payload.get("interpretation"),
        "per_event": payload.get("per_event", []),
    }


def _collect_artifacts(run_root: Path) -> dict[str, Any]:
    report = run_root / "report.html"
    images = sorted(
        _relative(path)
        for path in run_root.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )
    return {
        "report_path": _relative(report) if report.exists() else None,
        "image_paths": images,
        "raw_files": sorted(_relative(path) for path in run_root.iterdir() if path.is_file()),
    }


def build_run_summary(selected_path: Path) -> dict[str, Any] | None:
    run_root = find_run_root(selected_path.resolve())
    if run_root is None:
        return None

    metadata = _json_load(run_root / "metadata.json")
    network_metrics = _load_network_metrics(run_root / "network_metrics.csv")
    accidents = _load_accidents(run_root / "accident_reports.json") if (run_root / "accident_reports.json").exists() else None
    antifragility = _load_antifragility(run_root / "antifragility_index.json") if (run_root / "antifragility_index.json").exists() else None

    return {
        "run_root": _relative(run_root),
        "metadata": metadata,
        "summary": metadata.get("summary", {}) if isinstance(metadata, dict) else {},
        "config_snapshot": metadata.get("config", {}) if isinstance(metadata, dict) else {},
        "metrics": network_metrics,
        "accidents": accidents,
        "antifragility": antifragility,
        "artifacts": _collect_artifacts(run_root),
    }
