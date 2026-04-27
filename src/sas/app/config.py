from __future__ import annotations

from copy import deepcopy
import logging
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "thessaloniki" / "default.yaml"

_CONFIG_ALIASES = {
    "config.yaml": "thessaloniki/default.yaml",
    "config.seattle.yaml": "seattle/default.yaml",
    "config_thessaloniki_postmetro_50kph.yaml": "thessaloniki/postmetro_50kph.yaml",
    "default.yaml": "thessaloniki/default.yaml",
    "seattle.yaml": "seattle/default.yaml",
    "thessaloniki_postmetro_50kph.yaml": "thessaloniki/postmetro_50kph.yaml",
}


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    """Resolve a config path across legacy and new repository locations."""
    if config_path in (None, ""):
        return DEFAULT_CONFIG_PATH

    raw_path = Path(str(config_path)).expanduser()
    if raw_path.is_absolute():
        return raw_path

    aliased = Path(_CONFIG_ALIASES.get(raw_path.as_posix(), raw_path.as_posix()))
    candidates = [
        Path.cwd() / raw_path,
        CONFIGS_DIR / aliased,
        PROJECT_ROOT / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    if raw_path.parent == Path("."):
        return (CONFIGS_DIR / aliased).resolve()
    return (PROJECT_ROOT / raw_path).resolve()


def _resolve_project_path(path_value: str | Path) -> str:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def _prepare_loaded_config(config: dict[str, Any]) -> dict[str, Any]:
    prepared = deepcopy(config)

    sumo_cfg = prepared.get("sumo", {})
    if isinstance(sumo_cfg.get("config_file"), str):
        sumo_cfg["config_file"] = _resolve_project_path(sumo_cfg["config_file"])

    output_cfg = prepared.get("output", {})
    if isinstance(output_cfg.get("output_folder"), str):
        output_cfg["output_folder"] = _resolve_project_path(output_cfg["output_folder"])

    return prepared


def prepare_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw config dict for execution-time use."""
    return _prepare_loaded_config(config)


def load_config_raw(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config without rewriting repository-relative paths."""
    resolved_path = resolve_config_path(config_path)
    with resolved_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return raw


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config and normalize repository-relative paths."""
    raw = load_config_raw(config_path)
    return _prepare_loaded_config(raw)


def save_config(config: dict[str, Any], path: str | Path | None = None) -> Path:
    """Persist a config for CLI or GUI workflows."""
    target = resolve_config_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return target


def validate_config(config: dict[str, Any]) -> None:
    """
    Check critical config values before starting SUMO.

    This lives in the app layer so GUI code can validate configs without
    importing the simulation runtime or TraCI.
    """
    errors = []

    def _as_float(value: Any, label: str) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            errors.append(f"{label} must be numeric, got {value!r}")
            return None

    def _as_int(value: Any, label: str) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            errors.append(f"{label} must be an integer, got {value!r}")
            return None

    sumo = config.get("sumo", {})
    step_length = _as_float(sumo.get("step_length", 0), "sumo.step_length")
    if step_length is not None and step_length <= 0:
        errors.append(f"sumo.step_length must be > 0, got {step_length}")
    total_steps = _as_int(sumo.get("total_steps", 0), "sumo.total_steps")
    if total_steps is not None and total_steps <= 0:
        errors.append(f"sumo.total_steps must be > 0, got {total_steps}")

    risk = config.get("risk", {})
    bp = risk.get("base_probability", -1)
    if not (0 < bp < 1):
        errors.append(f"risk.base_probability must be in (0, 1), got {bp}")
    th = risk.get("trigger_threshold", -1)
    if not (0 < th < 1):
        errors.append(f"risk.trigger_threshold must be in (0, 1), got {th}")

    acc = config.get("accident", {})
    effect_mode = str(acc.get("incident_effect_mode", "hybrid")).lower()
    if effect_mode not in {"speed_limit", "lane_closure", "hybrid"}:
        errors.append(
            "accident.incident_effect_mode must be one of "
            "['hybrid', 'lane_closure', 'speed_limit']"
        )
    reroute_radius = _as_float(acc.get("reroute_radius_m", 750.0), "accident.reroute_radius_m")
    if reroute_radius is not None and reroute_radius < 0:
        errors.append(f"accident.reroute_radius_m must be >= 0, got {reroute_radius}")
    reroute_interval = _as_int(
        acc.get("reroute_interval_s", 60), "accident.reroute_interval_s"
    )
    if reroute_interval is not None and reroute_interval <= 0:
        errors.append(f"accident.reroute_interval_s must be > 0, got {reroute_interval}")

    severity = acc.get("severity", {})
    if not severity:
        errors.append(
            "accident.severity must define at least one tier "
            "(e.g. minor, moderate, major, critical)"
        )
    else:
        required_tier_keys = {
            "weight",
            "duration_min_s",
            "duration_max_s",
            "lane_capacity_fraction",
            "response_time_s",
            "secondary_risk_radius_m",
            "secondary_risk_multiplier",
        }
        for tier_name, params in severity.items():
            if not isinstance(params, dict):
                errors.append(f"accident.severity.{tier_name} must be a key-value mapping")
                continue
            missing = required_tier_keys - set(params.keys())
            if missing:
                errors.append(f"accident.severity.{tier_name} is missing required keys: {missing}")
                continue
            if params.get("weight", 0) <= 0:
                errors.append(f"accident.severity.{tier_name}.weight must be > 0")
            cf = params.get("lane_capacity_fraction", -1)
            if not (0.0 <= cf <= 1.0):
                errors.append(
                    f"accident.severity.{tier_name}.lane_capacity_fraction "
                    f"must be in [0.0, 1.0], got {cf}"
                )
            d_min = params.get("duration_min_s", 0)
            d_max = params.get("duration_max_s", 0)
            if d_min >= d_max:
                errors.append(
                    f"accident.severity.{tier_name}: duration_min_s ({d_min}) "
                    f"must be < duration_max_s ({d_max})"
                )

    cfg_file = sumo.get("config_file", "")
    if not os.path.exists(cfg_file):
        errors.append(f"sumo.config_file not found: {cfg_file}")

    output = config.get("output", {})
    metrics_interval = _as_int(
        output.get("metrics_interval_steps", 0), "output.metrics_interval_steps"
    )
    if metrics_interval is not None and metrics_interval <= 0:
        errors.append(f"output.metrics_interval_steps must be > 0, got {metrics_interval}")
    pre_window = output.get("pre_window_seconds", 0)
    if pre_window is not None:
        pre_window_value = _as_int(pre_window, "output.pre_window_seconds")
        if pre_window_value is not None and pre_window_value <= 0:
            errors.append(f"output.pre_window_seconds must be > 0, got {pre_window_value}")
    post_window = output.get("post_window_seconds", 0)
    if post_window is not None:
        post_window_value = _as_int(post_window, "output.post_window_seconds")
        if post_window_value is not None and post_window_value <= 0:
            errors.append(f"output.post_window_seconds must be > 0, got {post_window_value}")
    baseline_window = output.get("baseline_window_steps", 0)
    if baseline_window is not None:
        baseline_window_value = _as_int(baseline_window, "output.baseline_window_steps")
        if baseline_window_value is not None and baseline_window_value <= 0:
            errors.append(
                f"output.baseline_window_steps must be > 0, got {baseline_window_value}"
            )
    live_refresh = output.get("live_progress_refresh_steps")
    if live_refresh is not None:
        try:
            live_refresh_value = int(live_refresh)
        except (TypeError, ValueError):
            errors.append(
                "output.live_progress_refresh_steps must be a positive integer when provided"
            )
        else:
            if live_refresh_value <= 0:
                errors.append("output.live_progress_refresh_steps must be > 0")

    if errors:
        for error in errors:
            logging.error("Config validation failed: %s", error)
        raise SystemExit(1)
