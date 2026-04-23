from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG_PATH, load_config, save_config, validate_config


class SimulationService:
    """
    Thin programmatic entry point intended for future GUI usage.

    The GUI can load, mutate, validate, and run configs directly in Python
    without shelling out to CLI wrappers or editing YAML by hand.
    """

    def load(self, config_path: str | Path | None = None) -> dict[str, Any]:
        return load_config(config_path or DEFAULT_CONFIG_PATH)

    def save(self, config: dict[str, Any], path: str | Path | None = None) -> Path:
        return save_config(config, path)

    def validate(self, config: dict[str, Any]) -> None:
        validate_config(config)

    def run_once(
        self,
        config: dict[str, Any],
        *,
        seed: int | None = None,
        output_folder: str | None = None,
        enable_live_progress: bool = False,
    ) -> tuple[dict[str, Any], str]:
        from ..simulation.runner import run_once

        run_seed = int(seed if seed is not None else config["sumo"].get("seed", 42))
        run_output = output_folder or config["output"]["output_folder"]
        return run_once(
            config,
            run_seed,
            run_output,
            enable_live_progress=enable_live_progress,
        )


class AssessmentService:
    """Programmatic entry point for resilience assessments."""

    def load(self, config_path: str | Path | None = None) -> dict[str, Any]:
        return load_config(config_path or DEFAULT_CONFIG_PATH)

    def save(self, config: dict[str, Any], path: str | Path | None = None) -> Path:
        return save_config(config, path)

    def run_cli(self, argv: list[str] | None = None) -> None:
        from ..analysis import resilience_assessment as module

        module.main(argv)
