"""
parallel_runner.py
==================

Parallel execution engine for SAS resilience assessments.

Uses concurrent.futures.ProcessPoolExecutor to run multiple SUMO simulation
instances simultaneously, each with its own TraCI TCP port to avoid conflicts.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker function (executed in child processes)
# ---------------------------------------------------------------------------


def _run_scenario_worker(
    scenario_config: dict,
    seed: int,
    output_folder: str,
    traci_port: int,
    worker_id: int,
) -> dict:
    """
    Worker function executed in a child process.

    Each worker:
      1. Sets up file-based logging (no stdout to avoid interleaving).
      2. Calls runner.run_once() with its assigned TraCI port.
      3. Returns a result dict with status and summary.

    Args:
        scenario_config: Full config dict for this scenario.
        seed:            Random seed for this run.
        output_folder:   Output directory for this run.
        traci_port:      Unique TCP port for TraCI connection.
        worker_id:       Numeric worker identifier.

    Returns:
        Dict with keys: status ("success"|"error"), summary, output_folder,
        and optionally error message.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Configure logging to file only (avoid stdout interleaving).
    log_path = os.path.join(output_folder, "simulation.log")
    worker_logger = logging.getLogger()
    worker_logger.handlers.clear()
    fh = logging.FileHandler(log_path)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    worker_logger.addHandler(fh)
    worker_logger.setLevel(logging.INFO)

    label = f"worker_{worker_id}"

    try:
        from runner import run_once, write_metadata

        summary, sumo_version = run_once(
            scenario_config,
            seed,
            output_folder,
            traci_port=traci_port,
            traci_label=label,
        )
        write_metadata(output_folder, scenario_config, seed, summary, sumo_version)
        return {
            "status": "success",
            "summary": summary,
            "output_folder": output_folder,
        }
    except Exception as exc:
        logging.getLogger(__name__).error(
            "Worker %d failed: %s",
            worker_id,
            exc,
            exc_info=True,
        )
        return {
            "status": "error",
            "error": str(exc),
            "output_folder": output_folder,
        }


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


def default_progress_callback(completed: int, total: int, result: dict) -> None:
    """Print a text-based progress bar to stdout."""
    status = result.get("status", "unknown")
    folder = os.path.basename(result.get("output_folder", "?"))
    symbol = "OK" if status == "success" else "FAIL"
    pct = completed * 100 / total
    bar_width = 40
    filled = int(bar_width * completed / total)
    bar = "#" * filled + "-" * (bar_width - filled)
    end = "" if completed < total else "\n"
    print(
        f"\r  [{bar}] {pct:5.1f}%  ({completed}/{total})  {folder}: {symbol}",
        end=end,
        flush=True,
    )


# ---------------------------------------------------------------------------
# Parallel executor
# ---------------------------------------------------------------------------


class ParallelExecutor:
    """
    Manages parallel execution of multiple SUMO scenarios.

    Each SUMO instance runs in its own process with a unique TraCI TCP port.
    Resource estimates: ~400 MB RAM per SUMO instance (Thessaloniki network).
    """

    def __init__(
        self,
        max_workers: int | None = None,
        base_port: int = 10000,
    ):
        """
        Args:
            max_workers: Number of parallel SUMO instances.
                         Defaults to min(cpu_count - 1, 8).
            base_port:   Starting port number for TraCI connections.
        """
        if max_workers is None:
            max_workers = min(max(1, (os.cpu_count() or 4) - 1), 8)
        self.max_workers = max_workers
        self.base_port = base_port

    def execute_scenarios(
        self,
        scenarios: list[tuple[dict, int, str]],
        progress_callback: Any = None,
    ) -> list[dict]:
        """
        Run all scenarios in parallel.

        Args:
            scenarios: List of (config, seed, output_folder) tuples.
            progress_callback: Optional callable(completed, total, result)
                               called after each scenario finishes.
                               Defaults to default_progress_callback.

        Returns:
            List of result dicts in submission order.
        """
        if progress_callback is None:
            progress_callback = default_progress_callback

        total = len(scenarios)
        results: list[dict | None] = [None] * total
        completed = 0

        logger.info(
            "Launching %d scenarios across %d workers (ports %d–%d)",
            total,
            self.max_workers,
            self.base_port,
            self.base_port + self.max_workers - 1,
        )

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx: dict = {}

            for idx, (config, seed, output_folder) in enumerate(scenarios):
                port = self.base_port + (idx % self.max_workers)
                future = executor.submit(
                    _run_scenario_worker,
                    config,
                    seed,
                    output_folder,
                    port,
                    idx,
                )
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result(timeout=7200)  # 2 hours max per scenario
                except Exception as exc:
                    result = {
                        "status": "error",
                        "error": str(exc),
                        "output_folder": scenarios[idx][2],
                    }

                results[idx] = result
                completed += 1

                if progress_callback:
                    progress_callback(completed, total, result)

        # Replace any None entries (shouldn't happen, but be safe).
        return [r if r is not None else {"status": "error", "error": "unknown"} for r in results]
