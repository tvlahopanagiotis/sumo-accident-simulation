"""
parallel_runner.py
==================

Parallel execution engine for SAS resilience assessments.

Uses concurrent.futures.ProcessPoolExecutor to run multiple SUMO simulation
instances simultaneously, each with its own TraCI TCP port to avoid conflicts.

Progress reporting
------------------
Two complementary mechanisms keep you informed during long runs:

1. **Completion bar** — printed immediately when each scenario finishes,
   showing elapsed time, ETA, and a ✓/✗ per result.

2. **Heartbeat thread** — a daemon thread that wakes every 60 s and prints
   a timestamped status line *even when no scenario has just completed*.
   Also writes ``status.json`` into the output directory so you can check
   progress with::

       cat results/resilience_.../status.json

   or watch it live::

       while true; do cat results/resilience_.../status.json; sleep 30; done
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
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
        from .runner import run_once, write_metadata

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


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as 'Xh YYm ZZs' or 'YYm ZZs'."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _fmt_eta(elapsed: float, completed: int, total: int) -> str:
    """Return a human-readable ETA string based on average scenario time."""
    if completed == 0:
        return "calculating..."
    avg = elapsed / completed
    remaining = total - completed
    eta_sec = avg * remaining
    return f"~{_fmt_duration(eta_sec)} remaining"


class ProgressCallback:
    """
    Stateful progress callback — tracks elapsed time and ETA.

    Prints a one-line updating bar when each scenario finishes::

        [11:42:01]  [########--------]  50.0%  (15/30)  ✓ high_incident_p0p10_s42
                    elapsed: 8m 32s  |  est. ~8m 30s remaining

    The newline is suppressed on intermediate updates so the bar
    stays on one line in terminals that support ``\\r``.
    """

    def __init__(self, bar_width: int = 36) -> None:
        self._start = time.time()
        self._bar_width = bar_width

    def __call__(self, completed: int, total: int, result: dict) -> None:
        elapsed = time.time() - self._start
        status = result.get("status", "unknown")
        folder = os.path.basename(result.get("output_folder", "?"))
        icon = "✓" if status == "success" else "✗"

        pct = completed * 100 / total
        filled = int(self._bar_width * completed / total)
        bar = "█" * filled + "░" * (self._bar_width - filled)

        eta_str = _fmt_eta(elapsed, completed, total)
        elapsed_str = _fmt_duration(elapsed)
        ts = time.strftime("%H:%M:%S")

        # End with \n on the last scenario so the prompt appears cleanly below.
        end = "\n" if completed >= total else "\n"

        print(
            f"  [{ts}]  [{bar}]  {pct:5.1f}%  ({completed}/{total})"
            f"  {icon} {folder}\n"
            f"           elapsed: {elapsed_str}  |  {eta_str}",
            end=end,
            flush=True,
        )


# ---------------------------------------------------------------------------
# Heartbeat thread
# ---------------------------------------------------------------------------


def _heartbeat_fn(
    state: dict,
    lock: threading.Lock,
    stop_event: threading.Event,
    max_workers: int,
    output_dir: str | None,
    interval: int = 60,
) -> None:
    """
    Daemon function: wake every *interval* seconds and print a status line.

    Also writes ``status.json`` to *output_dir* (if provided) so the user
    can inspect progress without watching the terminal.
    """
    while not stop_event.wait(interval):
        with lock:
            completed = state["completed"]
            total = state["total"]
            failed = state["failed"]
            start = state["start"]

        elapsed = time.time() - start
        remaining = total - completed
        # Approximate: at most max_workers running at once
        active = min(max_workers, remaining)

        elapsed_str = _fmt_duration(elapsed)
        eta_str = _fmt_eta(elapsed, completed, total)
        pct = completed * 100 / total if total else 0

        ts = time.strftime("%H:%M:%S")
        ok_count = completed - failed
        fail_str = f", {failed} failed" if failed else ""

        print(
            f"\n  [{ts}] ⏳  Still running — {completed}/{total} done"
            f" ({pct:.0f}%){fail_str}"
            f" | {active} worker(s) active"
            f" | elapsed: {elapsed_str}"
            f" | est. {eta_str}"
            f" | {ok_count} OK",
            flush=True,
        )

        # Write status.json for external monitoring.
        if output_dir:
            status_data = {
                "timestamp": ts,
                "completed": completed,
                "total": total,
                "failed": failed,
                "ok": ok_count,
                "remaining": remaining,
                "active_workers": active,
                "elapsed_seconds": round(elapsed, 1),
                "elapsed_human": elapsed_str,
                "eta_human": eta_str,
                "pct_complete": round(pct, 1),
            }
            try:
                Path(output_dir, "status.json").write_text(
                    json.dumps(status_data, indent=2), encoding="utf-8"
                )
            except OSError:
                pass  # non-critical — don't crash the heartbeat thread


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
        output_dir: str | None = None,
        heartbeat_interval: int = 60,
    ) -> list[dict]:
        """
        Run all scenarios in parallel.

        Args:
            scenarios:          List of (config, seed, output_folder) tuples.
            progress_callback:  Optional callable(completed, total, result)
                                called after each scenario finishes.
                                Defaults to ProgressCallback().
            output_dir:         If provided, the heartbeat thread writes a
                                ``status.json`` file here every
                                *heartbeat_interval* seconds.
            heartbeat_interval: Seconds between heartbeat prints (default 60).

        Returns:
            List of result dicts in submission order.
        """
        if progress_callback is None:
            progress_callback = ProgressCallback()

        total = len(scenarios)
        results: list[dict | None] = [None] * total
        completed = 0

        logger.info(
            "Launching %d scenarios across %d workers (ports %d–%d)",
            total,
            self.max_workers,
            self.base_port,
            self.base_port + total - 1,
        )

        # ── Shared state for heartbeat thread ────────────────────────────
        _lock = threading.Lock()
        _state: dict = {
            "completed": 0,
            "total": total,
            "failed": 0,
            "start": time.time(),
        }
        _stop = threading.Event()

        _hb = threading.Thread(
            target=_heartbeat_fn,
            args=(_state, _lock, _stop, self.max_workers, output_dir, heartbeat_interval),
            daemon=True,
            name="sas-heartbeat",
        )
        _hb.start()
        logger.debug("Heartbeat thread started (interval=%ds)", heartbeat_interval)

        # Print a clear header so the user knows what's about to run.
        print(
            f"\n  {'─' * 60}\n"
            f"  Running {total} scenarios  |  {self.max_workers} worker(s)  "
            f"|  ports {self.base_port}–{self.base_port + total - 1}\n"
            f"  Heartbeat every {heartbeat_interval}s  "
            f"|  monitor: cat {output_dir or '.'}/status.json\n"
            f"  {'─' * 60}",
            flush=True,
        )

        # ── Parallel execution ────────────────────────────────────────────
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx: dict = {}

            for idx, (config, seed, output_folder) in enumerate(scenarios):
                # Unique port per scenario — avoids TraCI conflicts when a slow
                # scenario from batch N is still running when batch N+1 starts.
                port = self.base_port + idx
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

                # Update shared heartbeat state.
                with _lock:
                    _state["completed"] = completed
                    if result.get("status") != "success":
                        _state["failed"] += 1

                if progress_callback:
                    progress_callback(completed, total, result)

        # ── Stop heartbeat ────────────────────────────────────────────────
        _stop.set()
        _hb.join(timeout=2)

        # Write final status.json.
        if output_dir:
            elapsed = time.time() - _state["start"]
            final_status = {
                "timestamp": time.strftime("%H:%M:%S"),
                "status": "complete",
                "completed": completed,
                "total": total,
                "failed": _state["failed"],
                "ok": completed - _state["failed"],
                "elapsed_seconds": round(elapsed, 1),
                "elapsed_human": _fmt_duration(elapsed),
            }
            try:
                Path(output_dir, "status.json").write_text(
                    json.dumps(final_status, indent=2), encoding="utf-8"
                )
            except OSError:
                pass

        # Replace any None entries (shouldn't happen, but be safe).
        return [r if r is not None else {"status": "error", "error": "unknown"} for r in results]
