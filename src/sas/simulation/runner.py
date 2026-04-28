"""
sas/runner.py
=============
Main Simulation Runner

Entry point for the SUMO Accident Simulation (SAS).
Orchestrates the TraCI connection, risk model, accident lifecycle,
and metrics collection into a reproducible simulation loop.

Usage
-----
Single run:
    sas

Multiple independent runs — different seeds, results aggregated:
    sas --runs 10

Custom config:
    sas --config configs/thessaloniki/default.yaml --runs 5

How it works (every simulation step)
-------------------------------------
  1. Advance SUMO by step_length seconds of simulated time
  2. Subscribe new vehicles to all required variables (once per vehicle)
  3. Accumulate arrived-vehicle count for throughput measurement
  4. Fetch ALL vehicle data in 2 TraCI batch calls (getAllSubscriptionResults
     + getAllContextSubscriptionResults) — replaces ~15,000 individual calls
  5. Ask RiskModel.should_trigger_accident_fast() for each vehicle using the
     pre-fetched data — zero additional TraCI calls in the hot loop
  6. If triggered, and concurrency limit allows, create accident via AccidentManager
  7. Advance all active accident lifecycles
  8. Periodically reroute nearby vehicles around active incidents
  9. Every metrics_interval seconds, record a network snapshot
 10. At the end, export results + metadata.json

Performance
-----------
Before optimisation : ~14,950–20,700 TraCI calls per step
                      (~21 M calls for a 2-hour / 1,440-step simulation)

After optimisation  : 2 batch calls per step + one TraCI call per *unique edge*
                      on first encounter (static edge properties are cached).
                      Total individual-call overhead is O(unique_edges) ≈ 8,848
                      for the Thessaloniki network — incurred only once.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import platform
import random
import statistics
import sys
from collections import Counter
from typing import Any

from ..app.config import DEFAULT_CONFIG_PATH, load_config as load_repo_config, validate_config as validate_repo_config

traci = None
_tc = None


def _require_traci():
    """Import TraCI lazily so CLI help works without SUMO installed."""
    global traci, _tc
    if traci is None:
        try:
            import traci as traci_module  # noqa: PLC0415
        except ImportError:
            print("ERROR: traci not found. Install SUMO 1.20+ and add it to PYTHONPATH.")
            print("  See README.md for setup instructions.")
            sys.exit(1)
        traci = traci_module
        _tc = traci.constants
    return traci

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(output_folder: str, level: int = logging.INFO):
    """Configure console + file logging for the simulation."""
    os.makedirs(output_folder, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(output_folder, "simulation.log"), mode="w"),
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)


logger = logging.getLogger("sas.runner")


def _serialize_accidents(accidents: list[Any]) -> list[dict[str, Any]]:
    """Convert Accident objects into lightweight plotting records."""
    return [
        {
            "accident_id": acc.accident_id,
            "x": float(acc.x),
            "y": float(acc.y),
            "severity": str(acc.severity),
            "phase": str(acc.phase),
        }
        for acc in accidents
    ]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict[str, Any]:
    """Load and return the YAML configuration file."""
    return load_repo_config(config_path)


def validate_config(config: dict[str, Any]) -> None:
    validate_repo_config(config)
    logger.info("Config validated OK.")


# ---------------------------------------------------------------------------
# Single simulation run
# ---------------------------------------------------------------------------


def run_once(
    config: dict,
    run_seed: int,
    output_folder: str,
    traci_port: int | None = None,
    traci_label: str | None = None,
    enable_live_progress: bool = False,
) -> tuple[dict, str]:
    """
    Execute one complete simulation run.

    Args:
        config:        Full config dict (sumo / risk / accident / output).
        run_seed:      Random seed for this run.
        output_folder: Directory to write all results into.
        traci_port:    Optional TCP port for TraCI (for parallel execution).
        traci_label:   Optional TraCI connection label (for parallel execution).
        enable_live_progress:
                      If True, show a live Matplotlib dashboard and keep
                      refreshing output_folder/live_progress.png during the run.

    Returns:
        (summary dict, sumo_version string)
    """
    traci_module = _require_traci()
    from ..core.accident_manager import AccidentManager  # noqa: PLC0415
    from ..core.metrics import MetricsCollector  # noqa: PLC0415
    from ..core.risk_model import RiskModel  # noqa: PLC0415

    sumo_cfg = config["sumo"]
    total_steps = sumo_cfg["total_steps"]
    step_length = sumo_cfg.get("step_length", 5)
    neighbor_radius = config["risk"].get("neighbor_radius_m", 150.0)

    random.seed(run_seed)

    # Inject the per-run output folder into a shallow copy of config
    run_config = {**config, "output": {**config["output"], "output_folder": output_folder}}

    risk_model = RiskModel(config["risk"])
    accident_manager = AccidentManager(config["accident"])
    metrics = MetricsCollector(run_config, run_config["output"])
    live_progress = None

    if enable_live_progress:
        try:
            from ..visualization.visualize import LiveProgressDashboard, resolve_net_file

            refresh_steps = int(
                config["output"].get("live_progress_refresh_steps", metrics.metrics_interval)
            )
            live_progress = LiveProgressDashboard(
                output_dir=output_folder,
                total_steps=total_steps,
                refresh_interval_steps=refresh_steps,
                net_xml_path=resolve_net_file(sumocfg_path=sumo_cfg["config_file"]),
            )
            logger.info(
                "Live progress dashboard enabled → %s",
                os.path.join(output_folder, "live_progress.png"),
            )
        except Exception as exc:
            logger.warning("Failed to start live progress dashboard: %s", exc)

    # Variables to subscribe on every departing vehicle
    # (fetched in bulk via getAllSubscriptionResults each step)
    # Note: VAR_MAXSPEED (vehicle mechanical max) is intentionally omitted —
    # speed risk is now normalised against the road's posted limit, which is
    # fetched once per unique edge via _get_road_speed_limit_cached().
    _VEHICLE_VARS = [
        _tc.VAR_SPEED,  # current speed  (m/s)
        _tc.VAR_ROAD_ID,  # edge ID the vehicle is on
        _tc.VAR_POSITION,  # (x, y) Cartesian position
        _tc.VAR_LANE_ID,  # lane ID
        _tc.VAR_LANEPOSITION,  # position along the lane (m)
    ]

    # ── Start SUMO ──────────────────────────────────────────────────────
    sumo_cmd = [
        sumo_cfg.get("binary", "sumo"),
        "-c",
        sumo_cfg["config_file"],
        "--seed",
        str(run_seed),
        "--step-length",
        str(step_length),
        "--no-warnings",
        "true",
        "--collision.action",
        "none",
        "--ignore-route-errors",
        "true",
        "--no-step-log",
        "true",
        "--duration-log.disable",
        "true",
    ]
    logger.info("Starting SUMO: %s", " ".join(sumo_cmd))
    start_kwargs: dict = {}
    if traci_port is not None:
        start_kwargs["port"] = traci_port
    if traci_label is not None:
        start_kwargs["label"] = traci_label
    traci_module.start(sumo_cmd, **start_kwargs)

    sumo_version = traci_module.getVersion()[1]
    logger.info(
        "SUMO %s  |  seed=%d  |  total_steps=%d  |  step_length=%ds",
        sumo_version,
        run_seed,
        total_steps,
        step_length,
    )
    logger.info("=" * 60)
    logger.info("  SAS — %d simulation seconds  (seed %d)", total_steps, run_seed)
    logger.info("=" * 60)

    step = 0
    last_edge_vehicle_counts: dict[str, int] = {}

    try:
        while step < total_steps:
            traci_module.simulationStep()
            step += step_length

            # ── Subscribe newly-departed vehicles ──────────────────────────
            # Each vehicle is subscribed exactly once at departure.
            #   subscribe()        → populates getAllSubscriptionResults()
            #   subscribeContext() → populates getAllContextSubscriptionResults()
            for vid in traci_module.simulation.getDepartedIDList():
                try:
                    # Direct subscription: vehicle-level variables
                    traci_module.vehicle.subscribe(vid, _VEHICLE_VARS)
                    # Context subscription: speed of neighbours within radius
                    traci_module.vehicle.subscribeContext(
                        vid,
                        _tc.CMD_GET_VEHICLE_VARIABLE,
                        neighbor_radius,
                        [_tc.VAR_SPEED],
                    )
                except traci_module.exceptions.TraCIException:
                    pass

            # Accumulate arrivals every step for accurate throughput computation
            metrics.accumulate_arrivals(traci_module.simulation.getArrivedNumber())

            # ── Batch-fetch ALL vehicle data — 2 TraCI calls total ──────────
            # getAllSubscriptionResults() returns:
            #   {vehicle_id: {VAR_SPEED: float, VAR_MAX_SPEED: float,
            #                 VAR_ROAD_ID: str,  VAR_POSITION: (x,y), ...}}
            # getAllContextSubscriptionResults() returns:
            #   {vehicle_id: {neighbour_id: {VAR_SPEED: float}, ...}}
            # Together these replace ~14,950–20,700 individual TraCI calls/step.
            all_sub = traci_module.vehicle.getAllSubscriptionResults()
            all_ctx = traci_module.vehicle.getAllContextSubscriptionResults()

            # Pre-compute edge densities from subscription data (no TraCI calls)
            risk_model.prepare_step(all_sub)

            edge_vehicle_counts: Counter[str] | None = None
            if live_progress is not None:
                edge_vehicle_counts = Counter(
                    str(vdata[_tc.VAR_ROAD_ID])
                    for vdata in all_sub.values()
                    if _tc.VAR_ROAD_ID in vdata
                    and str(vdata[_tc.VAR_ROAD_ID])
                    and not str(vdata[_tc.VAR_ROAD_ID]).startswith(":")
                )
                last_edge_vehicle_counts = dict(edge_vehicle_counts)

            # ── Risk evaluation & accident triggering ──────────────────────
            if accident_manager.can_trigger_accident():
                # Build the set of vehicles already blocking a lane
                accident_vehicle_ids = {
                    acc.vehicle_id for acc in accident_manager.active_accidents.values()
                }

                for vehicle_id, vdata in all_sub.items():
                    # Skip vehicles already involved in an active accident
                    if vehicle_id in accident_vehicle_ids:
                        continue

                    # Position from subscription data — no extra TraCI call
                    x, y = vdata.get(_tc.VAR_POSITION, (0.0, 0.0))

                    secondary_mult = accident_manager.get_secondary_multiplier(x, y)

                    # Build neighbour speed map from context subscription
                    ctx = all_ctx.get(vehicle_id) or {}
                    neighbor_speeds = {
                        nid: nd[_tc.VAR_SPEED]
                        for nid, nd in ctx.items()
                        if nid != vehicle_id and _tc.VAR_SPEED in nd
                    }

                    # Evaluate risk using pre-fetched data — zero TraCI calls
                    if risk_model.should_trigger_accident(
                        vehicle_id, vdata, neighbor_speeds, secondary_mult
                    ):
                        accident = accident_manager.trigger_accident(vehicle_id, step)
                        if accident:
                            break  # one accident trigger per step

            # ── Advance accident lifecycles ────────────────────────────────
            prev_active = set(accident_manager.active_accidents.keys())
            accident_manager.update(step)
            accident_manager.refresh_rerouting(step, all_sub)
            curr_active = set(accident_manager.active_accidents.keys())
            active_accident_points = _serialize_accidents(list(accident_manager.active_accidents.values()))
            resolved_accident_points = _serialize_accidents(
                list(accident_manager.resolved_accidents)
            )

            for acc_id in prev_active - curr_active:
                resolved = next(
                    (a for a in accident_manager.resolved_accidents if a.accident_id == acc_id),
                    None,
                )
                if resolved:
                    metrics.record_accident_resolved(resolved)

            # ── Network snapshot ───────────────────────────────────────────
            if step % metrics.metrics_interval == 0:
                # Pass pre-fetched subscription data — record_step uses it
                # instead of re-fetching speeds via individual TraCI calls.
                metrics.record_step(
                    step,
                    len(accident_manager.active_accidents),
                    all_sub,
                    active_blocked_lanes=sum(
                        len(acc.blocked_lane_ids)
                        for acc in accident_manager.active_accidents.values()
                    ),
                    cumulative_accidents=accident_manager._accident_counter,
                    resolved_accidents=len(accident_manager.resolved_accidents),
                )
                if live_progress is not None:
                    live_progress.update(
                        metrics.snapshots,
                        current_step=step,
                        active_accident_count=len(accident_manager.active_accidents),
                        resolved_accidents=len(accident_manager.resolved_accidents),
                        total_accidents=accident_manager._accident_counter,
                        edge_vehicle_counts=edge_vehicle_counts,
                        accident_points=active_accident_points,
                        resolved_accident_points=resolved_accident_points,
                    )

                if step % 600 == 0:
                    logger.info(
                        "[%3d min] active_accidents=%d  resolved=%d  vehicles=%d",
                        step // 60,
                        len(accident_manager.active_accidents),
                        len(accident_manager.resolved_accidents),
                        len(all_sub),
                    )
            elif live_progress is not None:
                live_progress.update(
                    metrics.snapshots,
                    current_step=step,
                    active_accident_count=len(accident_manager.active_accidents),
                    resolved_accidents=len(accident_manager.resolved_accidents),
                    total_accidents=accident_manager._accident_counter,
                    edge_vehicle_counts=edge_vehicle_counts,
                    accident_points=active_accident_points,
                    resolved_accident_points=resolved_accident_points,
                )

    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user at step %d.", step)

    finally:
        traci_module.close()
        logger.info("SUMO connection closed.")

    # ── Export results ────────────────────────────────────────────────────
    exported = metrics.export_all()

    if live_progress is not None:
        try:
            live_progress.update(
                metrics.snapshots,
                current_step=step,
                active_accident_count=len(accident_manager.active_accidents),
                resolved_accidents=len(accident_manager.resolved_accidents),
                total_accidents=accident_manager._accident_counter,
                edge_vehicle_counts=last_edge_vehicle_counts,
                accident_points=_serialize_accidents(list(accident_manager.active_accidents.values())),
                resolved_accident_points=_serialize_accidents(
                    list(accident_manager.resolved_accidents)
                ),
                force=True,
            )
            live_progress.close()
        except Exception as exc:
            logger.warning("Failed to refresh live progress dashboard: %s", exc)

    # Build summary for metadata
    summary: dict = {
        "steps_run": step,
        "total_accidents": accident_manager._accident_counter,
    }
    ai_data = exported.get("antifragility") or {}
    if ai_data:
        summary["antifragility_index"] = ai_data.get("antifragility_index")
        summary["interpretation"] = ai_data.get("interpretation", "")
        summary["n_events_measured"] = ai_data.get("n_events_measured", 0)
        summary["ci_95_low"] = ai_data.get("ci_95_low")
        summary["ci_95_high"] = ai_data.get("ci_95_high")

    simulation_summary = exported.get("simulation_summary") or {}
    network_summary = simulation_summary.get("network", {})
    accident_summary = simulation_summary.get("accidents", {})
    if isinstance(network_summary, dict):
        summary["peak_vehicle_count"] = network_summary.get("peak_vehicle_count")
        summary["peak_active_accidents"] = network_summary.get("peak_active_accidents")
        summary["peak_active_blocked_lanes"] = network_summary.get("peak_active_blocked_lanes")
        summary["peak_throughput_per_hour"] = network_summary.get("peak_throughput_per_hour")
        summary["mean_speed_kmh"] = network_summary.get("mean_speed_kmh")
        summary["mean_delay_seconds"] = network_summary.get("mean_delay_seconds")
        summary["min_speed_ratio"] = network_summary.get("min_speed_ratio")
    if isinstance(accident_summary, dict):
        summary["accidents_by_severity"] = accident_summary.get("by_severity", {})
        summary["total_rerouted_vehicles"] = accident_summary.get("total_rerouted_vehicles")

    logger.info(
        "Run complete — %d steps, %d accidents, AI=%s",
        step,
        accident_manager._accident_counter,
        summary.get("antifragility_index", "N/A"),
    )

    # ── Generate visualizations ───────────────────────────────────────────
    if config["output"].get("save_accident_heatmap", False):
        try:
            from ..visualization.visualize import (
                generate_html_report,
                plot_accident_heatmap,
                plot_before_after_speeds,
                plot_network_metrics,
                plot_severity_distribution,
            )

            metrics_csv = os.path.join(output_folder, "network_metrics.csv")
            accidents_json = os.path.join(output_folder, "accident_reports.json")

            plot_network_metrics(metrics_csv, output_folder, run_id=str(run_seed))
            plot_severity_distribution(accidents_json, output_folder, run_id=str(run_seed))
            plot_before_after_speeds(
                accidents_json, metrics_csv, output_folder, run_id=str(run_seed)
            )
            plot_accident_heatmap(accidents_json, output_folder, run_id=str(run_seed))
            generate_html_report(output_folder, run_id=str(run_seed), config=config)

            logger.info("Visualizations generated → %s/", output_folder)
        except Exception as exc:
            logger.warning("Failed to generate visualizations: %s", exc)

    return summary, sumo_version


# ---------------------------------------------------------------------------
# Metadata export
# ---------------------------------------------------------------------------


def write_metadata(
    output_folder: str,
    config: dict,
    run_seed: int,
    summary: dict,
    sumo_version: str,
):
    """Write a metadata.json alongside every run's results."""
    ts = datetime.datetime.now(datetime.timezone.utc)
    metadata = {
        "run_id": f"run_{run_seed:04d}_{ts.strftime('%Y%m%d_%H%M%S')}",
        "timestamp_utc": ts.isoformat(),
        "seed": run_seed,
        "sumo_version": sumo_version,
        "python_version": platform.python_version(),
        "config": config,
        "summary": summary,
    }
    path = os.path.join(output_folder, "metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata → %s", path)


# ---------------------------------------------------------------------------
# Multi-run aggregation
# ---------------------------------------------------------------------------


def aggregate_runs(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute aggregate statistics (mean, std, 95% CI) over N simulation runs.

    Returns a dict suitable for serialising to aggregate_summary.json.
    """
    ai_values = [
        r["antifragility_index"] for r in run_summaries if r.get("antifragility_index") is not None
    ]
    acc_counts = [r["total_accidents"] for r in run_summaries]

    agg: dict = {
        "n_runs": len(run_summaries),
        "accident_mean": round(statistics.mean(acc_counts), 2),
        "accident_std": round(statistics.stdev(acc_counts), 2) if len(acc_counts) > 1 else None,
    }

    if ai_values:
        agg["ai_mean"] = round(statistics.mean(ai_values), 4)
        if len(ai_values) >= 2:
            from ..core.metrics import _t_critical  # noqa: PLC0415

            std = statistics.stdev(ai_values)
            n = len(ai_values)
            margin = _t_critical(n) * std / math.sqrt(n)
            agg["ai_std"] = round(std, 4)
            agg["ai_ci_95_low"] = round(agg["ai_mean"] - margin, 4)
            agg["ai_ci_95_high"] = round(agg["ai_mean"] + margin, 4)
        else:
            agg["note"] = "CI requires ≥2 runs with valid AI values."

    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Output folder naming
# ---------------------------------------------------------------------------


def _generate_output_folder_name(
    base_path: str,
    network_name_from_config: str,
    run_number: int = 1,
    is_batch: bool = False,
) -> str:
    """
    Generate a coherent output folder name with network name, run number, and timestamp.

    Args:
        base_path: Base output directory (from config.yaml)
        network_name_from_config: Path to network config file, e.g., '/path/to/thessaloniki.sumocfg'
        run_number: Run number (1, 2, 3, ...) for single-run naming
        is_batch: If True, generate batch folder name instead

    Returns:
        Full path to output folder
    """
    # Extract network name from config path
    config_filename = os.path.basename(network_name_from_config)
    network_name = os.path.splitext(config_filename)[0].capitalize()

    # Generate timestamp
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H:%M")

    # Generate folder name
    if is_batch:
        folder_name = f"{network_name}_Batch_{timestamp}"
    else:
        folder_name = f"{network_name}_Run{run_number}_{timestamp}"

    return os.path.join(base_path, folder_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="sas",
        description=(
            "SAS — SUMO Accident Simulation\n"
            "Probabilistic traffic accident simulator with Antifragility Index measurement.\n"
            "\n"
            "Triggers accidents stochastically based on vehicle speed, speed variance,\n"
            "and road density (Nilsson Power Model). Measures how the network recovers\n"
            "using the Antifragility Index (AI > 0 = improved, AI ≈ 0 = resilient,\n"
            "AI < 0 = degraded)."
        ),
        epilog=(
            "examples:\n"
            "  sas                                           # single run, default config\n"
            "  sas --runs 10                                 # 10 runs, aggregate statistics\n"
            "  sas --config configs/thessaloniki/default.yaml --runs 5\n"
            "  sas --log-level DEBUG                         # verbose output\n"
            "\n"
            "output files (written to output_folder in config.yaml):\n"
            "  network_metrics.csv        step-by-step speed, throughput, accidents\n"
            "  accident_reports.json      per-accident impact (queue length, duration)\n"
            "  antifragility_index.json   AI score + 95%% confidence interval\n"
            "  metadata.json              full config + summary for reproducibility\n"
            "\n"
            "config:\n"
            "  Edit configs/thessaloniki/default.yaml to change the network, accident rate,\n"
            "  distribution, and output settings. Every parameter is documented inline.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH.name,
        metavar="FILE",
        help=f"path to YAML configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "number of independent simulation runs to execute, each with a "
            "different random seed (seed, seed+1, …, seed+N-1). "
            "Results are aggregated into aggregate/aggregate_summary.json. "
            "(default: 1)"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help=(
            "logging verbosity: DEBUG shows every risk score and TraCI call; "
            "INFO shows step summaries and accident events; "
            "WARNING shows only errors; "
            "(default: INFO)"
        ),
    )
    parser.add_argument(
        "--live-progress",
        action="store_true",
        help=(
            "show a live Matplotlib dashboard during a single headless run and "
            "refresh live_progress.png in the output folder"
        ),
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    base_output_path = config["output"]["output_folder"]  # base directory
    base_seed = config["sumo"].get("seed", 42)
    live_progress_enabled = bool(config.get("output", {}).get("live_progress", False))
    if args.live_progress:
        live_progress_enabled = True

    # Generate output folder name with timestamp and run/batch info
    if args.runs == 1:
        # Single run: Network_Run1_YYYY-MM-DD_HH:MM
        output_folder = _generate_output_folder_name(
            base_output_path,
            config["sumo"]["config_file"],
            run_number=1,
            is_batch=False,
        )
    else:
        # Multi-run batch: Network_Batch_YYYY-MM-DD_HH:MM
        output_folder = _generate_output_folder_name(
            base_output_path,
            config["sumo"]["config_file"],
            run_number=1,  # not used for batch
            is_batch=True,
        )

    setup_logging(output_folder, level=getattr(logging, args.log_level))
    logger.info("Config: %s", os.path.abspath(args.config))

    validate_config(config)

    # ── Single run ────────────────────────────────────────────────────────
    if args.runs == 1:
        summary, sumo_version = run_once(
            config,
            base_seed,
            output_folder,
            enable_live_progress=live_progress_enabled,
        )
        write_metadata(output_folder, config, base_seed, summary, sumo_version)
        logger.info(
            "\n  Antifragility Index : %s\n"
            "  95%% CI             : [%s, %s]\n"
            "  Interpretation      : %s",
            summary.get("antifragility_index", "N/A"),
            summary.get("ci_95_low", "—"),
            summary.get("ci_95_high", "—"),
            summary.get("interpretation", ""),
        )
        return

    # ── Multi-run batch ───────────────────────────────────────────────────
    if live_progress_enabled:
        logger.warning(
            "Live progress dashboard is only supported for single runs; ignoring it for --runs %d.",
            args.runs,
        )
    logger.info(
        "Starting %d-run batch (seeds %d … %d)",
        args.runs,
        base_seed,
        base_seed + args.runs - 1,
    )
    all_summaries = []
    sumo_version = "unknown"

    for i in range(args.runs):
        seed = base_seed + i
        run_folder = os.path.join(output_folder, f"run_{seed:04d}")
        logger.info("── Run %d / %d  (seed %d) ──", i + 1, args.runs, seed)

        summary, sumo_version = run_once(config, seed, run_folder)
        write_metadata(run_folder, config, seed, summary, sumo_version)
        all_summaries.append(summary)

    # Aggregate and save
    agg = aggregate_runs(all_summaries)
    agg_folder = os.path.join(output_folder, "aggregate")
    os.makedirs(agg_folder, exist_ok=True)

    agg_path = os.path.join(agg_folder, "aggregate_summary.json")
    with open(agg_path, "w") as f:
        json.dump({"runs": all_summaries, "aggregate": agg}, f, indent=2)

    logger.info(
        "\n%s\n  BATCH COMPLETE  (%d runs)\n"
        "  Accidents        : %.1f ± %.1f\n"
        "  AI mean          : %s  (95%%CI [%s, %s])\n"
        "  Aggregate report : %s\n%s",
        "=" * 60,
        args.runs,
        agg.get("accident_mean", 0),
        agg.get("accident_std") or 0,
        agg.get("ai_mean", "N/A"),
        agg.get("ai_ci_95_low", "—"),
        agg.get("ai_ci_95_high", "—"),
        agg_path,
        "=" * 60,
    )

    # ── Generate batch-level visualizations ─────────────────────────────────
    if config["output"].get("save_accident_heatmap", False):
        try:
            from ..visualization.visualize import visualize_batch_results

            visualize_batch_results(output_folder, all_summaries)
            logger.info("Batch visualization saved → %s/batch_ai_distribution.png", output_folder)
        except Exception as exc:
            logger.warning("Failed to generate batch visualization: %s", exc)


if __name__ == "__main__":
    main()
