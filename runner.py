"""
sas/runner.py
=============
Main Simulation Runner

Entry point for the SUMO Accident Simulation (SAS).
Orchestrates the TraCI connection, risk model, accident lifecycle,
and metrics collection into a reproducible simulation loop.

Usage
-----
Single run (seed from config.yaml):
    python runner.py

Multiple independent runs — different seeds, results aggregated:
    python runner.py --runs 10

Custom config:
    python runner.py --config experiments/high_risk.yaml --runs 5

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
  8. Every metrics_interval seconds, record a network snapshot
  9. At the end, export results + metadata.json

Performance
-----------
Before optimisation : ~14,950–20,700 TraCI calls per step
                      (~21 M calls for a 2-hour / 1,440-step simulation)

After optimisation  : 2 batch calls per step + one TraCI call per *unique edge*
                      on first encounter (static edge properties are cached).
                      Total individual-call overhead is O(unique_edges) ≈ 8,848
                      for the Thessaloniki network — incurred only once.
"""

import sys
import os
import argparse
import datetime
import json
import logging
import math
import platform
import random
import statistics

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import traci
except ImportError:
    print("ERROR: traci not found. Install SUMO 1.20+ and add it to PYTHONPATH.")
    print("  See README.md for setup instructions.")
    sys.exit(1)

# Short alias for TraCI constants (used heavily in the hot loop)
_tc = traci.constants

from risk_model import RiskModel
from accident_manager import AccidentManager
from metrics import MetricsCollector, _t_critical


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_folder: str, level: int = logging.INFO):
    """Configure console + file logging for the simulation."""
    os.makedirs(output_folder, exist_ok=True)
    fmt      = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(output_folder, "simulation.log"), mode="w"),
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)


logger = logging.getLogger("sas.runner")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load and return the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_config(config: dict):
    """
    Check critical config values before starting SUMO.
    Logs a clear error and exits if anything is invalid.
    """
    errors = []

    risk = config.get("risk", {})
    bp = risk.get("base_probability", -1)
    if not (0 < bp < 1):
        errors.append(f"risk.base_probability must be in (0, 1), got {bp}")
    th = risk.get("trigger_threshold", -1)
    if not (0 < th < 1):
        errors.append(f"risk.trigger_threshold must be in (0, 1), got {th}")

    acc = config.get("accident", {})
    mn  = acc.get("min_duration_seconds", 0)
    mx  = acc.get("max_duration_seconds", 0)
    if mn >= mx:
        errors.append(
            f"accident.min_duration_seconds ({mn}) must be < max_duration_seconds ({mx})"
        )
    rt = acc.get("response_time_seconds", 0)
    if rt >= mx:
        errors.append(
            f"accident.response_time_seconds ({rt}) must be < max_duration_seconds ({mx})"
        )

    sumo     = config.get("sumo", {})
    cfg_file = sumo.get("config_file", "")
    if not os.path.exists(cfg_file):
        errors.append(f"sumo.config_file not found: {cfg_file}")

    if errors:
        for e in errors:
            logging.error("Config validation failed: %s", e)
        sys.exit(1)

    logger.info("Config validated OK.")


# ---------------------------------------------------------------------------
# Single simulation run
# ---------------------------------------------------------------------------

def run_once(config: dict, run_seed: int, output_folder: str) -> tuple[dict, str]:
    """
    Execute one complete simulation run.

    Args:
        config:        Full config dict (sumo / risk / accident / output).
        run_seed:      Random seed for this run.
        output_folder: Directory to write all results into.

    Returns:
        (summary dict, sumo_version string)
    """
    sumo_cfg        = config["sumo"]
    total_steps     = sumo_cfg["total_steps"]
    step_length     = sumo_cfg.get("step_length", 5)
    neighbor_radius = config["risk"].get("neighbor_radius_m", 150.0)

    random.seed(run_seed)

    # Inject the per-run output folder into a shallow copy of config
    run_config = {**config, "output": {**config["output"], "output_folder": output_folder}}

    risk_model       = RiskModel(config["risk"])
    accident_manager = AccidentManager(config["accident"])
    metrics          = MetricsCollector(run_config, run_config["output"])

    # Variables to subscribe on every departing vehicle
    # (fetched in bulk via getAllSubscriptionResults each step)
    _VEHICLE_VARS = [
        _tc.VAR_SPEED,        # current speed  (m/s)
        _tc.VAR_MAXSPEED,     # vehicle max speed (m/s)
        _tc.VAR_ROAD_ID,      # edge ID the vehicle is on
        _tc.VAR_POSITION,     # (x, y) Cartesian position
        _tc.VAR_LANE_ID,      # lane ID
        _tc.VAR_LANEPOSITION, # position along the lane (m)
    ]

    # ── Start SUMO ──────────────────────────────────────────────────────
    sumo_cmd = [
        sumo_cfg.get("binary", "sumo"),
        "-c",                     sumo_cfg["config_file"],
        "--seed",                 str(run_seed),
        "--step-length",          str(step_length),
        "--no-warnings",          "true",
        "--collision.action",     "none",
        "--ignore-route-errors",  "true",
        "--no-step-log",          "true",
        "--duration-log.disable", "true",
    ]
    logger.info("Starting SUMO: %s", " ".join(sumo_cmd))
    traci.start(sumo_cmd)

    sumo_version = traci.getVersion()[1]
    logger.info(
        "SUMO %s  |  seed=%d  |  total_steps=%d  |  step_length=%ds",
        sumo_version, run_seed, total_steps, step_length,
    )
    logger.info("=" * 60)
    logger.info("  SAS — %d simulation seconds  (seed %d)", total_steps, run_seed)
    logger.info("=" * 60)

    step = 0

    try:
        while step < total_steps:
            traci.simulationStep()
            step += step_length

            # ── Subscribe newly-departed vehicles ──────────────────────────
            # Each vehicle is subscribed exactly once at departure.
            #   subscribe()        → populates getAllSubscriptionResults()
            #   subscribeContext() → populates getAllContextSubscriptionResults()
            for vid in traci.simulation.getDepartedIDList():
                try:
                    # Direct subscription: vehicle-level variables
                    traci.vehicle.subscribe(vid, _VEHICLE_VARS)
                    # Context subscription: speed of neighbours within radius
                    traci.vehicle.subscribeContext(
                        vid,
                        _tc.CMD_GET_VEHICLE_VARIABLE,
                        neighbor_radius,
                        [_tc.VAR_SPEED],
                    )
                except traci.exceptions.TraCIException:
                    pass

            # Accumulate arrivals every step for accurate throughput computation
            metrics.accumulate_arrivals(traci.simulation.getArrivedNumber())

            # ── Batch-fetch ALL vehicle data — 2 TraCI calls total ──────────
            # getAllSubscriptionResults() returns:
            #   {vehicle_id: {VAR_SPEED: float, VAR_MAX_SPEED: float,
            #                 VAR_ROAD_ID: str,  VAR_POSITION: (x,y), ...}}
            # getAllContextSubscriptionResults() returns:
            #   {vehicle_id: {neighbour_id: {VAR_SPEED: float}, ...}}
            # Together these replace ~14,950–20,700 individual TraCI calls/step.
            all_sub = traci.vehicle.getAllSubscriptionResults()
            all_ctx = traci.vehicle.getAllContextSubscriptionResults()

            # Pre-compute edge densities from subscription data (no TraCI calls)
            risk_model.prepare_step(all_sub)

            # ── Risk evaluation & accident triggering ──────────────────────
            if accident_manager.can_trigger_accident():

                # Build the set of vehicles already blocking a lane
                accident_vehicle_ids = {
                    acc.vehicle_id
                    for acc in accident_manager.active_accidents.values()
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
                    if risk_model.should_trigger_accident_fast(
                        vehicle_id, vdata, neighbor_speeds, secondary_mult
                    ):
                        accident = accident_manager.trigger_accident(vehicle_id, step)
                        if accident:
                            break   # one accident trigger per step

            # ── Advance accident lifecycles ────────────────────────────────
            prev_active = set(accident_manager.active_accidents.keys())
            accident_manager.update(step)
            curr_active = set(accident_manager.active_accidents.keys())

            for acc_id in prev_active - curr_active:
                resolved = next(
                    (a for a in accident_manager.resolved_accidents
                     if a.accident_id == acc_id),
                    None,
                )
                if resolved:
                    metrics.record_accident_resolved(resolved)

            # ── Network snapshot ───────────────────────────────────────────
            if step % metrics.metrics_interval == 0:
                # Pass pre-fetched subscription data — record_step uses it
                # instead of re-fetching speeds via individual TraCI calls.
                metrics.record_step(
                    step, len(accident_manager.active_accidents), all_sub
                )

                if step % 600 == 0:
                    logger.info(
                        "[%3d min] active_accidents=%d  resolved=%d  vehicles=%d",
                        step // 60,
                        len(accident_manager.active_accidents),
                        len(accident_manager.resolved_accidents),
                        len(all_sub),
                    )

    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user at step %d.", step)

    finally:
        traci.close()
        logger.info("SUMO connection closed.")

    # ── Export results ────────────────────────────────────────────────────
    metrics.export_all()

    # Build summary for metadata
    summary: dict = {
        "steps_run":       step,
        "total_accidents": accident_manager._accident_counter,
    }
    ai_file = os.path.join(output_folder, "antifragility_index.json")
    if os.path.exists(ai_file):
        with open(ai_file) as f:
            ai_data = json.load(f)
        summary["antifragility_index"] = ai_data.get("antifragility_index")
        summary["interpretation"]      = ai_data.get("interpretation", "")
        summary["n_events_measured"]   = ai_data.get("n_events_measured", 0)
        summary["ci_95_low"]           = ai_data.get("ci_95_low")
        summary["ci_95_high"]          = ai_data.get("ci_95_high")

    logger.info(
        "Run complete — %d steps, %d accidents, AI=%s",
        step,
        accident_manager._accident_counter,
        summary.get("antifragility_index", "N/A"),
    )
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
    ts = datetime.datetime.utcnow()
    metadata = {
        "run_id":         f"run_{run_seed:04d}_{ts.strftime('%Y%m%d_%H%M%S')}",
        "timestamp_utc":  ts.isoformat() + "Z",
        "seed":           run_seed,
        "sumo_version":   sumo_version,
        "python_version": platform.python_version(),
        "config":         config,
        "summary":        summary,
    }
    path = os.path.join(output_folder, "metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata → %s", path)


# ---------------------------------------------------------------------------
# Multi-run aggregation
# ---------------------------------------------------------------------------

def aggregate_runs(run_summaries: list[dict]) -> dict:
    """
    Compute aggregate statistics (mean, std, 95% CI) over N simulation runs.

    Returns a dict suitable for serialising to aggregate_summary.json.
    """
    ai_values  = [r["antifragility_index"] for r in run_summaries
                  if r.get("antifragility_index") is not None]
    acc_counts = [r["total_accidents"] for r in run_summaries]

    agg: dict = {
        "n_runs":        len(run_summaries),
        "accident_mean": round(statistics.mean(acc_counts), 2),
        "accident_std":  round(statistics.stdev(acc_counts), 2) if len(acc_counts) > 1 else None,
    }

    if ai_values:
        agg["ai_mean"] = round(statistics.mean(ai_values), 4)
        if len(ai_values) >= 2:
            std    = statistics.stdev(ai_values)
            n      = len(ai_values)
            margin = _t_critical(n) * std / math.sqrt(n)
            agg["ai_std"]        = round(std, 4)
            agg["ai_ci_95_low"]  = round(agg["ai_mean"] - margin, 4)
            agg["ai_ci_95_high"] = round(agg["ai_mean"] + margin, 4)
        else:
            agg["note"] = "CI requires ≥2 runs with valid AI values."

    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SUMO Accident Simulation (SAS) — probabilistic accident modelling"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of independent runs with consecutive seeds (default: 1)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    config      = load_config(args.config)
    base_output = config["output"]["output_folder"]
    base_seed   = config["sumo"].get("seed", 42)

    setup_logging(base_output, level=getattr(logging, args.log_level))
    logger.info("Config: %s", os.path.abspath(args.config))

    validate_config(config)

    # ── Single run ────────────────────────────────────────────────────────
    if args.runs == 1:
        summary, sumo_version = run_once(config, base_seed, base_output)
        write_metadata(base_output, config, base_seed, summary, sumo_version)
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
    logger.info(
        "Starting %d-run batch (seeds %d … %d)",
        args.runs, base_seed, base_seed + args.runs - 1,
    )
    all_summaries = []
    sumo_version  = "unknown"

    for i in range(args.runs):
        seed       = base_seed + i
        run_folder = os.path.join(base_output, f"run_{seed:04d}")
        logger.info("── Run %d / %d  (seed %d) ──", i + 1, args.runs, seed)

        summary, sumo_version = run_once(config, seed, run_folder)
        write_metadata(run_folder, config, seed, summary, sumo_version)
        all_summaries.append(summary)

    # Aggregate and save
    agg        = aggregate_runs(all_summaries)
    agg_folder = os.path.join(base_output, "aggregate")
    os.makedirs(agg_folder, exist_ok=True)

    agg_path = os.path.join(agg_folder, "aggregate_summary.json")
    with open(agg_path, "w") as f:
        json.dump({"runs": all_summaries, "aggregate": agg}, f, indent=2)

    logger.info(
        "\n%s\n  BATCH COMPLETE  (%d runs)\n"
        "  Accidents        : %.1f ± %.1f\n"
        "  AI mean          : %s  (95%%CI [%s, %s])\n"
        "  Aggregate report : %s\n%s",
        "=" * 60, args.runs,
        agg.get("accident_mean", 0), agg.get("accident_std") or 0,
        agg.get("ai_mean", "N/A"),
        agg.get("ai_ci_95_low", "—"), agg.get("ai_ci_95_high", "—"),
        agg_path,
        "=" * 60,
    )


if __name__ == "__main__":
    main()
