"""
sas/metrics.py
==============
Network Performance & Antifragility Metrics Collector

Records network-wide performance data throughout the simulation,
producing structured, reproducible outputs for scientific analysis.

Key metrics
-----------
- Mean network speed and speed ratio vs free-flow baseline
- Vehicle throughput (vehicles/hour completing their routes)
- Mean travel delay vs free-flow baseline
- Per-accident disruption summary (queue length, vehicles affected)
- Antifragility Index (AI) — per-event and aggregate with 95% CI

Antifragility Index methodology
--------------------------------
For each resolved accident event the AI is computed as:

    AI_event = (V_post / V_pre) - 1

where:
    V_pre   = mean network speed in `pre_window_seconds` immediately
              before the accident trigger (accident-free snapshots preferred;
              falls back to all snapshots if none are clean)
    V_post  = mean network speed in `post_window_seconds` immediately
              after the accident resolves (accident-free snapshots only)

The aggregate AI is:
    AI      = mean(AI_event_1, AI_event_2, …)
    95% CI  = AI ± t_crit(n-1) * std(AI_events) / sqrt(n)

Interpretation:
    AI >  0.05  → ANTIFRAGILE — network improved after disruption
    AI ≈  0     → RESILIENT   — network returned to baseline
    AI > -0.20  → FRAGILE     — network did not fully recover
    AI ≤ -0.20  → BRITTLE     — network suffered lasting damage
"""

import csv
import math
import os
import json
import statistics
import logging
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional

import traci

# Short alias for TraCI constants
_tc = traci.constants

logger = logging.getLogger("sas.metrics")


# ---------------------------------------------------------------------------
# t critical values (95% CI, two-tailed) for df = n-1
# Source: standard t-distribution tables
# ---------------------------------------------------------------------------

_T_TABLE = {
    1: 12.706, 2: 4.303,  3: 3.182,  4: 2.776,  5: 2.571,
    6: 2.447,  7: 2.365,  8: 2.306,  9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    20: 2.086, 25: 2.060, 30: 2.042,
}


def _t_critical(n: int) -> float:
    """Two-tailed 95% CI t critical value for sample size n (df = n-1)."""
    df = n - 1
    if df <= 0:
        return float("inf")
    if df >= 30:
        return 1.960   # normal approximation valid for large n
    for k in sorted(_T_TABLE.keys(), reverse=True):
        if df >= k:
            return _T_TABLE[k]
    return _T_TABLE[1]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NetworkSnapshot:
    """A single timestep snapshot of network-wide performance."""
    step: int
    timestamp_seconds: int
    vehicle_count: int
    mean_speed_ms: float          # m/s
    mean_speed_kmh: float         # km/h
    throughput_per_hour: float    # vehicles completing routes per hour
    mean_delay_seconds: float     # delay vs free-flow baseline (seconds)
    active_accidents: int
    speed_ratio: float            # actual / free-flow speed (1.0 = no degradation)


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Collects, stores, and exports all simulation performance data.
    """

    def __init__(self, config: dict, output_config: dict):
        """
        Args:
            config:        Full simulation config dict (needs 'sumo' section
                           for step_length).
            output_config: The 'output' section of config.yaml.
        """
        self.output_folder        = output_config["output_folder"]
        self.metrics_interval     = output_config["metrics_interval_steps"]   # seconds
        self.compute_antifragility = output_config.get("compute_antifragility_index", True)

        # Per-event AI window sizes (seconds of simulation time)
        self._pre_window_s  = output_config.get("pre_window_seconds",  300)
        self._post_window_s = output_config.get("post_window_seconds", 300)

        # Baseline establishment window (seconds)
        self._baseline_steps = output_config.get("baseline_window_steps", 1800)

        os.makedirs(self.output_folder, exist_ok=True)

        # ── Snapshot storage ──────────────────────────────────────────────
        self.snapshots: list[NetworkSnapshot] = []
        self._baseline_speed: Optional[float] = None

        # Rolling speed buffer for antifragility pre-window lookbacks.
        # Maxlen keeps enough history to cover pre_window_s at metrics_interval cadence.
        _maxlen = max(200, (self._pre_window_s // max(self.metrics_interval, 1)) + 20)
        self._speed_buffer: deque = deque(maxlen=_maxlen)

        # ── Throughput accumulator ────────────────────────────────────────
        # Arrivals are accumulated every step and rate-converted at record_step.
        self._arrived_interval: int = 0

        # ── Antifragility tracking ────────────────────────────────────────
        self._recovery_monitors: list[dict] = []   # events awaiting post-window
        self._per_event_ais:     list[dict] = []   # finalised per-event results

        # ── Accident report storage ───────────────────────────────────────
        self._disruption_events: list[dict] = []

        # ── CSV output ────────────────────────────────────────────────────
        self._snapshot_file       = os.path.join(self.output_folder, "network_metrics.csv")
        self._accident_report_file = os.path.join(self.output_folder, "accident_reports.json")
        self._init_csv()

    # -----------------------------------------------------------------------
    # Public API — called from runner.py
    # -----------------------------------------------------------------------

    def accumulate_arrivals(self, n: int):
        """
        Accumulate vehicles that completed their route in this simulation step.

        Must be called every simulation step (not just at metrics intervals).

        Args:
            n: Number of vehicles that arrived this step,
               i.e. traci.simulation.getArrivedNumber().
        """
        self._arrived_interval += n

    def record_step(
        self,
        current_step: int,
        active_accident_count: int,
        all_sub: Optional[dict] = None,
    ):
        """
        Record a network-wide snapshot at the current simulation time.
        Call this every self.metrics_interval seconds of simulation time.

        Args:
            current_step:          Current simulation time (seconds).
            active_accident_count: Number of currently active accidents.
            all_sub:               Optional pre-fetched result of
                                   traci.vehicle.getAllSubscriptionResults().
                                   When supplied, speeds are read from the dict
                                   instead of making N individual TraCI calls.
        """
        if all_sub is not None:
            # Fast path — use pre-fetched subscription data (zero TraCI calls).
            # Filter out SUMO's INVALID_DOUBLE_VALUE sentinel (≈ −1.07e9) which
            # is returned for teleporting / not-yet-inserted vehicles.
            speeds     = [vd[_tc.VAR_SPEED] for vd in all_sub.values()
                          if _tc.VAR_SPEED in vd and vd[_tc.VAR_SPEED] >= 0.0]
            n_vehicles = len(speeds)
        else:
            # Legacy fallback — individual TraCI calls
            vehicle_ids = traci.vehicle.getIDList()
            n_vehicles  = len(vehicle_ids)
            speeds      = [traci.vehicle.getSpeed(v) for v in vehicle_ids]

        if n_vehicles == 0:
            self._arrived_interval = 0
            return

        mean_speed = statistics.mean(speeds)

        logger.info(
            "step=%d  vehicles=%d  speed=%.1f km/h  accidents=%d",
            current_step, n_vehicles, mean_speed * 3.6, active_accident_count,
        )

        # ── Throughput ────────────────────────────────────────────────────
        # _arrived_interval holds all arrivals across the last metrics_interval
        # seconds of simulation time; convert to vehicles per hour.
        throughput = self._arrived_interval * (3600.0 / max(self.metrics_interval, 1))
        self._arrived_interval = 0   # reset for next interval

        # ── Speed ratio vs baseline ───────────────────────────────────────
        if self._baseline_speed and self._baseline_speed > 0:
            speed_ratio = mean_speed / self._baseline_speed
            mean_delay  = (1.0 - speed_ratio) * self.metrics_interval
        else:
            speed_ratio = 1.0
            mean_delay  = 0.0

        # ── Establish / update free-flow baseline ─────────────────────────
        # Use accident-free snapshots in the first baseline_window_steps.
        if current_step <= self._baseline_steps and active_accident_count == 0:
            clean_early = [
                e["mean_speed"] for e in self._speed_buffer
                if e["step"] <= self._baseline_steps and e["active_accidents"] == 0
            ] + [mean_speed]
            if len(clean_early) >= 5:
                self._baseline_speed = statistics.mean(clean_early)

        # ── Update rolling speed buffer ───────────────────────────────────
        self._speed_buffer.append({
            "step":             current_step,
            "mean_speed":       mean_speed,
            "active_accidents": active_accident_count,
        })

        # ── Process post-accident recovery monitors ───────────────────────
        still_monitoring = []
        for monitor in self._recovery_monitors:
            elapsed = current_step - monitor["resolved_step"]
            if elapsed <= self._post_window_s:
                # Only accumulate clean (no active accident) snapshots
                if active_accident_count == 0:
                    monitor["post_speeds"].append(mean_speed)
                still_monitoring.append(monitor)
            else:
                # Post-window expired — finalise this event
                self._finalise_monitor(monitor)
        self._recovery_monitors = still_monitoring

        # ── Build and store snapshot ──────────────────────────────────────
        snapshot = NetworkSnapshot(
            step                = current_step,
            timestamp_seconds   = current_step,
            vehicle_count       = n_vehicles,
            mean_speed_ms       = round(mean_speed, 3),
            mean_speed_kmh      = round(mean_speed * 3.6, 2),
            throughput_per_hour = round(throughput, 1),
            mean_delay_seconds  = round(mean_delay, 2),
            active_accidents    = active_accident_count,
            speed_ratio         = round(speed_ratio, 4),
        )
        self.snapshots.append(snapshot)
        self._write_snapshot_csv(snapshot)

    # -----------------------------------------------------------------------
    # Accident recording
    # -----------------------------------------------------------------------

    def record_accident_resolved(self, accident):
        """
        Record a resolved accident's metrics and register it for AI monitoring.
        Call immediately when an accident transitions to RESOLVED.

        Args:
            accident: The resolved Accident dataclass instance.
        """
        report = {
            "accident_id":    accident.accident_id,
            "trigger_step":   accident.trigger_step,
            "resolved_step":  accident.resolved_step,
            "duration_seconds": accident.resolved_step - accident.trigger_step,
            "location": {
                "edge_id":            accident.edge_id,
                "x":                  round(accident.x, 2),
                "y":                  round(accident.y, 2),
                "position_on_edge_m": round(accident.position, 1),
            },
            "impact": {
                "peak_queue_length_vehicles": accident.peak_queue_length,
                "vehicles_affected":          accident.vehicles_affected_count,
            },
        }
        self._disruption_events.append(report)
        logger.info("Accident %s recorded (duration %ds).",
                    accident.accident_id,
                    report["duration_seconds"])

        # Register for per-event antifragility measurement
        if self.compute_antifragility:
            self._register_recovery_monitor(accident)

    # -----------------------------------------------------------------------
    # Antifragility index
    # -----------------------------------------------------------------------

    def compute_antifragility_index(self) -> dict:
        """
        Compute the aggregate Antifragility Index from per-event measurements.

        Finalises any monitors still open at end of simulation (truncated
        post-window is still usable if ≥1 clean sample exists).

        Returns:
            Dict with antifragility_index, 95% CI, per-event breakdown,
            and an interpretation string.
        """
        # Finalise any monitors whose post-window hasn't expired yet
        for monitor in self._recovery_monitors:
            self._finalise_monitor(monitor)
        self._recovery_monitors = []

        if not self._per_event_ais:
            return {
                "antifragility_index": None,
                "note": (
                    "No per-event data — accidents may have resolved too close "
                    "to the end of simulation, or no accident-free post-window "
                    "observations were recorded."
                ),
            }

        event_values = [e["event_ai"] for e in self._per_event_ais]
        n      = len(event_values)
        mean_ai = statistics.mean(event_values)

        result: dict = {
            "antifragility_index": round(mean_ai, 4),
            "n_events_measured":   n,
            "total_accidents":     len(self._disruption_events),
            "per_event":           self._per_event_ais,
        }

        if n >= 2:
            std_ai = statistics.stdev(event_values)
            margin = _t_critical(n) * std_ai / math.sqrt(n)
            result["std_dev"]     = round(std_ai, 4)
            result["ci_95_low"]   = round(mean_ai - margin, 4)
            result["ci_95_high"]  = round(mean_ai + margin, 4)
        else:
            result["note"] = "Only one event measured — CI requires ≥2 events."

        # Interpretation
        if mean_ai > 0.05:
            result["interpretation"] = "ANTIFRAGILE — Network performance improved after disruptions"
        elif mean_ai > -0.05:
            result["interpretation"] = "RESILIENT — Network returned to near-baseline performance"
        elif mean_ai > -0.20:
            result["interpretation"] = "FRAGILE — Network degraded, did not fully recover"
        else:
            result["interpretation"] = "BRITTLE — Network suffered significant lasting damage"

        return result

    def export_all(self):
        """
        Write all final outputs to disk. Call once at the end of simulation.
        """
        # Sort accident reports chronologically for reproducible output
        self._disruption_events.sort(key=lambda r: r["trigger_step"])
        with open(self._accident_report_file, "w") as f:
            json.dump(self._disruption_events, f, indent=2)
        logger.info("Accident reports → %s", self._accident_report_file)

        if self.compute_antifragility:
            ai_result = self.compute_antifragility_index()
            ai_file   = os.path.join(self.output_folder, "antifragility_index.json")
            with open(ai_file, "w") as f:
                json.dump(ai_result, f, indent=2)

            ai_val  = ai_result.get("antifragility_index", "N/A")
            interp  = ai_result.get("interpretation", ai_result.get("note", ""))
            ci_low  = ai_result.get("ci_95_low",  "—")
            ci_high = ai_result.get("ci_95_high", "—")
            n_ev    = ai_result.get("n_events_measured", 0)
            logger.info(
                "\n%s\n  ANTIFRAGILITY INDEX : %s\n"
                "  95%% CI             : [%s,  %s]  (n=%s events)\n"
                "  %s\n%s",
                "=" * 55, ai_val, ci_low, ci_high, n_ev, interp, "=" * 55,
            )

        logger.info("All outputs saved → %s", self.output_folder)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _register_recovery_monitor(self, accident):
        """
        Set up a post-recovery speed monitor for a resolved accident.
        Looks back in the rolling speed buffer for pre-event speeds.
        """
        trigger = accident.trigger_step
        pre_lo  = trigger - self._pre_window_s

        # Prefer accident-free snapshots for a clean local baseline
        pre_speeds = [
            e["mean_speed"] for e in self._speed_buffer
            if pre_lo <= e["step"] < trigger and e["active_accidents"] == 0
        ]
        if not pre_speeds:
            # Fall back: use any snapshot in the pre-window
            pre_speeds = [
                e["mean_speed"] for e in self._speed_buffer
                if pre_lo <= e["step"] < trigger
            ]

        if not pre_speeds:
            logger.debug(
                "Accident %s: no pre-window snapshots in buffer — skipping AI.",
                accident.accident_id,
            )
            return

        self._recovery_monitors.append({
            "accident_id":    accident.accident_id,
            "resolved_step":  accident.resolved_step,
            "pre_mean_speed": statistics.mean(pre_speeds),
            "n_pre_samples":  len(pre_speeds),
            "post_speeds":    [],
        })

    def _finalise_monitor(self, monitor: dict):
        """Compute event AI from a completed monitor and append to results."""
        if not monitor["post_speeds"]:
            logger.debug(
                "Accident %s: no post-window clean snapshots — skipping AI.",
                monitor["accident_id"],
            )
            return

        pre_spd  = monitor["pre_mean_speed"]
        post_spd = statistics.mean(monitor["post_speeds"])

        if pre_spd is None or pre_spd <= 0:
            return

        event_ai = (post_spd / pre_spd) - 1.0
        self._per_event_ais.append({
            "accident_id":         monitor["accident_id"],
            "event_ai":            round(event_ai, 4),
            "pre_mean_speed_kmh":  round(pre_spd  * 3.6, 2),
            "post_mean_speed_kmh": round(post_spd * 3.6, 2),
            "n_pre_samples":       monitor["n_pre_samples"],
            "n_post_samples":      len(monitor["post_speeds"]),
        })

    def _init_csv(self):
        with open(self._snapshot_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "step", "timestamp_seconds", "vehicle_count",
                "mean_speed_ms", "mean_speed_kmh", "throughput_per_hour",
                "mean_delay_seconds", "active_accidents", "speed_ratio",
            ])
            writer.writeheader()

    def _write_snapshot_csv(self, snapshot: NetworkSnapshot):
        with open(self._snapshot_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(snapshot).keys())
            writer.writerow(asdict(snapshot))
