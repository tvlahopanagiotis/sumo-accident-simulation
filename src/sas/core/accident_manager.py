"""
sas/accident_manager.py
========================
Accident Lifecycle Manager

Handles everything that happens AFTER an accident is triggered:

  Phase 1 — ACTIVE:    The accident vehicle is stopped; the affected edge is
                       degraded according to the incident severity using a
                       hybrid of lane closures and reduced lane speeds.
                       Emergency services are dispatched (timer starts).

  Phase 2 — CLEARING:  After response_time_s seconds, the scene is attended
                       and speeds on the remaining open lanes ramp back up
                       linearly while explicit lane closures stay in place.

  Phase 3 — RESOLVED:  Full lane access and speeds are restored; the incident
                       is archived for reporting.

Severity model
--------------
Each accident is randomly assigned a severity tier at trigger time. Tiers are
defined in the 'accident.severity' section of config.yaml and are drawn from a
weighted distribution calibrated to NHTSA KABCO injury classification data:

    MINOR     ≈ 62 %  — property damage only; short blockage, quick clearance
    MODERATE  ≈ 28 %  — non-incapacitating injury; lane partially closed
    MAJOR     ≈  8 %  — incapacitating injury; near-total closure, long scene
    CRITICAL  ≈  2 %  — fatal; full closure, forensic investigation required

Every tier has its own duration range, capacity fraction, emergency response
time, and secondary-risk zone. All parameters are user-configurable via
config.yaml.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any

import traci

logger = logging.getLogger("sas.accident")
_tc = traci.constants

# SUMO speedMode bit-flags (see SUMO docs: TraCI / Change Vehicle State)
# Bit pattern 31 = 11111₂ restores all SUMO safety checks.
# Bit pattern  0 = 00000₂ disables all checks, allowing forced speed override.
_SPEED_MODE_DEFAULT = 31
_SPEED_MODE_FROZEN = 0

_INCIDENT_EFFECT_MODES = {"speed_limit", "lane_closure", "hybrid"}
_CLOSED_LANE_VCLASSES = [
    "private",
    "passenger",
    "taxi",
    "bus",
    "coach",
    "delivery",
    "truck",
    "trailer",
    "motorcycle",
    "moped",
    "evehicle",
    "authority",
    "vip",
]


class Severity:
    """String constants for the four severity tiers."""

    MINOR = "MINOR"
    MODERATE = "MODERATE"
    MAJOR = "MAJOR"
    CRITICAL = "CRITICAL"


@dataclass
class Accident:
    """
    A single accident event and its complete lifecycle state.

    Severity-specific parameters (capacity_fraction, response_time_steps,
    secondary_risk_radius_m, secondary_risk_multiplier) are sampled at
    trigger time and stored on the instance so each accident is self-contained.
    """

    # Identity
    accident_id: str
    trigger_step: int
    vehicle_id: str
    edge_id: str
    lane_id: str
    position: float
    x: float
    y: float

    # Severity
    severity: str = Severity.MINOR

    # Timing
    duration_steps: int = 0
    response_time_steps: int = 0
    clearance_start_step: int = 0
    resolved_step: int = 0
    last_reroute_step: int = -1

    # Traffic impact
    capacity_fraction: float = 0.70
    secondary_risk_radius_m: float = 75.0
    secondary_risk_multiplier: float = 1.5
    managed_lane_ids: list[str] = field(default_factory=list)
    blocked_lane_ids: list[str] = field(default_factory=list)

    # Lifecycle state
    phase: str = "ACTIVE"
    affected_vehicles: list[str] = field(default_factory=list)

    # Impact metrics
    peak_queue_length: int = 0
    vehicles_affected_count: int = 0
    rerouted_vehicle_count: int = 0


class AccidentManager:
    """
    Manages the full lifecycle of all active accidents in the simulation.

    Exposes these methods to runner.py:
        can_trigger_accident()  — True if another accident is allowed
        trigger_accident()      — Initiate a new accident
        update()                — Advance all active accidents one step
        refresh_rerouting()     — Periodically reroute nearby traffic
        get_secondary_multiplier() — Risk elevation near active scenes
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.max_concurrent: int = int(config["max_concurrent_accidents"])
        self.secondary_enabled: bool = bool(config.get("secondary_accident_enabled", True))
        self.effect_mode: str = str(config.get("incident_effect_mode", "hybrid")).lower()
        if self.effect_mode not in _INCIDENT_EFFECT_MODES:
            raise ValueError(
                "accident.incident_effect_mode must be one of "
                f"{sorted(_INCIDENT_EFFECT_MODES)}, got {self.effect_mode!r}"
            )
        self.reroute_enabled: bool = bool(config.get("reroute_affected_vehicles", True))
        self.reroute_radius_m: float = float(config.get("reroute_radius_m", 750.0))
        self.reroute_interval_s: int = int(config.get("reroute_interval_s", 60))

        self._severity_tiers: list[dict] = self._load_severity_tiers(config["severity"])
        self._severity_weights: list[float] = [t["weight"] for t in self._severity_tiers]

        self.active_accidents: dict[str, Accident] = {}
        self.resolved_accidents: list[Accident] = []
        self._accident_counter: int = 0

    def can_trigger_accident(self) -> bool:
        """Return True if another concurrent accident is allowed."""
        return len(self.active_accidents) < self.max_concurrent

    def trigger_accident(self, vehicle_id: str, current_step: int) -> Accident | None:
        """
        Initiate a new accident at the vehicle's current position.

        Samples a severity tier, draws a clipped log-normal duration, applies
        the incident effects, and registers the accident for lifecycle tracking.
        """
        try:
            edge_id = traci.vehicle.getRoadID(vehicle_id)
            lane_id = traci.vehicle.getLaneID(vehicle_id)
            position = traci.vehicle.getLanePosition(vehicle_id)
            x, y = traci.vehicle.getPosition(vehicle_id)
        except traci.exceptions.TraCIException:
            return None

        if not edge_id or edge_id.startswith(":"):
            return None
        if any(acc.edge_id == edge_id for acc in self.active_accidents.values()):
            return None

        tier = random.choices(self._severity_tiers, weights=self._severity_weights, k=1)[0]
        duration = self._sample_duration(tier["duration_min_s"], tier["duration_max_s"])

        self._accident_counter += 1
        accident = Accident(
            accident_id=f"ACC_{self._accident_counter:04d}",
            trigger_step=current_step,
            vehicle_id=vehicle_id,
            edge_id=edge_id,
            lane_id=lane_id,
            position=position,
            x=x,
            y=y,
            severity=tier["name"],
            duration_steps=duration,
            response_time_steps=tier["response_time_s"],
            capacity_fraction=tier["lane_capacity_fraction"],
            secondary_risk_radius_m=tier["secondary_risk_radius_m"],
            secondary_risk_multiplier=tier["secondary_risk_multiplier"],
        )

        self._apply_accident_effects(accident)
        self.active_accidents[accident.accident_id] = accident

        logger.info(
            "ACCIDENT %s — severity: %s | vehicle: %s | edge: %s | "
            "duration: %ds | remaining_capacity: %.0f%% | blocked_lanes: %d",
            accident.accident_id,
            tier["name"],
            vehicle_id,
            edge_id,
            duration,
            tier["lane_capacity_fraction"] * 100,
            len(accident.blocked_lane_ids),
        )
        return accident

    def update(self, current_step: int) -> None:
        """Advance all active accident lifecycles by one simulation second."""
        resolved_ids = []

        for acc_id, accident in self.active_accidents.items():
            elapsed = current_step - accident.trigger_step

            if accident.phase == "ACTIVE":
                self._update_active(accident, current_step, elapsed)
            elif accident.phase == "CLEARING":
                self._update_clearing(accident, current_step)

            if elapsed >= accident.duration_steps:
                self._resolve_accident(accident, current_step)
                resolved_ids.append(acc_id)

        for acc_id in resolved_ids:
            self.resolved_accidents.append(self.active_accidents.pop(acc_id))

    def refresh_rerouting(self, current_step: int, all_sub: dict[str, dict]) -> None:
        """
        Periodically reroute nearby traffic around each active incident.

        The rerouting scope is local rather than network-wide: only vehicles
        within reroute_radius_m of an active incident are candidates.
        """
        if not self.reroute_enabled or not all_sub:
            return

        for accident in self.active_accidents.values():
            if accident.phase == "RESOLVED":
                continue
            if (
                accident.last_reroute_step >= 0
                and current_step - accident.last_reroute_step < self.reroute_interval_s
            ):
                continue

            rerouted = self._reroute_nearby_vehicles(accident, all_sub)
            accident.last_reroute_step = current_step
            if rerouted:
                logger.info(
                    "ACCIDENT %s — rerouted %d nearby vehicles within %.0f m.",
                    accident.accident_id,
                    rerouted,
                    self.reroute_radius_m,
                )

    def get_secondary_multiplier(self, x: float, y: float) -> float:
        """Return the highest secondary-risk multiplier at position (x, y)."""
        if not self.secondary_enabled:
            return 1.0

        best = 1.0
        for accident in self.active_accidents.values():
            dist = math.sqrt((x - accident.x) ** 2 + (y - accident.y) ** 2)
            if dist <= accident.secondary_risk_radius_m:
                best = max(best, accident.secondary_risk_multiplier)
        return best

    @staticmethod
    def _load_severity_tiers(severity_cfg: dict) -> list[dict]:
        required = {
            "weight",
            "duration_min_s",
            "duration_max_s",
            "lane_capacity_fraction",
            "response_time_s",
            "secondary_risk_radius_m",
            "secondary_risk_multiplier",
        }
        tiers = []
        for tier_name, params in severity_cfg.items():
            missing = required - set(params.keys())
            if missing:
                raise ValueError(f"Severity tier '{tier_name}' is missing keys: {missing}")
            tiers.append({"name": tier_name.upper(), **params})

        if not tiers:
            raise ValueError("No severity tiers defined in accident.severity config.")
        return tiers

    @staticmethod
    def _sample_duration(min_s: int, max_s: int) -> int:
        mu = math.log(math.sqrt(min_s * max_s))
        sigma = 0.5
        raw = random.lognormvariate(mu, sigma)
        return int(max(min_s, min(max_s, raw)))

    def _apply_accident_effects(self, accident: Accident) -> None:
        """Freeze the vehicle and degrade the affected edge immediately."""
        try:
            traci.vehicle.setSpeed(accident.vehicle_id, 0.0)
            traci.vehicle.setSpeedMode(accident.vehicle_id, _SPEED_MODE_FROZEN)
        except traci.exceptions.TraCIException as exc:
            logger.warning(
                "Failed to freeze vehicle %s for %s: %s",
                accident.vehicle_id,
                accident.accident_id,
                exc,
            )

        lane_ids, original_speeds = self._collect_edge_lane_speeds(accident.edge_id)
        accident.managed_lane_ids = lane_ids
        accident.blocked_lane_ids = self._select_blocked_lanes(
            lane_ids, accident.lane_id, accident.capacity_fraction
        )
        accident.__dict__["_original_lane_speeds"] = original_speeds
        accident.__dict__["_original_lane_disallowed"] = self._collect_disallowed_classes(
            accident.blocked_lane_ids
        )

        open_speed_factor = self._compute_open_lane_speed_factor(
            lane_count=len(lane_ids),
            blocked_lane_count=len(accident.blocked_lane_ids),
            capacity_fraction=accident.capacity_fraction,
        )
        blocked_speed_factor = (
            min(open_speed_factor, 0.15) if accident.blocked_lane_ids else open_speed_factor
        )
        accident.__dict__["_open_lane_speed_factor"] = open_speed_factor
        accident.__dict__["_blocked_lane_speed_factor"] = blocked_speed_factor

        if self.effect_mode in {"lane_closure", "hybrid"}:
            self._apply_lane_closures(accident.blocked_lane_ids)

        if self.effect_mode in {"speed_limit", "hybrid"}:
            self._set_lane_speeds_for_phase(
                accident,
                open_speed_factor=open_speed_factor,
                blocked_speed_factor=blocked_speed_factor,
            )

    def _collect_edge_lane_speeds(self, edge_id: str) -> tuple[list[str], dict[str, float]]:
        lane_ids: list[str] = []
        speeds: dict[str, float] = {}

        try:
            lane_count = int(traci.edge.getLaneNumber(edge_id))
        except traci.exceptions.TraCIException as exc:
            logger.warning("Failed to read lane count for edge %s: %s", edge_id, exc)
            lane_count = 1

        for lane_index in range(max(lane_count, 1)):
            lane_id = f"{edge_id}_{lane_index}"
            try:
                speeds[lane_id] = float(traci.lane.getMaxSpeed(lane_id))
                lane_ids.append(lane_id)
            except traci.exceptions.TraCIException as exc:
                logger.debug("Cannot read max speed for lane %s: %s", lane_id, exc)

        if not lane_ids:
            fallback_lane = f"{edge_id}_0"
            lane_ids = [fallback_lane]
            speeds[fallback_lane] = 13.9

        return lane_ids, speeds

    def _select_blocked_lanes(
        self,
        lane_ids: list[str],
        incident_lane_id: str,
        capacity_fraction: float,
    ) -> list[str]:
        """
        Translate remaining capacity into a discrete set of blocked lanes.

        Single-lane edges are only fully closed for critical incidents.
        Multi-lane edges lose floor((1 - cf) * n_lanes) lanes.
        """
        if self.effect_mode == "speed_limit" or not lane_ids:
            return []

        lane_count = len(lane_ids)
        if capacity_fraction <= 0.0:
            blocked_count = lane_count
        elif lane_count == 1:
            blocked_count = 0
        else:
            blocked_count = int(math.floor((1.0 - capacity_fraction) * lane_count))

        blocked_count = max(0, min(blocked_count, lane_count))
        if blocked_count == 0:
            return []

        incident_index = self._lane_index(incident_lane_id)
        ranked = sorted(
            lane_ids,
            key=lambda lane_id: (abs(self._lane_index(lane_id) - incident_index), self._lane_index(lane_id)),
        )
        blocked = ranked[:blocked_count]
        if incident_lane_id in lane_ids and incident_lane_id not in blocked:
            blocked = [incident_lane_id, *blocked[:-1]]
        return blocked

    @staticmethod
    def _lane_index(lane_id: str) -> int:
        try:
            return int(lane_id.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return 0

    @staticmethod
    def _compute_open_lane_speed_factor(
        lane_count: int,
        blocked_lane_count: int,
        capacity_fraction: float,
    ) -> float:
        if lane_count <= 0:
            return 1.0
        open_lane_count = max(lane_count - blocked_lane_count, 0)
        if open_lane_count == 0:
            return 0.0
        open_lane_share = open_lane_count / lane_count
        return max(0.05, min(capacity_fraction / max(open_lane_share, 1e-9), 1.0))

    def _collect_disallowed_classes(self, lane_ids: list[str]) -> dict[str, list[str]]:
        stored: dict[str, list[str]] = {}
        for lane_id in lane_ids:
            try:
                stored[lane_id] = list(traci.lane.getDisallowed(lane_id))
            except traci.exceptions.TraCIException as exc:
                logger.debug("Cannot read disallowed classes for lane %s: %s", lane_id, exc)
                stored[lane_id] = []
        return stored

    def _apply_lane_closures(self, lane_ids: list[str]) -> None:
        for lane_id in lane_ids:
            try:
                traci.lane.setDisallowed(lane_id, _CLOSED_LANE_VCLASSES)
            except traci.exceptions.TraCIException as exc:
                logger.warning("Failed to close lane %s: %s", lane_id, exc)

    def _set_lane_speeds_for_phase(
        self,
        accident: Accident,
        open_speed_factor: float,
        blocked_speed_factor: float,
    ) -> None:
        original_speeds: dict[str, float] = accident.__dict__.get("_original_lane_speeds", {})
        for lane_id in accident.managed_lane_ids:
            original = original_speeds.get(lane_id)
            if original is None:
                continue

            factor = blocked_speed_factor if lane_id in accident.blocked_lane_ids else open_speed_factor
            target_speed = self._scaled_lane_speed(original, factor)
            try:
                traci.lane.setMaxSpeed(lane_id, target_speed)
            except traci.exceptions.TraCIException as exc:
                logger.debug(
                    "Could not set lane speed on %s for %s: %s",
                    lane_id,
                    accident.accident_id,
                    exc,
                )

    @staticmethod
    def _scaled_lane_speed(original_speed: float, factor: float) -> float:
        if factor >= 1.0:
            return original_speed
        return max(original_speed * max(factor, 0.0), 0.1)

    def _update_active(self, accident: Accident, current_step: int, elapsed: int) -> None:
        try:
            vehicles_on_edge = traci.edge.getLastStepVehicleIDs(accident.edge_id)
            blocked = [
                v
                for v in vehicles_on_edge
                if traci.vehicle.getLanePosition(v) < accident.position and v != accident.vehicle_id
            ]
            accident.vehicles_affected_count = max(accident.vehicles_affected_count, len(blocked))
            accident.peak_queue_length = max(accident.peak_queue_length, len(blocked))
        except traci.exceptions.TraCIException as exc:
            logger.debug("Could not update queue metrics for %s: %s", accident.accident_id, exc)

        if elapsed >= accident.response_time_steps:
            accident.phase = "CLEARING"
            accident.clearance_start_step = current_step
            logger.info(
                "ACCIDENT %s (%s) — emergency services on scene, clearance begins.",
                accident.accident_id,
                accident.severity,
            )

    def _update_clearing(self, accident: Accident, current_step: int) -> None:
        original_speeds: dict[str, float] = accident.__dict__.get("_original_lane_speeds", {})
        if not original_speeds:
            return

        clearing_total = accident.duration_steps - accident.response_time_steps
        clearing_elapsed = current_step - accident.clearance_start_step
        if clearing_total <= 0:
            return

        fraction = min(clearing_elapsed / clearing_total, 1.0)
        base_open_factor = float(accident.__dict__.get("_open_lane_speed_factor", 1.0))
        open_factor = base_open_factor + (1.0 - base_open_factor) * fraction
        blocked_factor = float(accident.__dict__.get("_blocked_lane_speed_factor", open_factor))

        self._set_lane_speeds_for_phase(
            accident,
            open_speed_factor=open_factor,
            blocked_speed_factor=blocked_factor,
        )

    def _resolve_accident(self, accident: Accident, current_step: int) -> None:
        accident.phase = "RESOLVED"
        accident.resolved_step = current_step

        original_speeds: dict[str, float] = accident.__dict__.get("_original_lane_speeds", {})
        for lane_id, original in original_speeds.items():
            try:
                traci.lane.setMaxSpeed(lane_id, original)
            except traci.exceptions.TraCIException as exc:
                logger.warning(
                    "Failed to restore lane speed on %s for %s: %s",
                    lane_id,
                    accident.accident_id,
                    exc,
                )

        original_disallowed: dict[str, list[str]] = accident.__dict__.get(
            "_original_lane_disallowed", {}
        )
        for lane_id in accident.blocked_lane_ids:
            try:
                traci.lane.setDisallowed(lane_id, original_disallowed.get(lane_id, []))
            except traci.exceptions.TraCIException as exc:
                logger.warning(
                    "Failed to restore lane permissions on %s for %s: %s",
                    lane_id,
                    accident.accident_id,
                    exc,
                )

        try:
            traci.vehicle.setSpeedMode(accident.vehicle_id, _SPEED_MODE_DEFAULT)
            traci.vehicle.setSpeed(accident.vehicle_id, -1)
        except traci.exceptions.TraCIException:
            try:
                traci.vehicle.remove(accident.vehicle_id)
            except traci.exceptions.TraCIException:
                logger.debug(
                    "Vehicle %s already left network during %s resolution.",
                    accident.vehicle_id,
                    accident.accident_id,
                )

        logger.info(
            "ACCIDENT %s (%s) — RESOLVED | duration: %ds | peak queue: %d vehicles | "
            "affected: %d vehicles | rerouted: %d vehicles",
            accident.accident_id,
            accident.severity,
            current_step - accident.trigger_step,
            accident.peak_queue_length,
            accident.vehicles_affected_count,
            accident.rerouted_vehicle_count,
        )

    def _reroute_nearby_vehicles(self, accident: Accident, all_sub: dict[str, dict]) -> int:
        rerouted = 0
        rerouted_now: set[str] = set()
        radius_sq = self.reroute_radius_m * self.reroute_radius_m

        for vehicle_id, vdata in all_sub.items():
            if vehicle_id == accident.vehicle_id:
                continue

            edge_id = str(vdata.get(_tc.VAR_ROAD_ID, ""))
            if not edge_id or edge_id.startswith(":"):
                continue

            x, y = vdata.get(_tc.VAR_POSITION, (None, None))
            if x is None or y is None:
                continue
            if (x - accident.x) ** 2 + (y - accident.y) ** 2 > radius_sq:
                continue

            if edge_id == accident.edge_id:
                lane_pos = float(vdata.get(_tc.VAR_LANEPOSITION, float("inf")))
                if lane_pos >= accident.position:
                    continue

            try:
                traci.vehicle.rerouteTraveltime(vehicle_id)
            except traci.exceptions.TraCIException as exc:
                logger.debug(
                    "Could not reroute vehicle %s near %s: %s",
                    vehicle_id,
                    accident.accident_id,
                    exc,
                )
                continue

            rerouted += 1
            rerouted_now.add(vehicle_id)

        if rerouted_now:
            existing = set(accident.affected_vehicles)
            accident.affected_vehicles.extend(sorted(rerouted_now - existing))
            accident.rerouted_vehicle_count = len(set(accident.affected_vehicles))

        return rerouted
