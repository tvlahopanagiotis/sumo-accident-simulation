"""
sas/accident_manager.py
========================
Accident Lifecycle Manager

Handles everything that happens AFTER an accident is triggered:

  Phase 1 — ACTIVE:    The accident vehicle is stopped; lane capacity is
                       reduced according to the accident's severity tier.
                       Emergency services are dispatched (timer starts).

  Phase 2 — CLEARING:  After response_time_s seconds, the scene is attended
                       and capacity begins ramping back up linearly.

  Phase 3 — RESOLVED:  Full capacity restored; accident archived for reporting.

Severity model
--------------
Each accident is randomly assigned a severity tier at trigger time.  Tiers are
defined in the 'accident.severity' section of config.yaml and are drawn from a
weighted distribution calibrated to NHTSA KABCO injury classification data:

    MINOR     ≈ 62 %  — property damage only; short blockage, quick clearance
    MODERATE  ≈ 28 %  — non-incapacitating injury; lane partially closed
    MAJOR     ≈  8 %  — incapacitating injury; near-total closure, long scene
    CRITICAL  ≈  2 %  — fatal; full closure, forensic investigation required

Every tier has its own duration range, capacity fraction, emergency response
time, and secondary-risk zone.  All parameters are user-configurable via
config.yaml.
"""

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Optional

import traci

logger = logging.getLogger("sas.accident")

# SUMO speedMode bit-flags (see SUMO docs: TraCI / Change Vehicle State)
# Bit pattern 31 = 11111₂ restores all SUMO safety checks.
# Bit pattern  0 = 00000₂ disables all checks, allowing forced speed override.
_SPEED_MODE_DEFAULT = 31
_SPEED_MODE_FROZEN  = 0


# ---------------------------------------------------------------------------
# Severity tier
# ---------------------------------------------------------------------------

class Severity:
    """String constants for the four severity tiers."""
    MINOR    = "MINOR"
    MODERATE = "MODERATE"
    MAJOR    = "MAJOR"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Accident dataclass
# ---------------------------------------------------------------------------

@dataclass
class Accident:
    """
    A single accident event and its complete lifecycle state.

    Severity-specific parameters (capacity_fraction, response_time_steps,
    secondary_risk_radius_m, secondary_risk_multiplier) are sampled at
    trigger time and stored on the instance so each accident is self-contained.
    """

    # Identity
    accident_id: str          # Unique identifier, e.g. "ACC_0042"
    trigger_step: int         # Simulation step when the accident occurred
    vehicle_id: str           # Primary vehicle involved
    edge_id: str              # Road segment (SUMO edge ID)
    lane_id: str              # Lane ID
    position: float           # Distance from edge start (metres)
    x: float                  # Cartesian x position
    y: float                  # Cartesian y position

    # Severity
    severity: str = Severity.MINOR  # MINOR / MODERATE / MAJOR / CRITICAL

    # Timing (set at trigger time from severity tier)
    duration_steps: int = 0         # How long the accident lasts (simulated seconds)
    response_time_steps: int = 0    # Seconds until clearance begins
    clearance_start_step: int = 0   # Filled when clearance begins
    resolved_step: int = 0          # Filled when fully resolved

    # Traffic impact (set at trigger time from severity tier)
    capacity_fraction: float = 0.70          # Lane capacity remaining during active phase
    secondary_risk_radius_m: float = 75.0    # Radius of elevated-risk zone
    secondary_risk_multiplier: float = 1.5   # Risk multiplier within that zone

    # Lifecycle state
    phase: str = "ACTIVE"                          # ACTIVE → CLEARING → RESOLVED
    affected_vehicles: list = field(default_factory=list)

    # Impact metrics (updated during simulation)
    peak_queue_length: int = 0
    vehicles_affected_count: int = 0


# ---------------------------------------------------------------------------
# AccidentManager
# ---------------------------------------------------------------------------

class AccidentManager:
    """
    Manages the full lifecycle of all active accidents in the simulation.

    Exposes three methods to runner.py:
        can_trigger_accident()     — True if another accident is allowed
        trigger_accident()         — Initiate a new accident
        update()                   — Advance all active accidents one step
        get_secondary_multiplier() — Risk elevation near active scenes
    """

    def __init__(self, config: dict):
        """
        Args:
            config: The 'accident' section of config.yaml.
        """
        self.max_concurrent      = config["max_concurrent_accidents"]
        self.secondary_enabled   = config.get("secondary_accident_enabled", True)

        # Parse severity tiers from config
        self._severity_tiers     = self._load_severity_tiers(config["severity"])
        self._severity_names     = [t["name"]   for t in self._severity_tiers]
        self._severity_weights   = [t["weight"] for t in self._severity_tiers]

        self.active_accidents:   dict[str, Accident] = {}
        self.resolved_accidents: list[Accident]      = []
        self._accident_counter:  int                 = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_trigger_accident(self) -> bool:
        """Return True if another concurrent accident is allowed."""
        return len(self.active_accidents) < self.max_concurrent

    def trigger_accident(
        self,
        vehicle_id: str,
        current_step: int,
    ) -> Optional[Accident]:
        """
        Initiate a new accident at the vehicle's current position.

        Samples a severity tier from the configured distribution, draws a
        log-normally distributed duration within that tier's bounds, applies
        immediate traffic effects (stopped vehicle, reduced lane speed), and
        registers the accident for lifecycle tracking.

        Args:
            vehicle_id:   The SUMO vehicle ID that triggers the accident.
            current_step: Current simulation timestep (simulated seconds).

        Returns:
            The created Accident object, or None if the vehicle could not
            be located (e.g. already left the network).
        """
        # Fetch vehicle state via TraCI
        try:
            edge_id  = traci.vehicle.getRoadID(vehicle_id)
            lane_id  = traci.vehicle.getLaneID(vehicle_id)
            position = traci.vehicle.getLanePosition(vehicle_id)
            x, y     = traci.vehicle.getPosition(vehicle_id)
        except traci.exceptions.TraCIException:
            return None  # Vehicle left the network before we could act

        # Do not trigger on internal junction edges
        if not edge_id or edge_id.startswith(":"):
            return None

        # Sample severity tier
        tier = random.choices(self._severity_tiers, weights=self._severity_weights, k=1)[0]

        # Draw duration from a log-normal distribution.
        # Log-normal is appropriate here: most incidents clear quickly, but
        # the tail is heavy (some accidents take much longer than average).
        duration = self._sample_duration(tier["duration_min_s"], tier["duration_max_s"])

        self._accident_counter += 1
        accident_id = f"ACC_{self._accident_counter:04d}"

        accident = Accident(
            accident_id               = accident_id,
            trigger_step              = current_step,
            vehicle_id                = vehicle_id,
            edge_id                   = edge_id,
            lane_id                   = lane_id,
            position                  = position,
            x                         = x,
            y                         = y,
            severity                  = tier["name"],
            duration_steps            = duration,
            response_time_steps       = tier["response_time_s"],
            capacity_fraction         = tier["lane_capacity_fraction"],
            secondary_risk_radius_m   = tier["secondary_risk_radius_m"],
            secondary_risk_multiplier = tier["secondary_risk_multiplier"],
        )

        self._apply_accident_effects(accident)
        self.active_accidents[accident_id] = accident

        logger.info(
            "ACCIDENT %s — severity: %s | vehicle: %s | edge: %s | "
            "duration: %ds | capacity: %.0f%%",
            accident_id, tier["name"], vehicle_id, edge_id,
            duration, tier["lane_capacity_fraction"] * 100,
        )
        return accident

    def update(self, current_step: int):
        """
        Advance all active accident lifecycles by one step.

        Call this every simulation step.  Transitions accidents through
        ACTIVE → CLEARING → RESOLVED and removes resolved accidents from
        the active dict.

        Args:
            current_step: Current simulation timestep (simulated seconds).
        """
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

    def get_secondary_multiplier(self, x: float, y: float) -> float:
        """
        Return the risk multiplier for a vehicle at position (x, y).

        Returns the highest multiplier from any active accident whose
        secondary-risk zone contains the point.  Returns 1.0 (no elevation)
        if no active accident is nearby or secondary risk is disabled.

        Args:
            x, y: Cartesian coordinates of the vehicle.

        Returns:
            Float ≥ 1.0.
        """
        if not self.secondary_enabled:
            return 1.0

        best = 1.0
        for accident in self.active_accidents.values():
            dist = math.sqrt((x - accident.x) ** 2 + (y - accident.y) ** 2)
            if dist <= accident.secondary_risk_radius_m:
                best = max(best, accident.secondary_risk_multiplier)

        return best

    # ------------------------------------------------------------------
    # Private — severity config parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _load_severity_tiers(severity_cfg: dict) -> list[dict]:
        """
        Parse the 'severity' block from config.yaml into a list of tier dicts.

        Each tier dict has keys:
            name, weight, duration_min_s, duration_max_s,
            lane_capacity_fraction, response_time_s,
            secondary_risk_radius_m, secondary_risk_multiplier

        Raises ValueError if a required key is missing or if no tiers are defined.
        """
        required = {
            "weight", "duration_min_s", "duration_max_s",
            "lane_capacity_fraction", "response_time_s",
            "secondary_risk_radius_m", "secondary_risk_multiplier",
        }
        tiers = []
        for tier_name, params in severity_cfg.items():
            missing = required - set(params.keys())
            if missing:
                raise ValueError(
                    f"Severity tier '{tier_name}' is missing keys: {missing}"
                )
            tiers.append({"name": tier_name.upper(), **params})

        if not tiers:
            raise ValueError("No severity tiers defined in accident.severity config.")

        return tiers

    # ------------------------------------------------------------------
    # Private — duration sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_duration(min_s: int, max_s: int) -> int:
        """
        Draw a duration from a log-normal distribution clipped to [min_s, max_s].

        The log-normal mean is set to the geometric mean of the bounds;
        sigma = 0.5 gives a realistic right-skewed distribution (most
        incidents resolve quickly, rare ones take much longer).
        """
        mu    = math.log(math.sqrt(min_s * max_s))   # geometric mean
        sigma = 0.5
        raw   = random.lognormvariate(mu, sigma)
        return int(max(min_s, min(max_s, raw)))

    # ------------------------------------------------------------------
    # Private — lifecycle phases
    # ------------------------------------------------------------------

    def _apply_accident_effects(self, accident: Accident):
        """Stop the accident vehicle and reduce lane capacity immediately."""
        # Freeze the accident vehicle at zero speed.
        # Disabling safety checks prevents SUMO from overriding the forced speed.
        try:
            traci.vehicle.setSpeed(accident.vehicle_id, 0.0)
            traci.vehicle.setSpeedMode(accident.vehicle_id, _SPEED_MODE_FROZEN)
        except traci.exceptions.TraCIException:
            pass

        # Reduce lane max speed proportional to remaining capacity.
        try:
            original = traci.lane.getMaxSpeed(accident.lane_id)
            reduced  = original * accident.capacity_fraction
            traci.lane.setMaxSpeed(accident.lane_id, reduced)
            accident.__dict__["_original_lane_speed"] = original
        except traci.exceptions.TraCIException:
            accident.__dict__["_original_lane_speed"] = None

    def _update_active(
        self,
        accident: Accident,
        current_step: int,
        elapsed: int,
    ):
        """Track impact metrics and transition to CLEARING after response time."""
        try:
            vehicles_on_edge = traci.edge.getLastStepVehicleIDs(accident.edge_id)
            blocked = [
                v for v in vehicles_on_edge
                if traci.vehicle.getLanePosition(v) < accident.position
                and v != accident.vehicle_id
            ]
            accident.vehicles_affected_count = max(
                accident.vehicles_affected_count, len(blocked)
            )
            accident.peak_queue_length = max(accident.peak_queue_length, len(blocked))
        except traci.exceptions.TraCIException:
            pass

        if elapsed >= accident.response_time_steps:
            accident.phase              = "CLEARING"
            accident.clearance_start_step = current_step
            logger.info(
                "ACCIDENT %s (%s) — emergency services on scene, clearance begins.",
                accident.accident_id, accident.severity,
            )

    def _update_clearing(self, accident: Accident, current_step: int):
        """
        Ramp lane capacity back up linearly during the clearance phase.

        Speed limit is interpolated from the reduced value back to the original
        over the remaining accident duration.
        """
        original = accident.__dict__.get("_original_lane_speed")
        if original is None:
            return

        reduced          = original * accident.capacity_fraction
        clearing_total   = accident.duration_steps - accident.response_time_steps
        clearing_elapsed = current_step - accident.clearance_start_step

        if clearing_total <= 0:
            return

        fraction      = min(clearing_elapsed / clearing_total, 1.0)
        current_speed = reduced + (original - reduced) * fraction

        try:
            traci.lane.setMaxSpeed(accident.lane_id, current_speed)
        except traci.exceptions.TraCIException:
            pass

    def _resolve_accident(self, accident: Accident, current_step: int):
        """Restore full lane capacity and release the accident vehicle."""
        accident.phase         = "RESOLVED"
        accident.resolved_step = current_step

        original = accident.__dict__.get("_original_lane_speed")
        if original is not None:
            try:
                traci.lane.setMaxSpeed(accident.lane_id, original)
            except traci.exceptions.TraCIException:
                pass

        try:
            traci.vehicle.setSpeedMode(accident.vehicle_id, _SPEED_MODE_DEFAULT)
            traci.vehicle.setSpeed(accident.vehicle_id, -1)   # return control to SUMO
        except traci.exceptions.TraCIException:
            try:
                traci.vehicle.remove(accident.vehicle_id)
            except traci.exceptions.TraCIException:
                pass   # Already left the network — that is fine

        logger.info(
            "ACCIDENT %s (%s) — RESOLVED | "
            "duration: %ds | peak queue: %d vehicles | affected: %d vehicles",
            accident.accident_id,
            accident.severity,
            current_step - accident.trigger_step,
            accident.peak_queue_length,
            accident.vehicles_affected_count,
        )
