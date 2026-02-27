"""
sas/accident_manager.py
========================
Accident Lifecycle Manager

This module handles everything that happens AFTER an accident is triggered:

  Phase 1 — ACTIVE:      Vehicles stopped/slowed. Lane capacity reduced.
                          Emergency services dispatched (timer starts).
  Phase 2 — CLEARING:    After response_time, capacity gradually restored.
                          Stopped vehicles released one by one.
  Phase 3 — RECOVERED:   Full capacity restored. Accident archived.

Each accident is represented as an Accident dataclass, giving us a clean
record that feeds directly into the metrics and reporting modules.
"""

import random
import math
import traci
from dataclasses import dataclass, field
from typing import Optional

# SUMO speedMode bit-flags (see SUMO docs: TraCI/Change Vehicle State)
# Bit 0: right-of-way check, 1: safe speed, 2: safe lane-change,
# 3: brake-light, 4: emergency braking — 11111₂ = 31 restores all defaults.
_SPEED_MODE_DEFAULT = 31   # all safety checks active (SUMO built-in default)
_SPEED_MODE_FROZEN  = 0    # no safety checks — forces vehicle to hold speed


@dataclass
class Accident:
    """
    A single accident event and its full lifecycle state.
    """
    accident_id: str              # Unique identifier e.g. "ACC_0042"
    trigger_step: int             # Simulation step when accident occurred
    vehicle_id: str               # Primary vehicle involved
    edge_id: str                  # Road segment where it happened
    lane_id: str                  # Specific lane
    position: float               # Position along the edge (metres from start)
    x: float                      # Geographic x coordinate
    y: float                      # Geographic y coordinate

    # Timing
    duration_steps: int           # How long the accident lasts (in steps)
    response_time_steps: int      # Steps until clearance begins
    clearance_start_step: int = 0 # Filled in when clearance begins
    resolved_step: int = 0        # Filled in when fully resolved

    # State
    phase: str = "ACTIVE"         # ACTIVE → CLEARING → RESOLVED
    affected_vehicles: list = field(default_factory=list)  # Blocked vehicles

    # Impact metrics (filled during simulation)
    peak_queue_length: int = 0
    total_delay_seconds: float = 0.0
    vehicles_affected_count: int = 0


class AccidentManager:
    """
    Manages the full lifecycle of all active accidents in the simulation.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: The 'accident' section of config.yaml
        """
        self.min_duration = config["min_duration_seconds"]
        self.max_duration = config["max_duration_seconds"]
        self.capacity_fraction = config["lane_capacity_fraction"]
        self.response_time = config["response_time_seconds"]
        self.max_concurrent = config["max_concurrent_accidents"]
        self.secondary_enabled = config.get("secondary_accident_enabled", True)
        self.secondary_radius = config.get("secondary_risk_radius_meters", 200)
        self.secondary_multiplier = config.get("secondary_risk_multiplier", 3.0)

        self.active_accidents: dict[str, Accident] = {}  # accident_id → Accident
        self.resolved_accidents: list[Accident] = []
        self._accident_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_trigger_accident(self) -> bool:
        """Returns True if we're allowed to add another accident."""
        return len(self.active_accidents) < self.max_concurrent

    def trigger_accident(self, vehicle_id: str, current_step: int) -> Optional[Accident]:
        """
        Initiate a new accident at the vehicle's current position.

        Args:
            vehicle_id:    The vehicle that triggers the accident.
            current_step:  Current simulation timestep.

        Returns:
            The created Accident object, or None if it failed.
        """
        try:
            edge_id = traci.vehicle.getRoadID(vehicle_id)
            lane_id = traci.vehicle.getLaneID(vehicle_id)
            position = traci.vehicle.getLanePosition(vehicle_id)
            x, y = traci.vehicle.getPosition(vehicle_id)
        except traci.exceptions.TraCIException:
            return None  # Vehicle left the network before we could act

        # Don't trigger on internal junction edges
        if edge_id.startswith(":"):
            return None

        # Draw accident duration from a log-normal distribution
        # (reflects real-world data: most accidents clear quickly, some last long)
        mu = math.log((self.min_duration + self.max_duration) / 2)
        sigma = 0.6
        duration = int(min(max(
            random.lognormvariate(mu, sigma),
            self.min_duration
        ), self.max_duration))

        self._accident_counter += 1
        accident_id = f"ACC_{self._accident_counter:04d}"

        accident = Accident(
            accident_id=accident_id,
            trigger_step=current_step,
            vehicle_id=vehicle_id,
            edge_id=edge_id,
            lane_id=lane_id,
            position=position,
            x=x,
            y=y,
            duration_steps=duration,
            response_time_steps=self.response_time,
        )

        # Apply immediate traffic effects
        self._apply_accident_effects(accident)

        self.active_accidents[accident_id] = accident
        print(f"[Step {current_step}] ⚠️  ACCIDENT {accident_id}: "
              f"Vehicle {vehicle_id} on {edge_id} | Duration: {duration}s")
        return accident

    def update(self, current_step: int):
        """
        Called every simulation step. Advances accident lifecycle phases.

        Args:
            current_step: Current simulation timestep.
        """
        resolved = []

        for acc_id, accident in self.active_accidents.items():
            elapsed = current_step - accident.trigger_step

            if accident.phase == "ACTIVE":
                self._update_active(accident, current_step, elapsed)

            elif accident.phase == "CLEARING":
                self._update_clearing(accident, current_step, elapsed)

            # Check if fully resolved
            if elapsed >= accident.duration_steps:
                self._resolve_accident(accident, current_step)
                resolved.append(acc_id)

        for acc_id in resolved:
            self.resolved_accidents.append(self.active_accidents.pop(acc_id))

    def get_secondary_multiplier(self, x: float, y: float) -> float:
        """
        Returns an elevated risk multiplier if (x, y) is near an active accident.
        Used by the risk model to create secondary accident zones.

        Args:
            x, y: Geographic coordinates to check.

        Returns:
            A float multiplier (1.0 = no elevation, >1.0 = elevated risk).
        """
        if not self.secondary_enabled:
            return 1.0

        for accident in self.active_accidents.values():
            dist = math.sqrt((x - accident.x) ** 2 + (y - accident.y) ** 2)
            if dist <= self.secondary_radius:
                return self.secondary_multiplier

        return 1.0

    # ------------------------------------------------------------------
    # Private lifecycle helpers
    # ------------------------------------------------------------------

    def _apply_accident_effects(self, accident: Accident):
        # Stop the accident vehicle; disable safety checks so SUMO doesn't
        # override the forced zero speed (avoids brake-distance errors).
        try:
            traci.vehicle.setSpeed(accident.vehicle_id, 0.0)
            traci.vehicle.setSpeedMode(accident.vehicle_id, _SPEED_MODE_FROZEN)
        except traci.exceptions.TraCIException:
            pass

        # Reduce max speed on the accident lane
        try:
            original_speed = traci.lane.getMaxSpeed(accident.lane_id)
            reduced_speed = original_speed * self.capacity_fraction
            traci.lane.setMaxSpeed(accident.lane_id, reduced_speed)
            accident.__dict__["_original_lane_speed"] = original_speed
        except traci.exceptions.TraCIException:
            accident.__dict__["_original_lane_speed"] = None

    def _update_active(self, accident: Accident, current_step: int, elapsed: int):
        """Track blocked vehicles and transition to CLEARING after response time."""
        # Count vehicles stuck behind the accident
        try:
            vehicles_on_edge = traci.edge.getLastStepVehicleIDs(accident.edge_id)
            blocked = [v for v in vehicles_on_edge
                       if traci.vehicle.getLanePosition(v) < accident.position
                       and v != accident.vehicle_id]
            accident.vehicles_affected_count = max(
                accident.vehicles_affected_count, len(blocked)
            )
            accident.peak_queue_length = max(accident.peak_queue_length, len(blocked))
        except traci.exceptions.TraCIException:
            pass

        # Transition to CLEARING after response time
        if elapsed >= accident.response_time_steps:
            accident.phase = "CLEARING"
            accident.clearance_start_step = current_step
            print(f"[Step {current_step}] 🚨 {accident.accident_id}: "
                  f"Emergency services arrived — clearance begins.")

    def _update_clearing(self, accident: Accident, current_step: int, elapsed: int):
        """
        Gradually restore lane capacity during the clearing phase.
        Speed limit ramps back up linearly over the remaining accident duration.
        """
        if accident.__dict__.get("_original_lane_speed") is None:
            return

        original_speed = accident.__dict__["_original_lane_speed"]
        reduced_speed = original_speed * self.capacity_fraction

        # How far through the clearing phase are we?
        clearing_duration = accident.duration_steps - accident.response_time_steps
        clearing_elapsed = current_step - accident.clearance_start_step
        if clearing_duration <= 0:
            return

        fraction = min(clearing_elapsed / clearing_duration, 1.0)
        current_speed = reduced_speed + (original_speed - reduced_speed) * fraction

        try:
            traci.lane.setMaxSpeed(accident.lane_id, current_speed)
        except traci.exceptions.TraCIException:
            pass

    def _resolve_accident(self, accident: Accident, current_step: int):
        accident.phase = "RESOLVED"
        accident.resolved_step = current_step

        # Restore original lane speed
        original_speed = accident.__dict__.get("_original_lane_speed")
        if original_speed is not None:
            try:
                traci.lane.setMaxSpeed(accident.lane_id, original_speed)
            except traci.exceptions.TraCIException:
                pass

        # Release or remove the accident vehicle
        try:
            traci.vehicle.setSpeedMode(accident.vehicle_id, _SPEED_MODE_DEFAULT)
            traci.vehicle.setSpeed(accident.vehicle_id, -1)   # hand control back to SUMO
        except traci.exceptions.TraCIException:
            try:
                traci.vehicle.remove(accident.vehicle_id)
            except traci.exceptions.TraCIException:
                pass  # already left the network, that's fine

        print(f"[Step {current_step}] ✅ {accident.accident_id}: RESOLVED | "
            f"Peak queue: {accident.peak_queue_length} vehicles | "
            f"Duration: {current_step - accident.trigger_step}s")