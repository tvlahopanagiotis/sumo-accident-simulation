"""
sas/risk_model.py
=================
Probabilistic Accident Risk Calculator

This module answers the question: "At any given moment, how likely is
a specific vehicle on a specific road segment to be involved in an accident?"

The risk score is a weighted combination of:

  1. Speed risk       — Nilsson (1981) Power Model:
                          risk  ∝  (v / v_limit) ^ speed_exponent
                        where v_limit is the road's *posted speed limit*
                        (not the vehicle's mechanical maximum).  Exponents:
                          2.0  — property-damage-only (default)
                          3.0  — injury crashes
                          4.0  — fatal crashes
  2. Speed variance   — dangerous speed differentials between nearby vehicles
  3. Density risk     — peaks at medium-high traffic density (bell-curve model)
  4. Road type        — intersections and highways are inherently more risky

The final probability is: base_probability * composite_risk_score

Performance note
----------------
Two code-paths are provided:

  Fast path (recommended):
    call prepare_step(all_sub) once per step, then
    call should_trigger_accident_fast() / get_risk_score_fast() per vehicle.
    Uses only the pre-fetched TraCI subscription dicts — zero individual
    TraCI calls in the hot loop (~2 calls/step total vs ~15,000 before).

  Legacy path:
    get_risk_score() / should_trigger_accident() make individual TraCI calls
    and are retained for testing / debugging.
"""

import math
import random
import traci

# Short alias used throughout this module
_tc = traci.constants


class RiskModel:
    """
    Calculates accident risk for vehicles in the SUMO network.
    """

    def __init__(self, config: dict):
        """
        Initialise the risk model with parameters from config.yaml.

        Args:
            config: The 'risk' section of the loaded config dictionary.
        """
        self.base_probability = config["base_probability"]
        self.speed_weight = config["speed_weight"]
        self.speed_exponent = config["speed_exponent"]
        self.speed_variance_weight = config["speed_variance_weight"]
        self.speed_variance_threshold = config["speed_variance_threshold_ms"]
        self.density_weight = config["density_weight"]
        self.peak_density = config["peak_density_vehicles_per_km"]
        self.road_type_multipliers = config["road_type_multipliers"]
        self.trigger_threshold = config["trigger_threshold"]

        # ── Performance caches ────────────────────────────────────────────
        # Static properties (never change once SUMO has loaded the network):
        self._lane_multiplier_cache: dict[str, float] = {}   # edge_id → road multiplier
        self._road_speed_cache:      dict[str, float] = {}   # edge_id → speed limit (m/s)
        self._edge_length_cache:     dict[str, float] = {}   # edge_id → length in km

        # Per-step density cache — refreshed each step by prepare_step().
        self._edge_density_cache: dict[str, float] = {}      # edge_id → vehicles/km

    # ------------------------------------------------------------------
    # Fast-path API  (zero TraCI calls in the hot loop)
    # ------------------------------------------------------------------

    def prepare_step(self, all_sub: dict):
        """
        Pre-compute per-edge vehicle density from this step's subscription data.
        Must be called once per simulation step, before evaluating any vehicles.

        Internally counts vehicles per edge from the subscription dict, then
        converts to vehicles/km using a cached edge-length lookup (one TraCI
        call per *unique edge*, only on first encounter — never repeated).

        Args:
            all_sub: Return value of traci.vehicle.getAllSubscriptionResults().
                     Maps vehicle_id → {TraCI_var_constant: value, ...}.
        """
        # Count vehicles per edge from subscription data
        edge_counts: dict[str, int] = {}
        for vdata in all_sub.values():
            eid = vdata.get(_tc.VAR_ROAD_ID, "")
            if eid and not eid.startswith(":"):
                edge_counts[eid] = edge_counts.get(eid, 0) + 1

        # Convert raw counts → vehicles/km using cached edge lengths
        self._edge_density_cache = {}
        for eid, count in edge_counts.items():
            length_km = self._get_edge_length_km(eid)
            if length_km > 0:
                self._edge_density_cache[eid] = count / length_km

    def get_risk_score_fast(
        self,
        vehicle_id: str,
        vdata: dict,
        neighbor_speeds: dict,
    ) -> float:
        """
        Compute a risk score (0.0–1.0) using pre-fetched subscription data.
        Makes zero TraCI calls; reads from vdata and the step-level density cache.

        Args:
            vehicle_id:     The SUMO vehicle ID (used for error context only).
            vdata:          Subscription result for this vehicle:
                            {VAR_SPEED: float, VAR_ROAD_ID: str,
                             VAR_POSITION: (x, y), ...}
            neighbor_speeds: Mapping {neighbor_id: speed_float} built from the
                             context subscription results for this vehicle.

        Returns:
            A float in [0, 1] representing composite risk.
        """
        speed   = vdata.get(_tc.VAR_SPEED,  -1.0)
        edge_id = vdata.get(_tc.VAR_ROAD_ID, "")

        # Guard: SUMO returns INVALID_DOUBLE_VALUE (≈ −1.07e9) for vehicles
        # that are teleporting or not yet fully inserted into the network.
        if speed < 0.0 or not edge_id:
            return 0.0

        # --- 1. Speed Risk (Nilsson Power Model) ---
        # Normalise against the road's *posted speed limit*, not the vehicle's
        # mechanical maximum.  A vehicle driving at the limit scores 1.0;
        # a vehicle exceeding it is clamped to 1.0 (risk fully saturated).
        road_limit       = self._get_road_speed_limit_cached(edge_id)
        normalised_speed = (min(speed / road_limit, 1.0)
                            if road_limit > 0 else 0.0)
        speed_risk = normalised_speed ** self.speed_exponent

        # --- 2. Speed Variance Risk ---
        if neighbor_speeds:
            mean_neighbor_speed = sum(neighbor_speeds.values()) / len(neighbor_speeds)
            speed_diff = abs(speed - mean_neighbor_speed)
            variance_risk = min(
                speed_diff / max(self.speed_variance_threshold, 0.1), 1.0
            )
        else:
            variance_risk = 0.0

        # --- 3. Density Risk ---
        density      = self._edge_density_cache.get(edge_id, 0.0)
        density_risk = self._density_risk_curve(density)

        # --- 4. Composite Score (weighted sum, clamped) ---
        composite = (
            self.speed_weight          * speed_risk
            + self.speed_variance_weight * variance_risk
            + self.density_weight        * density_risk
        )
        composite = max(0.0, min(1.0, composite))

        # --- 5. Road Type Multiplier (static — fetched once, then cached) ---
        road_multiplier = self._get_road_multiplier_cached(edge_id)
        composite = min(composite * road_multiplier, 1.0)

        return composite

    def should_trigger_accident_fast(
        self,
        vehicle_id: str,
        vdata: dict,
        neighbor_speeds: dict,
        secondary_multiplier: float = 1.0,
    ) -> bool:
        """
        Decide whether an accident occurs this timestep using pre-fetched data.
        Makes zero TraCI calls.

        Args:
            vehicle_id:           The vehicle to evaluate.
            vdata:                Subscription result dict for this vehicle.
            neighbor_speeds:      {neighbor_id: speed} from context subscription.
            secondary_multiplier: Elevated risk if near an existing accident.

        Returns:
            True if an accident is triggered, False otherwise.
        """
        risk_score = self.get_risk_score_fast(vehicle_id, vdata, neighbor_speeds)

        # Only roll the dice if risk exceeds the threshold
        if risk_score < self.trigger_threshold:
            return False

        excess_risk    = risk_score - self.trigger_threshold
        effective_prob = (
            self.base_probability * (1 + excess_risk * 10) * secondary_multiplier
        )
        return random.random() < effective_prob

    # ------------------------------------------------------------------
    # Legacy API  (individual TraCI calls — kept for debugging)
    # ------------------------------------------------------------------

    def get_risk_score(self, vehicle_id: str, neighbor_ids: list) -> float:
        """
        Compute a risk score (0.0 to 1.0) for a given vehicle.

        Args:
            vehicle_id:   The SUMO vehicle ID to evaluate.
            neighbor_ids: List of vehicle IDs within proximity (same edge).

        Returns:
            A float between 0.0 and 1.0 representing composite risk.
        """
        try:
            speed   = traci.vehicle.getSpeed(vehicle_id)    # m/s
            edge_id = traci.vehicle.getRoadID(vehicle_id)
        except traci.exceptions.TraCIException:
            return 0.0  # Vehicle may have left the network

        # --- 1. Speed Risk (Nilsson Power Model) ---
        # Use the road's posted speed limit, not the vehicle's mechanical max.
        road_limit       = self._get_road_speed_limit_cached(edge_id)
        normalised_speed = (min(speed / road_limit, 1.0)
                            if road_limit > 0 else 0.0)
        speed_risk = normalised_speed ** self.speed_exponent

        # --- 2. Speed Variance Risk ---
        neighbor_speeds = []
        for nid in neighbor_ids:
            if nid != vehicle_id:
                try:
                    neighbor_speeds.append(traci.vehicle.getSpeed(nid))
                except traci.exceptions.TraCIException:
                    continue

        if neighbor_speeds:
            mean_neighbor_speed = sum(neighbor_speeds) / len(neighbor_speeds)
            speed_diff = abs(speed - mean_neighbor_speed)
            variance_risk = min(speed_diff / max(self.speed_variance_threshold, 0.1), 1.0)
        else:
            variance_risk = 0.0

        # --- 3. Density Risk ---
        density      = self._get_edge_density(edge_id)
        density_risk = self._density_risk_curve(density)

        # --- 4. Composite Score (weighted sum) ---
        composite = (
            self.speed_weight          * speed_risk
            + self.speed_variance_weight * variance_risk
            + self.density_weight        * density_risk
        )
        composite = max(0.0, min(1.0, composite))

        # --- 5. Road Type Multiplier ---
        road_multiplier = self._get_road_multiplier(edge_id)
        composite = min(composite * road_multiplier, 1.0)

        return composite

    def should_trigger_accident(
        self,
        vehicle_id: str,
        neighbor_ids: list,
        secondary_multiplier: float = 1.0,
    ) -> bool:
        """
        Decide whether an accident occurs this timestep for the given vehicle.

        Args:
            vehicle_id:           The vehicle to evaluate.
            neighbor_ids:         Nearby vehicles on the same edge.
            secondary_multiplier: Elevated risk if near an existing accident.

        Returns:
            True if an accident is triggered, False otherwise.
        """
        risk_score = self.get_risk_score(vehicle_id, neighbor_ids)

        if risk_score < self.trigger_threshold:
            return False

        excess_risk    = risk_score - self.trigger_threshold
        effective_prob = (
            self.base_probability * (1 + excess_risk * 10) * secondary_multiplier
        )
        return random.random() < effective_prob

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_edge_length_km(self, edge_id: str) -> float:
        """
        Return edge length in km.
        Result is cached after the first TraCI call (static property).
        """
        if edge_id not in self._edge_length_cache:
            try:
                length_m = traci.lane.getLength(edge_id + "_0")
                self._edge_length_cache[edge_id] = length_m / 1000.0
            except traci.exceptions.TraCIException:
                self._edge_length_cache[edge_id] = 0.0
        return self._edge_length_cache[edge_id]

    def _get_road_multiplier_cached(self, edge_id: str) -> float:
        """
        Return road type multiplier.
        Cached after first lookup — lane max speed is static.
        """
        if edge_id not in self._lane_multiplier_cache:
            self._lane_multiplier_cache[edge_id] = self._get_road_multiplier(edge_id)
        return self._lane_multiplier_cache[edge_id]

    def _get_edge_density(self, edge_id: str) -> float:
        """
        Estimate vehicle density on an edge in vehicles per km.
        Returns 0.0 for internal edges (junctions).
        """
        if edge_id.startswith(":"):
            return 0.0
        try:
            vehicle_count  = traci.edge.getLastStepVehicleNumber(edge_id)
            edge_length_km = self._get_edge_length_km(edge_id)
            if edge_length_km > 0:
                return vehicle_count / edge_length_km
        except traci.exceptions.TraCIException:
            pass
        return 0.0

    def _density_risk_curve(self, density: float) -> float:
        """
        Bell-curve style risk based on density.
        Risk peaks at self.peak_density and falls off on either side.
        Uses a Gaussian-like shape.
        """
        sigma    = self.peak_density * 0.5
        exponent = -((density - self.peak_density) ** 2) / (2 * sigma ** 2)
        return math.exp(exponent)

    def _get_road_speed_limit_cached(self, edge_id: str) -> float:
        """
        Return the posted speed limit for the primary lane of an edge (m/s).

        Cached after the first TraCI call — lane speed limits are a static
        network property and never change during simulation.  Returns 1.0 as
        a safe fallback on error (prevents division-by-zero).
        """
        if edge_id not in self._road_speed_cache:
            try:
                self._road_speed_cache[edge_id] = traci.lane.getMaxSpeed(
                    edge_id + "_0"
                )
            except traci.exceptions.TraCIException:
                self._road_speed_cache[edge_id] = 1.0   # safe fallback
        return self._road_speed_cache[edge_id]

    def _get_road_multiplier(self, edge_id: str) -> float:
        """
        Determine road type multiplier based on the edge's posted speed limit.

        SUMO does not expose road functional class directly, so the speed
        limit is used as a proxy (standard practice in macroscopic models):
            ≥ 25 m/s (90 km/h) → highway
            ≥ 13.9 m/s (50 km/h) → arterial
            <  13.9 m/s           → local street
        Internal junction edges receive the intersection premium.

        Delegates to _get_road_speed_limit_cached() so both the road-type
        lookup and the Nilsson normalisation share a single TraCI call per
        unique edge.
        """
        if not edge_id or edge_id.startswith(":"):
            return self.road_type_multipliers.get("intersection", 2.0)

        speed_mps = self._get_road_speed_limit_cached(edge_id)
        if speed_mps >= 25.0:    # ≥ 90 km/h → highway
            return self.road_type_multipliers.get("highway", 1.5)
        elif speed_mps >= 13.9:  # ≥ 50 km/h → arterial
            return self.road_type_multipliers.get("arterial", 1.0)
        else:                    # < 50 km/h  → local
            return self.road_type_multipliers.get("local", 0.6)
