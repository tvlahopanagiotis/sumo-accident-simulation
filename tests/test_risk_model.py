"""
tests/test_risk_model.py
========================
Tests for the RiskModel class (risk_model.py).

All tests use the mock traci injected by conftest.py.
"""

import math
import random

import traci  # This is the mock injected by conftest.py

from sas.core.risk_model import RiskModel

# Shorthand for the mock constants
_tc = traci.constants


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vdata(speed, edge_id="edge_1", position=(100.0, 200.0)):
    """Build a subscription-result dict for a single vehicle."""
    return {
        _tc.VAR_SPEED: speed,
        _tc.VAR_ROAD_ID: edge_id,
        _tc.VAR_POSITION: position,
    }


def _make_model(risk_config, road_limit=13.89, edge_length_m=1000.0):
    """
    Create a RiskModel and pre-populate its caches so that tests do not
    depend on traci.lane.getMaxSpeed / traci.lane.getLength being called.
    """
    model = RiskModel(risk_config)
    # Pre-fill caches for 'edge_1' so the fast-path never hits traci
    model._road_speed_cache["edge_1"] = road_limit
    model._edge_length_cache["edge_1"] = edge_length_m / 1000.0  # km
    # Also cache the road-type multiplier so _get_road_multiplier_cached works
    # For 13.89 m/s the road is classified as 'arterial' (>= 13.9 is arterial)
    # 13.89 < 13.9, so it is 'local'.  Use exactly 13.9 for arterial.
    return model


# ---------------------------------------------------------------------------
# Speed risk tests
# ---------------------------------------------------------------------------


class TestSpeedRisk:
    """Tests for the speed-risk component of get_risk_score_fast."""

    def test_speed_risk_at_limit(self, risk_config):
        """Vehicle travelling at exactly the speed limit => speed_risk = 1.0."""
        road_limit = 13.9  # m/s — classified as arterial (multiplier 1.0)
        model = _make_model(risk_config, road_limit=road_limit)

        vdata = _make_vdata(speed=road_limit)
        # No neighbours, no density => only speed contributes
        score = model.get_risk_score("v0", vdata, {})
        # speed_risk = (13.9 / 13.9)^2 = 1.0
        # composite  = 0.40 * 1.0 + 0.30 * 0.0 + 0.30 * density_risk
        # density_risk for edge not in cache = 0 density => exp(-peak^2/(2*sigma^2))
        # which is exp(-(15)^2 / (2*(7.5)^2)) = exp(-2) ~ 0.135
        # BUT density cache is empty, so density = 0.0 for edge_1
        expected_speed_component = 0.40 * 1.0
        assert score >= expected_speed_component - 0.01  # speed part dominates

    def test_speed_risk_below_limit(self, risk_config):
        """Vehicle at 50% of speed limit => speed_risk = 0.25."""
        road_limit = 20.0
        model = _make_model(risk_config, road_limit=road_limit)

        vdata = _make_vdata(speed=10.0)  # 50% of limit
        score = model.get_risk_score("v0", vdata, {})
        # speed_risk = (10/20)^2 = 0.25
        # composite includes density_risk at density=0
        speed_component = 0.40 * 0.25
        # The composite should contain the speed component plus density contribution
        assert score >= speed_component - 0.01

    def test_speed_risk_zero_speed(self, risk_config):
        """Vehicle at speed 0 => speed_risk = 0."""
        model = _make_model(risk_config, road_limit=13.9)

        vdata = _make_vdata(speed=0.0)
        score = model.get_risk_score("v0", vdata, {})
        # speed_risk = 0^2 = 0
        # Only density_risk contributes (small amount from density=0 curve)
        assert score < 0.20  # dominated by near-zero components

    def test_speed_risk_above_limit_clamped(self, risk_config):
        """Vehicle exceeding speed limit => normalised speed clamped to 1.0."""
        road_limit = 13.9
        model = _make_model(risk_config, road_limit=road_limit)

        vdata_at_limit = _make_vdata(speed=road_limit)
        vdata_above = _make_vdata(speed=road_limit * 2.0)

        score_at_limit = model.get_risk_score("v0", vdata_at_limit, {})
        score_above = model.get_risk_score("v0", vdata_above, {})

        # Both should be equal because speed is clamped to 1.0
        assert abs(score_at_limit - score_above) < 1e-9

    def test_speed_risk_negative_speed_returns_zero(self, risk_config):
        """SUMO INVALID_DOUBLE_VALUE (negative) => risk = 0.0."""
        model = _make_model(risk_config)
        vdata = _make_vdata(speed=-1073741824.0)  # SUMO sentinel value
        score = model.get_risk_score("v0", vdata, {})
        assert score == 0.0


# ---------------------------------------------------------------------------
# Variance risk tests
# ---------------------------------------------------------------------------


class TestVarianceRisk:
    """Tests for the speed-variance component."""

    def test_variance_risk_no_neighbors(self, risk_config):
        """No neighbours => variance_risk = 0.0."""
        model = _make_model(risk_config, road_limit=13.9)
        vdata = _make_vdata(speed=13.9)
        score_no_neighbors = model.get_risk_score("v0", vdata, {})

        # Add neighbours at the SAME speed => variance_risk still = 0
        neighbors_same = {"n1": 13.9, "n2": 13.9}
        score_same = model.get_risk_score("v0", vdata, neighbors_same)

        # Scores should be equal — zero variance in both cases
        assert abs(score_no_neighbors - score_same) < 1e-9

    def test_variance_risk_high_differential(self, risk_config):
        """Large speed differential => variance_risk near 1.0."""
        model = _make_model(risk_config, road_limit=13.9)
        vdata = _make_vdata(speed=13.9)

        # Vehicle at 13.9, neighbours at 0.0 => diff = 13.9, threshold = 5.0
        # variance_risk = min(13.9 / 5.0, 1.0) = 1.0
        neighbors_slow = {"n1": 0.0, "n2": 0.0}
        score_high_var = model.get_risk_score("v0", vdata, neighbors_slow)

        # Compare with no-variance case
        score_no_var = model.get_risk_score("v0", vdata, {"n1": 13.9})
        assert score_high_var > score_no_var

    def test_variance_risk_same_speed(self, risk_config):
        """All neighbours at same speed as ego => variance_risk = 0.0."""
        model = _make_model(risk_config, road_limit=13.9)
        vdata = _make_vdata(speed=10.0)
        neighbors = {"n1": 10.0, "n2": 10.0, "n3": 10.0}

        score = model.get_risk_score("v0", vdata, neighbors)

        # Recompute without neighbours to verify variance added nothing
        score_alone = model.get_risk_score("v0", vdata, {})
        assert abs(score - score_alone) < 1e-9


# ---------------------------------------------------------------------------
# Density risk tests
# ---------------------------------------------------------------------------


class TestDensityRisk:
    """Tests for the density-risk component."""

    def test_density_risk_at_peak(self, risk_config):
        """Density = peak_density => density_risk = 1.0 (Gaussian peak)."""
        model = _make_model(risk_config, road_limit=13.9, edge_length_m=1000.0)
        # peak_density = 15 veh/km.  Edge is 1 km long.
        # Need 15 vehicles on edge_1 to hit peak density.
        peak = risk_config["peak_density_vehicles_per_km"]
        density_risk = model._density_risk_curve(peak)
        assert abs(density_risk - 1.0) < 1e-9

    def test_density_risk_far_from_peak(self, risk_config):
        """Density = 0 => density_risk is low (far from peak)."""
        model = _make_model(risk_config, road_limit=13.9)
        density_risk = model._density_risk_curve(0.0)
        # exp(-(15)^2 / (2 * 7.5^2)) = exp(-2) ≈ 0.135
        expected = math.exp(-2.0)
        assert abs(density_risk - expected) < 1e-6


# ---------------------------------------------------------------------------
# Composite risk tests
# ---------------------------------------------------------------------------


class TestCompositeRisk:
    """Tests for the weighted composite score."""

    def test_composite_risk_weights_sum(self, risk_config):
        """Verify that the three weights sum to 1.0 by default."""
        total = (
            risk_config["speed_weight"]
            + risk_config["speed_variance_weight"]
            + risk_config["density_weight"]
        )
        assert abs(total - 1.0) < 1e-9

    def test_risk_score_fast_complete(self, risk_config):
        """Full integration: get_risk_score_fast returns a value in [0, 1]."""
        model = _make_model(risk_config, road_limit=13.9)
        # Pre-fill density cache for a realistic scenario
        model._edge_density_cache["edge_1"] = 15.0  # peak density

        vdata = _make_vdata(speed=13.9)
        neighbors = {"n1": 5.0, "n2": 7.0}
        score = model.get_risk_score("v0", vdata, neighbors)

        assert 0.0 <= score <= 1.0

    def test_empty_edge_id_returns_zero(self, risk_config):
        """If edge_id is empty string, get_risk_score_fast returns 0.0."""
        model = _make_model(risk_config)
        vdata = _make_vdata(speed=10.0, edge_id="")
        assert model.get_risk_score("v0", vdata, {}) == 0.0


# ---------------------------------------------------------------------------
# Trigger tests
# ---------------------------------------------------------------------------


class TestTrigger:
    """Tests for should_trigger_accident_fast."""

    def test_trigger_below_threshold_never_fires(self, risk_config):
        """Risk below trigger_threshold => should_trigger always returns False."""
        # Use a very low speed so composite risk is well below threshold (0.35)
        model = _make_model(risk_config, road_limit=13.9)
        vdata = _make_vdata(speed=1.0)  # very slow => low risk

        results = [model.should_trigger_accident("v0", vdata, {}) for _ in range(1000)]
        assert not any(results), "Should never trigger when risk < threshold"

    def test_trigger_above_threshold_can_fire(self, risk_config):
        """Risk above threshold with high base_prob => can fire."""
        # Increase base_probability to make triggering very likely
        config = dict(risk_config)
        config["base_probability"] = 1.0  # guaranteed (almost)
        config["trigger_threshold"] = 0.01  # very low threshold
        model = _make_model(config, road_limit=13.9)
        model._edge_density_cache["edge_1"] = 15.0  # peak density

        vdata = _make_vdata(speed=13.9)
        neighbors = {"n1": 0.0}  # high variance

        random.seed(42)
        results = [model.should_trigger_accident("v0", vdata, neighbors) for _ in range(100)]
        assert any(results), "Should trigger at least once with high probability"


# ---------------------------------------------------------------------------
# prepare_step tests
# ---------------------------------------------------------------------------


class TestPrepareStep:
    """Tests for the prepare_step density pre-computation."""

    def test_prepare_step_computes_density(self, risk_config):
        """prepare_step should populate _edge_density_cache from subscription data."""
        model = RiskModel(risk_config)
        # Pre-fill edge length cache (1 km)
        model._edge_length_cache["edge_A"] = 1.0

        # Simulate 10 vehicles on edge_A
        all_sub = {}
        for i in range(10):
            all_sub[f"v{i}"] = {
                _tc.VAR_ROAD_ID: "edge_A",
                _tc.VAR_SPEED: 10.0,
                _tc.VAR_POSITION: (float(i * 100), 0.0),
            }

        model.prepare_step(all_sub)

        assert "edge_A" in model._edge_density_cache
        assert abs(model._edge_density_cache["edge_A"] - 10.0) < 1e-9

    def test_prepare_step_ignores_junction_edges(self, risk_config):
        """Edges starting with ':' (junctions) should be excluded from density."""
        model = RiskModel(risk_config)
        all_sub = {
            "v0": {_tc.VAR_ROAD_ID: ":junction_1", _tc.VAR_SPEED: 5.0},
        }
        model.prepare_step(all_sub)
        assert ":junction_1" not in model._edge_density_cache
