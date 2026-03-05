"""
tests/test_accident_manager.py
==============================
Tests for the AccidentManager class (accident_manager.py).

All tests use the mock traci injected by conftest.py.
"""

import math
import random
import statistics
from collections import Counter
from unittest.mock import MagicMock, patch

import pytest

import traci  # mock from conftest.py
from accident_manager import AccidentManager, Accident, Severity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_severity_config():
    """Return a minimal accident config with one tier."""
    return {
        "max_concurrent_accidents": 2,
        "secondary_accident_enabled": True,
        "severity": {
            "minor": {
                "weight": 62,
                "duration_min_s": 120,
                "duration_max_s": 900,
                "lane_capacity_fraction": 0.70,
                "response_time_s": 300,
                "secondary_risk_radius_m": 75,
                "secondary_risk_multiplier": 1.5,
            },
        },
    }


def _setup_traci_for_trigger(
    edge_id="edge_1",
    lane_id="edge_1_0",
    position=50.0,
    x=100.0,
    y=200.0,
    lane_max_speed=13.9,
):
    """Configure mock traci returns for a successful trigger_accident call."""
    traci.vehicle.getRoadID.return_value = edge_id
    traci.vehicle.getLaneID.return_value = lane_id
    traci.vehicle.getLanePosition.return_value = position
    traci.vehicle.getPosition.return_value = (x, y)
    traci.vehicle.setSpeed.return_value = None
    traci.vehicle.setSpeedMode.return_value = None
    traci.lane.getMaxSpeed.return_value = lane_max_speed
    traci.lane.setMaxSpeed.return_value = None


# ---------------------------------------------------------------------------
# Concurrency tests
# ---------------------------------------------------------------------------

class TestConcurrency:
    """Tests for can_trigger_accident concurrency control."""

    def test_can_trigger_when_empty(self, accident_config):
        """No active accidents => can_trigger_accident returns True."""
        mgr = AccidentManager(accident_config)
        assert mgr.can_trigger_accident() is True

    def test_can_trigger_at_max(self, accident_config):
        """Active == max_concurrent => can_trigger_accident returns False."""
        mgr = AccidentManager(accident_config)
        # Fill up active accidents to max_concurrent (2)
        for i in range(accident_config["max_concurrent_accidents"]):
            acc_id = f"ACC_{i:04d}"
            mgr.active_accidents[acc_id] = Accident(
                accident_id=acc_id,
                trigger_step=0,
                vehicle_id=f"v{i}",
                edge_id="edge_1",
                lane_id="edge_1_0",
                position=50.0,
                x=100.0,
                y=200.0,
            )
        assert mgr.can_trigger_accident() is False


# ---------------------------------------------------------------------------
# Severity tier parsing tests
# ---------------------------------------------------------------------------

class TestSeverityTiers:
    """Tests for _load_severity_tiers configuration parsing."""

    def test_severity_tier_loading(self, accident_config):
        """Verify _load_severity_tiers correctly parses all tiers from config."""
        tiers = AccidentManager._load_severity_tiers(accident_config["severity"])
        tier_names = [t["name"] for t in tiers]
        assert "MINOR" in tier_names
        assert "MODERATE" in tier_names
        assert "MAJOR" in tier_names
        assert "CRITICAL" in tier_names
        assert len(tiers) == 4

        # Check that a specific tier has the expected values
        minor = next(t for t in tiers if t["name"] == "MINOR")
        assert minor["weight"] == 62
        assert minor["duration_min_s"] == 120
        assert minor["duration_max_s"] == 900
        assert minor["lane_capacity_fraction"] == 0.70

    def test_severity_tier_missing_key_raises(self):
        """An incomplete tier definition should raise ValueError."""
        bad_config = {
            "broken_tier": {
                "weight": 10,
                # Missing all other required keys
            },
        }
        with pytest.raises(ValueError, match="missing keys"):
            AccidentManager._load_severity_tiers(bad_config)


# ---------------------------------------------------------------------------
# Duration sampling tests
# ---------------------------------------------------------------------------

class TestDurationSampling:
    """Tests for _sample_duration log-normal distribution."""

    def test_sample_duration_within_bounds(self):
        """All sampled durations must lie within [min_s, max_s]."""
        random.seed(12345)
        min_s, max_s = 120, 900
        samples = [AccidentManager._sample_duration(min_s, max_s) for _ in range(1000)]
        assert all(min_s <= s <= max_s for s in samples), (
            f"Out-of-bounds sample found: min={min(samples)}, max={max(samples)}"
        )

    def test_sample_duration_is_lognormal_shaped(self):
        """Log-normal: median should be less than mean (right-skewed)."""
        random.seed(42)
        min_s, max_s = 120, 900
        samples = [AccidentManager._sample_duration(min_s, max_s) for _ in range(5000)]
        median = statistics.median(samples)
        mean = statistics.mean(samples)
        assert median < mean, (
            f"Expected right-skewed distribution (median < mean), "
            f"got median={median:.1f}, mean={mean:.1f}"
        )


# ---------------------------------------------------------------------------
# Trigger accident tests
# ---------------------------------------------------------------------------

class TestTriggerAccident:
    """Tests for the trigger_accident method."""

    def test_trigger_accident_creates_object(self, accident_config):
        """A successful trigger should return an Accident with correct fields."""
        _setup_traci_for_trigger()
        mgr = AccidentManager(accident_config)

        accident = mgr.trigger_accident("veh_42", current_step=100)

        assert accident is not None
        assert accident.vehicle_id == "veh_42"
        assert accident.trigger_step == 100
        assert accident.edge_id == "edge_1"
        assert accident.phase == "ACTIVE"
        assert accident.accident_id == "ACC_0001"
        assert accident.severity in ("MINOR", "MODERATE", "MAJOR", "CRITICAL")

    def test_trigger_on_junction_returns_none(self, accident_config):
        """Edge ID starting with ':' (junction) => trigger returns None."""
        _setup_traci_for_trigger(edge_id=":junction_node_0")
        mgr = AccidentManager(accident_config)

        result = mgr.trigger_accident("veh_1", current_step=50)
        assert result is None


# ---------------------------------------------------------------------------
# Severity distribution tests
# ---------------------------------------------------------------------------

class TestSeverityDistribution:
    """Tests that the weighted random draw approximates the configured weights."""

    def test_severity_distribution_approximate(self, accident_config):
        """Over 10000 draws, severity proportions should approximate weights."""
        _setup_traci_for_trigger()
        mgr = AccidentManager(accident_config)

        random.seed(42)
        counts = Counter()
        n_draws = 10000
        for i in range(n_draws):
            acc = mgr.trigger_accident(f"v{i}", current_step=i)
            if acc is not None:
                counts[acc.severity] += 1

        total = sum(counts.values())
        # Expected: MINOR ~62%, MODERATE ~28%, MAJOR ~8%, CRITICAL ~2%
        minor_pct = counts.get("MINOR", 0) / total * 100
        moderate_pct = counts.get("MODERATE", 0) / total * 100
        major_pct = counts.get("MAJOR", 0) / total * 100
        critical_pct = counts.get("CRITICAL", 0) / total * 100

        # Allow +/- 5 percentage points tolerance for stochastic sampling
        assert abs(minor_pct - 62) < 5, f"MINOR: expected ~62%, got {minor_pct:.1f}%"
        assert abs(moderate_pct - 28) < 5, f"MODERATE: expected ~28%, got {moderate_pct:.1f}%"
        assert abs(major_pct - 8) < 5, f"MAJOR: expected ~8%, got {major_pct:.1f}%"
        assert abs(critical_pct - 2) < 3, f"CRITICAL: expected ~2%, got {critical_pct:.1f}%"


# ---------------------------------------------------------------------------
# Update lifecycle tests
# ---------------------------------------------------------------------------

class TestUpdateLifecycle:
    """Tests for the update() method and phase transitions."""

    def test_update_transitions_to_clearing(self, accident_config):
        """After response_time_steps, accident phase changes to CLEARING."""
        _setup_traci_for_trigger()
        traci.edge.getLastStepVehicleIDs.return_value = []

        mgr = AccidentManager(accident_config)
        random.seed(0)
        acc = mgr.trigger_accident("v0", current_step=0)
        assert acc is not None
        assert acc.phase == "ACTIVE"

        response_time = acc.response_time_steps
        # Advance to just at response_time
        mgr.update(current_step=response_time)
        assert acc.phase == "CLEARING"

    def test_update_resolves_accident(self, accident_config):
        """After full duration, accident transitions to RESOLVED and is archived."""
        _setup_traci_for_trigger()
        traci.edge.getLastStepVehicleIDs.return_value = []
        traci.vehicle.setSpeed.return_value = None
        traci.vehicle.setSpeedMode.return_value = None
        traci.lane.setMaxSpeed.return_value = None

        mgr = AccidentManager(accident_config)
        random.seed(0)
        acc = mgr.trigger_accident("v0", current_step=0)
        assert acc is not None

        duration = acc.duration_steps
        # Advance past response time first
        mgr.update(current_step=acc.response_time_steps)
        # Now advance past full duration
        mgr.update(current_step=duration)

        assert acc.phase == "RESOLVED"
        assert len(mgr.active_accidents) == 0
        assert len(mgr.resolved_accidents) == 1
        assert mgr.resolved_accidents[0].accident_id == acc.accident_id


# ---------------------------------------------------------------------------
# Secondary multiplier tests
# ---------------------------------------------------------------------------

class TestSecondaryMultiplier:
    """Tests for get_secondary_multiplier proximity calculation."""

    def test_secondary_multiplier_inside_radius(self, accident_config):
        """A point within the secondary risk radius gets an elevated multiplier."""
        mgr = AccidentManager(accident_config)

        # Place a MINOR accident at (100, 200) with radius 75m, multiplier 1.5
        acc = Accident(
            accident_id="ACC_TEST",
            trigger_step=0,
            vehicle_id="v0",
            edge_id="edge_1",
            lane_id="edge_1_0",
            position=50.0,
            x=100.0,
            y=200.0,
            secondary_risk_radius_m=75.0,
            secondary_risk_multiplier=1.5,
        )
        mgr.active_accidents["ACC_TEST"] = acc

        # Point 10m away from the accident
        mult = mgr.get_secondary_multiplier(110.0, 200.0)
        assert mult == 1.5

    def test_secondary_multiplier_outside_radius(self, accident_config):
        """A point far from any accident gets multiplier 1.0."""
        mgr = AccidentManager(accident_config)

        acc = Accident(
            accident_id="ACC_TEST",
            trigger_step=0,
            vehicle_id="v0",
            edge_id="edge_1",
            lane_id="edge_1_0",
            position=50.0,
            x=100.0,
            y=200.0,
            secondary_risk_radius_m=75.0,
            secondary_risk_multiplier=1.5,
        )
        mgr.active_accidents["ACC_TEST"] = acc

        # Point 1000m away
        mult = mgr.get_secondary_multiplier(1100.0, 200.0)
        assert mult == 1.0

    def test_secondary_disabled_returns_one(self):
        """When secondary_accident_enabled is False, multiplier is always 1.0."""
        config = _minimal_severity_config()
        config["secondary_accident_enabled"] = False
        mgr = AccidentManager(config)

        # Place an accident
        acc = Accident(
            accident_id="ACC_TEST",
            trigger_step=0,
            vehicle_id="v0",
            edge_id="edge_1",
            lane_id="edge_1_0",
            position=50.0,
            x=100.0,
            y=200.0,
            secondary_risk_radius_m=75.0,
            secondary_risk_multiplier=1.5,
        )
        mgr.active_accidents["ACC_TEST"] = acc

        # Even very close, multiplier should be 1.0
        mult = mgr.get_secondary_multiplier(100.0, 200.0)
        assert mult == 1.0
