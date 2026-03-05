"""
tests/test_metrics.py
=====================
Tests for MetricsCollector and helper functions (metrics.py).

All tests use the mock traci injected by conftest.py.
"""

import json
import os
import statistics

import pytest

import traci  # mock from conftest.py
from metrics import MetricsCollector, _t_critical, NetworkSnapshot

_tc = traci.constants


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_all_sub(n_vehicles, speed=10.0):
    """
    Build a mock getAllSubscriptionResults() dict with n_vehicles
    all travelling at the given speed.
    """
    return {
        f"v{i}": {_tc.VAR_SPEED: speed}
        for i in range(n_vehicles)
    }


def _make_collector(output_config, sample_config):
    """Create a MetricsCollector with the test output_config."""
    return MetricsCollector(sample_config, output_config)


# ---------------------------------------------------------------------------
# _t_critical tests
# ---------------------------------------------------------------------------

class TestTCritical:
    """Tests for the t-distribution critical value lookup."""

    def test_t_critical_small_n(self):
        """n=2 (df=1) => t_critical = 12.706."""
        assert _t_critical(2) == 12.706

    def test_t_critical_large_n(self):
        """n > 30 (df >= 30) => normal approximation 1.960."""
        assert _t_critical(31) == 1.960
        assert _t_critical(100) == 1.960
        assert _t_critical(1000) == 1.960

    def test_t_critical_n_1(self):
        """n=1 (df=0) => infinity (cannot compute CI with one sample)."""
        result = _t_critical(1)
        assert result == float("inf")

    def test_t_critical_intermediate(self):
        """n=6 (df=5) => 2.571 from the lookup table."""
        assert _t_critical(6) == 2.571


# ---------------------------------------------------------------------------
# accumulate_arrivals tests
# ---------------------------------------------------------------------------

class TestAccumulateArrivals:
    """Tests for the arrival accumulator."""

    def test_accumulate_arrivals(self, output_config, sample_config):
        """accumulate_arrivals should additively increase the internal counter."""
        mc = _make_collector(output_config, sample_config)
        assert mc._arrived_interval == 0

        mc.accumulate_arrivals(5)
        assert mc._arrived_interval == 5

        mc.accumulate_arrivals(3)
        assert mc._arrived_interval == 8

        mc.accumulate_arrivals(0)
        assert mc._arrived_interval == 8


# ---------------------------------------------------------------------------
# record_step tests
# ---------------------------------------------------------------------------

class TestRecordStep:
    """Tests for the record_step snapshot recording."""

    def test_record_step_empty_network(self, output_config, sample_config):
        """No vehicles => record_step returns early, no snapshot created."""
        mc = _make_collector(output_config, sample_config)
        all_sub = {}  # empty network

        initial_snapshot_count = len(mc.snapshots)
        mc.record_step(60, active_accident_count=0, all_sub=all_sub)
        assert len(mc.snapshots) == initial_snapshot_count

    def test_record_step_creates_snapshot(self, output_config, sample_config):
        """With vehicles present, record_step should create a NetworkSnapshot."""
        mc = _make_collector(output_config, sample_config)
        all_sub = _make_all_sub(10, speed=12.0)

        mc.record_step(60, active_accident_count=0, all_sub=all_sub)

        assert len(mc.snapshots) == 1
        snap = mc.snapshots[0]
        assert snap.step == 60
        assert snap.vehicle_count == 10
        assert abs(snap.mean_speed_ms - 12.0) < 0.01

    def test_record_step_resets_arrival_counter(self, output_config, sample_config):
        """After record_step, the arrival counter should be reset to 0."""
        mc = _make_collector(output_config, sample_config)
        mc.accumulate_arrivals(15)
        assert mc._arrived_interval == 15

        all_sub = _make_all_sub(5, speed=10.0)
        mc.record_step(60, active_accident_count=0, all_sub=all_sub)
        assert mc._arrived_interval == 0


# ---------------------------------------------------------------------------
# Baseline speed tests
# ---------------------------------------------------------------------------

class TestBaseline:
    """Tests for the free-flow baseline speed establishment."""

    def test_baseline_speed_established(self, output_config, sample_config):
        """After enough clean snapshots in the baseline window, baseline is set."""
        mc = _make_collector(output_config, sample_config)

        # Record 10 clean snapshots within baseline_window_steps (1800s)
        for step in range(60, 660, 60):
            all_sub = _make_all_sub(20, speed=15.0)
            mc.record_step(step, active_accident_count=0, all_sub=all_sub)

        # After 10 snapshots (>= 5 clean), baseline should be established
        assert mc._baseline_speed is not None
        assert abs(mc._baseline_speed - 15.0) < 0.1


# ---------------------------------------------------------------------------
# Antifragility Index tests
# ---------------------------------------------------------------------------

class TestAntifragilityIndex:
    """Tests for the per-event and aggregate AI computation."""

    def test_ai_positive_when_post_faster(self, output_config, sample_config):
        """If post-accident speed > pre-accident speed => AI > 0."""
        mc = _make_collector(output_config, sample_config)

        # Manually inject a finalised per-event AI where post > pre
        mc._per_event_ais.append({
            "accident_id": "ACC_0001",
            "event_ai": 0.10,  # post was 10% faster than pre
            "pre_mean_speed_kmh": 50.0,
            "post_mean_speed_kmh": 55.0,
            "n_pre_samples": 5,
            "n_post_samples": 5,
        })

        result = mc.compute_antifragility_index()
        assert result["antifragility_index"] is not None
        assert result["antifragility_index"] > 0

    def test_ai_negative_when_post_slower(self, output_config, sample_config):
        """If post-accident speed < pre-accident speed => AI < 0."""
        mc = _make_collector(output_config, sample_config)

        mc._per_event_ais.append({
            "accident_id": "ACC_0001",
            "event_ai": -0.15,  # post was 15% slower
            "pre_mean_speed_kmh": 50.0,
            "post_mean_speed_kmh": 42.5,
            "n_pre_samples": 5,
            "n_post_samples": 5,
        })

        result = mc.compute_antifragility_index()
        assert result["antifragility_index"] < 0

    def test_ai_no_events(self, output_config, sample_config):
        """No accidents => antifragility_index is None."""
        mc = _make_collector(output_config, sample_config)

        result = mc.compute_antifragility_index()
        assert result["antifragility_index"] is None
        assert "note" in result

    def test_ai_confidence_interval_with_multiple_events(self, output_config, sample_config):
        """With >= 2 events, a 95% CI should be computed."""
        mc = _make_collector(output_config, sample_config)

        mc._per_event_ais = [
            {"accident_id": "ACC_0001", "event_ai": 0.05,
             "pre_mean_speed_kmh": 50.0, "post_mean_speed_kmh": 52.5,
             "n_pre_samples": 5, "n_post_samples": 5},
            {"accident_id": "ACC_0002", "event_ai": -0.03,
             "pre_mean_speed_kmh": 50.0, "post_mean_speed_kmh": 48.5,
             "n_pre_samples": 5, "n_post_samples": 5},
            {"accident_id": "ACC_0003", "event_ai": 0.02,
             "pre_mean_speed_kmh": 50.0, "post_mean_speed_kmh": 51.0,
             "n_pre_samples": 5, "n_post_samples": 5},
        ]

        result = mc.compute_antifragility_index()
        assert "ci_95_low" in result
        assert "ci_95_high" in result
        assert result["ci_95_low"] < result["ci_95_high"]
        assert result["ci_95_low"] <= result["antifragility_index"] <= result["ci_95_high"]


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------

class TestExport:
    """Tests for file export functionality."""

    def test_export_creates_files(self, output_config, sample_config):
        """export_all should write accident_reports.json and network_metrics.csv."""
        mc = _make_collector(output_config, sample_config)

        # Record a few snapshots so CSV has content
        for step in range(60, 240, 60):
            all_sub = _make_all_sub(10, speed=12.0)
            mc.record_step(step, active_accident_count=0, all_sub=all_sub)

        mc.export_all()

        output_folder = output_config["output_folder"]
        assert os.path.isfile(os.path.join(output_folder, "accident_reports.json"))
        assert os.path.isfile(os.path.join(output_folder, "network_metrics.csv"))

        # Verify JSON is valid
        with open(os.path.join(output_folder, "accident_reports.json")) as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_export_creates_antifragility_file(self, output_config, sample_config):
        """When compute_antifragility_index is True, the AI JSON is written."""
        mc = _make_collector(output_config, sample_config)

        # Add a fake event so AI computes
        mc._per_event_ais.append({
            "accident_id": "ACC_0001",
            "event_ai": 0.05,
            "pre_mean_speed_kmh": 50.0,
            "post_mean_speed_kmh": 52.5,
            "n_pre_samples": 5,
            "n_post_samples": 5,
        })

        mc.export_all()

        ai_file = os.path.join(output_config["output_folder"], "antifragility_index.json")
        assert os.path.isfile(ai_file)

        with open(ai_file) as f:
            ai_data = json.load(f)
        assert "antifragility_index" in ai_data
