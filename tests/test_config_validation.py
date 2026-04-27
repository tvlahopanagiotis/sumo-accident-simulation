"""
tests/test_config_validation.py
===============================
Tests for validate_config from sas.simulation.runner.

All tests use the mock traci injected by conftest.py.
os.path.exists is also mocked where needed so tests do not depend on the
filesystem having a real .sumocfg file.
"""

from unittest.mock import patch

import pytest

from sas.simulation.runner import validate_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_config():
    """Return a config dict that passes all validation checks."""
    return {
        "sumo": {
            "config_file": "/tmp/test_network.sumocfg",
            "binary": "sumo",
            "total_steps": 7200,
            "seed": 42,
            "step_length": 1,
        },
        "risk": {
            "base_probability": 3.0e-06,
            "speed_weight": 0.40,
            "speed_exponent": 2.0,
            "speed_variance_weight": 0.30,
            "speed_variance_threshold_ms": 5.0,
            "density_weight": 0.30,
            "peak_density_vehicles_per_km": 15,
            "road_type_multipliers": {
                "highway": 1.5,
                "arterial": 1.0,
                "local": 0.6,
                "intersection": 2.0,
            },
            "trigger_threshold": 0.35,
            "neighbor_radius_m": 150,
        },
        "accident": {
            "max_concurrent_accidents": 2,
            "secondary_accident_enabled": True,
            "incident_effect_mode": "hybrid",
            "reroute_affected_vehicles": True,
            "reroute_radius_m": 750,
            "reroute_interval_s": 60,
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
        },
        "output": {
            "output_folder": "results/",
            "metrics_interval_steps": 60,
            "pre_window_seconds": 300,
            "post_window_seconds": 300,
            "baseline_window_steps": 3600,
        },
    }


# ---------------------------------------------------------------------------
# Valid config test
# ---------------------------------------------------------------------------


class TestValidConfig:
    """Verify that a correct config passes validation without error."""

    @patch("os.path.exists", return_value=True)
    def test_valid_config_passes(self, mock_exists):
        """A complete, valid config should not call sys.exit."""
        config = _valid_config()
        # Should complete without raising SystemExit
        validate_config(config)


# ---------------------------------------------------------------------------
# Risk parameter validation
# ---------------------------------------------------------------------------


class TestRiskValidation:
    """Tests for risk section parameter validation."""

    @patch("os.path.exists", return_value=True)
    def test_invalid_probability_exits(self, mock_exists):
        """base_probability > 1 should cause sys.exit(1)."""
        config = _valid_config()
        config["risk"]["base_probability"] = 1.5  # Invalid: > 1
        with pytest.raises(SystemExit):
            validate_config(config)

    @patch("os.path.exists", return_value=True)
    def test_invalid_probability_zero_exits(self, mock_exists):
        """base_probability = 0 should cause sys.exit(1)."""
        config = _valid_config()
        config["risk"]["base_probability"] = 0  # Invalid: must be > 0
        with pytest.raises(SystemExit):
            validate_config(config)

    @patch("os.path.exists", return_value=True)
    def test_invalid_threshold_exits(self, mock_exists):
        """trigger_threshold > 1 should cause sys.exit(1)."""
        config = _valid_config()
        config["risk"]["trigger_threshold"] = 1.5  # Invalid: > 1
        with pytest.raises(SystemExit):
            validate_config(config)


class TestSumoRuntimeValidation:
    """Tests for runtime settings with new 1-second default semantics."""

    @patch("os.path.exists", return_value=True)
    def test_nonpositive_step_length_exits(self, mock_exists):
        config = _valid_config()
        config["sumo"]["step_length"] = 0
        with pytest.raises(SystemExit):
            validate_config(config)


# ---------------------------------------------------------------------------
# Severity tier validation
# ---------------------------------------------------------------------------


class TestSeverityValidation:
    """Tests for accident.severity tier validation."""

    @patch("os.path.exists", return_value=True)
    def test_missing_severity_keys_exits(self, mock_exists):
        """A tier missing required keys should cause sys.exit(1)."""
        config = _valid_config()
        config["accident"]["severity"]["minor"] = {
            "weight": 62,
            # Missing: duration_min_s, duration_max_s, lane_capacity_fraction,
            #          response_time_s, secondary_risk_radius_m,
            #          secondary_risk_multiplier
        }
        with pytest.raises(SystemExit):
            validate_config(config)

    @patch("os.path.exists", return_value=True)
    def test_duration_min_exceeds_max_exits(self, mock_exists):
        """duration_min_s >= duration_max_s should cause sys.exit(1)."""
        config = _valid_config()
        tier = config["accident"]["severity"]["minor"]
        tier["duration_min_s"] = 1000
        tier["duration_max_s"] = 500  # Invalid: min > max
        with pytest.raises(SystemExit):
            validate_config(config)

    @patch("os.path.exists", return_value=True)
    def test_invalid_capacity_fraction_exits(self, mock_exists):
        """lane_capacity_fraction > 1.0 should cause sys.exit(1)."""
        config = _valid_config()
        tier = config["accident"]["severity"]["minor"]
        tier["lane_capacity_fraction"] = 1.5  # Invalid: > 1.0
        with pytest.raises(SystemExit):
            validate_config(config)

    @patch("os.path.exists", return_value=True)
    def test_empty_severity_exits(self, mock_exists):
        """No severity tiers defined should cause sys.exit(1)."""
        config = _valid_config()
        config["accident"]["severity"] = {}  # Empty
        with pytest.raises(SystemExit):
            validate_config(config)

    @patch("os.path.exists", return_value=True)
    def test_invalid_incident_effect_mode_exits(self, mock_exists):
        config = _valid_config()
        config["accident"]["incident_effect_mode"] = "mystery"
        with pytest.raises(SystemExit):
            validate_config(config)

    @patch("os.path.exists", return_value=True)
    def test_invalid_reroute_interval_exits(self, mock_exists):
        config = _valid_config()
        config["accident"]["reroute_interval_s"] = 0
        with pytest.raises(SystemExit):
            validate_config(config)


# ---------------------------------------------------------------------------
# SUMO config file validation
# ---------------------------------------------------------------------------


class TestSumoConfigValidation:
    """Tests for sumo.config_file path validation."""

    @patch("os.path.exists", return_value=False)
    def test_missing_config_file_exits(self, mock_exists):
        """A nonexistent .sumocfg path should cause sys.exit(1)."""
        config = _valid_config()
        config["sumo"]["config_file"] = "/nonexistent/path/network.sumocfg"
        with pytest.raises(SystemExit):
            validate_config(config)


class TestLiveProgressValidation:
    """Tests for output.live_progress_refresh_steps validation."""

    @patch("os.path.exists", return_value=True)
    def test_nonpositive_live_refresh_exits(self, mock_exists):
        """live_progress_refresh_steps must be positive when configured."""
        config = _valid_config()
        config["output"]["live_progress_refresh_steps"] = 0
        with pytest.raises(SystemExit):
            validate_config(config)
