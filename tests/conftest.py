"""
tests/conftest.py
=================
Shared fixtures and traci mocking for the SAS test suite.

The traci module is only available when SUMO is installed.  Every source module
in this project (risk_model, accident_manager, metrics, runner) imports traci
at module level.  We inject a MagicMock into sys.modules BEFORE any of those
imports run, so the entire test suite can execute without SUMO.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock the traci module hierarchy BEFORE any project imports
# ---------------------------------------------------------------------------

mock_traci = MagicMock()

# TraCI constants — actual integer values used in subscription dicts
mock_traci.constants.VAR_SPEED = 0x40
mock_traci.constants.VAR_ROAD_ID = 0x50
mock_traci.constants.VAR_POSITION = 0x42
mock_traci.constants.VAR_LANE_ID = 0x51
mock_traci.constants.VAR_LANEPOSITION = 0x56
mock_traci.constants.VAR_MAXSPEED = 0x41
mock_traci.constants.CMD_GET_VEHICLE_VARIABLE = 0xA4

# TraCI exception type — must be a real exception class for `except` clauses
mock_traci.exceptions.TraCIException = type("TraCIException", (Exception,), {})

# Register the mock in sys.modules so `import traci` resolves to it
sys.modules["traci"] = mock_traci
sys.modules["traci.constants"] = mock_traci.constants
sys.modules["traci.exceptions"] = mock_traci.exceptions

# Ensure the src tree is on sys.path so `from sas... import ...` works.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_project_root, "src")
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config():
    """A complete config dict matching the structure of config.yaml."""
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
                "moderate": {
                    "weight": 28,
                    "duration_min_s": 900,
                    "duration_max_s": 2700,
                    "lane_capacity_fraction": 0.40,
                    "response_time_s": 600,
                    "secondary_risk_radius_m": 150,
                    "secondary_risk_multiplier": 2.5,
                },
                "major": {
                    "weight": 8,
                    "duration_min_s": 2700,
                    "duration_max_s": 7200,
                    "lane_capacity_fraction": 0.10,
                    "response_time_s": 1200,
                    "secondary_risk_radius_m": 300,
                    "secondary_risk_multiplier": 4.0,
                },
                "critical": {
                    "weight": 2,
                    "duration_min_s": 3600,
                    "duration_max_s": 18000,
                    "lane_capacity_fraction": 0.00,
                    "response_time_s": 1800,
                    "secondary_risk_radius_m": 500,
                    "secondary_risk_multiplier": 6.0,
                },
            },
        },
        "output": {
            "output_folder": "results/",
            "metrics_interval_steps": 60,
            "save_vehicle_snapshots": True,
            "save_accident_reports": True,
            "compute_antifragility_index": True,
            "save_accident_heatmap": True,
            "pre_window_seconds": 300,
            "post_window_seconds": 300,
            "baseline_window_steps": 3600,
        },
    }


@pytest.fixture
def risk_config(sample_config):
    """Just the 'risk' section of the config."""
    return sample_config["risk"]


@pytest.fixture
def accident_config(sample_config):
    """Just the 'accident' section of the config."""
    return sample_config["accident"]


@pytest.fixture
def output_config(tmp_path):
    """The 'output' section with a tmp_path output folder for filesystem tests."""
    return {
        "output_folder": str(tmp_path / "test_output"),
        "metrics_interval_steps": 60,
        "save_vehicle_snapshots": True,
        "save_accident_reports": True,
        "compute_antifragility_index": True,
        "save_accident_heatmap": True,
        "pre_window_seconds": 300,
        "post_window_seconds": 300,
        "baseline_window_steps": 3600,
    }
