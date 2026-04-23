"""
tests/test_resilience.py
========================

Unit tests for the one-click resilience assessment modules:
  - scenario_generator
  - parallel_runner
  - mfd_analysis
  - resilience_report

Uses the existing conftest.py TraCI mock so no SUMO installation is needed.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from sas.analysis.mfd_analysis import (
    EdgeVulnerability,
    ResilienceScore,
    _greenshields_speed,
    compute_network_lane_km,
    compute_resilience_score,
    compute_weak_points,
    extract_mfd_data,
    fit_greenshields_model,
    interpret_resilience_score,
    score_to_dict,
)
from sas.simulation.parallel_runner import ParallelExecutor
from sas.analysis.scenario_generator import (
    DEFAULT_DEMAND_LEVELS,
    DEFAULT_INCIDENT_CONFIGS,
    QUICK_DEMAND_LEVELS,
    QUICK_INCIDENT_CONFIGS,
    Scenario,
    assign_route_files,
    build_scenario_config,
    generate_scenario_matrix,
    matrix_to_dict,
)

# ---------------------------------------------------------------------------
# TestScenarioGenerator
# ---------------------------------------------------------------------------


class TestScenarioGenerator:
    """Tests for scenario_generator.py."""

    def test_generate_scenario_matrix_default(self, sample_config):
        """Default matrix: 7 demand × 5 incident × 3 seeds = 105."""
        matrix = generate_scenario_matrix(sample_config, "/tmp/test_out")
        expected = len(DEFAULT_DEMAND_LEVELS) * len(DEFAULT_INCIDENT_CONFIGS) * 3
        assert len(matrix.scenarios) == expected
        assert matrix.demand_levels == DEFAULT_DEMAND_LEVELS

    def test_generate_scenario_matrix_quick(self, sample_config):
        """Quick mode: 3 demand × 2 incident × 2 seeds = 12."""
        matrix = generate_scenario_matrix(sample_config, "/tmp/test_out", quick=True)
        expected = len(QUICK_DEMAND_LEVELS) * len(QUICK_INCIDENT_CONFIGS) * 2
        assert len(matrix.scenarios) == expected
        assert matrix.demand_levels == QUICK_DEMAND_LEVELS

    def test_generate_scenario_matrix_custom(self, sample_config):
        """Custom demand levels and seeds."""
        matrix = generate_scenario_matrix(
            sample_config,
            "/tmp/test_out",
            demand_levels=[1.0, 2.0],
            seeds=[42, 43, 44, 45],
        )
        expected = 2 * len(DEFAULT_INCIDENT_CONFIGS) * 4
        assert len(matrix.scenarios) == expected

    def test_scenario_ids_unique(self, sample_config):
        """All scenario IDs must be unique."""
        matrix = generate_scenario_matrix(sample_config, "/tmp/test_out")
        ids = [s.scenario_id for s in matrix.scenarios]
        assert len(ids) == len(set(ids))

    def test_build_scenario_config_overrides(self, sample_config):
        """Config overrides are correctly applied."""
        scenario = Scenario(
            scenario_id="test_scenario",
            scenario_type="high_incident",
            period=1.0,
            seed=99,
            base_probability=5.0e-04,
            output_folder="/tmp/test_run",
            sumocfg_path="/tmp/test.sumocfg",
        )
        cfg = build_scenario_config(sample_config, scenario)
        assert cfg["sumo"]["config_file"] == "/tmp/test.sumocfg"
        assert cfg["sumo"]["seed"] == 99
        assert cfg["risk"]["base_probability"] == 5.0e-04
        assert cfg["output"]["output_folder"] == "/tmp/test_run"
        # Original should not be mutated.
        assert sample_config["risk"]["base_probability"] == 1.5e-04

    def test_assign_route_files(self, sample_config):
        """Route files are assigned to all matching scenarios."""
        matrix = generate_scenario_matrix(
            sample_config,
            "/tmp/test_out",
            demand_levels=[1.0, 2.0],
            seeds=[42],
        )
        route_map = {
            1.0: ("/tmp/routes/1p00.sumocfg", "/tmp/routes/1p00.rou.xml"),
            2.0: ("/tmp/routes/2p00.sumocfg", "/tmp/routes/2p00.rou.xml"),
        }
        assign_route_files(matrix, route_map)
        for s in matrix.scenarios:
            assert s.sumocfg_path != ""
            assert s.sumocfg_path.endswith(".sumocfg")

    def test_matrix_to_dict(self, sample_config):
        """Serialization produces expected keys."""
        matrix = generate_scenario_matrix(
            sample_config,
            "/tmp/test_out",
            demand_levels=[1.0],
            seeds=[42],
        )
        d = matrix_to_dict(matrix)
        assert "demand_levels" in d
        assert "total_scenarios" in d
        assert "scenarios" in d
        assert len(d["scenarios"]) == d["total_scenarios"]


# ---------------------------------------------------------------------------
# TestMFDAnalysis
# ---------------------------------------------------------------------------


class TestMFDAnalysis:
    """Tests for mfd_analysis.py."""

    def test_compute_network_lane_km(self, tmp_path):
        """Lane-km correctly computed from a minimal .net.xml."""
        net_xml = tmp_path / "test.net.xml"
        net_xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<net version="1.20">
  <edge id="E1">
    <lane id="E1_0" length="1000.0"/>
    <lane id="E1_1" length="1000.0"/>
  </edge>
  <edge id="E2">
    <lane id="E2_0" length="500.0"/>
  </edge>
  <edge id=":J1_0">
    <lane id=":J1_0_0" length="10.0"/>
  </edge>
</net>""")
        result = compute_network_lane_km(str(net_xml))
        # E1: 2 × 1000m = 2000m, E2: 500m. Internal :J1 excluded.
        assert abs(result - 2.5) < 0.01

    def test_greenshields_speed_model(self):
        """Greenshields formula: v = vf × (1 - k/kj)."""
        k = np.array([0, 50, 100])
        v = _greenshields_speed(k, vf=60.0, kj=100.0)
        np.testing.assert_allclose(v, [60.0, 30.0, 0.0])

    def test_fit_greenshields_model(self):
        """Fit recovers known parameters from synthetic data."""
        np.random.seed(42)
        vf_true, kj_true = 50.0, 80.0
        k = np.linspace(1, 70, 200)
        v = vf_true * (1 - k / kj_true) + np.random.normal(0, 1.5, len(k))
        v = np.clip(v, 0, None)

        df = pd.DataFrame(
            {
                "density_veh_per_km": k,
                "speed_kmh": v,
                "scenario_type": "baseline",
            }
        )
        result = fit_greenshields_model(df)
        assert "error" not in result
        assert abs(result["free_flow_speed_kmh"] - vf_true) < 3.0
        assert abs(result["jam_density_veh_per_km"] - kj_true) < 5.0
        assert result["r_squared"] > 0.9

    def test_fit_greenshields_no_baseline(self):
        """Returns error when no baseline data."""
        df = pd.DataFrame(
            {
                "density_veh_per_km": [1, 2],
                "speed_kmh": [50, 45],
                "scenario_type": "incident",
            }
        )
        result = fit_greenshields_model(df)
        assert "error" in result

    def test_extract_mfd_data(self, tmp_path):
        """MFD extraction from sample CSV."""
        # Create a scenario output folder with a metrics CSV.
        run_dir = tmp_path / "runs" / "baseline_p1p00_s42"
        run_dir.mkdir(parents=True)
        csv_path = run_dir / "network_metrics.csv"
        csv_path.write_text(
            "step,timestamp_seconds,vehicle_count,mean_speed_ms,mean_speed_kmh,"
            "throughput_per_hour,mean_delay_seconds,active_accidents,speed_ratio\n"
            "60,300,100,10.0,36.0,500,5.0,0,0.9\n"
            "120,600,150,8.0,28.8,600,8.0,0,0.85\n"
            "240,1200,200,6.0,21.6,550,12.0,0,0.75\n"
            "360,1800,250,5.0,18.0,500,15.0,0,0.7\n"
        )
        scenario = Scenario(
            scenario_id="baseline_p1p00_s42",
            scenario_type="baseline",
            period=1.0,
            seed=42,
            base_probability=0.0,
            output_folder=str(run_dir),
            sumocfg_path="/tmp/test.sumocfg",
        )
        df = extract_mfd_data([scenario], network_lane_km=10.0, warmup_seconds=500)
        # Only rows with timestamp > 500 should be included.
        assert len(df) == 3
        assert "density_veh_per_km" in df.columns
        # density = vehicle_count / 10.0
        expected_density = 150 / 10.0
        assert abs(df.iloc[0]["density_veh_per_km"] - expected_density) < 0.01

    def test_interpret_resilience_score(self):
        """Grading boundaries."""
        assert interpret_resilience_score(0.95)[0] == "A+"
        assert interpret_resilience_score(0.85)[0] == "A"
        assert interpret_resilience_score(0.75)[0] == "B"
        assert interpret_resilience_score(0.65)[0] == "C"
        assert interpret_resilience_score(0.55)[0] == "D"
        assert interpret_resilience_score(0.40)[0] == "F"

    def test_compute_resilience_score_perfect(self):
        """A network with no degradation scores near 1.0."""
        # Create MFD data where incident speeds ≈ baseline speeds.
        rows = []
        for stype in ["baseline", "default_incident"]:
            for seed in [42, 43]:
                for ts in [1500, 2000, 2500]:
                    rows.append(
                        {
                            "period": 1.0,
                            "scenario_type": stype,
                            "seed": seed,
                            "timestamp_seconds": ts,
                            "density_veh_per_km": 10.0,
                            "flow_veh_per_hour": 500.0,
                            "speed_kmh": 40.0,
                            "active_accidents": 0,
                        }
                    )
        mfd_data = pd.DataFrame(rows)

        scenarios = [
            Scenario("b_s42", "baseline", 1.0, 42, 0.0, "/tmp/b42", "/tmp/t.cfg"),
            Scenario("b_s43", "baseline", 1.0, 43, 0.0, "/tmp/b43", "/tmp/t.cfg"),
            Scenario("i_s42", "default_incident", 1.0, 42, 1.5e-4, "/tmp/i42", "/tmp/t.cfg"),
            Scenario("i_s43", "default_incident", 1.0, 43, 1.5e-4, "/tmp/i43", "/tmp/t.cfg"),
        ]
        results = [
            {"status": "success", "summary": {"antifragility_index": 0.0}},
            {"status": "success", "summary": {"antifragility_index": 0.0}},
            {"status": "success", "summary": {"antifragility_index": 0.0}},
            {"status": "success", "summary": {"antifragility_index": 0.0}},
        ]
        mfd_params = {"free_flow_speed_kmh": 50, "jam_density_veh_per_km": 80}

        score = compute_resilience_score(mfd_data, results, scenarios, [], mfd_params)
        assert score.overall_score >= 0.75  # Speed & throughput = 1.0, recovery = 0.8.
        assert score.grade in ("A+", "A", "B")

    def test_compute_resilience_score_brittle(self):
        """A severely degraded network scores low."""
        rows = []
        for seed in [42]:
            for ts in [1500, 2000]:
                rows.append(
                    {
                        "period": 1.0,
                        "scenario_type": "baseline",
                        "seed": seed,
                        "timestamp_seconds": ts,
                        "density_veh_per_km": 10.0,
                        "flow_veh_per_hour": 500.0,
                        "speed_kmh": 40.0,
                        "active_accidents": 0,
                    }
                )
                rows.append(
                    {
                        "period": 1.0,
                        "scenario_type": "default_incident",
                        "seed": seed,
                        "timestamp_seconds": ts,
                        "density_veh_per_km": 15.0,
                        "flow_veh_per_hour": 100.0,
                        "speed_kmh": 10.0,
                        "active_accidents": 3,
                    }
                )
        mfd_data = pd.DataFrame(rows)

        scenarios = [
            Scenario("b_s42", "baseline", 1.0, 42, 0.0, "/tmp/b42", "/tmp/t.cfg"),
            Scenario("i_s42", "default_incident", 1.0, 42, 1.5e-4, "/tmp/i42", "/tmp/t.cfg"),
        ]
        results = [
            {"status": "success", "summary": {"antifragility_index": -0.3}},
            {"status": "success", "summary": {"antifragility_index": -0.3}},
        ]

        score = compute_resilience_score(mfd_data, results, scenarios, [], {})
        assert score.overall_score < 0.6
        assert score.grade in ("D", "F")

    def test_score_to_dict(self):
        """Serialization produces valid dict."""
        score = ResilienceScore(
            overall_score=0.75,
            grade="B",
            interpretation="Good",
            speed_resilience=0.8,
            throughput_resilience=0.7,
            recovery_resilience=0.75,
            robustness=0.7,
            weights={"speed": 0.3, "throughput": 0.25, "recovery": 0.25, "robustness": 0.2},
            mfd_parameters={},
            ai_aggregate=-0.01,
            weak_points=[],
        )
        d = score_to_dict(score)
        assert d["overall_score"] == 0.75
        assert d["grade"] == "B"


# ---------------------------------------------------------------------------
# TestWeakPoints
# ---------------------------------------------------------------------------


class TestWeakPoints:
    """Tests for weak point analysis."""

    def test_compute_weak_points_no_data(self):
        """Returns empty list when no accident reports exist."""
        scenario = Scenario("b_s42", "baseline", 1.0, 42, 0.0, "/tmp/no_exist", "/tmp/t.cfg")
        result = compute_weak_points([scenario], top_n=5)
        assert result == []

    def test_compute_weak_points_with_data(self, tmp_path):
        """Returns ranked weak points from accident report data."""
        run_dir = tmp_path / "runs" / "incident_p1p00_s42"
        run_dir.mkdir(parents=True)
        reports = [
            {
                "accident_id": "ACC_0001",
                "severity": "MINOR",
                "duration_seconds": 500,
                "location": {"edge_id": "E1", "x": 100.0, "y": 200.0},
                "impact": {"vehicles_affected": 5, "peak_queue_length_vehicles": 3},
            },
            {
                "accident_id": "ACC_0002",
                "severity": "MODERATE",
                "duration_seconds": 1200,
                "location": {"edge_id": "E1", "x": 100.0, "y": 200.0},
                "impact": {"vehicles_affected": 10, "peak_queue_length_vehicles": 7},
            },
            {
                "accident_id": "ACC_0003",
                "severity": "MINOR",
                "duration_seconds": 300,
                "location": {"edge_id": "E2", "x": 300.0, "y": 400.0},
                "impact": {"vehicles_affected": 2, "peak_queue_length_vehicles": 1},
            },
        ]
        with open(run_dir / "accident_reports.json", "w") as f:
            json.dump(reports, f)

        scenario = Scenario(
            "incident_p1p00_s42",
            "default_incident",
            1.0,
            42,
            1.5e-4,
            str(run_dir),
            "/tmp/t.cfg",
        )
        result = compute_weak_points([scenario], top_n=5)
        assert len(result) >= 1
        # E1 should rank higher (more incidents, more affected vehicles).
        assert result[0].edge_id == "E1"
        assert result[0].accident_count == 2


# ---------------------------------------------------------------------------
# TestParallelRunner
# ---------------------------------------------------------------------------


class TestParallelRunner:
    """Tests for parallel_runner.py."""

    def test_default_workers(self):
        """Default max_workers is reasonable."""
        executor = ParallelExecutor()
        assert 1 <= executor.max_workers <= 8
        assert executor.base_port == 10000

    def test_custom_workers(self):
        """Custom max_workers and base_port."""
        executor = ParallelExecutor(max_workers=4, base_port=20000)
        assert executor.max_workers == 4
        assert executor.base_port == 20000

    def test_port_assignment_unique(self):
        """Workers get unique ports (modulo max_workers)."""
        executor = ParallelExecutor(max_workers=3, base_port=10000)
        ports = set()
        for idx in range(3):
            port = executor.base_port + (idx % executor.max_workers)
            ports.add(port)
        assert len(ports) == 3


# ---------------------------------------------------------------------------
# TestResilienceReport
# ---------------------------------------------------------------------------


class TestResilienceReport:
    """Tests for resilience_report.py."""

    def test_report_generation(self, tmp_path):
        """Report generates valid HTML with expected sections."""
        from sas.analysis.resilience_report import generate_resilience_report

        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()

        score = ResilienceScore(
            overall_score=0.72,
            grade="B",
            interpretation="Good resilience",
            speed_resilience=0.8,
            throughput_resilience=0.7,
            recovery_resilience=0.65,
            robustness=0.7,
            weights={"speed": 0.3, "throughput": 0.25, "recovery": 0.25, "robustness": 0.2},
            mfd_parameters={
                "free_flow_speed_kmh": 45.0,
                "jam_density_veh_per_km": 80.0,
                "capacity_veh_per_hour": 900.0,
                "r_squared": 0.92,
            },
            ai_aggregate=-0.02,
            weak_points=[
                EdgeVulnerability(
                    edge_id="E1",
                    x=100,
                    y=200,
                    accident_count=5,
                    mean_duration_seconds=600,
                    mean_vehicles_affected=8,
                    mean_speed_drop_ratio=0.7,
                    edge_importance=0.9,
                    vulnerability_index=0.45,
                ),
            ],
        )

        scenarios = [
            Scenario("b_s42", "baseline", 1.0, 42, 0.0, str(tmp_path / "b42"), "/tmp/t.cfg"),
            Scenario(
                "i_s42", "default_incident", 1.0, 42, 1.5e-4, str(tmp_path / "i42"), "/tmp/t.cfg"
            ),
        ]
        results = [
            {"status": "success", "summary": {"total_accidents": 0, "antifragility_index": None}},
            {"status": "success", "summary": {"total_accidents": 3, "antifragility_index": -0.05}},
        ]
        config = {
            "sumo": {"config_file": "test.sumocfg"},
            "risk": {},
            "accident": {},
            "output": {},
        }

        report_path = generate_resilience_report(
            str(tmp_path),
            score,
            scenarios,
            results,
            config,
            {},
        )
        assert os.path.exists(report_path)

        with open(report_path) as f:
            html = f.read()
        assert "Resilience Assessment Report" in html
        assert "Executive Summary" in html
        assert "Good resilience" in html
        assert "B" in html  # Grade
        assert "Weak Point" in html
        assert "Recommendations" in html
