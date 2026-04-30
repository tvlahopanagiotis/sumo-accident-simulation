from __future__ import annotations

import json
from pathlib import Path

from sas.gui import results as results_module


def _write_run(run_root: Path) -> None:
    run_root.mkdir(parents=True)
    (run_root / "metadata.json").write_text(
        json.dumps(
            {
                "created_at": "2026-04-30T10:00:00Z",
                "summary": {
                    "total_accidents": 2,
                    "antifragility_index": 0.12,
                    "mean_speed_kmh": 24.5,
                },
                "config": {
                    "sumo": {
                        "config_file": "data/cities/sample/network/sample.sumocfg",
                        "total_steps": 7200,
                        "step_length": 1,
                        "seed": 7,
                    },
                    "output": {"output_folder": "results/sample"},
                },
            }
        ),
        encoding="utf-8",
    )
    (run_root / "network_metrics.csv").write_text(
        "step,timestamp_seconds,timestamp_minutes,vehicle_count,mean_speed_kmh,"
        "throughput_per_hour,mean_delay_seconds,active_accidents,"
        "active_blocked_lanes,cumulative_accidents,resolved_accidents,speed_ratio\n"
        "30,30,0.5,10,25,120,1,0,0,0,0,1\n",
        encoding="utf-8",
    )


def test_list_run_registry_reads_metadata(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path.resolve()
    results_root = project_root / "results"
    run_root = results_root / "sample" / "run_1"
    _write_run(run_root)

    monkeypatch.setattr(results_module, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(results_module, "_RESULTS_ROOT", results_root.resolve())

    runs = results_module.list_run_registry()

    assert len(runs) == 1
    assert runs[0]["run_root"] == "results/sample/run_1"
    assert runs[0]["city"] == "sample"
    assert runs[0]["total_accidents"] == 2
    assert runs[0]["antifragility_index"] == 0.12


def test_delete_run_removes_only_detected_run_root(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path.resolve()
    results_root = project_root / "results"
    run_root = results_root / "sample" / "run_1"
    _write_run(run_root)

    monkeypatch.setattr(results_module, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(results_module, "_RESULTS_ROOT", results_root.resolve())

    result = results_module.delete_run(run_root / "network_metrics.csv")

    assert result["deleted"] is True
    assert result["path"] == "results/sample/run_1"
    assert not run_root.exists()
