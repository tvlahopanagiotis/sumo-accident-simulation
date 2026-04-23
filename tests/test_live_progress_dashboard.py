from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

from sas.visualization.visualize import LiveProgressDashboard, resolve_net_file


def _snapshot(
    timestamp_seconds: int,
    vehicle_count: int,
    mean_speed_kmh: float,
    throughput_per_hour: float,
    active_accidents: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        timestamp_seconds=timestamp_seconds,
        vehicle_count=vehicle_count,
        mean_speed_kmh=mean_speed_kmh,
        throughput_per_hour=throughput_per_hour,
        active_accidents=active_accidents,
    )


def test_live_progress_dashboard_writes_png(tmp_path: Path) -> None:
    net_path = tmp_path / "toy.net.xml"
    net_path.write_text(
        """
<net>
  <edge id="edge_a" from="n0" to="n1">
    <lane id="edge_a_0" index="0" speed="13.9" length="100.0" shape="0.0,0.0 100.0,0.0"/>
  </edge>
  <edge id="edge_b" from="n1" to="n2">
    <lane id="edge_b_0" index="0" speed="13.9" length="100.0" shape="100.0,0.0 100.0,100.0"/>
  </edge>
</net>
""".strip(),
        encoding="utf-8",
    )

    dashboard = LiveProgressDashboard(
        output_dir=str(tmp_path),
        total_steps=7200,
        refresh_interval_steps=60,
        prefer_window=False,
        net_xml_path=str(net_path),
    )

    snapshots = [
        _snapshot(60, 120, 28.5, 950.0, 0),
        _snapshot(120, 180, 25.0, 1100.0, 1),
        _snapshot(180, 165, 23.2, 1025.0, 2),
    ]

    dashboard.update(
        snapshots,
        current_step=180,
        active_accident_count=2,
        resolved_accidents=1,
        total_accidents=3,
        edge_vehicle_counts={"edge_a": 4, "edge_b": 1},
        accident_points=[{"x": 10.0, "y": 0.0, "severity": "MAJOR"}],
        resolved_accident_points=[{"x": 5.0, "y": -3.0, "severity": "MINOR"}],
        force=True,
    )
    dashboard.update(
        snapshots,
        current_step=240,
        active_accident_count=1,
        resolved_accidents=2,
        total_accidents=3,
        edge_vehicle_counts={"edge_a": 2},
        accident_points=[{"x": 100.0, "y": 20.0, "severity": "MINOR"}],
        resolved_accident_points=[
            {"x": 5.0, "y": -3.0, "severity": "MINOR"},
            {"x": 60.0, "y": 25.0, "severity": "MAJOR"},
        ],
        force=True,
    )
    dashboard.close()

    assert (tmp_path / "live_progress.png").exists()


def test_resolve_net_file_from_sumocfg(tmp_path: Path) -> None:
    net_path = tmp_path / "toy.net.xml"
    net_path.write_text(
        '<net><edge id="e1"><lane id="e1_0" shape="0,0 1,1"/></edge></net>',
        encoding="utf-8",
    )
    cfg_path = tmp_path / "toy.sumocfg"
    cfg_path.write_text(
        """
<configuration>
  <input>
    <net-file value="toy.net.xml"/>
  </input>
</configuration>
""".strip(),
        encoding="utf-8",
    )

    assert resolve_net_file(sumocfg_path=str(cfg_path)) == str(net_path)
