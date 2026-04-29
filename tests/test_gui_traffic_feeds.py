from __future__ import annotations

import json
from pathlib import Path

from sas.gui import traffic_feeds as traffic_feeds_module
from sas.gui import cities as cities_module


def test_discover_traffic_feeds_finds_govgr_city(tmp_path: Path, monkeypatch) -> None:
    city_root = tmp_path / "data" / "cities" / "thessaloniki"
    govgr_root = city_root / "govgr"
    (govgr_root / "downloads").mkdir(parents=True)
    (govgr_root / "targets").mkdir(parents=True)
    (govgr_root / "thessaloniki_vehicle_speed").mkdir(parents=True)
    (city_root / "city_metadata.json").write_text(
        json.dumps({"place": "Thessaloniki, Greece"}),
        encoding="utf-8",
    )
    (govgr_root / "thessaloniki_vehicle_speed" / "datapackage.json").write_text(
        json.dumps({"title": "Vehicle Speed", "resources": []}),
        encoding="utf-8",
    )

    monkeypatch.setattr(traffic_feeds_module, "PROJECT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(
        traffic_feeds_module,
        "_CITIES_ROOT",
        (tmp_path / "data" / "cities").resolve(),
    )

    records = traffic_feeds_module.discover_traffic_feeds()

    assert len(records) == 1
    assert records[0]["slug"] == "thessaloniki"
    assert records[0]["provider"] == "govgr"
    assert records[0]["catalog_count"] == 1


def test_build_traffic_feed_preview_reads_catalog_runs_and_targets(
    tmp_path: Path, monkeypatch
) -> None:
    city_root = tmp_path / "data" / "cities" / "thessaloniki"
    govgr_root = city_root / "govgr"
    catalog_root = govgr_root / "thessaloniki_vehicle_speed"
    download_root = govgr_root / "downloads" / "2026-04-01_120000"
    target_root = govgr_root / "targets" / "post_metro_2025_2026"

    catalog_root.mkdir(parents=True)
    (download_root / "clean").mkdir(parents=True)
    target_root.mkdir(parents=True)
    (city_root / "city_metadata.json").write_text(
        json.dumps({"place": "Thessaloniki, Greece"}),
        encoding="utf-8",
    )
    (catalog_root / "datapackage.json").write_text(
        json.dumps(
            {
                "title": "Vehicle Speed of Thessaloniki",
                "description": "Realtime and historical speed feed.",
                "version": "15 minutes",
                "keywords": ["Speed"],
                "resources": [{"title": "CSV", "format": "CSV", "path": "speed.csv"}],
            }
        ),
        encoding="utf-8",
    )
    (catalog_root / "speed.csv").write_text(
        '"Link_id";"Link_Direction";"Timestamp";"Speed"\n'
        '"1";"1";"2026-01-01 08:00:00.000";"30"\n',
        encoding="utf-8",
    )
    (download_root / "quality_report.json").write_text(
        json.dumps(
            {
                "started_utc": "2026-04-01T12:00:00Z",
                "finished_utc": "2026-04-01T12:10:00Z",
                "args": {"dataset": "speed"},
                "datasets": {
                    "speed": {
                        "realtime": {
                            "rows_clean": 100,
                            "pages_downloaded": 2,
                            "baseline_files": [
                                str(download_root / "baselines" / "baseline_speed_by_hour.csv")
                            ],
                            "clean_csv": str(download_root / "clean" / "speed.csv"),
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (download_root / "clean" / "speed.csv").write_text("Speed\n30\n", encoding="utf-8")
    (target_root / "targets_summary.json").write_text(
        json.dumps(
            {
                "calibration_year": 2025,
                "validation_year": 2026,
                "outputs": {
                    "calibration": {
                        "files": [str(target_root / "calibration_speed_network_hourly.csv")],
                        "speed_meta": {"rows_read": 50},
                        "travel_time_meta": {"rows_read": 10},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    network_root = city_root / "network"
    network_root.mkdir(parents=True, exist_ok=True)
    (network_root / "thessaloniki.osm").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="1" lat="40.0" lon="22.0" />
  <node id="2" lat="40.001" lon="22.001" />
  <way id="1">
    <nd ref="1" />
    <nd ref="2" />
    <tag k="highway" v="primary" />
    <tag k="name" v="Example Road" />
    <tag k="maxspeed" v="50" />
  </way>
</osm>
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(traffic_feeds_module, "PROJECT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(
        traffic_feeds_module,
        "_CITIES_ROOT",
        (tmp_path / "data" / "cities").resolve(),
    )
    monkeypatch.setattr(cities_module, "PROJECT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(
        cities_module,
        "_CITIES_ROOT",
        (tmp_path / "data" / "cities").resolve(),
    )
    monkeypatch.setattr(
        cities_module,
        "_CONFIGS_ROOT",
        (tmp_path / "configs").resolve(),
    )
    cities_module._build_preview_cached.cache_clear()

    preview = traffic_feeds_module.build_traffic_feed_preview("thessaloniki")

    assert preview["source"]["slug"] == "thessaloniki"
    assert preview["target_city"]["slug"] == "thessaloniki"
    assert preview["catalog_datasets"][0]["title"] == "Vehicle Speed of Thessaloniki"
    assert preview["catalog_datasets"][0]["sample_csv"]["columns"] == [
        "Link_id",
        "Link_Direction",
        "Timestamp",
        "Speed",
    ]
    assert preview["download_runs"][0]["datasets"][0]["realtime_rows_clean"] == 100
    assert preview["target_exports"][0]["calibration_year"] == 2025
    assert preview["target_exports"][0]["sets"][0]["name"] == "calibration"
    assert preview["linked_network"] is not None
    assert preview["linked_network"]["stats"]["matched_link_count"] == 1
    assert preview["linked_network"]["features"][0]["speed_current_kph"] == 30.0


def test_build_traffic_feed_preview_can_mix_source_catalogs_with_target_exports(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "data" / "cities" / "thessaloniki"
    target_root = tmp_path / "data" / "cities" / "thessaloniki_centre"
    source_govgr_root = source_root / "govgr"
    target_govgr_root = target_root / "govgr"
    catalog_root = source_govgr_root / "thessaloniki_vehicle_speed"
    download_root = target_govgr_root / "downloads" / "2026-04-01_120000"
    target_export_root = target_govgr_root / "targets" / "calibration_2025_validation_2026"

    catalog_root.mkdir(parents=True)
    (download_root / "clean").mkdir(parents=True)
    target_export_root.mkdir(parents=True)
    (source_root / "city_metadata.json").write_text(
        json.dumps({"place": "Thessaloniki, Greece"}),
        encoding="utf-8",
    )
    (target_root / "city_metadata.json").write_text(
        json.dumps({"place": "Thessaloniki Centre, Greece"}),
        encoding="utf-8",
    )
    (catalog_root / "datapackage.json").write_text(
        json.dumps({"title": "Vehicle Speed", "resources": []}),
        encoding="utf-8",
    )
    (download_root / "quality_report.json").write_text(
        json.dumps({"datasets": {"speed": {"realtime": {"rows_clean": 12}}}}),
        encoding="utf-8",
    )
    (target_export_root / "targets_summary.json").write_text(
        json.dumps({"calibration_year": 2025, "validation_year": 2026, "outputs": {}}),
        encoding="utf-8",
    )
    network_root = target_root / "network"
    network_root.mkdir(parents=True, exist_ok=True)
    (network_root / "thessaloniki_centre.osm").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="1" lat="40.0" lon="22.0" />
  <node id="2" lat="40.001" lon="22.001" />
  <way id="1">
    <nd ref="1" />
    <nd ref="2" />
    <tag k="highway" v="primary" />
  </way>
</osm>
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(traffic_feeds_module, "PROJECT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(
        traffic_feeds_module,
        "_CITIES_ROOT",
        (tmp_path / "data" / "cities").resolve(),
    )
    monkeypatch.setattr(cities_module, "PROJECT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(
        cities_module,
        "_CITIES_ROOT",
        (tmp_path / "data" / "cities").resolve(),
    )
    monkeypatch.setattr(
        cities_module,
        "_CONFIGS_ROOT",
        (tmp_path / "configs").resolve(),
    )
    cities_module._build_preview_cached.cache_clear()

    preview = traffic_feeds_module.build_traffic_feed_preview(
        "thessaloniki",
        target_city_slug="thessaloniki_centre",
    )

    assert preview["source"]["slug"] == "thessaloniki"
    assert preview["target_city"]["slug"] == "thessaloniki_centre"
    assert len(preview["catalog_datasets"]) == 1
    assert preview["download_runs"][0]["datasets"][0]["realtime_rows_clean"] == 12
    assert preview["target_exports"][0]["name"] == "calibration_2025_validation_2026"
