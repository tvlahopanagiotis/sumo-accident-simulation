from __future__ import annotations

from pathlib import Path

from sas.gui.cities import _build_preview_cached, _parse_lane_count, _parse_speed_kph, update_city_speed_limits


def test_parse_speed_kph_handles_metric_and_mph() -> None:
    assert _parse_speed_kph("50") == 50.0
    assert _parse_speed_kph("35 mph") == 56.33
    assert _parse_speed_kph("signals") is None


def test_parse_lane_count_uses_first_numeric_value() -> None:
    assert _parse_lane_count("3") == 3
    assert _parse_lane_count("2;3") == 2
    assert _parse_lane_count("unknown") is None


def test_build_preview_cached_reads_highway_geometry(tmp_path: Path) -> None:
    osm_path = tmp_path / "sample.osm"
    osm_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="1" lat="40.0000" lon="22.0000" />
  <node id="2" lat="40.0005" lon="22.0005" />
  <node id="3" lat="40.0010" lon="22.0010" />
  <way id="100">
    <nd ref="1" />
    <nd ref="2" />
    <nd ref="3" />
    <tag k="highway" v="primary" />
    <tag k="maxspeed" v="50" />
    <tag k="lanes" v="2" />
    <tag k="oneway" v="yes" />
    <tag k="name" v="Example Avenue" />
  </way>
</osm>
""",
        encoding="utf-8",
    )

    preview = _build_preview_cached(str(osm_path), osm_path.stat().st_mtime_ns)

    assert preview["source_path"].endswith("sample.osm")
    assert preview["stats"]["feature_count"] == 1
    assert preview["stats"]["with_speed_limit"] == 1
    assert preview["stats"]["oneway_count"] == 1
    assert preview["stats"]["road_type_counts"] == {"primary": 1}
    feature = preview["features"][0]
    assert feature["road_type"] == "primary"
    assert feature["speed_kph"] == 50.0
    assert feature["lane_count"] == 2
    assert feature["oneway"] is True
    assert feature["node_ids"] == ["1", "2", "3"]
    assert feature["coords"][0] == [40.0, 22.0]


def test_build_preview_cached_reads_signalized_intersections(tmp_path: Path) -> None:
    osm_path = tmp_path / "signals.osm"
    osm_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="1" lat="40.0000" lon="22.0000" />
  <node id="2" lat="40.0005" lon="22.0005">
    <tag k="highway" v="traffic_signals" />
  </node>
  <node id="3" lat="40.0010" lon="22.0010" />
  <way id="100">
    <nd ref="1" />
    <nd ref="2" />
    <tag k="highway" v="primary" />
  </way>
  <way id="200">
    <nd ref="2" />
    <nd ref="3" />
    <tag k="highway" v="secondary" />
  </way>
</osm>
""",
        encoding="utf-8",
    )

    preview = _build_preview_cached(str(osm_path), osm_path.stat().st_mtime_ns)

    assert preview["stats"]["signalized_intersection_count"] == 1
    assert preview["intersections"][0]["id"] == "2"
    assert preview["intersections"][0]["connected_road_types"] == ["primary", "secondary"]


def test_update_city_speed_limits_writes_maxspeed_tag(tmp_path: Path, monkeypatch) -> None:
    from sas.gui import cities as cities_module

    data_root = tmp_path / "data" / "cities" / "sample" / "network"
    data_root.mkdir(parents=True)
    configs_root = tmp_path / "configs" / "sample"
    configs_root.mkdir(parents=True)
    (configs_root / "default.yaml").write_text("sumo: {}\n", encoding="utf-8")
    (tmp_path / "data" / "cities" / "sample" / "city_metadata.json").write_text("{}", encoding="utf-8")
    osm_path = data_root / "sample.osm"
    osm_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="1" lat="40.0000" lon="22.0000" />
  <node id="2" lat="40.0005" lon="22.0005" />
  <way id="100">
    <nd ref="1" />
    <nd ref="2" />
    <tag k="highway" v="primary" />
  </way>
</osm>
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(cities_module, "_CITIES_ROOT", (tmp_path / "data" / "cities").resolve())
    monkeypatch.setattr(cities_module, "_CONFIGS_ROOT", (tmp_path / "configs").resolve())
    monkeypatch.setattr(cities_module, "PROJECT_ROOT", tmp_path.resolve())
    cities_module._build_preview_cached.cache_clear()

    result = update_city_speed_limits("sample", ["100"], 50)

    assert result["updated_way_count"] == 1
    text = osm_path.read_text(encoding="utf-8")
    assert 'k="maxspeed"' in text
    assert 'v="50"' in text
