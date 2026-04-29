from __future__ import annotations

from pathlib import Path

from sas.integrations.download_osm_place import DEFAULT_ROAD_TYPES, _bootstrap_city_layout, _build_overpass_query, _default_output_path, _expand_bbox, _slugify


def test_slugify_place_name() -> None:
    assert _slugify("Seattle, Washington, USA") == "seattle_washington_usa"


def test_default_output_path_uses_slugified_name() -> None:
    assert str(_default_output_path("New York, USA")) == "data/cities/new_york/network/new_york.osm"


def test_expand_bbox_with_padding() -> None:
    south, west, north, east = _expand_bbox(47.5, -122.5, 47.7, -122.3, 1.0)
    assert south < 47.5
    assert west < -122.5
    assert north > 47.7
    assert east > -122.3


def test_build_overpass_query_uses_highway_filter_by_default() -> None:
    query = _build_overpass_query(47.5, -122.5, 47.7, -122.3)
    assert 'way["highway"~"' in query
    assert DEFAULT_ROAD_TYPES[0] in query
    assert 'relation["type"="restriction"]' in query


def test_build_overpass_query_supports_all_features() -> None:
    query = _build_overpass_query(47.5, -122.5, 47.7, -122.3, include_all_features=True)
    assert "node(" in query
    assert "way(" in query
    assert "relation(" in query


def test_build_overpass_query_supports_custom_road_types() -> None:
    query = _build_overpass_query(47.5, -122.5, 47.7, -122.3, road_types=["primary", "secondary"])
    assert "primary|secondary" in query
    assert "motorway" not in query


def test_bootstrap_city_layout_creates_network_and_config(tmp_path: Path) -> None:
    template = tmp_path / "template.yaml"
    template.write_text(
        "sumo:\n  config_file: data/cities/__CITY_SLUG__/network/__CITY_SLUG__.sumocfg\n"
        "output:\n  output_folder: results/__CITY_SLUG__/default\n",
        encoding="utf-8",
    )

    cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        _bootstrap_city_layout(
            city_slug="athens",
            place="Athens, Greece",
            out_path=Path("data/cities/athens/network/athens.osm"),
            config_out=Path("configs/athens/default.yaml"),
            config_template=template,
            bootstrap_config=True,
        )
    finally:
        os.chdir(cwd)

    assert (tmp_path / "data/cities/athens/network").is_dir()
    assert (tmp_path / "data/cities/athens/govgr/downloads").is_dir()
    assert (tmp_path / "data/cities/athens/govgr/targets").is_dir()
    config_text = (tmp_path / "configs/athens/default.yaml").read_text(encoding="utf-8")
    assert "data/cities/athens/network/athens.sumocfg" in config_text
    assert "results/athens/default" in config_text
    assert (tmp_path / "data/cities/athens/city_metadata.json").exists()
