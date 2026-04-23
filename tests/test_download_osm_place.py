from __future__ import annotations

from sas.integrations.download_osm_place import _build_overpass_query, _default_output_path, _expand_bbox, _slugify


def test_slugify_place_name() -> None:
    assert _slugify("Seattle, Washington, USA") == "seattle_washington_usa"


def test_default_output_path_uses_slugified_name() -> None:
    assert str(_default_output_path("New York, USA")) == "new_york_usa.osm"


def test_expand_bbox_with_padding() -> None:
    south, west, north, east = _expand_bbox(47.5, -122.5, 47.7, -122.3, 1.0)
    assert south < 47.5
    assert west < -122.5
    assert north > 47.7
    assert east > -122.3


def test_build_overpass_query_uses_highway_filter_by_default() -> None:
    query = _build_overpass_query(47.5, -122.5, 47.7, -122.3, highways_only=True)
    assert 'way["highway"]' in query
    assert 'relation["type"="restriction"]' in query


def test_build_overpass_query_supports_all_features() -> None:
    query = _build_overpass_query(47.5, -122.5, 47.7, -122.3, highways_only=False)
    assert "node(" in query
    assert "way(" in query
    assert "relation(" in query
