from __future__ import annotations

import json
from pathlib import Path

from sas.generators import generate_city as generate_city_module


def test_resolve_osm_source_prefers_standard_city_network_file(
    tmp_path: Path, monkeypatch
) -> None:
    city_root = tmp_path / "data" / "cities" / "athens" / "network"
    city_root.mkdir(parents=True)
    osm_path = city_root / "athens.osm"
    osm_path.write_text("<osm />", encoding="utf-8")

    monkeypatch.setattr(generate_city_module, "PROJECT_ROOT", tmp_path.resolve())

    resolved = generate_city_module._resolve_osm_source("athens", None)

    assert resolved == osm_path.resolve()


def test_resolve_osm_source_can_follow_metadata_path(
    tmp_path: Path, monkeypatch
) -> None:
    city_root = tmp_path / "data" / "cities" / "athens"
    bundle_root = city_root / "bundle"
    bundle_root.mkdir(parents=True)
    osm_path = bundle_root / "Athens.osm"
    osm_path.write_text("<osm />", encoding="utf-8")
    (city_root / "city_metadata.json").write_text(
        json.dumps({"osm_extract": "data/cities/athens/bundle/Athens.osm"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(generate_city_module, "PROJECT_ROOT", tmp_path.resolve())

    resolved = generate_city_module._resolve_osm_source("athens", None)

    assert resolved == osm_path.resolve()


def test_resolve_support_csv_discovers_od_and_node_files(
    tmp_path: Path, monkeypatch
) -> None:
    city_root = tmp_path / "data" / "cities" / "sample" / "bundle"
    city_root.mkdir(parents=True)
    od_path = city_root / "sample_od.csv"
    node_path = city_root / "sample_node.csv"
    od_path.write_text("x", encoding="utf-8")
    node_path.write_text("x", encoding="utf-8")

    monkeypatch.setattr(generate_city_module, "PROJECT_ROOT", tmp_path.resolve())

    resolved_od = generate_city_module._resolve_support_csv("sample", None, "od")
    resolved_node = generate_city_module._resolve_support_csv("sample", None, "node")

    assert resolved_od == od_path.resolve()
    assert resolved_node == node_path.resolve()
