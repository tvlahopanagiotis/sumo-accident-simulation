from __future__ import annotations

from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

from ..app.config import PROJECT_ROOT

_CITIES_ROOT = (PROJECT_ROOT / "data" / "cities").resolve()
_CONFIGS_ROOT = (PROJECT_ROOT / "configs").resolve()
_PREVIEW_EXCLUDED_HIGHWAYS = {"footway", "path", "cycleway", "steps", "bridleway"}


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _display_name_from_slug(slug: str) -> str:
    return slug.replace("_", " ").replace("-", " ").title()


def _load_city_metadata(city_root: Path) -> dict[str, Any]:
    metadata_path = city_root / "city_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def discover_cities() -> list[dict[str, Any]]:
    if not _CITIES_ROOT.exists():
        return []

    cities: list[dict[str, Any]] = []
    for city_root in sorted(
        (path for path in _CITIES_ROOT.iterdir() if path.is_dir()),
        key=lambda path: path.name.lower(),
    ):
        metadata = _load_city_metadata(city_root)
        slug = city_root.name
        network_dir = city_root / "network"
        osm_files = sorted(network_dir.glob("*.osm")) if network_dir.exists() else []
        net_files = sorted(network_dir.glob("*.net.xml")) if network_dir.exists() else []
        sumocfg_files = sorted(network_dir.glob("*.sumocfg")) if network_dir.exists() else []
        config_path = _CONFIGS_ROOT / slug / "default.yaml"

        cities.append(
            {
                "slug": slug,
                "display_name": str(metadata.get("display_name") or _display_name_from_slug(slug)),
                "city_root": _relative(city_root),
                "network_dir": _relative(network_dir) if network_dir.exists() else None,
                "osm_path": _relative(osm_files[0]) if osm_files else None,
                "net_path": _relative(net_files[0]) if net_files else None,
                "sumocfg_path": _relative(sumocfg_files[0]) if sumocfg_files else None,
                "config_path": _relative(config_path) if config_path.exists() else None,
                "has_osm": bool(osm_files),
                "has_network": bool(net_files),
                "metadata": metadata,
            }
        )
    return cities


def _parse_speed_kph(raw: str | None) -> float | None:
    if not raw:
        return None
    value = raw.strip().lower()
    if not value:
        return None
    if ";" in value:
        value = value.split(";")[0].strip()
    match = re.search(r"(\d+(?:\.\d+)?)", value)
    if not match:
        return None
    speed = float(match.group(1))
    if "mph" in value:
        return round(speed * 1.60934, 2)
    return round(speed, 2)


def _parse_lane_count(raw: str | None) -> int | None:
    if not raw:
        return None
    match = re.search(r"\d+", raw)
    if not match:
        return None
    return int(match.group(0))


def _format_speed_tag(speed_kph: float) -> str:
    if float(speed_kph).is_integer():
        return str(int(speed_kph))
    return f"{speed_kph:.1f}".rstrip("0").rstrip(".")


@lru_cache(maxsize=16)
def _build_preview_cached(osm_path_str: str, mtime_ns: int) -> dict[str, Any]:
    del mtime_ns
    osm_path = Path(osm_path_str)
    tree = ET.parse(osm_path)
    root = tree.getroot()

    nodes: dict[str, tuple[float, float]] = {}
    signal_node_ids: set[str] = set()
    for node in root.findall("node"):
        node_id = node.get("id")
        lat = node.get("lat")
        lon = node.get("lon")
        if node_id and lat and lon:
            nodes[node_id] = (float(lat), float(lon))
            tags = {tag.get("k"): tag.get("v") for tag in node.findall("tag") if tag.get("k")}
            if tags.get("highway") == "traffic_signals":
                signal_node_ids.add(node_id)

    features: list[dict[str, Any]] = []
    intersections: list[dict[str, Any]] = []
    road_type_counts: Counter[str] = Counter()
    all_points: list[tuple[float, float]] = []
    with_speed_limit = 0
    with_lane_count = 0
    oneway_count = 0
    node_road_types: dict[str, set[str]] = {}

    for way in root.findall("way"):
        tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag") if tag.get("k")}
        road_type = tags.get("highway")
        if not road_type or road_type in _PREVIEW_EXCLUDED_HIGHWAYS:
            continue

        refs = [nd.get("ref") for nd in way.findall("nd") if nd.get("ref")]
        coords = [nodes[ref] for ref in refs if ref in nodes]
        if len(coords) < 2:
            continue

        speed_kph = _parse_speed_kph(tags.get("maxspeed"))
        lane_count = _parse_lane_count(tags.get("lanes"))
        oneway_value = (tags.get("oneway") or "").lower()
        oneway = oneway_value in {"yes", "true", "1", "-1"}
        reverse = oneway_value == "-1"

        if speed_kph is not None:
            with_speed_limit += 1
        if lane_count is not None:
            with_lane_count += 1
        if oneway:
            oneway_count += 1

        road_type_counts[road_type] += 1
        all_points.extend(coords)
        for ref in refs:
            node_road_types.setdefault(ref, set()).add(road_type)

        features.append(
            {
                "id": way.get("id") or f"way_{len(features)}",
                "name": tags.get("name"),
                "road_type": road_type,
                "speed_kph": speed_kph,
                "lane_count": lane_count,
                "oneway": oneway,
                "reverse_oneway": reverse,
                "node_ids": refs,
                "coords": [[lat, lon] for lat, lon in coords],
            }
        )

    for node_id in sorted(signal_node_ids):
        coords = nodes.get(node_id)
        if coords is None:
            continue
        connected = sorted(node_road_types.get(node_id, set()))
        intersections.append(
            {
                "id": node_id,
                "coords": [coords[0], coords[1]],
                "connected_road_types": connected,
                "connected_road_count": len(connected),
            }
        )

    if all_points:
        lats = [point[0] for point in all_points]
        lons = [point[1] for point in all_points]
        bbox = [min(lats), min(lons), max(lats), max(lons)]
    else:
        bbox = None

    return {
        "source_path": _relative(osm_path),
        "bbox": bbox,
        "stats": {
            "feature_count": len(features),
            "road_type_counts": dict(sorted(road_type_counts.items())),
            "with_speed_limit": with_speed_limit,
            "with_lane_count": with_lane_count,
            "oneway_count": oneway_count,
            "signalized_intersection_count": len(intersections),
        },
        "features": features,
        "intersections": intersections,
    }


def build_city_network_preview(city_slug: str) -> dict[str, Any]:
    city = next((item for item in discover_cities() if item["slug"] == city_slug), None)
    if city is None:
        raise FileNotFoundError(f"Unknown city slug: {city_slug}")
    osm_path = city.get("osm_path")
    if not osm_path:
        raise FileNotFoundError(f"No OSM extract found for city: {city_slug}")

    osm_file = (PROJECT_ROOT / str(osm_path)).resolve()
    preview = _build_preview_cached(str(osm_file), osm_file.stat().st_mtime_ns)
    return {
        "city": city,
        **preview,
    }


def update_city_speed_limits(city_slug: str, way_ids: list[str], speed_kph: float) -> dict[str, Any]:
    city = next((item for item in discover_cities() if item["slug"] == city_slug), None)
    if city is None:
        raise FileNotFoundError(f"Unknown city slug: {city_slug}")
    osm_path = city.get("osm_path")
    if not osm_path:
        raise FileNotFoundError(f"No OSM extract found for city: {city_slug}")
    if not way_ids:
        raise ValueError("At least one way id is required")
    if speed_kph <= 0:
        raise ValueError("speed_kph must be > 0")

    osm_file = (PROJECT_ROOT / str(osm_path)).resolve()
    tree = ET.parse(osm_file)
    root = tree.getroot()
    targets = set(str(way_id) for way_id in way_ids)
    changed = 0
    rendered_speed = _format_speed_tag(speed_kph)

    for way in root.findall("way"):
        if way.get("id") not in targets:
            continue
        maxspeed_tag = next((tag for tag in way.findall("tag") if tag.get("k") == "maxspeed"), None)
        if maxspeed_tag is None:
            maxspeed_tag = ET.SubElement(way, "tag")
            maxspeed_tag.set("k", "maxspeed")
        maxspeed_tag.set("v", rendered_speed)
        changed += 1

    if changed == 0:
        raise ValueError("None of the selected way ids were found in the OSM extract")

    tree.write(osm_file, encoding="utf-8", xml_declaration=True)
    _build_preview_cached.cache_clear()
    return {
        "city_slug": city_slug,
        "updated_way_count": changed,
        "speed_kph": speed_kph,
        "osm_path": str(osm_path),
    }


def delete_city_ways(city_slug: str, way_ids: list[str]) -> dict[str, Any]:
    city = next((item for item in discover_cities() if item["slug"] == city_slug), None)
    if city is None:
        raise FileNotFoundError(f"Unknown city slug: {city_slug}")
    osm_path = city.get("osm_path")
    if not osm_path:
        raise FileNotFoundError(f"No OSM extract found for city: {city_slug}")
    if not way_ids:
        raise ValueError("At least one way id is required")

    osm_file = (PROJECT_ROOT / str(osm_path)).resolve()
    tree = ET.parse(osm_file)
    root = tree.getroot()
    targets = set(str(way_id) for way_id in way_ids)
    removed = 0

    for way in list(root.findall("way")):
        if way.get("id") not in targets:
            continue
        root.remove(way)
        removed += 1

    if removed == 0:
        raise ValueError("None of the selected way ids were found in the OSM extract")

    tree.write(osm_file, encoding="utf-8", xml_declaration=True)
    _build_preview_cached.cache_clear()
    return {
        "city_slug": city_slug,
        "deleted_way_count": removed,
        "osm_path": str(osm_path),
    }
