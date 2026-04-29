from __future__ import annotations

from pathlib import Path
from typing import Any

from ..app.config import PROJECT_ROOT
from ..generators import generate_city as generate_city_module

_CITIES_ROOT = (PROJECT_ROOT / "data" / "cities").resolve()


def _relative(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_city_demand_preview(city_slug: str, row_limit: int = 50, flow_limit: int = 100) -> dict[str, Any]:
    city_root = _CITIES_ROOT / city_slug
    if not city_root.exists():
        raise FileNotFoundError(f"Unknown city slug: {city_slug}")

    od_file = generate_city_module._resolve_support_csv(city_slug, None, "od")
    node_file = generate_city_module._resolve_support_csv(city_slug, None, "node")

    preview: dict[str, Any] = {
        "city_slug": city_slug,
        "od_file": _relative(od_file),
        "node_file": _relative(node_file),
        "supported": False,
        "issues": [],
        "summary": None,
        "sample_rows": [],
        "top_flows": [],
        "nodes": [],
    }

    if od_file is None:
        preview["issues"].append("No OD CSV was found under the city folder.")
        return preview
    if node_file is None:
        preview["issues"].append("No node centroid CSV was found under the city folder.")
        return preview

    try:
        od_rows = generate_city_module._read_od_rows(od_file)
    except Exception as exc:
        preview["issues"].append(f"Failed to read OD file: {exc}")
        return preview

    try:
        centroids = generate_city_module._read_zone_centroids(node_file)
    except Exception as exc:
        preview["issues"].append(f"Failed to read node file: {exc}")
        return preview

    total_od = 0.0
    intrazonal_raw = 0.0
    missing_zone_ids: set[str] = set()
    sample_rows: list[dict[str, Any]] = []
    external_rows: list[tuple[str, str, float]] = []

    for index, (origin, destination, od_number) in enumerate(od_rows):
        total_od += od_number
        if origin == destination:
            intrazonal_raw += od_number
        else:
            external_rows.append((origin, destination, od_number))
        if origin not in centroids:
            missing_zone_ids.add(origin)
        if destination not in centroids:
            missing_zone_ids.add(destination)
        if index < row_limit:
            sample_rows.append(
                {
                    "origin": origin,
                    "destination": destination,
                    "od_number": od_number,
                    "intrazonal": origin == destination,
                }
            )

    top_flows: list[dict[str, Any]] = []
    for origin, destination, od_number in sorted(external_rows, key=lambda item: item[2], reverse=True):
        origin_coords = centroids.get(origin)
        destination_coords = centroids.get(destination)
        if origin_coords is None or destination_coords is None:
            continue
        top_flows.append(
            {
                "origin": origin,
                "destination": destination,
                "od_number": od_number,
                "origin_coords": [origin_coords[1], origin_coords[0]],
                "destination_coords": [destination_coords[1], destination_coords[0]],
            }
        )
        if len(top_flows) >= flow_limit:
            break

    node_rows = [
        {
            "zone_id": zone_id,
            "coords": [lat, lon],
        }
        for zone_id, (lon, lat) in sorted(centroids.items())
    ]

    preview["supported"] = True
    preview["summary"] = {
        "zone_count": len(centroids),
        "od_row_count": len(od_rows),
        "external_od_row_count": len(external_rows),
        "total_od": total_od,
        "intrazonal_raw": intrazonal_raw,
        "missing_zone_count": len(missing_zone_ids),
        "mapped_top_flow_count": len(top_flows),
    }
    preview["sample_rows"] = sample_rows
    preview["top_flows"] = top_flows
    preview["nodes"] = node_rows
    if missing_zone_ids:
        sample = ", ".join(sorted(missing_zone_ids)[:8])
        preview["issues"].append(
            f"{len(missing_zone_ids)} OD zone ids are missing centroid coordinates (sample: {sample})."
        )
    return preview
