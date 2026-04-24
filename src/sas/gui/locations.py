from __future__ import annotations

from typing import Any

import requests

from ..integrations.download_osm_place import DEFAULT_NOMINATIM_URL

DEFAULT_GUI_USER_AGENT = "antifragicity-sas-gui/0.2"


def search_locations(query: str, *, limit: int = 6, nominatim_url: str = DEFAULT_NOMINATIM_URL) -> list[dict[str, Any]]:
    headers = {"User-Agent": DEFAULT_GUI_USER_AGENT}
    params = {
        "q": query,
        "format": "jsonv2",
        "addressdetails": 1,
        "polygon_geojson": 1,
        "limit": max(1, min(limit, 10)),
    }
    response = requests.get(nominatim_url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    results = response.json()

    normalized: list[dict[str, Any]] = []
    for item in results:
        bbox = item.get("boundingbox") or []
        if len(bbox) != 4:
            continue
        address = item.get("address", {}) if isinstance(item.get("address"), dict) else {}
        normalized.append(
            {
                "display_name": item.get("display_name"),
                "lat": float(item.get("lat")),
                "lon": float(item.get("lon")),
                "boundingbox": [float(value) for value in bbox],
                "country_code": str(address.get("country_code", "")).lower(),
                "country": address.get("country"),
                "city": address.get("city") or address.get("town") or address.get("municipality") or address.get("county"),
                "state": address.get("state"),
                "osm_type": item.get("osm_type"),
                "osm_id": item.get("osm_id"),
                "class": item.get("class"),
                "type": item.get("type"),
                "geojson": item.get("geojson"),
            }
        )
    return normalized
