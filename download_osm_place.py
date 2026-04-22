"""
download_osm_place.py
=====================

Download a raw OSM XML extract for a named place using:
  1. Nominatim for place lookup / bounding box resolution
  2. Overpass API for OSM XML extraction

This is intended for SUMO network generation workflows where you need an
`.osm` file for a city but only know the human-readable place name.

Examples
--------
  python download_osm_place.py \
      --place "Seattle, Washington, USA" \
      --out seattle_bundle/traffic_dataset/02_Seattle/01_Input_data/Seattle.osm

  python download_osm_place.py \
      --place "Athens, Greece" \
      --out athens_network/athens.osm \
      --pad-km 1.5
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

DEFAULT_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
DEFAULT_USER_AGENT = "antifragicity-sas-osm-downloader/0.1"


def _slugify(text: str) -> str:
    """Convert a free-form place string into a safe default filename stem."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "place"


def _expand_bbox(
    south: float,
    west: float,
    north: float,
    east: float,
    pad_km: float,
) -> tuple[float, float, float, float]:
    """Expand a lat/lon bounding box by approximately `pad_km` on each side."""
    if pad_km <= 0:
        return south, west, north, east

    lat_pad = pad_km / 111.0
    mid_lat = (south + north) / 2.0
    cos_lat = max(0.2, abs(__import__("math").cos(__import__("math").radians(mid_lat))))
    lon_pad = pad_km / (111.0 * cos_lat)

    return south - lat_pad, west - lon_pad, north + lat_pad, east + lon_pad


def _build_overpass_query(
    south: float,
    west: float,
    north: float,
    east: float,
    highways_only: bool,
) -> str:
    """Build the Overpass query used to download the OSM XML extract."""
    bbox = f"({south:.6f},{west:.6f},{north:.6f},{east:.6f})"
    if highways_only:
        body = f"""
(
  way["highway"]{bbox};
  relation["type"="restriction"]{bbox};
);
(._;>;);
"""
    else:
        body = f"""
(
  node{bbox};
  way{bbox};
  relation{bbox};
);
(._;>;);
"""

    return f"""
[out:xml][timeout:240];
{body}
out body;
""".strip()


def _resolve_place(
    place: str,
    nominatim_url: str,
    user_agent: str,
    email: str | None,
) -> dict[str, Any]:
    """Resolve a place name to a Nominatim result with a bounding box."""
    headers = {"User-Agent": user_agent}
    params = {
        "q": place,
        "format": "jsonv2",
        "limit": 1,
    }
    if email:
        params["email"] = email

    response = requests.get(nominatim_url, params=params, headers=headers, timeout=60)
    response.raise_for_status()
    results = response.json()
    if not results:
        raise RuntimeError(f"Place not found via Nominatim: {place}")

    result = results[0]
    bbox = result.get("boundingbox")
    if not bbox or len(bbox) != 4:
        raise RuntimeError(f"Nominatim result for '{place}' did not include a bounding box.")
    return result


def _download_osm(
    query: str,
    out_path: Path,
    overpass_url: str,
    user_agent: str,
) -> None:
    """Execute an Overpass query and write the raw XML response."""
    headers = {"User-Agent": user_agent}
    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            response = requests.post(
                overpass_url,
                data={"data": query},
                headers=headers,
                timeout=300,
            )
            response.raise_for_status()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(response.content)
            return
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 3:
                break
            print(f"    Overpass attempt {attempt} failed: {exc}. Retrying in 10 s ...")
            time.sleep(10)

    raise RuntimeError(f"Overpass download failed after 3 attempts: {last_error}")


def _default_output_path(place: str) -> Path:
    """Pick a simple default output filename from the place name."""
    return Path(f"{_slugify(place)}.osm")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download an OSM XML extract for a named place using Nominatim + Overpass."
    )
    parser.add_argument(
        "--place",
        required=True,
        help='Free-form place query, e.g. "Seattle, Washington, USA".',
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output .osm path. Default: ./<slugified-place>.osm",
    )
    parser.add_argument(
        "--pad-km",
        type=float,
        default=0.0,
        help="Expand the resolved place bounding box by this many km on each side.",
    )
    parser.add_argument(
        "--all-features",
        action="store_true",
        help="Download all OSM nodes/ways/relations in the bbox instead of roads-only.",
    )
    parser.add_argument(
        "--nominatim-url",
        default=DEFAULT_NOMINATIM_URL,
        help="Nominatim search endpoint.",
    )
    parser.add_argument(
        "--overpass-url",
        default=DEFAULT_OVERPASS_URL,
        help="Overpass interpreter endpoint.",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="HTTP User-Agent for Nominatim/Overpass requests.",
    )
    parser.add_argument(
        "--email",
        default=None,
        help="Optional contact email for Nominatim requests.",
    )
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else _default_output_path(args.place)

    try:
        print(f"Resolving place via Nominatim: {args.place}")
        place_info = _resolve_place(
            place=args.place,
            nominatim_url=args.nominatim_url,
            user_agent=args.user_agent,
            email=args.email,
        )

        display_name = str(place_info.get("display_name", args.place))
        south, north, west, east = (
            float(place_info["boundingbox"][0]),
            float(place_info["boundingbox"][1]),
            float(place_info["boundingbox"][2]),
            float(place_info["boundingbox"][3]),
        )
        south, west, north, east = _expand_bbox(south, west, north, east, args.pad_km)

        print(f"  Matched: {display_name}")
        print(f"  Bounding box: south={south:.6f}, west={west:.6f}, north={north:.6f}, east={east:.6f}")
        if args.pad_km > 0:
            print(f"  Padding applied: {args.pad_km:.2f} km")

        # Stay under the public Nominatim usage limit of 1 request/second.
        time.sleep(1.0)

        query = _build_overpass_query(
            south=south,
            west=west,
            north=north,
            east=east,
            highways_only=not args.all_features,
        )
        print(f"Downloading OSM XML via Overpass -> {out_path}")
        _download_osm(
            query=query,
            out_path=out_path,
            overpass_url=args.overpass_url,
            user_agent=args.user_agent,
        )

        size_mb = out_path.stat().st_size / 1_048_576
        print(f"Saved {size_mb:.1f} MB -> {out_path}")
        print("Attribution: OpenStreetMap contributors (ODbL)")
    except Exception as exc:
        print(f"❌  {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
