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
  sas-fetch-osm \
      --place "Seattle, Washington, USA" \
      --out data/cities/seattle/bundle/traffic_dataset/02_Seattle/01_Input_data/Seattle.osm

  sas-fetch-osm \
      --place "Athens, Greece" \
      --out data/cities/athens/network/athens.osm \
      --pad-km 1.5
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
DEFAULT_USER_AGENT = "antifragicity-suma-osm-downloader/0.3"
DEFAULT_CONFIG_TEMPLATE = PROJECT_ROOT / "configs" / "templates" / "city_default.yaml"
DEFAULT_ROAD_TYPES = (
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "road",
)


def _slugify(text: str) -> str:
    """Convert a free-form place string into a safe default filename stem."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "place"


def _default_city_slug(place: str) -> str:
    """Use the first place token as the default city folder slug."""
    return _slugify(place.split(",")[0])


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
    road_types: list[str] | tuple[str, ...] | None = None,
    include_all_features: bool = False,
) -> str:
    """Build the Overpass query used to download the OSM XML extract."""
    bbox = f"({south:.6f},{west:.6f},{north:.6f},{east:.6f})"
    if not include_all_features:
        selected_types = [item for item in (road_types or DEFAULT_ROAD_TYPES) if item]
        if not selected_types:
            selected_types = list(DEFAULT_ROAD_TYPES)
        joined = "|".join(re.escape(item) for item in selected_types)
        body = f"""
(
  way["highway"~"^({joined})$"]{bbox};
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


def _default_output_path(place: str, city_slug: str | None = None) -> Path:
    """Place OSM extracts under the standard city/network layout."""
    slug = city_slug or _default_city_slug(place)
    return Path("data") / "cities" / slug / "network" / f"{slug}.osm"


def _default_config_path(city_slug: str) -> Path:
    return Path("configs") / city_slug / "default.yaml"


def _city_root(city_slug: str) -> Path:
    return Path("data") / "cities" / city_slug


def _render_config_template(template_text: str, city_slug: str, place: str) -> str:
    return (
        template_text.replace("__CITY_SLUG__", city_slug)
        .replace("__CITY_TITLE__", place)
    )


def _write_city_metadata(city_slug: str, place: str, osm_out: Path, config_out: Path) -> None:
    payload = {
        "slug": city_slug,
        "display_name": place,
        "osm_extract": osm_out.as_posix(),
        "config_path": config_out.as_posix(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = _city_root(city_slug) / "city_metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_template_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute() or expanded.exists():
        return expanded
    return (PROJECT_ROOT / expanded).resolve()


def _bootstrap_city_layout(
    *,
    city_slug: str,
    place: str,
    out_path: Path,
    config_out: Path,
    config_template: Path,
    bootstrap_config: bool,
) -> None:
    """Create the standard city folder structure and a default config if missing."""
    network_dir = _city_root(city_slug) / "network"
    govgr_root = _city_root(city_slug) / "govgr"
    network_dir.mkdir(parents=True, exist_ok=True)
    (govgr_root / "downloads").mkdir(parents=True, exist_ok=True)
    (govgr_root / "targets").mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if bootstrap_config:
        config_out.parent.mkdir(parents=True, exist_ok=True)
        if not config_out.exists():
            template_text = config_template.read_text(encoding="utf-8")
            rendered = _render_config_template(template_text, city_slug, place)
            config_out.write_text(rendered, encoding="utf-8")

    _write_city_metadata(city_slug, place, out_path, config_out)


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
        help="Output .osm path. Default: data/cities/<city>/network/<city>.osm",
    )
    parser.add_argument(
        "--city-slug",
        default=None,
        help="Folder/config slug to use under data/cities/ and configs/.",
    )
    parser.add_argument(
        "--pad-km",
        type=float,
        default=0.0,
        help="Expand the resolved place bounding box by this many km on each side.",
    )
    parser.add_argument("--south", type=float, default=None, help="Optional south latitude override.")
    parser.add_argument("--west", type=float, default=None, help="Optional west longitude override.")
    parser.add_argument("--north", type=float, default=None, help="Optional north latitude override.")
    parser.add_argument("--east", type=float, default=None, help="Optional east longitude override.")
    parser.add_argument(
        "--road-types",
        nargs="+",
        default=list(DEFAULT_ROAD_TYPES),
        help="OSM highway tag values to include when downloading roads for SUMO.",
    )
    parser.add_argument(
        "--all-features",
        action="store_true",
        help="Download all OSM nodes/ways/relations in the bbox instead of filtering by road type.",
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
    parser.add_argument(
        "--bootstrap-layout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create the standard data/cities/<city>/network layout before download.",
    )
    parser.add_argument(
        "--bootstrap-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create configs/<city>/default.yaml from the city template when missing.",
    )
    parser.add_argument(
        "--config-out",
        default=None,
        help="Override the generated config path. Default: configs/<city>/default.yaml",
    )
    parser.add_argument(
        "--config-template",
        default=DEFAULT_CONFIG_TEMPLATE.relative_to(PROJECT_ROOT).as_posix(),
        help="Config template used for new city bootstrap.",
    )
    args = parser.parse_args()

    city_slug = args.city_slug or _default_city_slug(args.place)
    out_path = Path(args.out) if args.out else _default_output_path(args.place, city_slug)
    config_out = Path(args.config_out) if args.config_out else _default_config_path(city_slug)
    config_template = _resolve_template_path(Path(args.config_template))

    try:
        if args.bootstrap_layout:
            _bootstrap_city_layout(
                city_slug=city_slug,
                place=args.place,
                out_path=out_path,
                config_out=config_out,
                config_template=config_template,
                bootstrap_config=args.bootstrap_config,
            )
            print(f"City scaffold ready: data/cities/{city_slug} and {config_out}")

        if None not in {args.south, args.west, args.north, args.east}:
            display_name = args.place
            south, west, north, east = (
                float(args.south),
                float(args.west),
                float(args.north),
                float(args.east),
            )
            print(f"Using explicit bounding box overrides for: {display_name}")
        else:
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

        if None in {args.south, args.west, args.north, args.east}:
            # Stay under the public Nominatim usage limit of 1 request/second.
            time.sleep(1.0)

        query = _build_overpass_query(
            south=south,
            west=west,
            north=north,
            east=east,
            road_types=args.road_types,
            include_all_features=args.all_features,
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
        _write_city_metadata(city_slug, display_name, out_path, config_out)
        print("Attribution: OpenStreetMap contributors (ODbL)")
    except Exception as exc:
        print(f"❌  {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
