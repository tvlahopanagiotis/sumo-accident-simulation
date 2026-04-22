"""
sumo_paths.py
=============

Helpers for locating SUMO resources across common installation layouts.
"""

from __future__ import annotations

import os


def resolve_sumo_home(raw_sumo_home: str | None = None) -> str:
    """
    Resolve SUMO_HOME to the actual SUMO share directory when possible.
    """
    candidate = (raw_sumo_home or os.environ.get("SUMO_HOME") or "").strip()
    if candidate:
        normalized = os.path.normpath(candidate)
        if normalized.endswith(os.path.join("share", "sumo")):
            return candidate
        share_candidate = os.path.join(candidate, "share", "sumo")
        if os.path.isdir(share_candidate):
            return share_candidate
        if os.path.isdir(candidate):
            return candidate

    common_candidates = [
        "/opt/homebrew/share/sumo",
        "/usr/local/share/sumo",
        "/usr/share/sumo",
        "/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/share/sumo",
        "/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/share/sumo",
    ]
    for path in common_candidates:
        if os.path.isdir(path):
            return path

    return candidate


def find_typemap_path(sumo_home: str) -> str:
    """Return the expected path to SUMO's OSM typemap file."""
    return os.path.join(sumo_home, "data", "typemap", "osmNetconvert.typ.xml")


def find_random_trips_path(sumo_home: str) -> str:
    """Return the expected path to SUMO's randomTrips.py helper."""
    return os.path.join(sumo_home, "tools", "randomTrips.py")
