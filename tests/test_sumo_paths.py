from __future__ import annotations

from sas.simulation.sumo_paths import find_random_trips_path, find_typemap_path, resolve_sumo_home


def test_resolve_sumo_home_keeps_share_path() -> None:
    path = "/opt/homebrew/share/sumo"
    assert resolve_sumo_home(path) == path


def test_resolve_sumo_home_expands_framework_root() -> None:
    path = "/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO"
    expected = "/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/share/sumo"
    assert resolve_sumo_home(path) == expected


def test_find_typemap_path() -> None:
    assert find_typemap_path("/tmp/sumo") == "/tmp/sumo/data/typemap/osmNetconvert.typ.xml"


def test_find_random_trips_path() -> None:
    assert find_random_trips_path("/tmp/sumo") == "/tmp/sumo/tools/randomTrips.py"
