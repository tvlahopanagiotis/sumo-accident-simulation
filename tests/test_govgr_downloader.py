from __future__ import annotations

import datetime as dt

import pandas as pd

from sas.integrations.govgr_downloader import DATASETS, _clean_dataframe, _extract_archive, _parse_ftp_url, _resolve_output_root


def test_parse_ftp_url() -> None:
    host, port, path = _parse_ftp_url("ftp://62.217.125.152:2121/fcd_speed/")
    assert host == "62.217.125.152"
    assert port == 2121
    assert path == "/fcd_speed/"


def test_clean_dataframe_speed_dedup() -> None:
    cfg = DATASETS["speed"]
    df = pd.DataFrame(
        [
            {
                "Link_id": "1",
                "Link_Direction": "1",
                "Timestamp": "2026-01-01 12:00:00.000",
                "Speed": "15",
                "UniqueEntries": "1",
            },
            {
                "Link_id": "1",
                "Link_Direction": "1",
                "Timestamp": "2026-01-01 12:00:00.000",
                "Speed": "20",
                "UniqueEntries": "2",
            },
        ]
    )
    clean, info = _clean_dataframe(df, cfg)
    assert len(clean) == 1
    assert info["dedup_removed"] == 1
    assert clean["Speed"].iloc[0] == 20


def test_extract_archive_non_archive(tmp_path) -> None:
    p = tmp_path / "file.txt"
    p.write_text("x", encoding="utf-8")
    out = _extract_archive(p, tmp_path / "out")
    assert out["extracted"] is False
    assert out["error"] == "Not an archive type requiring extraction."


def test_resolve_output_root_uses_timestamp_under_downloads_root() -> None:
    started = dt.datetime(2026, 4, 29, 12, 30, 0, tzinfo=dt.timezone.utc)
    path = _resolve_output_root("data/cities/thermi/govgr/downloads", started)
    assert path.as_posix() == "data/cities/thermi/govgr/downloads/2026-04-29_12-30-00"


def test_resolve_output_root_preserves_explicit_run_directory() -> None:
    started = dt.datetime(2026, 4, 29, 12, 30, 0, tzinfo=dt.timezone.utc)
    path = _resolve_output_root("data/cities/thermi/govgr/downloads/manual_run", started)
    assert path.as_posix() == "data/cities/thermi/govgr/downloads/manual_run"
