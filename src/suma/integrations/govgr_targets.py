"""
govgr_targets.py
================

Build calibration/validation target tables from Thessaloniki historical FCD data.

Default split:
- calibration = 2025
- validation  = 2026
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SPEED_COLS = ["Link_id", "Link_Direction", "Timestamp", "Speed", "UniqueEntries"]
TRAVEL_COLS = ["Path_id", "Timestamp", "Duration"]
TS_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def _quantile_from_hist(hist: np.ndarray, q: float) -> int:
    total = int(hist.sum())
    if total <= 0:
        return -1
    target = int(np.ceil(total * q))
    csum = np.cumsum(hist)
    idx = int(np.searchsorted(csum, target, side="left"))
    return idx


@dataclass
class HistStats:
    max_bin: int

    def __post_init__(self) -> None:
        self.count = np.zeros(24, dtype=np.int64)
        self.sum = np.zeros(24, dtype=np.float64)
        self.hist = np.zeros((24, self.max_bin + 1), dtype=np.int64)

        # 0=weekday, 1=weekend
        self.w_count = np.zeros((2, 24), dtype=np.int64)
        self.w_sum = np.zeros((2, 24), dtype=np.float64)
        self.w_hist = np.zeros((2, 24, self.max_bin + 1), dtype=np.int64)

    def update(self, hour: np.ndarray, weekday: np.ndarray, value: np.ndarray) -> None:
        valid = (
            ~pd.isna(hour)
            & ~pd.isna(weekday)
            & ~pd.isna(value)
            & (value >= 0)
            & (hour >= 0)
            & (hour <= 23)
        )
        if not np.any(valid):
            return

        h = hour[valid].astype(np.int16)
        wd = weekday[valid].astype(bool)
        # clip for bounded histogram aggregation
        v = np.clip(value[valid].astype(np.int64), 0, self.max_bin)

        for hour_val in np.unique(h):
            m = h == hour_val
            vals = v[m]
            if vals.size == 0:
                continue
            self.count[hour_val] += vals.size
            self.sum[hour_val] += float(vals.sum())
            self.hist[hour_val] += np.bincount(vals, minlength=self.max_bin + 1)

        is_weekend = ~wd
        for wp_idx, mask_wp in enumerate([wd, is_weekend]):
            hh = h[mask_wp]
            vv = v[mask_wp]
            if vv.size == 0:
                continue
            for hour_val in np.unique(hh):
                m = hh == hour_val
                vals = vv[m]
                if vals.size == 0:
                    continue
                self.w_count[wp_idx, hour_val] += vals.size
                self.w_sum[wp_idx, hour_val] += float(vals.sum())
                self.w_hist[wp_idx, hour_val] += np.bincount(vals, minlength=self.max_bin + 1)

    def to_hourly_df(self, value_name: str) -> pd.DataFrame:
        rows = []
        for h in range(24):
            n = int(self.count[h])
            if n == 0:
                continue
            rows.append(
                {
                    "hour": h,
                    "rows": n,
                    f"mean_{value_name}": float(self.sum[h] / n),
                    f"p10_{value_name}": _quantile_from_hist(self.hist[h], 0.10),
                    f"p50_{value_name}": _quantile_from_hist(self.hist[h], 0.50),
                    f"p90_{value_name}": _quantile_from_hist(self.hist[h], 0.90),
                }
            )
        return pd.DataFrame(rows)

    def to_weekpart_hourly_df(self, value_name: str) -> pd.DataFrame:
        rows = []
        names = ["weekday", "weekend"]
        for wp_idx, wp in enumerate(names):
            for h in range(24):
                n = int(self.w_count[wp_idx, h])
                if n == 0:
                    continue
                rows.append(
                    {
                        "weekpart": wp,
                        "hour": h,
                        "rows": n,
                        f"mean_{value_name}": float(self.w_sum[wp_idx, h] / n),
                        f"p10_{value_name}": _quantile_from_hist(self.w_hist[wp_idx, h], 0.10),
                        f"p50_{value_name}": _quantile_from_hist(self.w_hist[wp_idx, h], 0.50),
                        f"p90_{value_name}": _quantile_from_hist(self.w_hist[wp_idx, h], 0.90),
                    }
                )
        return pd.DataFrame(rows)


def _list_files(base: Path, dataset: str, year: int) -> list[Path]:
    if dataset == "speed":
        pattern = f"fcd_speed_*_{year}.txt"
        root = base / "speed"
    else:
        pattern = f"fcd_traveltimes_*_{year}.txt"
        root = base / "travel_times"
    return sorted(root.glob(pattern))


def _process_speed_year(
    files: list[Path], chunksize: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    stats = HistStats(max_bin=200)
    grouped_parts: list[pd.DataFrame] = []
    row_total = 0

    for fp in files:
        for chunk in pd.read_csv(
            fp,
            sep="\t",
            names=SPEED_COLS,
            header=None,
            dtype={"Link_id": "string", "Link_Direction": "string"},
            chunksize=chunksize,
            low_memory=False,
        ):
            row_total += len(chunk)
            chunk["Timestamp"] = pd.to_datetime(
                chunk["Timestamp"], format=TS_FORMAT, errors="coerce"
            )
            chunk["hour"] = chunk["Timestamp"].dt.hour
            chunk["weekday"] = chunk["Timestamp"].dt.weekday < 5
            chunk["Speed"] = pd.to_numeric(chunk["Speed"], errors="coerce")
            chunk["UniqueEntries"] = pd.to_numeric(chunk["UniqueEntries"], errors="coerce")

            stats.update(
                hour=chunk["hour"].to_numpy(),
                weekday=chunk["weekday"].to_numpy(),
                value=chunk["Speed"].to_numpy(),
            )

            g = (
                chunk.dropna(subset=["hour", "Speed"])
                .groupby(["Link_id", "Link_Direction", "hour"], as_index=False)
                .agg(
                    rows=("Speed", "size"),
                    sum_speed=("Speed", "sum"),
                    sum_unique_entries=("UniqueEntries", "sum"),
                )
            )
            grouped_parts.append(g)

    if grouped_parts:
        link_hour = pd.concat(grouped_parts, ignore_index=True)
        link_hour = link_hour.groupby(["Link_id", "Link_Direction", "hour"], as_index=False).agg(
            rows=("rows", "sum"),
            sum_speed=("sum_speed", "sum"),
            sum_unique_entries=("sum_unique_entries", "sum"),
        )
        link_hour["mean_speed"] = link_hour["sum_speed"] / link_hour["rows"]
        link_hour["mean_unique_entries"] = link_hour["sum_unique_entries"] / link_hour["rows"]
        link_hour = link_hour[
            ["Link_id", "Link_Direction", "hour", "rows", "mean_speed", "mean_unique_entries"]
        ]
    else:
        link_hour = pd.DataFrame(
            columns=[
                "Link_id",
                "Link_Direction",
                "hour",
                "rows",
                "mean_speed",
                "mean_unique_entries",
            ]
        )

    hourly = stats.to_hourly_df("speed")
    weekpart = stats.to_weekpart_hourly_df("speed")
    summary = {"files": len(files), "rows_read": row_total, "link_hour_rows": int(len(link_hour))}
    return hourly, weekpart, link_hour, summary


def _process_travel_year(
    files: list[Path], chunksize: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    stats = HistStats(max_bin=20000)
    grouped_parts: list[pd.DataFrame] = []
    row_total = 0

    for fp in files:
        for chunk in pd.read_csv(
            fp,
            sep="\t",
            names=TRAVEL_COLS,
            header=None,
            dtype={"Path_id": "string"},
            chunksize=chunksize,
            low_memory=False,
        ):
            row_total += len(chunk)
            chunk["Timestamp"] = pd.to_datetime(
                chunk["Timestamp"], format=TS_FORMAT, errors="coerce"
            )
            chunk["hour"] = chunk["Timestamp"].dt.hour
            chunk["weekday"] = chunk["Timestamp"].dt.weekday < 5
            chunk["Duration"] = pd.to_numeric(chunk["Duration"], errors="coerce")

            stats.update(
                hour=chunk["hour"].to_numpy(),
                weekday=chunk["weekday"].to_numpy(),
                value=chunk["Duration"].to_numpy(),
            )

            g = (
                chunk.dropna(subset=["hour", "Duration"])
                .groupby(["Path_id", "hour"], as_index=False)
                .agg(rows=("Duration", "size"), sum_duration=("Duration", "sum"))
            )
            grouped_parts.append(g)

    if grouped_parts:
        path_hour = pd.concat(grouped_parts, ignore_index=True)
        path_hour = path_hour.groupby(["Path_id", "hour"], as_index=False).agg(
            rows=("rows", "sum"), sum_duration=("sum_duration", "sum")
        )
        path_hour["mean_duration_s"] = path_hour["sum_duration"] / path_hour["rows"]
        path_hour = path_hour[["Path_id", "hour", "rows", "mean_duration_s"]]
    else:
        path_hour = pd.DataFrame(columns=["Path_id", "hour", "rows", "mean_duration_s"])

    hourly = stats.to_hourly_df("duration_s")
    weekpart = stats.to_weekpart_hourly_df("duration_s")
    summary = {"files": len(files), "rows_read": row_total, "path_hour_rows": int(len(path_hour))}
    return hourly, weekpart, path_hour, summary


def _add_set_label(df: pd.DataFrame, set_name: str, year: int) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "set", set_name)
    out.insert(1, "year", year)
    return out


def build_targets(
    base_downloads: Path,
    calibration_year: int,
    validation_year: int,
    out_dir: Path,
    chunksize: int,
) -> dict:
    _ = out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "calibration_year": calibration_year,
        "validation_year": validation_year,
        "outputs": {},
    }

    for set_name, year in [("calibration", calibration_year), ("validation", validation_year)]:
        hist_root = base_downloads / f"historical_{year}" / "historical"

        speed_files = _list_files(hist_root, "speed", year)
        tt_files = _list_files(hist_root, "travel_times", year)

        sp_hour, sp_week, sp_link, sp_meta = _process_speed_year(speed_files, chunksize)
        tt_hour, tt_week, tt_path, tt_meta = _process_travel_year(tt_files, chunksize)

        sp_hour = _add_set_label(sp_hour, set_name, year)
        sp_week = _add_set_label(sp_week, set_name, year)
        sp_link = _add_set_label(sp_link, set_name, year)
        tt_hour = _add_set_label(tt_hour, set_name, year)
        tt_week = _add_set_label(tt_week, set_name, year)
        tt_path = _add_set_label(tt_path, set_name, year)

        p1 = out_dir / f"{set_name}_speed_network_hourly.csv"
        p2 = out_dir / f"{set_name}_speed_network_weekpart_hourly.csv"
        p3 = out_dir / f"{set_name}_speed_link_direction_hourly_mean.csv"
        p4 = out_dir / f"{set_name}_travel_time_network_hourly.csv"
        p5 = out_dir / f"{set_name}_travel_time_network_weekpart_hourly.csv"
        p6 = out_dir / f"{set_name}_travel_time_path_hourly_mean.csv"

        sp_hour.to_csv(p1, index=False)
        sp_week.to_csv(p2, index=False)
        sp_link.to_csv(p3, index=False)
        tt_hour.to_csv(p4, index=False)
        tt_week.to_csv(p5, index=False)
        tt_path.to_csv(p6, index=False)

        summary["outputs"][set_name] = {
            "speed_meta": sp_meta,
            "travel_time_meta": tt_meta,
            "files": [str(p1), str(p2), str(p3), str(p4), str(p5), str(p6)],
        }

    summary_path = out_dir / "targets_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build post-metro calibration/validation target tables."
    )
    p.add_argument(
        "--downloads-root",
        default="data/cities/thessaloniki/govgr/downloads",
        help="Root where historical_YYYY folders exist.",
    )
    p.add_argument("--calibration-year", type=int, default=2025)
    p.add_argument("--validation-year", type=int, default=2026)
    p.add_argument(
        "--output-dir",
        default="data/cities/thessaloniki/govgr/targets/post_metro_2025_2026",
        help="Output target folder.",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="Rows per chunk when reading large historical files.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    summary = build_targets(
        base_downloads=Path(args.downloads_root),
        calibration_year=args.calibration_year,
        validation_year=args.validation_year,
        out_dir=Path(args.output_dir),
        chunksize=args.chunksize,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
