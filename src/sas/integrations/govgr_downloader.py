"""
govgr_downloader.py
===================

Download Thessaloniki FCD datasets from:
1) Realtime paginated HTTP feeds
2) Historical FTP archives

Outputs
-------
- raw/            : untouched downloaded pages/files
- clean/          : merged/deduped realtime tables
- baselines/      : hourly baseline targets from realtime tables
- historical/     : extracted historical archives (best effort)
- quality_report.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import ftplib
import gzip
import io
import json
import logging
import shutil
import subprocess
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

logger = logging.getLogger("sas.govgr_downloader")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    realtime_url: str | None
    historical_ftp: str | None
    timestamp_col: str
    key_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]


DATASETS: dict[str, DatasetConfig] = {
    "speed": DatasetConfig(
        name="speed",
        realtime_url="https://feed.opendata.imetb.gr/fcd/speed.csv",
        historical_ftp="ftp://62.217.125.152:2121/fcd_speed/",
        timestamp_col="Timestamp",
        key_columns=("Link_id", "Link_Direction", "Timestamp"),
        numeric_columns=("Speed", "UniqueEntries"),
    ),
    "congestion": DatasetConfig(
        name="congestion",
        realtime_url="https://feed.opendata.imetb.gr/fcd/congestions.csv",
        historical_ftp=None,
        timestamp_col="Timestamp",
        key_columns=("Link_id", "Link_Direction", "Timestamp"),
        numeric_columns=(),
    ),
    "travel_times": DatasetConfig(
        name="travel_times",
        realtime_url="https://feed.opendata.imetb.gr/fcd/traveltimes.csv",
        historical_ftp="ftp://62.217.125.152:2121/fcd_traveltimes/",
        timestamp_col="Timestamp",
        key_columns=("Path_id", "Timestamp"),
        numeric_columns=("Duration",),
    ),
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Download Thessaloniki govgr datasets from realtime feeds and/or "
            "historical FTP archives."
        )
    )
    p.add_argument(
        "--dataset",
        default="all",
        choices=["all", "speed", "congestion", "travel_times"],
        help="Dataset to download (default: all).",
    )
    p.add_argument(
        "--source",
        default="realtime",
        choices=["realtime", "historical", "both"],
        help="Data source mode (default: realtime).",
    )
    p.add_argument("--limit", type=int, default=500, help="Realtime page size (default: 500).")
    p.add_argument("--offset-start", type=int, default=0, help="Realtime start offset.")
    p.add_argument("--max-pages", type=int, default=None, help="Realtime page cap per dataset.")
    p.add_argument("--timeout", type=float, default=20.0, help="HTTP/FTP timeout (seconds).")
    p.add_argument("--retries", type=int, default=4, help="HTTP retry attempts.")
    p.add_argument("--backoff", type=float, default=1.5, help="Retry base backoff in seconds.")
    p.add_argument(
        "--historical-max-files",
        type=int,
        default=None,
        help="Optional cap on number of FTP files per dataset.",
    )
    p.add_argument(
        "--historical-pattern",
        default=None,
        help="Only download FTP files containing this substring.",
    )
    p.add_argument(
        "--no-extract-historical",
        action="store_true",
        help="Download historical archives but do not extract them.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output root dir. Default: data/cities/thessaloniki/govgr/downloads/<timestamp>/",
    )
    p.add_argument("--dry-run", action="store_true", help="Print planned actions only.")
    p.add_argument(
        "--skip-parquet",
        action="store_true",
        help="Skip parquet export for realtime clean tables.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return p


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _request_text(
    url: str,
    *,
    timeout: float,
    retries: int,
    backoff: float,
    session: requests.Session,
) -> str:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text
        except requests.RequestException as exc:
            last_err = exc
            if attempt >= retries:
                break
            wait_s = backoff * (2**attempt)
            logger.warning("Request failed (%s). Retry in %.1fs: %s", exc, wait_s, url)
            time.sleep(wait_s)
    raise RuntimeError(f"HTTP fetch failed after retries: {url}") from last_err


def _parse_csv(csv_text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(csv_text), sep=";")


def _clean_dataframe(df: pd.DataFrame, cfg: DatasetConfig) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    info: dict = {}

    for col in cfg.numeric_columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if cfg.timestamp_col in out.columns:
        out[cfg.timestamp_col] = pd.to_datetime(out[cfg.timestamp_col], errors="coerce")

    rows_before = len(out)
    dedup_keys = [c for c in cfg.key_columns if c in out.columns]
    if dedup_keys:
        out = out.drop_duplicates(subset=dedup_keys, keep="last")
    rows_after = len(out)

    info["rows_before_dedup"] = rows_before
    info["rows_after_dedup"] = rows_after
    info["dedup_removed"] = rows_before - rows_after

    if cfg.timestamp_col in out.columns:
        ts = out[cfg.timestamp_col]
        info["timestamp_min"] = str(ts.min()) if not ts.empty else None
        info["timestamp_max"] = str(ts.max()) if not ts.empty else None
        info["unique_timestamps"] = int(ts.nunique(dropna=True))

    if "Link_id" in out.columns:
        info["unique_links"] = int(out["Link_id"].nunique(dropna=True))
    if "Path_id" in out.columns:
        info["unique_paths"] = int(out["Path_id"].nunique(dropna=True))

    if "UniqueEntries" in out.columns:
        ue = pd.to_numeric(out["UniqueEntries"], errors="coerce")
        info["uniqueentries_eq1_pct"] = float((ue == 1).mean() * 100.0)
        info["uniqueentries_lt3_pct"] = float((ue < 3).mean() * 100.0)

    if "Speed" in out.columns:
        sp = pd.to_numeric(out["Speed"], errors="coerce")
        info["speed_gt120_count"] = int((sp > 120).sum())

    return out, info


def _write_baselines(df: pd.DataFrame, cfg: DatasetConfig, out_dir: Path) -> list[str]:
    created: list[str] = []
    if cfg.timestamp_col not in df.columns or df.empty:
        return created

    base = df.copy()
    base["hour"] = pd.to_datetime(base[cfg.timestamp_col], errors="coerce").dt.hour
    base = base.dropna(subset=["hour"])
    if base.empty:
        return created

    if cfg.name == "speed" and "Speed" in base.columns:
        g = base.groupby("hour")["Speed"]
        hourly = pd.DataFrame(
            {
                "rows": g.size(),
                "mean_speed": g.mean(),
                "median_speed": g.median(),
                "p10_speed": g.quantile(0.10),
                "p90_speed": g.quantile(0.90),
            }
        ).reset_index()
        out = out_dir / "baseline_speed_by_hour.csv"
        hourly.to_csv(out, index=False)
        created.append(str(out))

    if cfg.name == "congestion" and "Congestion" in base.columns:
        counts = base.groupby(["hour", "Congestion"]).size().rename("n").reset_index()
        pivot = counts.pivot(index="hour", columns="Congestion", values="n").fillna(0)
        shares = pivot.div(pivot.sum(axis=1), axis=0)
        shares.columns = [f"share_{str(c).lower()}" for c in shares.columns]
        shares = shares.reset_index()
        out = out_dir / "baseline_congestion_share_by_hour.csv"
        shares.to_csv(out, index=False)
        created.append(str(out))

    if cfg.name == "travel_times" and "Duration" in base.columns:
        g = base.groupby("hour")["Duration"]
        hourly = pd.DataFrame(
            {
                "rows": g.size(),
                "mean_duration_s": g.mean(),
                "median_duration_s": g.median(),
                "p10_duration_s": g.quantile(0.10),
                "p90_duration_s": g.quantile(0.90),
            }
        ).reset_index()
        out = out_dir / "baseline_travel_time_by_hour.csv"
        hourly.to_csv(out, index=False)
        created.append(str(out))

    return created


def _download_realtime_dataset(
    cfg: DatasetConfig,
    *,
    out_root: Path,
    limit: int,
    offset_start: int,
    max_pages: int | None,
    timeout: float,
    retries: int,
    backoff: float,
    dry_run: bool,
    skip_parquet: bool,
    session: requests.Session,
) -> dict:
    if not cfg.realtime_url:
        return {"dataset": cfg.name, "realtime": {"skipped": "No realtime URL configured."}}

    raw_dir = out_root / "raw" / cfg.name
    clean_dir = out_root / "clean"
    base_dir = out_root / "baselines"
    _ensure_dir(raw_dir)
    _ensure_dir(clean_dir)
    _ensure_dir(base_dir)

    pages: list[dict] = []
    frames: list[pd.DataFrame] = []
    page = 0
    offset = offset_start

    while True:
        url = f"{cfg.realtime_url}?offset={offset}&limit={limit}"
        logger.info("[%s][realtime] page=%d offset=%d", cfg.name, page, offset)

        if dry_run:
            pages.append({"page": page, "offset": offset, "url": url, "rows": None})
            if max_pages is not None and page + 1 >= max_pages:
                break
            page += 1
            offset += limit
            continue

        text = _request_text(
            url,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
            session=session,
        )
        page_file = raw_dir / f"page_{page:05d}_offset_{offset:08d}.csv"
        page_file.write_text(text, encoding="utf-8")
        df = _parse_csv(text)
        rows = len(df)
        frames.append(df)
        pages.append(
            {
                "page": page,
                "offset": offset,
                "url": url,
                "rows": rows,
                "raw_file": str(page_file),
            }
        )

        if rows < limit:
            break
        if max_pages is not None and page + 1 >= max_pages:
            break
        page += 1
        offset += limit

    if dry_run:
        return {"dataset": cfg.name, "realtime": {"dry_run": True, "pages_planned": pages}}

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    clean, info = _clean_dataframe(merged, cfg)

    clean_csv = clean_dir / f"{cfg.name}.csv"
    clean.to_csv(clean_csv, index=False)

    parquet_path = clean_dir / f"{cfg.name}.parquet"
    parquet_saved = False
    if not skip_parquet:
        try:
            clean.to_parquet(parquet_path, index=False)
            parquet_saved = True
        except Exception as exc:
            logger.warning("[%s] parquet export skipped: %s", cfg.name, exc)

    baselines = _write_baselines(clean, cfg, base_dir)
    out: dict = {
        "dataset": cfg.name,
        "source_url": cfg.realtime_url,
        "rows_raw_merged": int(len(merged)),
        "rows_clean": int(len(clean)),
        "pages_downloaded": len(pages),
        "limit": limit,
        "offset_start": offset_start,
        "pages": pages,
        "clean_csv": str(clean_csv),
        "clean_parquet": str(parquet_path) if parquet_saved else None,
        "baseline_files": baselines,
    }
    out.update(info)
    return {"dataset": cfg.name, "realtime": out}


def _parse_ftp_url(url: str) -> tuple[str, int, str]:
    parsed = urlparse(url)
    if parsed.scheme != "ftp" or not parsed.hostname:
        raise ValueError(f"Invalid FTP URL: {url}")
    host = parsed.hostname
    port = parsed.port or 21
    path = parsed.path or "/"
    return host, port, path


def _extract_archive(src: Path, dst_dir: Path) -> dict:
    """Extract one archive file, best effort."""
    result = {"file": str(src), "extracted": False, "method": None, "error": None}
    _ensure_dir(dst_dir)
    name_low = src.name.lower()
    try:
        if name_low.endswith(".zip"):
            with zipfile.ZipFile(src, "r") as zf:
                zf.extractall(dst_dir)
            result["extracted"] = True
            result["method"] = "zipfile"
            return result
        if name_low.endswith(".gz"):
            out_name = src.name[:-3]
            out_path = dst_dir / out_name
            with gzip.open(src, "rb") as fin, out_path.open("wb") as fout:
                shutil.copyfileobj(fin, fout)
            result["extracted"] = True
            result["method"] = "gzip"
            return result
        if name_low.endswith(".7z"):
            tool = shutil.which("7zz") or shutil.which("7z")
            if not tool:
                result["error"] = "No 7z tool found (install p7zip / 7zz)."
                return result
            cmd = [tool, "x", "-y", f"-o{dst_dir}", str(src)]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                result["error"] = proc.stderr.strip() or proc.stdout.strip()
                return result
            result["extracted"] = True
            result["method"] = Path(tool).name
            return result
        # Not a compressed archive: keep as raw only.
        result["error"] = "Not an archive type requiring extraction."
        return result
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        return result


def _ftp_is_dir(ftp: ftplib.FTP, path: str) -> bool:
    """Return True if FTP path is a directory."""
    current = ftp.pwd()
    try:
        ftp.cwd(path)
        ftp.cwd(current)
        return True
    except ftplib.all_errors:
        try:
            ftp.cwd(current)
        except ftplib.all_errors:
            pass
        return False


def _ftp_walk_files(ftp: ftplib.FTP, root: str) -> list[str]:
    """Recursively list file paths under root (directories excluded)."""
    files: list[str] = []
    stack = [root]
    seen: set[str] = set()

    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)

        try:
            entries = ftp.nlst(cur)
        except ftplib.all_errors:
            continue

        for entry in entries:
            if not entry:
                continue
            name = entry.rstrip("/").split("/")[-1]
            if name in {".", ".."}:
                continue
            if _ftp_is_dir(ftp, entry):
                stack.append(entry)
            else:
                files.append(entry)

    return sorted(set(files))


def _download_historical_dataset(
    cfg: DatasetConfig,
    *,
    out_root: Path,
    timeout: float,
    dry_run: bool,
    max_files: int | None,
    filename_pattern: str | None,
    extract: bool,
) -> dict:
    if not cfg.historical_ftp:
        return {"dataset": cfg.name, "historical": {"skipped": "No historical FTP configured."}}

    host, port, remote_path = _parse_ftp_url(cfg.historical_ftp)
    raw_dir = out_root / "raw" / "historical" / cfg.name
    ext_dir = out_root / "historical" / cfg.name
    _ensure_dir(raw_dir)
    _ensure_dir(ext_dir)

    if dry_run:
        return {
            "dataset": cfg.name,
            "historical": {
                "dry_run": True,
                "ftp_url": cfg.historical_ftp,
                "host": host,
                "port": port,
                "remote_path": remote_path,
                "max_files": max_files,
                "pattern": filename_pattern,
            },
        }

    logger.info("[%s][historical] Connecting FTP %s:%d%s", cfg.name, host, port, remote_path)
    ftp = ftplib.FTP()
    ftp.connect(host=host, port=port, timeout=timeout)
    ftp.login()
    files = _ftp_walk_files(ftp, remote_path)
    if filename_pattern:
        files = [f for f in files if filename_pattern in f]
    if max_files is not None:
        files = files[:max_files]

    downloaded: list[dict] = []
    extracted: list[dict] = []
    for remote_file in files:
        rel = remote_file
        if rel.startswith(remote_path):
            rel = rel[len(remote_path) :]
        rel = rel.lstrip("/")
        local = raw_dir / rel
        _ensure_dir(local.parent)

        logger.info("[%s][historical] downloading %s", cfg.name, remote_file)
        with local.open("wb") as fout:
            ftp.retrbinary(f"RETR {remote_file}", fout.write)
        size = local.stat().st_size
        downloaded.append(
            {"remote_file": remote_file, "local_file": str(local), "size_bytes": int(size)}
        )

        if extract:
            extracted.append(_extract_archive(local, ext_dir))

    ftp.quit()
    return {
        "dataset": cfg.name,
        "historical": {
            "ftp_url": cfg.historical_ftp,
            "remote_path": remote_path,
            "files_downloaded": len(downloaded),
            "files": downloaded,
            "extraction": extracted if extract else [],
            "extracted_dir": str(ext_dir),
        },
    }


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    )

    started = dt.datetime.now(dt.timezone.utc)
    out_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path("data") / "cities" / "thessaloniki" / "govgr" / "downloads" / started.strftime("%Y-%m-%d_%H-%M-%S")
    )
    _ensure_dir(out_root)

    selected = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    run: dict = {
        "started_utc": started.isoformat(),
        "args": vars(args),
        "datasets": {},
        "dataset_configs": {k: asdict(v) for k, v in DATASETS.items()},
    }

    with requests.Session() as session:
        for name in selected:
            cfg = DATASETS[name]
            ds: dict = {"dataset": name}

            if args.source in {"realtime", "both"}:
                rt = _download_realtime_dataset(
                    cfg,
                    out_root=out_root,
                    limit=args.limit,
                    offset_start=args.offset_start,
                    max_pages=args.max_pages,
                    timeout=args.timeout,
                    retries=args.retries,
                    backoff=args.backoff,
                    dry_run=args.dry_run,
                    skip_parquet=args.skip_parquet,
                    session=session,
                )
                ds.update(rt)

            if args.source in {"historical", "both"}:
                hist = _download_historical_dataset(
                    cfg,
                    out_root=out_root,
                    timeout=args.timeout,
                    dry_run=args.dry_run,
                    max_files=args.historical_max_files,
                    filename_pattern=args.historical_pattern,
                    extract=not args.no_extract_historical,
                )
                ds.update(hist)

            run["datasets"][name] = ds

    run["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    q_path = out_root / "quality_report.json"
    q_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    logger.info("Quality report → %s", q_path)


if __name__ == "__main__":
    main()
