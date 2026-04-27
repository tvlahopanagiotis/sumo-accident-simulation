# Command: `sas-fetch-govgr`

Download govgr traffic feeds used for Thessaloniki calibration and validation
workflows.

## What It Does

1. selects realtime or historical govgr datasets
2. downloads the requested feed files
3. stores them under a local downloads directory

## Main Inputs

- `--source` such as `realtime` or `historical`
- `--dataset` such as `speed` or `travel_times`
- optional historical selectors and output directory

## Main Outputs

- downloaded govgr files under `data/cities/thessaloniki/govgr/downloads/`
  or a custom target directory

## Typical Usage

Realtime pull:

```bash
sas-fetch-govgr \
  --source realtime \
  --dataset all \
  --output-dir data/cities/thessaloniki/govgr/downloads/realtime_latest
```

Historical 2025 speed feed:

```bash
sas-fetch-govgr \
  --source historical \
  --dataset speed \
  --historical-pattern _2025 \
  --output-dir data/cities/thessaloniki/govgr/downloads/historical_2025
```

## Operational Notes

- This is a raw acquisition step.
- Use [`data-govgr-targets.md`](data-govgr-targets.md) next to transform the
  downloads into SAS-ready calibration targets.
