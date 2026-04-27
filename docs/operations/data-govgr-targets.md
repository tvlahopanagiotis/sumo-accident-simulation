# Command: `sas-build-govgr-targets`

Build SAS-ready calibration and validation targets from downloaded govgr data.

## What It Does

1. reads previously downloaded govgr feed files
2. assembles year-specific calibration and validation targets
3. writes normalized target outputs for later analysis

## Main Inputs

- `--downloads-root`
- `--calibration-year`
- `--validation-year`
- `--output-dir`

## Main Outputs

- processed target files under `data/cities/thessaloniki/govgr/targets/`
  or the selected output directory

## Typical Usage

```bash
sas-build-govgr-targets \
  --downloads-root data/cities/thessaloniki/govgr/downloads \
  --calibration-year 2025 \
  --validation-year 2026 \
  --output-dir data/cities/thessaloniki/govgr/targets/post_metro_2025_2026
```

## Operational Notes

- This is the bridge between raw govgr downloads and model calibration work.
- Keep the target directory together with the config and results used for any
  reported analysis.
