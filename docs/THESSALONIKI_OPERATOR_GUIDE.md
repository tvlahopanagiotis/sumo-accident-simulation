# Thessaloniki Operator Guide

This is the shortest practical workflow for transport-engineering use.

## 1. Open the correct folder

```bash
cd <repo-root>/.worktrees/main-clean
```

## 2. First-time setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If SUMO is missing on macOS:

```bash
brew install sumo
```

## 3. Run one simulation

Default config:

```bash
python runner.py
```

Post-metro Thessaloniki network with all `>90 km/h` links capped to `50 km/h`:

```bash
python runner.py --config config_thessaloniki_postmetro_50kph.yaml
```

Batch run:

```bash
python runner.py --config config_thessaloniki_postmetro_50kph.yaml --runs 10
```

## 4. Download govgr data

Realtime:

```bash
python govgr_downloader.py \
  --source realtime \
  --dataset all \
  --output-dir thessaloniki_govgr/downloads/realtime_latest
```

Historical 2025:

```bash
python govgr_downloader.py \
  --source historical \
  --dataset speed \
  --historical-pattern _2025 \
  --output-dir thessaloniki_govgr/downloads/historical_2025

python govgr_downloader.py \
  --source historical \
  --dataset travel_times \
  --historical-pattern _2025 \
  --output-dir thessaloniki_govgr/downloads/historical_2025
```

Historical 2026:

```bash
python govgr_downloader.py \
  --source historical \
  --dataset speed \
  --historical-pattern _2026 \
  --output-dir thessaloniki_govgr/downloads/historical_2026

python govgr_downloader.py \
  --source historical \
  --dataset travel_times \
  --historical-pattern _2026 \
  --output-dir thessaloniki_govgr/downloads/historical_2026
```

## 5. Build calibration/validation targets

```bash
python govgr_targets.py \
  --downloads-root thessaloniki_govgr/downloads \
  --calibration-year 2025 \
  --validation-year 2026 \
  --output-dir thessaloniki_govgr/targets/post_metro_2025_2026
```

Outputs go to:

`thessaloniki_govgr/targets/post_metro_2025_2026/`

## 6. What to share with colleagues

Share these folders/files for reproducible analysis:

- `config.yaml` (or another config used in the run)
- `thessaloniki_network_postmetro_50kph/` (network used)
- `thessaloniki_govgr/targets/post_metro_2025_2026/` (calibration targets)
- `results/` run folders with timestamps

## Optional shortcut commands

If `make` is installed, there are wrappers for the same actions:

```bash
make help
```
