# Thessaloniki Operator Guide

This is the shortest city-specific runbook in the repository. It stays focused
on the Thessaloniki operational path and delegates command details to
`docs/operations/`.

## 1. Open the correct folder

```bash
cd <repo-root>
```

If you use git worktrees, replace `<repo-root>` with the specific worktree you
intend to run from.

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

## 3. Build or refresh the Thessaloniki network

```bash
sas-generate-thessaloniki --update-config
```

For command details:

- [`operations/generator-thessaloniki.md`](operations/generator-thessaloniki.md)

## 4. Run one simulation

Default config:

```bash
sas
```

Post-metro Thessaloniki network with all `>90 km/h` links capped to `50 km/h`:

```bash
sas --config configs/thessaloniki/postmetro_50kph.yaml
```

Batch run:

```bash
sas --config configs/thessaloniki/postmetro_50kph.yaml --runs 10
```

For simulator behavior and outputs:

- [`operations/simulation-runner.md`](operations/simulation-runner.md)
- [`REFERENCE.md`](REFERENCE.md)

## 5. Download govgr data

Realtime:

```bash
sas-fetch-govgr \
  --source realtime \
  --dataset all \
  --output-dir data/cities/thessaloniki/govgr/downloads/realtime_latest
```

Historical 2025:

```bash
sas-fetch-govgr \
  --source historical \
  --dataset speed \
  --historical-pattern _2025 \
  --output-dir data/cities/thessaloniki/govgr/downloads/historical_2025

sas-fetch-govgr \
  --source historical \
  --dataset travel_times \
  --historical-pattern _2025 \
  --output-dir data/cities/thessaloniki/govgr/downloads/historical_2025
```

Historical 2026:

```bash
sas-fetch-govgr \
  --source historical \
  --dataset speed \
  --historical-pattern _2026 \
  --output-dir data/cities/thessaloniki/govgr/downloads/historical_2026

sas-fetch-govgr \
  --source historical \
  --dataset travel_times \
  --historical-pattern _2026 \
  --output-dir data/cities/thessaloniki/govgr/downloads/historical_2026
```

For command details:

- [`operations/data-govgr-download.md`](operations/data-govgr-download.md)

## 6. Build calibration/validation targets

```bash
sas-build-govgr-targets \
  --downloads-root data/cities/thessaloniki/govgr/downloads \
  --calibration-year 2025 \
  --validation-year 2026 \
  --output-dir data/cities/thessaloniki/govgr/targets/post_metro_2025_2026
```

Outputs go to:

`data/cities/thessaloniki/govgr/targets/post_metro_2025_2026/`

For command details:

- [`operations/data-govgr-targets.md`](operations/data-govgr-targets.md)

## 7. What to share with colleagues

Share these folders/files for reproducible analysis:

- `configs/thessaloniki/postmetro_50kph.yaml` (or another config used in the run)
- `data/cities/thessaloniki/network/` (network used)
- `data/cities/thessaloniki/govgr/targets/post_metro_2025_2026/` (calibration targets)
- `results/` run folders with timestamps

## Optional shortcut commands

If `make` is installed, there are wrappers for the same actions:

```bash
make help
```
