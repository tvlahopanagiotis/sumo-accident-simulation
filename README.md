# SUMO Accident Simulation (SAS)

[![CI](https://github.com/tvlahopanagiotis/sumo-accident-simulation/actions/workflows/ci.yml/badge.svg)](https://github.com/tvlahopanagiotis/sumo-accident-simulation/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

SAS is a probabilistic traffic accident simulation layer for
[SUMO](https://sumo.dlr.de/). It adds stochastic accident triggering,
accident lifecycle management, resilience measurement, and reporting on top of
standard SUMO traffic simulation.

The repository is organized around a package-first layout:

- code in `src/sas/`
- configs in `configs/`
- city and benchmark assets in `data/`
- outputs in `results/`

For the canonical repo map, see [docs/STRUCTURE.md](/Users/kgrizos/Documents/GitHub/sumo-accident-simulation/docs/STRUCTURE.md).
For the full documentation index, see [docs/README.md](/Users/kgrizos/Documents/GitHub/sumo-accident-simulation/docs/README.md).

## Core Capabilities

- Probabilistic accident triggering using speed, speed variance, density, and road type.
- Four severity tiers with configurable durations, capacity loss, and response timing.
- Accident lifecycle management: `ACTIVE -> CLEARING -> RESOLVED`.
- Antifragility Index and resilience assessment workflows.
- Batch execution and aggregate reporting.
- Live Python dashboard for headless runs.
- OSM and govgr ingestion tooling.
- Network generation workflows for Thessaloniki, Seattle, Sioux Falls, and Riverside.

## Installation

### 1. Install SUMO

macOS:

```bash
brew install sumo
```

Ubuntu / Debian:

```bash
sudo apt-get install sumo sumo-tools sumo-doc
```

Verify:

```bash
sumo --version
```

### 2. Set `SUMO_HOME`

macOS / Linux:

```bash
export SUMO_HOME="/opt/homebrew/share/sumo"
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```

If you use the official macOS installer instead of Homebrew, point `SUMO_HOME`
at the installer `share/sumo` directory.

### 3. Install the package

```bash
git clone https://github.com/tvlahopanagiotis/sumo-accident-simulation.git
cd sumo-accident-simulation
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Main Commands

After installation:

```bash
sas
sas --runs 10
sas --config configs/thessaloniki/postmetro_50kph.yaml
sas --config configs/seattle/default.yaml
sas-assess
```

Generator and data commands:

```bash
sas-generate-thessaloniki
sas-generate-seattle
sas-generate-sioux-falls
sas-generate-riverside
sas-fetch-osm
sas-fetch-govgr
sas-build-govgr-targets
```

Analysis commands:

```bash
sas-analyse-batch
sas-sweep
sas-visualise-sweep
sas-merge-report
sas-compare-seattle-real
```

## Quick Start

Default Thessaloniki run:

```bash
sas
```

Batch run:

```bash
sas --runs 10
```

Seattle run:

```bash
sas --config configs/seattle/default.yaml
```

Thessaloniki post-metro variant:

```bash
sas --config configs/thessaloniki/postmetro_50kph.yaml
```

Headless run with the Python dashboard:

```bash
sas --live-progress
```

## Workflows

### Thessaloniki

```bash
sas-generate-thessaloniki --update-config
sas
```

### Seattle

The large Seattle OSM and `net.xml` artifacts are kept local and are not
committed to git because they exceed GitHub's file size limit.

Download the missing OSM extract if needed:

```bash
sas-fetch-osm \
  --place "Seattle, Washington, USA" \
  --out data/cities/seattle/bundle/traffic_dataset/02_Seattle/01_Input_data/Seattle.osm
```

Then:

```bash
sas-generate-seattle --update-config --config configs/seattle/default.yaml
sas --config configs/seattle/default.yaml
```

This generates local `data/cities/seattle/network/seattle.osm` and
`data/cities/seattle/network/seattle.net.xml` files as needed.

### Sioux Falls

```bash
sas-generate-sioux-falls --update-config
sas
```

### Riverside

```bash
sas-generate-riverside --update-config
sas
```

## Repository Layout

- `src/sas/`: application package
- `configs/`: simulation configs grouped by city
- `data/cities/`: real-city networks and datasets
- `data/benchmarks/`: benchmark networks
- `data/synthetic/`: synthetic/dev networks
- `results/`: generated outputs
- `docs/`: operational, reference, and maintenance docs

## Docs

- [docs/README.md](/Users/kgrizos/Documents/GitHub/sumo-accident-simulation/docs/README.md): documentation index
- [docs/STRUCTURE.md](/Users/kgrizos/Documents/GitHub/sumo-accident-simulation/docs/STRUCTURE.md): canonical repository layout
- [docs/REFERENCE.md](/Users/kgrizos/Documents/GitHub/sumo-accident-simulation/docs/REFERENCE.md): outputs, risk model, severity model, development notes
- [docs/THESSALONIKI_OPERATOR_GUIDE.md](/Users/kgrizos/Documents/GitHub/sumo-accident-simulation/docs/THESSALONIKI_OPERATOR_GUIDE.md): Thessaloniki workflow
- [docs/MACOS_INSTALL.md](/Users/kgrizos/Documents/GitHub/sumo-accident-simulation/docs/MACOS_INSTALL.md): macOS setup
- [docs/SEATTLE_DATA.md](/Users/kgrizos/Documents/GitHub/sumo-accident-simulation/docs/SEATTLE_DATA.md): Seattle bundle and network notes
- [docs/CHANGELOG.md](/Users/kgrizos/Documents/GitHub/sumo-accident-simulation/docs/CHANGELOG.md): changelog

## GUI Direction

The repository is already structured for a future GUI:

- `src/sas/app/config.py` centralizes config loading, saving, and validation.
- `src/sas/app/services.py` exposes programmatic simulation and assessment entry points.
- The CLI is now just a thin interface over importable package logic.
