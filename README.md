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
- GUI backend in `src/sas/gui/`
- React frontend in `frontend/`
- configs in `configs/`
- city and benchmark assets in `data/`
- outputs in `results/`

For the canonical repo map, see [docs/STRUCTURE.md](docs/STRUCTURE.md).
For the full documentation index, see [docs/README.md](docs/README.md).
For narrative module guides, see [docs/modules/README.md](docs/modules/README.md).
For workflow docs, see
[docs/operations/README.md](docs/operations/README.md).

## Core Capabilities

- Probabilistic accident triggering using speed, speed variance, density, and road type.
- Four severity tiers with configurable durations, edge-capacity loss, lane closures, and response timing.
- Optional local rerouting around active incidents for system-wide resilience analysis.
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

### 4. GUI prerequisites

For the React frontend:

```bash
cd frontend
npm install
```

Node `20+` is recommended.

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

GUI backend:

```bash
sas-gui-api
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
- `src/sas/gui/`: FastAPI backend for the GUI
- `frontend/`: React operator interface
- `configs/`: simulation configs grouped by city
- `data/cities/`: real-city networks and datasets
- `data/benchmarks/`: benchmark networks
- `data/synthetic/`: synthetic/dev networks
- `results/`: generated outputs
- `docs/`: operational, reference, and maintenance docs

## Docs

- [docs/README.md](docs/README.md): documentation index
- [docs/modules/README.md](docs/modules/README.md): narrative guides for the major SAS modules
- [docs/operations/README.md](docs/operations/README.md): consolidated operator and analyst workflow docs
- [docs/STRUCTURE.md](docs/STRUCTURE.md): canonical repository layout
- [docs/REFERENCE.md](docs/REFERENCE.md): outputs, risk model, severity model, development notes
- [docs/THESSALONIKI_OPERATOR_GUIDE.md](docs/THESSALONIKI_OPERATOR_GUIDE.md): Thessaloniki workflow
- [docs/MACOS_INSTALL.md](docs/MACOS_INSTALL.md): macOS setup
- [docs/SEATTLE_DATA.md](docs/SEATTLE_DATA.md): Seattle bundle and network notes
- [docs/GUI.md](docs/GUI.md): how the GUI works, what it covers, and how to run it
- [docs/SUMO_ACCIDENT_SIMULATOR_REVIEW.md](docs/SUMO_ACCIDENT_SIMULATOR_REVIEW.md): technical review of the current incident model
- [docs/CHANGELOG.md](docs/CHANGELOG.md): changelog

## GUI Direction

The first GUI version is now in place:

- `src/sas/gui/` provides the FastAPI backend, workflow registry, and job manager.
- `frontend/` provides the React operator console.
- `src/sas/app/config.py` still centralizes config loading, saving, and validation.
- The CLI remains intact; the GUI is an additional interface over the same workflows.
- Workflow pages are now split into `Data & Integrations`, `Generators`, `Simulations`, and `Analysis`.
- The GUI can create configs from a clean starter template or by cloning an existing config.
- `Data & Integrations` now includes OSM place search, boundary preview, and explicit bbox overrides.
- `Results` now parses SAS run artifacts into interactive charts and tables instead of only static file previews.
- The GUI also includes a `Documentation` page for tabbed preview of the markdown docs under `docs/`.
- `Data & Integrations` now supports map-based locality boundaries, bbox editing, and custom drawn shapes for OSM extract setup.

For the actual GUI behavior, screen model, and execution flow, see
[`docs/GUI.md`](docs/GUI.md).
