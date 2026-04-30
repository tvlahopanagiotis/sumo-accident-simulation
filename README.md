# AntifragiCity SUMA

[![CI](https://github.com/tvlahopanagiotis/sumo-accident-simulation/actions/workflows/ci.yml/badge.svg)](https://github.com/tvlahopanagiotis/sumo-accident-simulation/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

SUMA is the AntifragiCity Simulator for Urban Mobility Antifragility. The
current implementation is an orchestration layer around
[SUMO](https://sumo.dlr.de/) for city data preparation, demand/network
generation, incident simulation, resilience assessment, and result inspection.

The canonical Python package and CLI namespace is now `suma`. A small `sas`
compatibility shim remains so older notebooks, scripts, and command aliases can
continue to run during the transition.

The repository is organized around a package-first layout:

- code in `src/suma/`
- GUI backend in `src/suma/gui/`
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
- OSM-based city bootstrapping, raw-network cleanup, and generic city
  generation.
- Network generation workflows for extracted cities plus benchmark and
  synthetic cases.

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
suma
suma --runs 10
suma --config configs/thessaloniki/postmetro_50kph.yaml
suma --config configs/seattle/default.yaml
suma-assess
```

Generator and data commands:

```bash
suma-generate-city
suma-generate-sioux-falls
suma-generate-riverside
suma-fetch-osm
suma-fetch-govgr
suma-build-govgr-targets
```

Analysis commands:

```bash
suma-analyse-batch
suma-sweep
suma-visualise-sweep
suma-merge-report
suma-compare-seattle-real
```

GUI backend:

```bash
suma-gui-api
```

## Quick Start

Default Thessaloniki run:

```bash
suma
```

Batch run:

```bash
suma --runs 10
```

Seattle run:

```bash
suma --config configs/seattle/default.yaml
```

Thessaloniki post-metro variant:

```bash
suma --config configs/thessaloniki/postmetro_50kph.yaml
```

Headless run with the Python dashboard:

```bash
suma --live-progress
```

## Workflows

### Thessaloniki

```bash
suma-generate-thessaloniki --update-config
suma
```

### Seattle

The large Seattle OSM and `net.xml` artifacts are kept local and are not
committed to git because they exceed GitHub's file size limit.

Download the missing OSM extract if needed:

```bash
suma-fetch-osm \
  --place "Seattle, Washington, USA" \
  --out data/cities/seattle/bundle/traffic_dataset/02_Seattle/01_Input_data/Seattle.osm
```

Then:

```bash
suma-generate-seattle --update-config --config configs/seattle/default.yaml
suma --config configs/seattle/default.yaml
```

This generates local `data/cities/seattle/network/seattle.osm` and
`data/cities/seattle/network/seattle.net.xml` files as needed.

### Sioux Falls

```bash
suma-generate-sioux-falls --update-config
suma
```

### Riverside

```bash
suma-generate-riverside --update-config
suma
```

## Repository Layout

- `src/suma/`: application package
- `src/suma/gui/`: FastAPI backend for the GUI
- `frontend/`: React operator interface
- `configs/`: simulation configs grouped by city
- `data/cities/`: real-city networks and datasets
- `data/benchmarks/`: benchmark networks
- `data/synthetic/`: synthetic/dev networks
- `results/`: generated outputs
- `docs/`: operational, reference, and maintenance docs

## Docs

- [docs/README.md](docs/README.md): documentation index
- [docs/modules/README.md](docs/modules/README.md): narrative guides for the major SUMA modules
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

- `src/suma/gui/` provides the FastAPI backend, workflow registry, and job manager.
- `frontend/` provides the React operator console.
- `src/suma/app/config.py` still centralizes config loading, saving, and validation.
- The CLI remains intact; the GUI is an additional interface over the same workflows.
- Workflow pages are now split into `Data & Integrations`, `Generators`, `Simulations`, and `Analysis`.
- The GUI can create configs from a clean starter template or by cloning an existing config.
- `Data & Integrations` now includes OSM place search, boundary preview, and explicit bbox overrides.
- `Results` now parses SUMA run artifacts into interactive charts and tables instead of only static file previews.
- The GUI also includes a `Documentation` page for tabbed preview of the markdown docs under `docs/`.
- `Data & Integrations` now supports map-based locality boundaries, bbox editing, and custom drawn shapes for OSM extract setup.

For the actual GUI behavior, screen model, and execution flow, see
[`docs/GUI.md`](docs/GUI.md).
