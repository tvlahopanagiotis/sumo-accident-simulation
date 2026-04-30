# SUMA GUI Guide

This document explains the current AntifragiCity SUMA GUI, how it relates to
the CLI, and what the operator interface currently does.

The GUI ships as:

- a FastAPI backend in `src/sas/gui/`
- a React frontend in `frontend/`

The CLI remains fully supported. The GUI is an additional operator interface,
not a replacement runtime.

## What It Is

The GUI is an operator console over the same SUMA/SAS workflows that already
power:

- `suma` / `sas`
- `suma-assess` / `sas-assess`
- the generator commands
- the OSM / gov.gr integration commands
- the analysis scripts

It does not introduce a separate simulation engine. It orchestrates the same
Python modules and subprocess flows already used by the CLI. The Python package
namespace remains `sas` for compatibility, while the product/API branding is now
SUMA.

## How It Works

### Backend

The backend lives in `src/sas/gui/`.

Main pieces:

- `app.py`
  - FastAPI application
  - exposes config, workflow, location-search, result-summary, job, and file APIs
- `workflows.py`
  - workflow registry used by both the launcher UI and the subprocess runner
  - maps form values to `python -m sas...` commands
- `jobs.py`
  - managed background subprocess execution
  - captures logs, progress hints, return codes, and key output files
- `locations.py`
  - OSM/Nominatim place search helper for the GUI
- `results.py`
  - parses existing run artifacts into a structured summary for the Results page

### Frontend

The frontend lives in `frontend/`.

It is a React app that:

- loads workflow metadata from the backend
- edits and saves configs through the API
- creates new configs from a clean starter or a clone of an existing config
- searches OSM locations and previews extract boundaries
- launches workflows as managed jobs
- polls jobs and results
- renders interactive post-run metrics from existing SUMA/SAS output files

### Shell And Navigation

The operator shell is organized around a consistent dashboard pattern:

- a collapsible left sidebar for the main workflow pages
- a sticky compact top bar with the product title, running-job summary, and
  user/settings placeholder
- primary tabs for major page families, such as `OSM Extract` versus
  `Traffic Feeds`
- secondary tabs for methods or tasks inside a page, such as `New Extract` and
  `Extracted Network`
- guide buttons with consistent labels: `Page Guide`, `Method Guide`,
  `Model Guide`, and `Tool Guide`
- responsive card layouts that collapse to one column on smaller screens

The current settings menu includes a light/dark theme toggle and a disabled
language selector. These are interface placeholders for later user-preference
and localization work, but the dashboard already includes the first dark-theme
contrast pass.

### Execution Model

Long-running actions are executed as subprocess jobs.

That means:

- the GUI stays aligned with the CLI code paths
- job logs are captured from stdout/stderr
- progress is inferred from current log/report artifacts
- run folders and result files remain the source of truth

This keeps GUI and CLI behavior close and avoids duplicating business logic.

## Main Screens

### Overview

High-level landing page with:

- counts for configs, workflows, jobs, and result runs
- active-job snapshot
- live progress image when present
- shortcuts into the main pipeline stages
- recent result runs

### Config Studio

Config Studio now supports:

- loading configs from `configs/`
- creating a new config from a clean starter template
- cloning the currently selected config into a new file
- deleting config files from the GUI
- validating configs through the shared backend validation layer
- saving in structured mode or raw YAML mode
- selecting configs from a folder-grouped config picker
- opening configs through a two-step folder/name selector
- selecting the network from discovered `.sumocfg` files while still allowing a
  manual path for a new city before its generator has written the file
- selecting the output folder from discovered output roots instead of typing every path manually

Structured mode is now organized into secondary tabs:

- `SUMO Runtime`
- `Risk Model`
- `Accident Model`
- `Outputs & Monitoring`
- `Resilience Assessment`

Each tab includes:

- introductory text
- grouped forms instead of one long flat column
- lightweight inline descriptions
- hover help for examples and practical meaning
- switch-style toggles for boolean settings

The `Risk Model`, `Accident Model`, and `Resilience Assessment` tabs also
include a dedicated `Model Guide` dialog that explains how to interpret their
parameters in simulation terms and points back to the relevant source and
documentation files.

The `Resilience Assessment` tab also uses a clearer structure for:

- demand levels as a vertical numeric list, with the model guide explaining
  that they are a variable stress ladder. In current random-demand assessment
  workflows they are route-generation periods, so lower values usually mean
  heavier demand.
- incident scenarios as a table with preset ladder, scenario name, and base
  probability side by side

### Data & Integrations

This is now a dedicated page rather than one card inside a generic workflow
grid.

Current capabilities:

- split the OSM workflow into:
  - `New Extract`
  - `Extracted Network`
- search a place through OSM/Nominatim
- pick a result from the returned matches
- bootstrap `data/cities/<city>/network/` and `configs/<city>/default.yaml`
  directly from the OSM workflow
- preview the result on a real map component
- keep boundary mode and map preview together:
  - locality boundary
  - bounding box
  - custom shape
- override south/west/north/east boundaries before launch
- draw a custom shape directly on the map; the GUI derives the enclosing bbox
  used by the current download backend
- pass those explicit bounds to the OSM download workflow
- edit advanced OSM fetch parameters such as endpoints and user-agent details
- browse extracted city `.osm` files directly from the same page
- inspect extracted roads by:
  - speed-limit tags
  - road type
  - lanes and direction
  - signalized intersections
- clean up speed-limit tags before generation by:
  - clicking one road
  - Shift-clicking multiple roads
  - selecting all roads that match a filter set
  - selecting multiple road groups at once
  - drag-box selection without moving the map
  - expanding a selected segment to the full connected same-name road
  - deleting selected road segments from the raw `.osm` extract

Greek traffic-feed tooling is shown separately.

Important current limitation:

- the present gov.gr integration in this repository is still effectively
  Thessaloniki-specific, because it uses the IMET/CERTH Thessaloniki feed
  sources already wired into `sas.integrations.govgr_downloader`

The traffic-feed page is now split into:

- `New Feed Pull`
  - choose the discovered city feed integration
  - choose the target city folder separately from the feed source
  - review provider workflow slots
  - launch the current gov.gr downloader
  - launch target-building for historical calibration/validation tables
  - keep or override the derived `data/cities/<city>/govgr/` output paths
- `Exported Feeds`
  - inspect published feed catalogs from the selected source integration
  - inspect downloaded export runs and generated target folders from the
    selected target city folder
  - browse exported files in a scrollable in-page browser
  - view the matched subset of feed `Link_id` values on top of the city OSM
    network through the feed-alignment map

The feed page is now also structured around provider workflow slots so future
city-specific traffic-feed adapters can be inserted without changing the page
layout again.

This matters for network variants such as `thessaloniki_centre`: you can keep
the Thessaloniki gov.gr feed as the source while writing and reviewing exports
under the variant city folder.

When the gov.gr downloader output path is left at a city `.../downloads` root,
the backend creates a timestamped run folder under that root. If you enter a
more specific folder manually, that exact folder is used.

The feed-alignment map currently works only where feed link identifiers match
OSM way identifiers in the selected city extract. In Thessaloniki that
alignment is partial, so the map is diagnostic rather than complete.

### OD Generators

OD Generators now have their own page.

The main real-city path is a generic city generator:

- use the family tabs to switch between:
  - `City`
  - `Benchmark`
  - `Synthetic`
- use the secondary task tabs to switch between:
  - `Build`
  - `View Inputs`
- choose any extracted city under `data/cities/<slug>/`
- build the network into that city's `network/` folder
- generate either:
  - random demand
  - OD-driven demand when compatible files exist
- show different build fields for random-demand and OD-demand choices
- optionally patch the city's default YAML config

The generator page now also includes a `View Inputs` tab for city demand inputs:

- inspect whether the selected city has discovered OD and node support files
- review a sample of the OD table
- view the top OD desire lines on a map between centroid nodes
- click/select a zone to inspect origin and destination demand totals
- estimate rough random-demand trip requests from route period and end time
- identify quickly when a city is currently limited to random-demand generation

The build tab also explains the random-demand control logic directly in the UI:

- the route period is the main request-rate control
- lower period means more requested departures
- network size and connectivity still affect how many valid trips survive
  validation and how many vehicles remain active at once

The benchmark and synthetic generators remain separate:

- Sioux Falls
- Riverside

### Simulations

Simulations now have their own page and currently expose:

- single or batch simulation runs
- resilience assessment runs
- sub-tabs for the simulator run and resilience assessment tools
- a shared config picker with folder/name selection
- a simulator guide dialog that explains both execution modes

### Analysis

Analysis now has its own page and exposes the current analyst tooling:

- batch analysis
- parameter sweeps
- sweep visualisation
- merge-report workflow
- Seattle real-data comparison
- tabs for batch, sweeps, reports, and validation tools
- a tool guide dialog for the current analysis families

### Jobs

Jobs remain the operational monitoring page:

- left third: job queue
- right side top: live image / HTML report area
- right side bottom: logs

Capabilities:

- inspect the exact command being run
- cancel running jobs
- clear finished jobs
- remove individual completed jobs from the visible registry
- keep completed-job history across browser refreshes while the backend remains
  available, with persisted history for completed records
- watch progress and output discovery

### Results

Results is no longer just a file browser.

It now has two layers:

- a run registry table with city, date, run-name, and search filters
- an interactive run dashboard when the selected file or folder belongs to a
  valid run directory

The interactive layer parses existing SUMA/SAS outputs such as:

- `metadata.json`
- `network_metrics.csv`
- `accident_reports.json`
- `antifragility_index.json`

It then renders:

- headline metrics
- vehicle-count, speed, throughput, delay, accident, and speed-ratio charts
- severity distribution
- accident table
- per-event antifragility table
- embedded report and figure previews
- CSV, JSON, and ZIP export actions for selected runs
- deletion of a selected run folder through the backend API

The underlying source of truth is still the normal SUMA run folder.

### Documentation

The sidebar now also includes a `Documentation` page.

This page:

- includes the repository-root `README.md`
- groups documents into a curated reading order rather than a flat file list
- separates module guides from command guides and maintenance notes
- renders each document as markdown rather than raw plain text

The current reading structure is:

- `Start Here`
- `Foundations`
- `Module Guides`
- `Command Guides`
- `City Notes And Reviews`
- `Maintenance`

The goal is to keep operator guidance and reference material available inside
the same workspace as config editing and workflow execution.

## Current Scope

The GUI currently exposes launchers for:

- simulation runs
- resilience assessment
- Thessaloniki / Seattle / Sioux Falls / Riverside generation
- OSM download
- gov.gr download
- gov.gr target building
- batch analysis
- sweep run / sweep visualisation
- MFD merge/report regeneration
- Seattle real-data comparison

## Branding And Footer

The GUI uses the AntifragiCity logo assets in `frontend/public/branding/`.

The UI footer now includes:

- AntifragiCity monogram branding
- SUMA subtitle: Simulator for Urban Mobility Antifragility
- project navigation links
- `https://antifragicity.eu`
- Rhoé as the development partner responsible for SUMA
- the Horizon Europe funding disclaimer
- the `Funded by the European Union` mark used by the current frontend build
- the visible release version

## Current Limitations

- Progress is still inferred partly from logs and existing artifacts rather
  than emitted through a dedicated event stream.
- Simulation live monitoring still relies on `live_progress.png` refreshes for
  in-progress visualisation.
- The backend/frontend still use polling rather than WebSockets or SSE.
- The Results page is now interactive, but it currently reflects the metrics
  already written by SUMA; if a future workflow needs richer post-run views, the
  underlying simulation/analysis outputs should be extended rather than
  bypassed.

## Relationship To The CLI

The CLI is still the operational baseline.

Use the CLI when:

- you want shell automation
- you want CI/cron/batch scripting
- you do not need interactive monitoring

Use the GUI when:

- you want guided config authoring
- you want place search and boundary preview for OSM ingestion
- you want a visible job queue and logs
- you want interactive post-run dashboards without manually opening result files

Both interfaces are intended to coexist.

## Run In Development

Backend:

```bash
pip install -e .
suma-gui-api
```

or without the entry point:

```bash
PYTHONPATH=src python -m sas.gui.app
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Then open:

`http://127.0.0.1:5173`

The frontend expects the backend at:

`http://127.0.0.1:12000`

Override with `VITE_API_BASE` if needed.
