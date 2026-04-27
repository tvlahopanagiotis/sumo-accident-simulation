# GUI Guide

This document explains the current SAS GUI, how it relates to the CLI, and
what the present first-generation interface actually does.

The GUI ships as:

- a FastAPI backend in `src/sas/gui/`
- a React frontend in `frontend/`

The CLI remains fully supported. The GUI is an additional operator interface,
not a replacement runtime.

## What It Is

The GUI is an operator console over the same SAS workflows that already power:

- `sas`
- `sas-assess`
- the generator commands
- the OSM / gov.gr integration commands
- the analysis scripts

It does not introduce a separate simulation engine. It orchestrates the same
Python modules and subprocess flows already used by the CLI.

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
- renders interactive post-run metrics from existing SAS output files

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

- counts for configs, workflows, jobs, and result roots
- active-job snapshot
- live progress image when present

### Config Studio

Config Studio now supports:

- loading configs from `configs/`
- creating a new config from a clean starter template
- cloning the currently selected config into a new file
- deleting config files from the GUI
- validating configs through the shared backend validation layer
- saving in structured mode or raw YAML mode
- selecting configs from a folder-grouped config picker
- selecting the network from discovered `.sumocfg` files instead of hand-typing paths
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
include a dedicated “About This Model” dialog that explains how to interpret
their parameters in simulation terms and points back to the relevant source and
documentation files.

The `Resilience Assessment` tab also uses a clearer structure for:

- demand levels as a vertical numeric list
- incident scenarios as a table with preset ladder, scenario name, and base
  probability side by side

### Data & Integrations

This is now a dedicated page rather than one card inside a generic workflow
grid.

Current capabilities:

- search a place through OSM/Nominatim
- pick a result from the returned matches
- preview the result on a real map component
- switch between locality boundary, bounding box, and custom shape modes
- override south/west/north/east boundaries before launch
- draw a custom shape directly on the map; the GUI derives the enclosing bbox
  used by the current download backend
- keep a stable map view while sketching a custom shape instead of constantly
  re-fitting to the in-progress polygon
- pass those explicit bounds to the OSM download workflow
- edit advanced OSM fetch parameters such as endpoints and user-agent details

Greek traffic-feed tooling is shown separately.

Important current limitation:

- the present gov.gr integration in this repository is still effectively
  Thessaloniki-specific, because it uses the IMET/CERTH Thessaloniki feed
  sources already wired into `sas.integrations.govgr_downloader`

So the GUI surfaces these workflows only in the Greek-location context and
marks Thessaloniki as the currently valid operational target.

### Generators

Generators now have their own page. The UI still exposes the existing bundled
generator workflows, but frames them as reusable generation tasks rather than
as root-level script substitutes.

### Simulations

Simulations now have their own page and currently expose:

- single or batch simulation runs
- resilience assessment runs

### Analysis

Analysis now has its own page and exposes the current analyst tooling:

- batch analysis
- parameter sweeps
- sweep visualisation
- merge-report workflow
- Seattle real-data comparison

### Jobs

Jobs remain the operational monitoring page:

- left third: job queue
- right side top: live image / HTML report area
- right side bottom: logs

Capabilities:

- inspect the exact command being run
- cancel running jobs
- watch progress and output discovery

### Results

Results is no longer just a file browser.

It now has two layers:

- a file/tree browser over result folders
- an interactive run dashboard when the selected file or folder belongs to a
  valid run directory

The interactive layer parses existing SAS outputs such as:

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

The underlying source of truth is still the normal SAS run folder.

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

- project navigation links
- `https://antifragicity.eu`
- the Horizon Europe funding disclaimer
- the `Funded by the European Union` mark used by the current frontend build

## Current Limitations

- Progress is still inferred partly from logs and existing artifacts rather
  than emitted through a dedicated event stream.
- Simulation live monitoring still relies on `live_progress.png` refreshes for
  in-progress visualisation.
- The backend/frontend still use polling rather than WebSockets or SSE.
- The Results page is now interactive, but it currently reflects the metrics
  already written by SAS; if a future workflow needs richer post-run views, the
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
sas-gui-api
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

`http://127.0.0.1:8000`

Override with `VITE_API_BASE` if needed.
