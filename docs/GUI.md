# GUI Guide

This document explains how the first SAS GUI works, what it currently covers,
and how it relates to the existing CLI.

The first GUI version ships as:

- a Python backend in `src/sas/gui/`
- a React frontend in `frontend/`

The CLI remains fully supported. The GUI is an additional operator interface,
not a replacement.

## What It Is

The GUI is an operator console for the existing SAS workflows.

It does not introduce a separate simulation engine. Instead, it sits on top of
the same Python modules and command flows that already power:

- `sas`
- `sas-assess`
- the network generators
- the data download / gov.gr tooling
- the analysis scripts

The design goal for this first version is practical control and visibility:

- edit configs without hand-editing YAML
- launch workflows without remembering CLI arguments
- monitor progress and logs in one place
- inspect generated images, reports, and results directly from the GUI

## How It Works

### Backend

The backend lives in `src/sas/gui/`.

Main pieces:

- `app.py`
  - FastAPI application
  - serves config, workflow, job, and file/result endpoints
- `workflows.py`
  - registry of GUI-exposed workflows
  - defines fields, labels, defaults, and how form input becomes CLI/module arguments
- `jobs.py`
  - background job manager
  - starts long-running tasks as managed subprocesses
  - captures logs, status, progress hints, and output paths

### Frontend

The frontend lives in `frontend/`.

It is a React app that:

- loads the workflow registry from the backend
- renders workflow forms dynamically
- loads and saves YAML configs through the API
- polls job state and result trees
- previews text files, images, and HTML reports

### Execution Model

For the first version, long-running actions are executed as subprocess jobs.

That means:

- the GUI does not replace the existing CLI code paths
- job logs are captured from process output
- progress is inferred from milestone log lines where possible
- existing result folders and output files remain the source of truth

This keeps the GUI aligned with the CLI and reduces duplication.

## Current Scope

The GUI currently exposes:

- config browsing, validation, and saving
- simulation runs
- resilience assessment runs
- network generators
- OSM and gov.gr data workflows
- analysis tools
- job status, logs, progress, and cancellation
- live image/report/result preview

The frontend is branded for the AntifragiCity project using the official logo
assets and the palette derived from the provided SVG brand files.

## Main Screens

### Overview

High-level landing page with:

- available config/workflow/result counts
- active job summary
- live progress image when present

### Config Studio

Used to work with YAML configs in two modes:

- structured mode
  - editable nested form view based on the YAML structure
- raw mode
  - direct YAML text editing

Current behavior:

- load a config from `configs/`
- validate it through the shared config validation layer
- save it back without rewriting paths into machine-specific absolute values

### Workflows

Workflow launcher grouped by category:

- Simulation
- Generators
- Data & Integrations
- Analysis

Each workflow form is built from backend metadata rather than hardcoded page
logic.

### Jobs

Operational monitoring view:

- left side: job queue
- right side top: live image / HTML report area
- right side bottom: logs

Capabilities:

- see running / failed / completed jobs
- inspect the exact command being run
- cancel running jobs
- watch progress and generated outputs

### Results

Simple file/result browser over allowed project output paths.

Current previews:

- text-like files
- generated PNG/SVG images
- HTML reports

## What The GUI Covers Today

The first version includes GUI launchers for:

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

## Current Limitations

- Progress is partly inferred from log output rather than emitted through a
  dedicated event stream.
- Simulation live visualization currently relies on `live_progress.png`
  refreshes rather than native browser charts.
- The frontend uses a clean fallback font stack because the official
  AntifragiCity font assets were not readable from the provided OneDrive path
  in this environment.
- The backend currently uses polling between frontend and API rather than
  WebSockets/SSE.
- The GUI is designed for local operator use in this first version, not for
  multi-user deployment.

## Relationship To The CLI

The CLI is still the stable operational baseline.

Use the CLI when:

- you want shell automation
- you want batch scripting / cron / CI integration
- you do not need interactive monitoring

Use the GUI when:

- you want guided configuration editing
- you want a visible job queue and logs
- you want result/report preview without hunting through folders

Both interfaces are intended to coexist.

## Run In Development

Backend:

```bash
pip install -e .
sas-gui-api
```

or without installing the package entry point:

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

The frontend expects the backend API at:

`http://127.0.0.1:8000`

To change that, set `VITE_API_BASE` when starting the frontend.

## First-Version Implementation Notes

- Long-running tasks are launched as managed subprocess jobs so the GUI can
  surface logs and progress without changing the CLI workflows.
- Simulation live progress currently uses the generated `live_progress.png`
  image when available.
- The official AntifragiCity font files could not be read from the provided
  OneDrive folder in this environment, so the first version uses a clean
  fallback stack. Once those files are accessible, the frontend can be updated
  to use the exact brand typography.
