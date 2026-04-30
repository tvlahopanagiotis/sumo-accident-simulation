# Repository Structure

This file is the canonical description of the repository layout.

Rule: whenever the repository structure changes, update this file in the same
change so it always matches the current on-disk layout.

## Goals

- Keep reusable code in `src/sas/`.
- Keep configuration in `configs/`.
- Keep large city / benchmark / synthetic assets in `data/`.
- Keep generated run outputs in `results/`.
- Avoid root-level Python wrappers and duplicated entry points.

## Current Layout

```text
.
├── .github/
│   └── workflows/
├── .worktrees/
├── frontend/
│   ├── public/
│   └── src/
├── configs/
│   ├── seattle/
│   │   └── default.yaml
│   └── thessaloniki/
│       ├── default.yaml
│       └── postmetro_50kph.yaml
├── data/
│   ├── benchmarks/
│   │   └── sioux_falls/
│   │       └── network/
│   ├── cities/
│   │   ├── seattle/
│   │   │   ├── bundle/
│   │   │   └── network/
│   │   └── thessaloniki/
│   │       ├── govgr/
│   │       └── network/
│   └── synthetic/
│       └── riverside/
│           └── network/
├── docs/
│   ├── modules/
│   ├── operations/
│   ├── README.md
│   ├── STRUCTURE.md
│   └── ...
├── results/
├── src/
│   └── sas/
│       ├── analysis/
│       ├── app/
│       ├── core/
│       ├── generators/
│       ├── gui/
│       ├── integrations/
│       ├── simulation/
│       ├── tools/
│       └── visualization/
├── tests/
├── Makefile
├── README.md
└── pyproject.toml
```

Hidden helper directories are shown because they exist in the repository today,
but `.worktrees/` is optional local workflow state rather than required project
source.

## Directory Responsibilities

### `src/sas/`

Application code only.

- `core/`: simulation primitives such as the accident manager, risk model, and
  metrics collection.
- `simulation/`: main runner, parallel execution, and SUMO path/runtime logic.
- `analysis/`: resilience assessment, MFD analysis, reports, and batch analysis.
- `integrations/`: govgr and OSM download/ingest tooling.
- `generators/`: network and route generation workflows.
- `gui/`: FastAPI backend for the React GUI, including workflow metadata,
  background job execution, config creation, OSM search, and result/file APIs.
- `tools/`: standalone operational or analyst-facing utilities.
- `visualization/`: plots, reports, and the live dashboard.
- `app/`: config resolution, persistence, validation, and GUI-friendly services.

### `frontend/`

React frontend for the operator GUI.

- `frontend/src/`: application views, dynamic workflow forms, config editor,
  job console, OSM search/boundary preview, and interactive result dashboards.
- `frontend/public/branding/`: AntifragiCity logo assets plus the EU funding
  mark used in the footer.

See `docs/GUI.md` for the functional explanation of how the frontend and
backend interact.

### `configs/`

Human-edited run configurations.

- `configs/thessaloniki/default.yaml`: default Thessaloniki simulation config.
- `configs/thessaloniki/postmetro_50kph.yaml`: Thessaloniki post-metro variant.
- `configs/seattle/default.yaml`: Seattle simulation config.

Configs are grouped by city/domain rather than kept flat at the repo root.

### `data/`

Input assets and generated datasets, grouped by provenance.

- `data/cities/`: real-city assets.
  - `data/cities/thessaloniki/network/`: Thessaloniki OSM/SUMO network files.
  - `data/cities/thessaloniki/govgr/`: govgr datasets, plus runtime
    `downloads/` and derived `targets/`.
  - `data/cities/seattle/bundle/`: bundled Seattle OD / validation / source data.
  - `data/cities/seattle/network/`: generated Seattle SUMO network files.
    Large generated artifacts such as `seattle.osm` and `seattle.net.xml`
    remain local and are regenerated on demand instead of being committed.
- `data/benchmarks/`: canonical benchmark networks.
  - `data/benchmarks/sioux_falls/network/`
- `data/synthetic/`: synthetic/generated networks used mainly for development.
  - `data/synthetic/riverside/network/`

This keeps domain data out of the repo root and makes future city additions
predictable.

### `results/`

Generated run outputs only. Nothing under `results/` should be treated as a
source input to the package unless explicitly documented.

### `docs/`

Project documentation lives here. The only markdown file kept at the repository
root is `README.md`; all other long-form docs belong under `docs/`.

- `docs/operations/`: consolidated workflow documentation for operators and
  analysts.
- `docs/modules/`: narrative guides for the simulator, generators,
  data/integrations, and analysis layers.
- top-level docs under `docs/`: reference, project notes, platform guides, and
  maintenance material.

## How To Run

After `pip install -e .`:

- `suma`
- `suma-assess`
- `suma-generate-city`
- `suma-fetch-osm`
- `suma-fetch-govgr`
- `suma-build-govgr-targets`
- `suma-analyse-batch`
- `suma-sweep`
- `suma-visualise-sweep`
- `suma-merge-report`
- `suma-compare-seattle-real`
- `suma-gui-api`
- `sas`
- `sas-assess`
- `sas-generate-city`
- `sas-generate-thessaloniki`
- `sas-generate-seattle`
- `sas-generate-sioux-falls`
- `sas-generate-riverside`
- `sas-fetch-osm`
- `sas-fetch-govgr`
- `sas-build-govgr-targets`
- `sas-analyse-batch`
- `sas-sweep`
- `sas-visualise-sweep`
- `sas-merge-report`
- `sas-compare-seattle-real`

The `sas-*` names are retained for compatibility. The visible app and new
operator-facing command aliases use the SUMA name.

## Structure Notes

- There are no root-level Python compatibility wrappers anymore.
- The only root-level markdown file is `README.md`.
- The GUI spans `src/sas/gui/` and `frontend/`; keep their contracts aligned.
- If a new city is added, prefer `data/cities/<city>/...` and
  `configs/<city>/...`.
- If a new benchmark is added, prefer `data/benchmarks/<name>/...`.
- If a new synthetic/dev network is added, prefer `data/synthetic/<name>/...`.
