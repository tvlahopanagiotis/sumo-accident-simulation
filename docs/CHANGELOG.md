# Changelog

All notable changes to this project will be documented in this file.

## [0.2.2] - 2026-04-27

### Changed
- Reworked the incident-impact model from a speed-only lane slowdown into a
  configurable `speed_limit` / `lane_closure` / `hybrid` incident effect model.
- Added explicit lane closures, edge-level speed degradation, clearing-phase
  recovery, and periodic local rerouting around active incidents.
- Normalized risk-model density to vehicles per lane-km instead of vehicles per
  edge-km.
- Moved the default scientific runtime to `sumo.step_length: 1` and rescaled
  the default incident probabilities and resilience-assessment ladders for the
  new time resolution.
- Extended accident reporting with blocked-lane, managed-lane, and rerouted
  vehicle counts.
- Expanded config validation and the GUI config editor to cover the new
  incident controls and corrected time-unit guidance.

### Added
- New accident config controls:
  - `incident_effect_mode`
  - `reroute_affected_vehicles`
  - `reroute_radius_m`
  - `reroute_interval_s`
- A command-by-command operations documentation set under `docs/operations/`.

### Documentation
- Rewrote the SUMO accident simulator review to match the current hybrid
  incident logic and to state the remaining scientific limitations clearly.
- Reorganized the docs index so workflow docs, reference docs, and
  data/project notes are separated by purpose.
- Clarified the Thessaloniki and Seattle docs so they point to the new command
  references instead of duplicating them loosely.

### Verification
- Verified the simulator and tests with `pytest -q` (`98 passed`).

## [0.2.1] - 2026-04-24

### Changed
- Reworked the GUI navigation so workflow categories now live on separate
  sidebar pages: `Data & Integrations`, `Generators`, `Simulations`, and
  `Analysis`.
- Replaced the old flat structured config editor with section tabs, grouped
  forms, parameter explanations, example values, and clearer input types.
- Improved Config Studio with folder-grouped config selection, `.sumocfg`
  discovery for the network field, output-folder discovery, hover help popups,
  switch-style booleans, config deletion, and section-level model guidance for
  the risk, accident, and resilience tabs.
- Added a `Documentation` sidebar page that previews the markdown files under
  `docs/` plus the root `README.md` as rendered markdown tabs.
- Upgraded `Data & Integrations` with OSM/Traffic Feed tabs and a real map
  preview that can show locality geometry, bounding boxes, and custom drawn
  shapes.
- Stabilized custom-shape map behavior so drawing no longer auto-zooms into the
  temporary shape extent on each point.
- Added the workflow module path label to the OSM extract card, matching the
  pattern used elsewhere in the GUI.
- Added config creation from either a clean starter template or a clone of the
  selected config.
- Reworked the Results page so it parses existing SAS artifacts into
  interactive charts, summary cards, accident tables, antifragility event
  tables, and report/image previews.
- Added an AntifragiCity/EU-compliant footer and removed the temporary sidebar
  branding disclaimer.

### Added
- OSM place search from the GUI via Nominatim, with embedded map preview and
  explicit south/west/north/east boundary overrides.
- Backend endpoints for location search, config creation, and parsed run
  summaries.
- Official `Funded by the European Union` branding asset in
  `frontend/public/branding/`.
- Frontend markdown rendering and map dependencies for documentation preview and
  boundary overlays.

### Documentation
- Updated `docs/GUI.md` to match the current screen model and execution flow.
- Refreshed README GUI notes to reflect the new information architecture and
  interactive results behavior.
- Updated `docs/STRUCTURE.md` to include the new GUI helper modules and footer
  branding asset.

### Verification
- Verified the GUI backend modules with `python -m compileall src/sas/gui src/sas/integrations/download_osm_place.py`.
- Verified the frontend with `cd frontend && npm run build`.
- Smoke-tested the new API flows for branding, config creation, and parsed run
  summaries through FastAPI `TestClient`.

## [0.2.0] - 2026-04-23

### Changed
- Reorganized the repository around a package-first `src/sas/` layout.
- Split run configuration into `configs/thessaloniki/` and `configs/seattle/`.
- Grouped datasets and generated network assets under `data/cities/`,
  `data/benchmarks/`, and `data/synthetic/`.
- Removed root-level Python wrapper scripts in favor of package entry points
  and `python -m sas...` module execution.
- Centralized config loading, validation, and save logic in `src/sas/app/`
  to support a future GUI-driven parameter editor.
- Updated the test suite and tooling for the new package layout.
- Added a first GUI stack with a FastAPI backend and React frontend while
  keeping the CLI workflows intact.

### Added
- Package entry points for simulation, assessment, generators, OSM downloads,
  govgr ingestion, and analyst tooling.
- `docs/STRUCTURE.md` as the canonical repository layout reference.
- `docs/REFERENCE.md` and `docs/SEATTLE_DATA.md` for operational and
  dataset-specific documentation.
- `docs/GUI.md` for GUI architecture and run instructions.

### Documentation
- Consolidated long-form markdown under `docs/` and kept only `README.md` at
  the repository root.
- Rewrote markdown links to use GitHub-friendly relative paths.
- Updated operator and installation guides to match the current command set and
  repository layout.
- Documented that large Seattle `seattle.osm` and `seattle.net.xml` artifacts
  are generated locally and are not committed to Git history.
- Added a dedicated GUI guide that explains the first-version architecture,
  screens, execution model, and CLI relationship.

### Verification
- Verified the restructured repository with `pytest -q tests` (`92 passed`).
- Verified the main CLI entry points with:
  - `PYTHONPATH=src python -m sas.simulation.runner --help`
  - `PYTHONPATH=src python -m sas.analysis.resilience_assessment --help`

## [0.1.0] - 2026-03-08

### Added
- Initial release of SUMO Accident Simulation (SAS) for the AntifragiCity research project
- Probabilistic risk model based on the Nilsson Power Model (speed, variance, density, road-type components)
- Four-tier severity classification (MINOR / MODERATE / MAJOR / CRITICAL) per NHTSA KABCO
- Realistic accident lifecycle management (ACTIVE → CLEARING → RESOLVED) with log-normal duration distributions
- Antifragility Index: quantifies whether the network recovers and adapts post-disruption
- Macroscopic Fundamental Diagram (MFD) analysis with theoretical Greenshields curves per incident level
- One-click resilience assessment CLI (`sas-assess`) with 105-scenario and 12-scenario quick modes
- HTML resilience report with optional Claude AI narrative analysis
- Parallel batch runner with live heartbeat and ETA tracking
- Publication-quality visualisations: MFD scatter plots, density-binned speed deficit, CoV degradation figures
- Bundled networks: Sioux Falls (benchmark), Thessaloniki (real OSM), Seattle (real OSM), Riverside District (synthetic)
- Network generators for all four bundled networks using OSM + SUMO netconvert
- Greek government data ingestion toolchain (`sas-fetch-govgr`, `sas-build-govgr-targets`)
- Real-world validation against Seattle crash data (`compare_seattle_real.py`)
- Batch analysis and comparative visualisation dashboard (`analyse_batch.py`)
- Config-driven parameters via YAML configuration files with full inline documentation
- Seeded, reproducible runs with `metadata.json` output per run
- Comprehensive test suite (55+ unit tests, TraCI fully mocked — no SUMO required)
- GitHub Actions CI: Ruff linting, Mypy type-checking, Pytest across Python 3.10–3.12
- PEP 621 packaging for the SAS command-line interface
- Makefile for common development and operator workflows
- Operator guide for Thessaloniki deployment
