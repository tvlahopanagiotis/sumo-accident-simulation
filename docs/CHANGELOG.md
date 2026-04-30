# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-04-30

### Changed
- Renamed the visible application and API branding from AntifragiCity SAS to
  AntifragiCity SUMA, while keeping the existing `sas` Python package namespace
  and CLI commands for compatibility.
- Added preferred `suma-*` command aliases for the main simulator, assessment,
  data, generator, analysis, and GUI API entry points.
- Reworked the Overview page into a more useful operator dashboard with current
  workspace counts, pipeline shortcuts, active-job state, and recent run links.
- Reworked Config Studio with a two-step config picker: target folder first,
  config file second.
- Moved Config Studio validate/delete/save actions into a dedicated action bar
  below the open/create controls.
- Kept Config Studio model families as secondary tabs and expanded the
  resilience-assessment model guide so demand levels are explained as a
  variable stress ladder with examples.
- Renamed the generator area to `OD Generators` and split the city-generator
  fields according to the selected demand source.
- Reworked simulator and analysis pages into tabbed tool groups with expanded
  guide dialogs.
- Reworked the Results page from a tree-first browser into a run registry with
  city/date/search filters and tabbed interactive summaries.
- Improved Jobs so logs are scrollable, live/report media is better centered,
  progress parsing reads simulation step logs, and completed job history can be
  cleared or removed item-by-item.

### Added
- Result export endpoints for selected runs:
  - summary JSON
  - `network_metrics.csv`
  - full run artifact ZIP
- Result deletion endpoint for removing a selected run folder from `results/`.
- Persistent GUI job history at `results/gui_job_history.json` for completed
  job records.
- OD input preview now includes per-zone origin and destination demand totals,
  and the map/table can focus on a selected zone.
- Random-demand preview now estimates requested trips from route period and end
  time so operators can judge run weight before generating.
- Rhoé partner logo in the GUI footer.
- SUMA project-development context added to the in-app documentation library.

### Documentation
- Updated the README, GUI guide, generator module guide, generator operations
  guide, documentation index, and changelog for the SUMA naming and 0.3.0 UI/API
  behavior.

## [0.2.4] - 2026-04-28

### Changed
- Reworked the generator layer around a generic city generator that can build
  SUMO assets for any extracted city under `data/cities/<slug>/`, while
  keeping Sioux Falls and Riverside as separate benchmark/synthetic workflows.
- Reworked the generator page UI into family tabs (`City`, `Benchmark`,
  `Synthetic`) plus task tabs (`Build`, `View Inputs`) so it follows the same
  operator flow pattern as the rest of the console.
- Expanded the generator GUI with more informative field-level help and a
  `View Inputs` tab for OD support-file inspection.
- Removed duplicated inline help text under workflow fields where the same
  explanation is already available through the `?` tooltip, and tightened the
  generator-page narrative around demand selection.
- Reworked the GUI `Traffic Feeds` page into a two-stage path:
  - `New Feed Pull`
  - `Exported Feeds`
- Adjusted the gov.gr feed workflow so target-city defaults no longer clear a
  manually chosen downloader output path, and the downloader now treats a
  `.../downloads` path as a city downloads root that receives timestamped run
  folders.
- Reworked the gov.gr preview model so published catalogs can come from one
  source integration while download runs, target exports, and feed-to-OSM
  alignment are inspected against a different target city folder.
- Cleaned up the traffic-feed page layout so export summaries stay in a left
  operator column while the feed-alignment map and export browser share the
  right working column.
- Made the traffic-feed export browser independently scrollable so large export
  trees and file previews stay usable inside the page.
- Polished the `Data & Integrations` UI so the OSM and traffic-feed pages use a
  cleaner operator flow, more consistent controls, and more informative
  summaries.
- Refined the OSM extracted-network editor with more granular road grouping,
  multi-group road filters, and destructive cleanup controls for raw OSM
  pruning.
- Reworked OSM extraction from a roads-only toggle into an explicit SUMO-leaning
  road-type selector with `all-features` kept as an advanced override.
- Improved `Config Studio` so `sumo.config_file` can be typed manually for a
  newly bootstrapped city while still offering discovered `.sumocfg`
  suggestions, and the GUI now refreshes config and `.sumocfg` discovery as
  jobs create new files.

### Added
- A new `sas-generate-city` CLI and matching GUI workflow that supports either
  `randomTrips` or OD-driven demand generation for extracted city folders.
- A city demand preview API and GUI viewer that can show OD samples and top
  centroid-to-centroid flows on a map when a city has compatible OD inputs.
- Backend discovery of published Thessaloniki feed bundles, downloader export
  runs, and built target folders under `data/cities/<city>/govgr/`.
- OSM bootstrap of `data/cities/<city>/govgr/downloads/` and
  `data/cities/<city>/govgr/targets/` alongside the network and config
  scaffold.
- A feed-alignment map that overlays the subset of feed `Link_id` values that
  currently match OSM way ids in the selected city extract.
- Feed-alignment views for:
  - current speed
  - congestion
  - coverage
- Provider workflow-slot metadata in the traffic-feed layer so future
  city-specific adapters can fit the same page structure.
- Raw OSM cleanup actions in `Extracted Network`, including drag-box selection
  and selected-road deletion.

### Documentation
- Updated the GUI and data-integration docs to describe the new traffic-feed
  workflow, export artifacts, partial feed-to-OSM alignment behavior, and the
  more capable extracted-network editing flow.
- Added a pilot-city traffic-data findings note to support later feed
  integration planning.

## [0.2.3] - 2026-04-28

### Changed
- Expanded run outputs so the simulator now exports richer time-series and
  summary artifacts for downstream analysis, including
  `simulation_summary.json`.
- Reworked the GUI Results page into an interactive dashboard with
  axis-labelled charts, legends, tooltips, brushing, incident timelines, and
  richer accident-impact views.
- Reworked the OSM workflow into a two-stage GUI path:
  - `New Extract`
  - `Extracted Network`
- Tightened the OSM page layout so boundary controls stay with the map and the
  extraction/config bootstrap controls stay in a separate operator panel.
- Added an extracted-network review step before generation, so operators can
  inspect and clean the raw `.osm` input instead of treating it as opaque.
- Added reusable nested-tab styling to distinguish secondary workflow tabs from
  top-level page tabs.

### Added
- Automatic city bootstrap during OSM fetch:
  - `data/cities/<city>/network/`
  - `configs/<city>/default.yaml`
  - `configs/templates/city_default.yaml`
  - city metadata for GUI discovery
- A GUI-discoverable extracted-city browser driven by the saved city `.osm`
  files.
- OSM network preview modes for:
  - speed limits
  - road type
  - lanes and direction
  - signalized intersections
- Road-selection editing in the extracted-network page:
  - single-click selection
  - Shift multi-selection
  - bulk filter-based selection
  - expansion to the connected same-name road
- Backend API support for writing edited OSM `maxspeed` tags back to the city
  extract.

### Documentation
- Updated the GUI and operations docs to describe the new OSM bootstrap,
  extracted-network editor, and pre-generation input-cleanup workflow.

### Verification
- Verified the frontend with `npm run build`.
- Verified the full test suite with `pytest -q` (`104 passed`).

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
- Reorganized the long-form documentation into a more explicit reading path:
  start here, foundations, module guides, command guides, city notes, and
  maintenance.
- Reworked the GUI `Documentation` page grouping so it now follows the same
  curated structure instead of a flatter topic bucket layout.
- Reduced documentation fragmentation by consolidating the operational markdown
  files into fewer workflow guides.
- Made the documentation library panel and the open document pane scroll
  independently in the GUI.
- Reordered the GUI sidebar and documentation command-guide sections to follow
  the recommended workflow order more closely.
- Surfaced the release version in the GUI footer and aligned the frontend,
  backend, and package metadata to `0.2.2`.

### Added
- New accident config controls:
  - `incident_effect_mode`
  - `reroute_affected_vehicles`
  - `reroute_radius_m`
  - `reroute_interval_s`
- An operations documentation set under `docs/operations/`.
- New module-overview guides for:
  - simulator
  - generators
  - data and integrations
  - analysis
- Improvement-recommendation sections under each module guide, based on the
  current scripts and architecture.
- A new end-to-end workflow guide for adding a brand-new location and taking it
  through setup, generation, simulation, and analysis.

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
