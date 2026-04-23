# Changelog

All notable changes to this project will be documented in this file.

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

### Added
- Package entry points for simulation, assessment, generators, OSM downloads,
  govgr ingestion, and analyst tooling.
- `docs/STRUCTURE.md` as the canonical repository layout reference.
- `docs/REFERENCE.md` and `docs/SEATTLE_DATA.md` for operational and
  dataset-specific documentation.

### Documentation
- Consolidated long-form markdown under `docs/` and kept only `README.md` at
  the repository root.
- Rewrote markdown links to use GitHub-friendly relative paths.
- Updated operator and installation guides to match the current command set and
  repository layout.
- Documented that large Seattle `seattle.osm` and `seattle.net.xml` artifacts
  are generated locally and are not committed to Git history.

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
