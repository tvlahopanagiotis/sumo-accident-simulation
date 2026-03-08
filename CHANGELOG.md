# Changelog

All notable changes to this project will be documented in this file.

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
- Config-driven parameters via `config.yaml` with full inline documentation
- Seeded, reproducible runs with `metadata.json` output per run
- Comprehensive test suite (55+ unit tests, TraCI fully mocked — no SUMO required)
- GitHub Actions CI: Ruff linting, Mypy type-checking, Pytest across Python 3.10–3.12
- PEP 621 packaging with four CLI entry points (`sas`, `sas-assess`, `sas-fetch-govgr`, `sas-build-govgr-targets`)
- Makefile for common development and operator workflows
- Operator guide for Thessaloniki deployment
