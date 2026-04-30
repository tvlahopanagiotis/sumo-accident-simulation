# Documentation

This directory contains the long-form project documentation. The root
`README.md` is intentionally kept as the high-level overview and quick-start.
From version `0.3.0`, the visible application name is AntifragiCity SUMA
(Simulator for Urban Mobility Antifragility). The internal Python package still
uses the historical `sas` namespace for compatibility.

All links in this directory use repository-relative paths so they render
correctly on GitHub.

## Start Here

- [README.md](../README.md):
  root project overview and quick start
- [modules/README.md](modules/README.md):
  narrative path through the main SUMA modules
- [operations/README.md](operations/README.md):
  consolidated workflow index
- [antifragicity/SUMA_Codex_Development_Instructions.md](antifragicity/SUMA_Codex_Development_Instructions.md):
  project and WP5 development context for SUMA

## Module Guides

- [modules/simulator.md](modules/simulator.md):
  runtime, incident logic, outputs, and simulator priorities
- [modules/generators.md](modules/generators.md):
  network and demand generation workflows
- [modules/data-integrations.md](modules/data-integrations.md):
  external data intake and target preparation
- [modules/analysis.md](modules/analysis.md):
  resilience, batch, sweep, and validation workflows

## Workflow Guides

- [THESSALONIKI_OPERATOR_GUIDE.md](THESSALONIKI_OPERATOR_GUIDE.md):
  shortest city-specific Thessaloniki runbook

## Reference

- [REFERENCE.md](REFERENCE.md):
  parameters, formulas, outputs, and core simulator semantics
- [SUMO_ACCIDENT_SIMULATOR_REVIEW.md](SUMO_ACCIDENT_SIMULATOR_REVIEW.md):
  technical review of the current incident model and its limitations
- [STRUCTURE.md](STRUCTURE.md):
  canonical repository layout

## Platform And Interface

- [GUI.md](GUI.md):
  GUI architecture, pages, and backend/frontend responsibilities
- [MACOS_INSTALL.md](MACOS_INSTALL.md):
  validated macOS setup notes

## Data Notes

- [SEATTLE_DATA.md](SEATTLE_DATA.md):
  Seattle bundle, local large artifacts, and Seattle-specific workflow notes

## Maintenance

- [WORKTREES.md](WORKTREES.md):
  practical guide to git worktrees in this repo
- [CHANGELOG.md](CHANGELOG.md):
  release history
