# Documentation

This directory is the curated long-form documentation for SUMA. The repository
root [README.md](../README.md) remains the public project overview and quick
start, while this index is the internal reading map for operators and
developers.

From version `0.3.1`, the visible application name is AntifragiCity SUMA
(Simulator for Urban Mobility Antifragility). The canonical Python package is
`suma`; the older `sas` namespace remains only as a compatibility shim.

All links in this directory use repository-relative paths so they render
correctly on GitHub and in the in-app documentation page.

## Recommended Reading Paths

### 1. Orientation

Read these first if you are new to the repository or the GUI:

1. [../README.md](../README.md) for the project overview and quick start
2. [GUI.md](GUI.md) for the interface model and page responsibilities
3. [STRUCTURE.md](STRUCTURE.md) for the canonical repository layout
4. [REFERENCE.md](REFERENCE.md) for parameters, outputs, and core semantics

Then return to this file when you need the workflow order.

### 2. Operator Workflow

Use this path when your goal is to run the system end-to-end:

1. [operations/README.md](operations/README.md) for the consolidated workflow index
2. [operations/new-location-workflow.md](operations/new-location-workflow.md) for a full new-city path
3. [operations/data-integrations.md](operations/data-integrations.md) for source-data intake
4. [operations/generators.md](operations/generators.md) for network and demand generation
5. [operations/simulation.md](operations/simulation.md) for SUMA execution
6. [operations/analysis.md](operations/analysis.md) for post-processing and resilience outputs
7. [THESSALONIKI_OPERATOR_GUIDE.md](THESSALONIKI_OPERATOR_GUIDE.md) for the shortest city-specific Thessaloniki runbook

### 3. System Understanding

Use this path when you want the architectural and modeling story behind the
operator workflow:

1. [modules/README.md](modules/README.md) for the module-level map
2. [modules/data-integrations.md](modules/data-integrations.md) for external-data intake and target preparation
3. [modules/generators.md](modules/generators.md) for network and OD generation behavior
4. [modules/simulator.md](modules/simulator.md) for runtime, incidents, and outputs
5. [modules/analysis.md](modules/analysis.md) for resilience, sweep, and validation workflows

### 4. Research, Validation, And Project Context

Read these when you need city-specific caveats, scientific limits, or project
context:

- [SEATTLE_DATA.md](SEATTLE_DATA.md):
  Seattle bundle, local large artifacts, and Seattle-specific workflow notes
- [SUMO_ACCIDENT_SIMULATOR_REVIEW.md](SUMO_ACCIDENT_SIMULATOR_REVIEW.md):
  technical review of the current incident model and its limitations
- [antifragicity/SUMA_Codex_Development_Instructions.md](antifragicity/SUMA_Codex_Development_Instructions.md):
  internal SUMA WP5 development context, Mini-GA preparation guide, roadmap, and partner-dependency register
- [antifragicity/SUMA_MiniGA_Diplomatic_Brief.md](antifragicity/SUMA_MiniGA_Diplomatic_Brief.md):
  partner-facing talking-point brief derived from the internal SUMA guide

### 5. Environment And Maintenance

Use these when you are setting up the repo or maintaining it:

- [MACOS_INSTALL.md](MACOS_INSTALL.md):
  validated macOS setup notes
- [WORKTREES.md](WORKTREES.md):
  practical guide to git worktrees in this repo
- [CHANGELOG.md](CHANGELOG.md):
  release history

## Narrative Summary

The documentation is intentionally consolidated around a simple progression:
understand the product, run the workflow, study the system internals, then
consult research notes and maintenance material as needed. The in-app
documentation page follows the same order so operators do not have to jump
between conceptual, operational, and maintenance material without a clear
narrative.
