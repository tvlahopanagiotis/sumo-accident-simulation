# Data And Integrations Module

This guide covers the workflows that bring external data into the repository
before network generation, calibration, or validation.

## Scope

The main integration code lives in:

- `src/sas/integrations/`

The current workflows cover:

- OSM place-based extraction
- govgr traffic feed download
- govgr target building

## What This Module Does

The data and integrations layer is responsible for controlled intake of
external data sources.

Its role is to:

- locate and fetch source data
- normalize it into local repository paths
- prepare analysis-ready targets for calibration or validation

## Current Integration Paths

### OSM acquisition

The OSM workflow uses place search and bounding boxes to produce `.osm`
extracts that later feed the city generators.

### govgr traffic feeds

The govgr workflow downloads traffic feed files and then converts them into
target datasets used by Thessaloniki-oriented calibration and validation work.

## Where It Fits In The Full Workflow

This module is the earliest stage in the full SAS pipeline.

Typical chain:

1. fetch raw data
2. normalize or build targets
3. generate network and demand
4. run simulation
5. run analysis or validation

## Related Command Guides

- [`../operations/data-integrations.md`](../operations/data-integrations.md)
- [`../operations/new-location-workflow.md`](../operations/new-location-workflow.md)

## Recommendations For Improvement

### Introduce explicit provenance metadata

Every ingestion workflow should produce a small provenance record that stores:

- source URL or endpoint
- query or bounding box
- download time
- transformation parameters
- output file inventory

### Formalize schemas for downloaded and derived data

The integration layer would be easier to maintain if raw inputs, normalized
tables, and calibration targets each had documented schemas and lightweight
validation checks.

### Generalize beyond Thessaloniki-specific feed assumptions

The current govgr path is operationally useful, but still strongly shaped by
the Thessaloniki feed context. A cleaner adapter model would make future city
integrations easier.

### Add quality checks before downstream use

Before generator or analysis workflows consume these inputs, the integration
module should verify file completeness, date coverage, and critical field
presence more explicitly.
