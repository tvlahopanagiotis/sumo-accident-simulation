# Data And Integrations Operations

This guide covers the commands that fetch external data and prepare it for SAS.

## OSM Extracts

### `sas-fetch-osm`

Download a raw OSM XML extract for a named place.

What it does:

1. resolves the place name with Nominatim
2. derives or expands a bounding box
3. downloads OSM XML through Overpass
4. writes a local `.osm` file

Typical usage:

```bash
sas-fetch-osm \
  --place "Seattle, Washington, USA" \
  --out data/cities/seattle/bundle/traffic_dataset/02_Seattle/01_Input_data/Seattle.osm
```

```bash
sas-fetch-osm \
  --place "Athens, Greece" \
  --out data/cities/athens/network/athens.osm \
  --pad-km 1.5
```

## govgr Downloads

### `sas-fetch-govgr`

Download govgr traffic feeds used for Thessaloniki calibration and validation.

Typical usage:

```bash
sas-fetch-govgr \
  --source realtime \
  --dataset all \
  --output-dir data/cities/thessaloniki/govgr/downloads/realtime_latest
```

```bash
sas-fetch-govgr \
  --source historical \
  --dataset speed \
  --historical-pattern _2025 \
  --output-dir data/cities/thessaloniki/govgr/downloads/historical_2025
```

## govgr Targets

### `sas-build-govgr-targets`

Build SAS-ready calibration and validation targets from downloaded govgr data.

Typical usage:

```bash
sas-build-govgr-targets \
  --downloads-root data/cities/thessaloniki/govgr/downloads \
  --calibration-year 2025 \
  --validation-year 2026 \
  --output-dir data/cities/thessaloniki/govgr/targets/post_metro_2025_2026
```

## Operational Notes

- OSM fetching prepares generator inputs. It does not build a SUMO network by itself.
- The govgr path is still effectively Thessaloniki-oriented in the current repository.

## Related Docs

- [`../modules/data-integrations.md`](../modules/data-integrations.md)
- [`new-location-workflow.md`](new-location-workflow.md)
