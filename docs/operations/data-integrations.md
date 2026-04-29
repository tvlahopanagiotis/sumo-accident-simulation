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
5. by default bootstraps:
   - `data/cities/<city>/network/`
   - `data/cities/<city>/govgr/downloads/`
   - `data/cities/<city>/govgr/targets/`
   - `configs/<city>/default.yaml`
   - city metadata for GUI discovery

Typical usage:

```bash
sas-fetch-osm \
  --place "Seattle, Washington, USA" \
  --out data/cities/seattle/bundle/traffic_dataset/02_Seattle/01_Input_data/Seattle.osm
```

```bash
sas-fetch-osm \
  --place "Athens, Greece" \
  --city-slug athens \
  --pad-km 1.5
```

Useful options:

- `--city-slug`
  - sets the folder name under `data/cities/` and `configs/`
- `--road-types`
  - selects the OSM `highway=*` classes to include
  - the default set focuses on road classes that are generally useful for SUMO
    network building
- `--all-features`
  - advanced override
  - when enabled, the explicit road-type filter is bypassed and the full raw
    OSM node / way / relation set is downloaded
- `--bootstrap-layout` / `--bootstrap-config`
  - keep both on for normal new-city setup
- `--nominatim-url`
  - place lookup endpoint
- `--overpass-url`
  - OSM XML extraction endpoint

GUI workflow:

- `New Extract`
  - search and bootstrap a new city
  - adjust the boundary
  - choose the road classes to extract
  - launch the OSM fetch
- `Extracted Network`
  - inspect the saved `.osm`
  - view speed tags, road type, lanes/direction, and signalized intersections
  - select roads individually, by filter, or with drag-box selection
  - combine multiple road-group filters
  - edit speed limits before running a generator
  - delete selected road segments from the raw `.osm` when needed

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

What it writes:

- `raw/`
  - untouched realtime pages or FTP files
- `clean/`
  - merged and deduplicated realtime CSV tables
- `baselines/`
  - hourly baseline summaries derived from realtime pulls
- `historical/`
  - extracted historical archives when extraction is enabled
- `quality_report.json`
  - machine-readable run summary for the download/export job

Output-directory behavior:

- if `--output-dir` is omitted, the downloader writes under
  `data/cities/thessaloniki/govgr/downloads/<timestamp>/`
- if `--output-dir` points to a city downloads root such as
  `data/cities/thermi/govgr/downloads`, the downloader creates a timestamped
  run folder under that root
- if `--output-dir` points to a more specific folder, that exact folder is used

## govgr Targets

### `sas-build-govgr-targets`

Build SAS-ready calibration and validation targets from downloaded govgr data.

Typical usage:

```bash
sas-build-govgr-targets \
  --downloads-root data/cities/thessaloniki/govgr/downloads \
  --calibration-year 2025 \
  --validation-year 2026 \
  --output-dir data/cities/thessaloniki/govgr/targets/calibration_2025_validation_2026
```

What it writes:

- `calibration_speed_network_hourly.csv`
- `calibration_speed_network_weekpart_hourly.csv`
- `calibration_speed_link_direction_hourly_mean.csv`
- `calibration_travel_time_network_hourly.csv`
- `calibration_travel_time_network_weekpart_hourly.csv`
- `calibration_travel_time_path_hourly_mean.csv`
- matching `validation_...` files
- `targets_summary.json`
  - machine-readable summary of the built target set

GUI workflow:

- `New Feed Pull`
  - choose the feed source separately from the target city folder
  - run the Thessaloniki downloader and target builder
  - review provider workflow slots that make future city adapters easier to add
  - keep the derived `downloads_root` and target export paths, or override them
    manually when needed
- `Exported Feeds`
  - inspect the published feed catalogs from the selected source integration
  - inspect downloaded run folders and built target exports from the selected
    target city folder
  - browse exported files from the GUI
  - view the matched subset of feed `Link_id` values on top of the selected
    target city OSM extract

## Operational Notes

- OSM fetching prepares generator inputs. It does not build a SUMO network by itself.
- The govgr path is still effectively Thessaloniki-oriented in the current repository.
- The target city folder can now differ from the feed source. This is useful
  for alternate Thessaloniki network variants, but cross-network reuse should
  be validated carefully.
- Feed-to-OSM alignment is currently partial. The feed map only shows links whose
  `Link_id` matches a way id in the selected city `.osm` extract.

## Related Docs

- [`../modules/data-integrations.md`](../modules/data-integrations.md)
- [`new-location-workflow.md`](new-location-workflow.md)
- [`../PILOT_CITY_TRAFFIC_DATA_FINDINGS.md`](../PILOT_CITY_TRAFFIC_DATA_FINDINGS.md)
