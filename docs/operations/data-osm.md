# Command: `sas-fetch-osm`

Download a raw OSM XML extract for a named place.

## What It Does

1. resolves the place name with Nominatim
2. derives or expands a bounding box
3. downloads OSM XML through Overpass
4. writes a local `.osm` file for later generator use

## Main Inputs

- `--place`
- `--out`
- optional padding or explicit bbox overrides

## Main Outputs

- a raw `.osm` file at the path you choose

## Typical Usage

Fetch Seattle for the bundled generator:

```bash
sas-fetch-osm \
  --place "Seattle, Washington, USA" \
  --out data/cities/seattle/bundle/traffic_dataset/02_Seattle/01_Input_data/Seattle.osm
```

Fetch a new city extract with extra padding:

```bash
sas-fetch-osm \
  --place "Athens, Greece" \
  --out data/cities/athens/network/athens.osm \
  --pad-km 1.5
```

## Operational Notes

- This command prepares data for generators. It does not build a SUMO network.
- The GUI `Data & Integrations` page uses the same general OSM acquisition idea
  for place search and boundary setup.
