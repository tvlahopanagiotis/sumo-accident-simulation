# Command: `sas-generate-seattle`

Build the Seattle SUMO network and demand from the local Seattle bundle.

## What It Does

1. reads the Seattle OSM extract and bundle files
2. converts the network with `netconvert`
3. generates demand from either Seattle OD inputs or random trips
4. writes `seattle.sumocfg`
5. optionally patches a YAML config

## Main Inputs

- `data/cities/seattle/bundle/.../Seattle.osm`
- `Seattle_od.csv`
- `Seattle_node.csv`
- optional demand-source and scaling arguments

## Main Outputs

Under `data/cities/seattle/network/`:

- `seattle.osm`
- `seattle.net.xml`
- `seattle.rou.xml`
- `seattle.sumocfg`

## Typical Usage

Generate the default Seattle network and patch the config:

```bash
sas-generate-seattle --update-config --config configs/seattle/default.yaml
```

Use OD-based demand at a different scale:

```bash
sas-generate-seattle --demand-source od --od-scale 0.02
```

Use random demand instead:

```bash
sas-generate-seattle --demand-source random --period 1.5
```

## Operational Notes

- Large Seattle `seattle.osm` and `seattle.net.xml` artifacts are generated
  locally and are not kept in git.
- If the source OSM extract is missing, fetch it first with
  [`data-osm.md`](data-osm.md).
- See [`../SEATTLE_DATA.md`](../SEATTLE_DATA.md) for bundle-specific notes.
