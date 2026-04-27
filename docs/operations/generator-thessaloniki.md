# Command: `sas-generate-thessaloniki`

Build the Thessaloniki SUMO network and demand set from OpenStreetMap.

## What It Does

1. downloads the OSM road network from Overpass
2. converts it to SUMO with `netconvert`
3. generates demand with `randomTrips.py`
4. writes `thessaloniki.sumocfg`
5. optionally patches a YAML config to point at the generated files

## Main Inputs

- built-in Thessaloniki bounding box
- SUMO typemap and `randomTrips.py`
- optional `--period` to change synthetic demand intensity

## Main Outputs

Under `data/cities/thessaloniki/network/`:

- `thessaloniki.osm`
- `thessaloniki.net.xml`
- `thessaloniki.rou.xml`
- `thessaloniki.sumocfg`

## Typical Usage

Generate the default network and demand:

```bash
sas-generate-thessaloniki
```

Generate and patch the default config:

```bash
sas-generate-thessaloniki --update-config
```

Generate a lighter-demand route set:

```bash
sas-generate-thessaloniki --period 2.0
```

## Operational Notes

- Demand is synthetic because this workflow uses `randomTrips.py`.
- Use this generator for reproducible network builds, not for observed O-D
  demand calibration.
- After generation, run [`simulation-runner.md`](simulation-runner.md).
