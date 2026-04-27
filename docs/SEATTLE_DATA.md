# Seattle Data And Network Notes

This file is the Seattle-specific data note. It describes the bundle and local
artifact policy. It is not the primary command reference.

For CLI workflow steps, use:

- [`operations/generators.md`](operations/generators.md)
- [`operations/data-integrations.md`](operations/data-integrations.md)
- [`operations/analysis.md`](operations/analysis.md)

## Data Bundle

The Seattle bundle lives under:

`data/cities/seattle/bundle/`

It contains the OD / TAZ / validation material used for Seattle network
generation and comparison workflows.

### Bundle layout

- `source/20_us_cities.zip`
  - upstream archive from Figshare
- `traffic_dataset/02_Seattle/`
  - `01_Input_data/Seattle_od.csv`
  - `01_Input_data/Seattle_node.csv`
  - `01_Input_data/Seattle_link.csv`
  - `02_TransCAD_results/...`
  - `03_AequilibraE_results/...`
- `crash_data/`
  - `sdot_collisions_all_years.csv`
  - `sdot_collisions_all_years.geojson`

### Upstream sources

- Traffic dataset:
  https://figshare.com/articles/dataset/A_unified_and_validated_traffic_dataset_for_20_U_S_cities/24235696
- Seattle collisions dataset:
  https://data-seattlecitygis.opendata.arcgis.com/datasets/SeattleCityGIS::sdot-collisions-all-years

## Seattle Network

The runnable Seattle SUMO setup lives under:

`data/cities/seattle/network/`

### Files

- `seattle.rou.xml`
- `seattle.sumocfg`

Generated locally, not committed:

- `seattle.osm`
- `seattle.net.xml`

Those two files exceed GitHub's normal file size limit in this repository, so
the committed Seattle setup keeps only the smaller reproducible artifacts under
version control. Regenerate them locally with `sas-generate-seattle`.

## Typical Workflow

Generate the Seattle network and demand:

```bash
sas-generate-seattle --update-config --config configs/seattle/default.yaml
```

If the source extract is missing, fetch it first:

```bash
sas-fetch-osm \
  --place "Seattle, Washington, USA" \
  --out data/cities/seattle/bundle/traffic_dataset/02_Seattle/01_Input_data/Seattle.osm
```

Run the simulation:

```bash
sas --config configs/seattle/default.yaml
```

Variants:

```bash
sas-generate-seattle --demand-source od --od-scale 0.02
sas-generate-seattle --demand-source random --period 1.5
sas-generate-seattle --skip-network --demand-source od
```
