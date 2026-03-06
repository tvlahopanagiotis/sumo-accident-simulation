# Seattle Data Bundle

This folder contains Seattle data downloaded on 2026-03-06 for OD/TAZ work and crash analysis.

## Folder layout

- `source/20_us_cities.zip`
  - Full upstream archive from Figshare (MD5 verified).
- `traffic_dataset/02_Seattle/`
  - Seattle subset extracted from the 20-city traffic dataset.
  - Includes:
    - `01_Input_data/Seattle_od.csv` (OD table)
    - `01_Input_data/Seattle_node.csv` (nodes / zones)
    - `01_Input_data/Seattle_link.csv` (links)
    - `02_TransCAD_results/od.mtx`
    - `03_AequilibraE_results/od_demand.aem`
    - plus supporting shapefiles and assignment outputs.
- `crash_data/`
  - `sdot_collisions_all_years.csv`
  - `sdot_collisions_all_years.geojson`

## Upstream sources

- Traffic dataset: https://figshare.com/articles/dataset/A_unified_and_validated_traffic_dataset_for_20_U_S_cities/24235696
- Seattle collisions dataset page: https://data-seattlecitygis.opendata.arcgis.com/datasets/SeattleCityGIS::sdot-collisions-all-years
- Seattle collisions direct CSV endpoint:
  - https://data-seattlecitygis.opendata.arcgis.com/api/download/v1/items/504838adcb124cf4a434e33bf420c4ad/csv?layers=0
- Seattle collisions direct GeoJSON endpoint:
  - https://data-seattlecitygis.opendata.arcgis.com/api/download/v1/items/504838adcb124cf4a434e33bf420c4ad/geojson?layers=0

## Integrity check

- `source/20_us_cities.zip` MD5:
  - `3f7632e00599588abecbcfc488f862b2`

