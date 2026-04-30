export type WorkflowField = {
  name: string;
  label: string;
  type: string;
  required: boolean;
  default: unknown;
  help: string;
  placeholder: string | null;
  options: string[] | null;
};

export type CityDemandPreview = {
  city_slug: string;
  od_file: string | null;
  node_file: string | null;
  supported: boolean;
  issues: string[];
  summary: {
    zone_count: number;
    od_row_count: number;
    external_od_row_count: number;
    total_od: number;
    intrazonal_raw: number;
    missing_zone_count: number;
    mapped_top_flow_count: number;
  } | null;
  sample_rows: Array<{
    origin: string;
    destination: string;
    od_number: number;
    intrazonal: boolean;
  }>;
  top_flows: Array<{
    origin: string;
    destination: string;
    od_number: number;
    origin_coords: [number, number];
    destination_coords: [number, number];
  }>;
  nodes: Array<{
    zone_id: string;
    coords: [number, number];
  }>;
  zone_demands: Array<{
    zone_id: string;
    origin_demand: number;
    destination_demand: number;
    total_demand: number;
    coords: [number, number];
  }>;
};

export type WorkflowSpec = {
  id: string;
  category: string;
  title: string;
  description: string;
  module: string;
  progress_mode: string;
  fields: WorkflowField[];
  command_preview: string[];
};

export type ConfigDocument = {
  path: string;
  raw_yaml: string;
  config: Record<string, unknown>;
};

export type JobRecord = {
  id: string;
  workflow_id: string;
  title: string;
  payload: Record<string, unknown>;
  command: string[];
  status: string;
  progress: number | null;
  progress_label: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  return_code: number | null;
  output_dir: string | null;
  log_path: string | null;
  error: string | null;
  log_lines: string[];
  phase: string | null;
  live_progress_path: string | null;
  report_path: string | null;
  figures: string[];
};

export type TreeNode = {
  name: string;
  path: string;
  kind: "file" | "directory";
  children?: TreeNode[];
};

export type Branding = {
  name: string;
  colors: Record<string, string>;
  logo_path: string;
  favicon_path: string;
  eu_logo_path: string;
  rhoe_logo_path?: string;
  project_url: string;
  footer_disclaimer: string;
  copyright: string;
};

export type LocationSearchResult = {
  display_name: string;
  lat: number;
  lon: number;
  boundingbox: [number, number, number, number];
  country_code: string;
  country?: string;
  city?: string;
  state?: string;
  osm_type?: string;
  osm_id?: number;
  class?: string;
  type?: string;
  geojson?: Record<string, unknown> | null;
};

export type CityRecord = {
  slug: string;
  display_name: string;
  city_root: string;
  network_dir: string | null;
  osm_path: string | null;
  net_path: string | null;
  sumocfg_path: string | null;
  config_path: string | null;
  has_osm: boolean;
  has_network: boolean;
  metadata: Record<string, unknown>;
};

export type CityNetworkPreview = {
  city: CityRecord;
  source_path: string;
  bbox: [number, number, number, number] | null;
  stats: {
    feature_count: number;
    road_type_counts: Record<string, number>;
    with_speed_limit: number;
    with_lane_count: number;
    oneway_count: number;
    signalized_intersection_count: number;
  };
  features: Array<{
    id: string;
    name?: string | null;
    road_type: string;
    speed_kph: number | null;
    lane_count: number | null;
    oneway: boolean;
    reverse_oneway: boolean;
    node_ids: string[];
    coords: Array<[number, number]>;
  }>;
  intersections: Array<{
    id: string;
    coords: [number, number];
    connected_road_types: string[];
    connected_road_count: number;
  }>;
};

export type TrafficFeedSourceRecord = {
  slug: string;
  display_name: string;
  provider: string;
  provider_label: string;
  integration_stage: string;
  coverage_note: string;
  workflow_slots: Array<{
    id: string;
    title: string;
    status: string;
    description: string;
  }>;
  city_root: string;
  provider_root: string;
  catalog_count: number;
  download_run_count: number;
  target_export_count: number;
  metadata: Record<string, unknown>;
};

export type TrafficFeedPreview = {
  source: {
    slug: string;
    display_name: string;
    provider: string;
    provider_label: string;
    integration_stage: string;
    coverage_note: string;
    workflow_slots: Array<{
      id: string;
      title: string;
      status: string;
      description: string;
    }>;
    city_root: string;
    provider_root: string;
    metadata: Record<string, unknown>;
  };
  target_city: {
    slug: string;
    display_name: string;
    city_root: string;
    provider_root: string;
    metadata: Record<string, unknown>;
  };
  catalog_datasets: Array<{
    id: string;
    title: string;
    description: string;
    version: string | null;
    keywords: string[];
    path: string;
    datapackage_path: string;
    resources: Array<{
      title: string | null;
      format: string | null;
      path: string | null;
      description: string;
      source_urls: string[];
    }>;
    sample_csv: {
      path: string;
      delimiter: string;
      columns: string[];
      rows: Array<Record<string, string>>;
    } | null;
  }>;
  download_runs: Array<{
    name: string;
    path: string;
    quality_report_path: string | null;
    started_utc: string | null;
    finished_utc: string | null;
    args: Record<string, unknown>;
    datasets: Array<{
      name: string;
      realtime_rows_clean: number | null;
      realtime_pages_downloaded: number | null;
      baseline_files: string[];
      clean_csv: string | null;
      historical_files_downloaded: number | null;
      historical_extracted_dir: string | null;
    }>;
    files: string[];
  }>;
  target_exports: Array<{
    name: string;
    path: string;
    summary_path: string | null;
    calibration_year: number | null;
    validation_year: number | null;
    sets: Array<{
      name: string;
      files: string[];
      speed_meta: Record<string, unknown>;
      travel_time_meta: Record<string, unknown>;
    }>;
    files: string[];
  }>;
  linked_network: {
    bbox: [number, number, number, number] | null;
    stats: {
      network_feature_count: number;
      feed_speed_link_count: number;
      feed_congestion_link_count: number;
      matched_link_count: number;
      match_ratio: number;
      unmatched_link_count: number;
    };
    features: Array<{
      id: string;
      name?: string | null;
      road_type: string;
      speed_limit_kph: number | null;
      coords: Array<[number, number]>;
      oneway: boolean;
      reverse_oneway: boolean;
      speed_current_kph: number | null;
      congestion_level: string | null;
      latest_timestamp: string | null;
      direction_values: {
        speed: Record<string, Record<string, string>>;
        congestion: Record<string, Record<string, string>>;
      };
    }>;
  } | null;
};

export type ResultRunSummary = {
  run_root: string;
  metadata: Record<string, unknown>;
  summary: Record<string, unknown>;
  config_snapshot: Record<string, unknown>;
  metrics: {
    rows: Array<Record<string, number>>;
    series: Record<string, number[]>;
    stats: Record<string, number>;
  };
  accidents: {
    count: number;
    by_severity: Record<string, number>;
    max_duration_seconds: number;
    max_queue_length_vehicles: number;
    max_vehicles_affected: number;
    total_rerouted_vehicles: number;
    total_blocked_lanes: number;
    total_managed_lanes: number;
    items: Array<Record<string, unknown>>;
  } | null;
  antifragility: {
    antifragility_index: number | null;
    n_events_measured: number | null;
    total_accidents: number | null;
    std_dev: number | null;
    ci_95_low: number | null;
    ci_95_high: number | null;
    interpretation: string | null;
    per_event: Array<Record<string, unknown>>;
  } | null;
  simulation_summary: {
    network?: Record<string, unknown>;
    accidents?: Record<string, unknown>;
    antifragility?: Record<string, unknown> | null;
  } | null;
  artifacts: {
    report_path: string | null;
    image_paths: string[];
    raw_files: string[];
  };
};

export type ResultRunRegistryItem = {
  run_root: string;
  name: string;
  city: string;
  created_at: string | null;
  modified_at: number;
  config_file: string | null;
  output_folder: string | null;
  total_steps: number | null;
  step_length: number | null;
  seed: number | null;
  total_accidents: number | null;
  antifragility_index: number | null;
  mean_speed_kmh: number | null;
  has_accidents: boolean;
  has_antifragility: boolean;
  has_simulation_summary: boolean;
  raw_file_count: number;
};
