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
