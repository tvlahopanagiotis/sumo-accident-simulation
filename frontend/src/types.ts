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
  project: string;
  colors: Record<string, string>;
  logo_path: string;
  favicon_path: string;
  font_note: string;
};

