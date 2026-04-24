import { useEffect, useMemo, useState } from "react";
import { GeoJSON, MapContainer, Polygon, Rectangle, TileLayer, useMap, useMapEvents } from "react-leaflet";
import type { ChangeEvent } from "react";
import type { LeafletMouseEvent } from "leaflet";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { CONFIG_SECTIONS, type ConfigFieldSpec, type ConfigSectionSpec } from "./configStudio";
import { api } from "./lib/api";
import type {
  Branding,
  ConfigDocument,
  JobRecord,
  LocationSearchResult,
  ResultRunSummary,
  TreeNode,
  WorkflowField,
  WorkflowSpec,
} from "./types";

type ViewKey =
  | "overview"
  | "configs"
  | "data_integrations"
  | "generators"
  | "simulations"
  | "analysis"
  | "jobs"
  | "results"
  | "documentation";
type ConfigMode = "structured" | "raw";
type ConfigSectionKey = ConfigSectionSpec["key"];
type BoundaryMode = "locality" | "bbox" | "shape";
type InfoModal = {
  title: string;
  sections: Array<{ heading: string; body: string[] }>;
} | null;

const DEFAULT_BRANDING: Branding = {
  name: "AntifragiCity SAS",
  colors: {
    primary: "#f93262",
    secondary: "#ffbea1",
    ink: "#29131b",
    surface: "#fff6f2",
    surface_alt: "#ffe7de",
    border: "#f2b9aa",
  },
  logo_path: "frontend/public/branding/antifragicity-logo-main-h.svg",
  favicon_path: "frontend/public/branding/antifragicity-favicon.svg",
  eu_logo_path: "frontend/public/branding/eu-funded-by-eu.png",
  project_url: "https://antifragicity.eu",
  footer_disclaimer:
    "This project has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement No. 101203052. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Climate, Infrastructure and Environment Executive Agency (CINEA). Neither the European Union nor the granting authority can be held responsible for them.",
  copyright: "© AntifragiCity. All rights reserved.",
};

const VIEW_LABELS: Array<{ key: ViewKey; label: string }> = [
  { key: "overview", label: "Overview" },
  { key: "configs", label: "Config Studio" },
  { key: "data_integrations", label: "Data & Integrations" },
  { key: "generators", label: "Generators" },
  { key: "simulations", label: "Simulations" },
  { key: "analysis", label: "Analysis" },
  { key: "jobs", label: "Jobs" },
  { key: "results", label: "Results" },
  { key: "documentation", label: "Documentation" },
];

const WORKFLOW_CATEGORY_BY_VIEW: Record<Exclude<ViewKey, "overview" | "configs" | "jobs" | "results" | "documentation">, string> = {
  data_integrations: "Data & Integrations",
  generators: "Generators",
  simulations: "Simulations",
  analysis: "Analysis",
};

function parseListValue(input: string, numeric: boolean): Array<string | number> {
  return input
    .split(/[\n, ]+/)
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => (numeric ? Number(item) : item));
}

function valueToListText(value: unknown): string {
  return Array.isArray(value) ? value.join("\n") : "";
}

function assetPath(source: string | undefined, fallback: string): string {
  if (!source) {
    return fallback;
  }
  const marker = "/public/";
  const normalized = source.replace(/\\/g, "/");
  if (normalized.includes(marker)) {
    return `/${normalized.split(marker)[1]}`;
  }
  return source.startsWith("/") ? source : fallback;
}

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "") || "location";
}

function formatNumber(value: unknown, digits = 2): string {
  const numeric = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(numeric)) {
    return "–";
  }
  return numeric.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits > 0 ? 0 : 0,
  });
}

function groupPathsByFolder(paths: string[], stripPrefix: string): Array<{ folder: string; items: string[] }> {
  const groups = new Map<string, string[]>();
  for (const path of paths) {
    const clean = path.startsWith(`${stripPrefix}/`) ? path.slice(stripPrefix.length + 1) : path;
    const segments = clean.split("/");
    const folder = segments.length > 1 ? segments.slice(0, -1).join(" / ") : "root";
    groups.set(folder, [...(groups.get(folder) ?? []), path]);
  }
  return Array.from(groups.entries())
    .map(([folder, items]) => ({ folder, items: items.sort() }))
    .sort((a, b) => a.folder.localeCompare(b.folder));
}

function buildConfigPath(folderChoice: string, customFolder: string, fileName: string): string {
  const folder = folderChoice === "__new__" ? customFolder.trim() : folderChoice;
  const cleanName = fileName.trim().replace(/\.ya?ml$/i, "");
  if (!folder || !cleanName) {
    return "";
  }
  return `configs/${folder}/${cleanName}.yaml`;
}

function normalizeGeoJsonCoordinates(input: unknown): Array<[number, number][]> {
  if (!input || typeof input !== "object") {
    return [];
  }
  const geometry = input as Record<string, unknown>;
  if (geometry.type === "Polygon" && Array.isArray(geometry.coordinates)) {
    return (geometry.coordinates as unknown[]).flatMap((ring) =>
      Array.isArray(ring)
        ? [
            (ring as unknown[])
              .map((point) => (Array.isArray(point) && point.length >= 2 ? [Number(point[1]), Number(point[0])] : null))
              .filter(Boolean) as [number, number][],
          ]
        : [],
    );
  }
  if (geometry.type === "MultiPolygon" && Array.isArray(geometry.coordinates)) {
    return (geometry.coordinates as unknown[]).flatMap((polygon) =>
      Array.isArray(polygon)
        ? (polygon as unknown[]).flatMap((ring) =>
            Array.isArray(ring)
              ? [
                  (ring as unknown[])
                    .map((point) => (Array.isArray(point) && point.length >= 2 ? [Number(point[1]), Number(point[0])] : null))
                    .filter(Boolean) as [number, number][],
                ]
              : [],
          )
        : [],
    );
  }
  return [];
}

function boundsFromPoints(points: Array<[number, number]>): [number, number, number, number] | null {
  if (points.length === 0) {
    return null;
  }
  const lats = points.map((point) => point[0]);
  const lngs = points.map((point) => point[1]);
  return [Math.min(...lats), Math.min(...lngs), Math.max(...lats), Math.max(...lngs)];
}

function topLevelFolders(paths: string[]): string[] {
  const values = new Set<string>();
  for (const path of paths) {
    const segments = path.split("/");
    if (segments.length >= 3 && segments[0] === "configs") {
      values.add(segments[1]);
    }
  }
  return Array.from(values).sort();
}

function BooleanSwitch({ checked, onChange }: { checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <label className="switch">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span className="switch-slider" />
    </label>
  );
}

function pathSegments(path: string): string[] {
  return path.split(".").filter(Boolean);
}

function getPathValue(root: unknown, path: string): unknown {
  return pathSegments(path).reduce<unknown>((current, segment) => {
    if (Array.isArray(current)) {
      return current[Number(segment)];
    }
    if (typeof current === "object" && current !== null) {
      return (current as Record<string, unknown>)[segment];
    }
    return undefined;
  }, root);
}

function cloneWithUpdate(value: unknown, path: string[], nextValue: unknown): unknown {
  if (path.length === 0) {
    return nextValue;
  }
  const [head, ...rest] = path;
  if (Array.isArray(value)) {
    const index = Number(head);
    const next = [...value];
    next[index] = cloneWithUpdate(next[index], rest, nextValue);
    return next;
  }
  const source = (value ?? {}) as Record<string, unknown>;
  return {
    ...source,
    [head]: cloneWithUpdate(source[head], rest, nextValue),
  };
}

function setPathValue(root: Record<string, unknown>, path: string, nextValue: unknown): Record<string, unknown> {
  return cloneWithUpdate(root, pathSegments(path), nextValue) as Record<string, unknown>;
}

function searchLocationOutputPath(location: LocationSearchResult): string {
  const place = slugify(location.city || location.state || location.country || location.display_name);
  return `data/cities/${place}/network/${place}.osm`;
}

function HelpBadge({ field }: { field: ConfigFieldSpec | WorkflowField }) {
  const hasContent = Boolean(
    ("help" in field ? field.help ?? "" : "") ||
      ("description" in field ? field.description : "") ||
      ("example" in field ? field.example ?? "" : "") ||
      ("impact" in field ? field.impact ?? "" : ""),
  );
  if (!hasContent) {
    return null;
  }
  return <span className="help-badge">?</span>;
}

function TooltipHelp({
  field,
}: {
  field: ConfigFieldSpec | WorkflowField;
}) {
  const description = "description" in field ? field.description : field.help ?? "";
  const example = "example" in field ? field.example : undefined;
  const impact = "impact" in field ? field.impact : undefined;
  if (!description && !example && !impact) {
    return null;
  }
  return (
    <span className="help-inline" tabIndex={0}>
      <HelpBadge field={field} />
      <span className="help-tooltip" role="tooltip">
        {description ? <strong>{description}</strong> : null}
        {example ? <span>Example: {example}</span> : null}
        {impact ? <span>Effect: {impact}</span> : null}
      </span>
    </span>
  );
}

function NumberListEditor({
  values,
  onChange,
}: {
  values: number[];
  onChange: (values: number[]) => void;
}) {
  const items = values.length ? values : [0];
  return (
    <div className="number-list-editor">
      {items.map((item, index) => (
        <div key={index} className="number-list-row">
          <input
            type="number"
            value={item}
            onChange={(event) => {
              const next = [...items];
              next[index] = Number(event.target.value);
              onChange(next);
            }}
          />
          <button
            type="button"
            className="secondary-button"
            onClick={() => onChange(items.filter((_, itemIndex) => itemIndex !== index))}
            disabled={items.length === 1}
          >
            Remove
          </button>
        </div>
      ))}
      <button type="button" className="secondary-button" onClick={() => onChange([...items, 0])}>
        Add Value
      </button>
    </div>
  );
}

function StructuredFieldInput({
  field,
  value,
  onChange,
  sumoConfigGroups,
  outputFolderGroups,
}: {
  field: ConfigFieldSpec;
  value: unknown;
  onChange: (value: unknown) => void;
  sumoConfigGroups: Array<{ folder: string; items: string[] }>;
  outputFolderGroups: Array<{ folder: string; items: string[] }>;
}) {
  return (
    <label className="field structured-field">
      <div className="field-heading">
        <span>{field.label}</span>
        <TooltipHelp field={field} />
      </div>
      {field.path === "sumo.config_file" ? (
        <select value={String(value ?? "")} onChange={(event) => onChange(event.target.value)}>
          {sumoConfigGroups.map((group) => (
            <optgroup key={group.folder} label={group.folder}>
              {group.items.map((path) => (
                <option key={path} value={path}>
                  {path.split("/").slice(-1)[0]}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      ) : field.path === "output.output_folder" ? (
        <select value={String(value ?? "")} onChange={(event) => onChange(event.target.value)}>
          {outputFolderGroups.map((group) => (
            <optgroup key={group.folder} label={group.folder}>
              {group.items.map((path) => (
                <option key={path} value={path}>
                  {path.split("/").slice(-1)[0]}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      ) : field.type === "boolean" ? (
        <BooleanSwitch checked={Boolean(value)} onChange={onChange} />
      ) : field.type === "choice" ? (
        <select value={String(value ?? field.options?.[0] ?? "")} onChange={(event) => onChange(event.target.value)}>
          {(field.options ?? []).map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      ) : field.type === "number" ? (
        <input
          type="number"
          value={value === undefined || value === null ? "" : String(value)}
          onChange={(event) => onChange(event.target.value === "" ? undefined : Number(event.target.value))}
        />
      ) : field.type === "number_list" ? (
        <NumberListEditor values={Array.isArray(value) ? (value as number[]) : []} onChange={onChange} />
      ) : (
        <input type="text" value={value === undefined || value === null ? "" : String(value)} onChange={(event) => onChange(event.target.value)} />
      )}
      <small>{field.description}</small>
    </label>
  );
}

function WorkflowInput({
  field,
  value,
  onChange,
  configPaths,
}: {
  field: WorkflowField;
  value: unknown;
  onChange: (value: unknown) => void;
  configPaths: string[];
}) {
  const placeholder = field.placeholder ?? field.help ?? "";
  return (
    <label className="field">
      <div className="field-heading">
        <span>{field.label}</span>
        <TooltipHelp field={field} />
      </div>
      {field.type === "boolean" ? (
        <BooleanSwitch checked={Boolean(value)} onChange={onChange} />
      ) : field.type === "choice" ? (
        <select value={String(value ?? field.default ?? "")} onChange={(event) => onChange(event.target.value)}>
          {(field.options ?? []).map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      ) : field.type === "config" ? (
        <select value={String(value ?? field.default ?? "")} onChange={(event) => onChange(event.target.value)}>
          {configPaths.map((path) => (
            <option key={path} value={path}>
              {path}
            </option>
          ))}
        </select>
      ) : field.type === "number" ? (
        <input
          type="number"
          value={value === undefined || value === null ? "" : String(value)}
          onChange={(event) => onChange(event.target.value === "" ? undefined : Number(event.target.value))}
          placeholder={placeholder}
        />
      ) : field.type === "number_list" ? (
        <NumberListEditor values={Array.isArray(value) ? (value as number[]) : []} onChange={onChange} />
      ) : (
        <input
          type="text"
          value={value === undefined || value === null ? "" : String(value)}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
        />
      )}
      {field.help ? <small>{field.help}</small> : null}
    </label>
  );
}

function MapViewport({
  bounds,
  center,
  mode,
}: {
  bounds: [number, number, number, number] | null;
  center: [number, number] | null;
  mode: BoundaryMode;
}) {
  const map = useMap();
  useEffect(() => {
    if (mode !== "shape" && bounds) {
      map.fitBounds(
        [
          [bounds[0], bounds[1]],
          [bounds[2], bounds[3]],
        ],
        { padding: [24, 24] },
      );
      return;
    }
    if (center) {
      map.setView(center, 12);
    }
  }, [map, bounds, center, mode]);
  return null;
}

function ShapeEditor({
  enabled,
  onAddPoint,
}: {
  enabled: boolean;
  onAddPoint: (point: [number, number]) => void;
}) {
  useMapEvents({
    click(event: LeafletMouseEvent) {
      if (!enabled) {
        return;
      }
      onAddPoint([event.latlng.lat, event.latlng.lng]);
    },
  });
  return null;
}

function TreeView({
  nodes,
  onSelect,
}: {
  nodes: TreeNode[];
  onSelect: (path: string) => void;
}) {
  return (
    <ul className="tree-list">
      {nodes.map((node) => (
        <li key={node.path}>
          <button className="tree-node" onClick={() => onSelect(node.path)}>
            <span>{node.kind === "directory" ? "▣" : "•"}</span>
            <span>{node.name}</span>
          </button>
          {node.children && node.children.length > 0 ? <TreeView nodes={node.children} onSelect={onSelect} /> : null}
        </li>
      ))}
    </ul>
  );
}

function SimpleLineChart({
  title,
  values,
  accentClass,
}: {
  title: string;
  values: number[];
  accentClass?: string;
}) {
  const width = 520;
  const height = 180;
  if (values.length < 2) {
    return (
      <div className="chart-card">
        <div className="chart-head">
          <h3>{title}</h3>
        </div>
        <p className="muted">Not enough data yet.</p>
      </div>
    );
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const points = values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * width;
      const y = height - ((value - min) / range) * height;
      return `${x},${y}`;
    })
    .join(" ");
  return (
    <div className="chart-card">
      <div className="chart-head">
        <h3>{title}</h3>
        <span>
          {formatNumber(min)} → {formatNumber(max)}
        </span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className={`chart-svg ${accentClass ?? ""}`} role="img" aria-label={title}>
        <polyline points={points} fill="none" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

function SeverityBars({ counts }: { counts: Record<string, number> }) {
  const entries = Object.entries(counts);
  const max = Math.max(...entries.map(([, value]) => value), 1);
  return (
    <div className="severity-bars">
      {entries.map(([label, value]) => (
        <div key={label} className="severity-row">
          <strong>{label}</strong>
          <div className="severity-track">
            <div className="severity-fill" style={{ width: `${(value / max) * 100}%` }} />
          </div>
          <span>{value}</span>
        </div>
      ))}
    </div>
  );
}

function WorkflowCard({
  workflow,
  values,
  onChange,
  onLaunch,
  configPaths,
  extraNote,
  disabled,
}: {
  workflow: WorkflowSpec;
  values: Record<string, unknown>;
  onChange: (name: string, value: unknown) => void;
  onLaunch: () => void;
  configPaths: string[];
  extraNote?: string;
  disabled?: boolean;
}) {
  return (
    <section className="workflow-card">
      <div className="workflow-head">
        <div>
          <h3>{workflow.title}</h3>
          <p className="workflow-description">{workflow.description}</p>
        </div>
        <code>{workflow.module}</code>
      </div>
      {extraNote ? <p className="workflow-note">{extraNote}</p> : null}
      <div className="workflow-fields">
        {workflow.fields.map((field) => (
          <WorkflowInput
            key={`${workflow.id}.${field.name}`}
            field={field}
            value={values[field.name]}
            onChange={(next) => onChange(field.name, next)}
            configPaths={configPaths}
          />
        ))}
      </div>
      <div className="button-row">
        <button className="primary-button" onClick={onLaunch} disabled={disabled}>
          Launch
        </button>
      </div>
    </section>
  );
}

export default function App() {
  const [branding, setBranding] = useState<Branding>(DEFAULT_BRANDING);
  const [view, setView] = useState<ViewKey>("overview");
  const [workflowSpecs, setWorkflowSpecs] = useState<WorkflowSpec[]>([]);
  const [workflowValues, setWorkflowValues] = useState<Record<string, Record<string, unknown>>>({});
  const [configPaths, setConfigPaths] = useState<string[]>([]);
  const [sumoConfigPaths, setSumoConfigPaths] = useState<string[]>([]);
  const [outputFolderPaths, setOutputFolderPaths] = useState<string[]>([]);
  const [selectedConfigPath, setSelectedConfigPath] = useState<string>("configs/thessaloniki/default.yaml");
  const [configDoc, setConfigDoc] = useState<ConfigDocument | null>(null);
  const [rawYaml, setRawYaml] = useState<string>("");
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [resultsTree, setResultsTree] = useState<TreeNode[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [selectedFileText, setSelectedFileText] = useState<string>("");
  const [selectedRunSummary, setSelectedRunSummary] = useState<ResultRunSummary | null>(null);
  const [message, setMessage] = useState<string>("Loading AntifragiCity SAS Console…");
  const [configMode, setConfigMode] = useState<ConfigMode>("structured");
  const [configSection, setConfigSection] = useState<ConfigSectionKey>("sumo");
  const [newConfigFolderChoice, setNewConfigFolderChoice] = useState<string>("custom");
  const [newConfigCustomFolder, setNewConfigCustomFolder] = useState<string>("custom");
  const [newConfigName, setNewConfigName] = useState<string>("new_config");
  const [sectionGuide, setSectionGuide] = useState<ConfigSectionSpec["guide"] | null>(null);
  const [docPaths, setDocPaths] = useState<string[]>([]);
  const [selectedDocPath, setSelectedDocPath] = useState<string>("README.md");
  const [selectedDocText, setSelectedDocText] = useState<string>("");
  const [locationQuery, setLocationQuery] = useState<string>("");
  const [locationResults, setLocationResults] = useState<LocationSearchResult[]>([]);
  const [selectedLocation, setSelectedLocation] = useState<LocationSearchResult | null>(null);
  const [customBounds, setCustomBounds] = useState<[number, number, number, number] | null>(null);
  const [boundaryMode, setBoundaryMode] = useState<BoundaryMode>("bbox");
  const [customShapePoints, setCustomShapePoints] = useState<Array<[number, number]>>([]);
  const [dataTab, setDataTab] = useState<"osm" | "feeds">("osm");
  const [infoModal, setInfoModal] = useState<InfoModal>(null);

  const selectedJob = useMemo(
    () => jobs.find((job) => job.id === selectedJobId) ?? jobs[0] ?? null,
    [jobs, selectedJobId],
  );

  const selectedFileIsImage = Boolean(selectedFile && /\.(png|jpg|jpeg|webp|svg)$/i.test(selectedFile));
  const selectedFileIsHtml = Boolean(selectedFile && /\.html$/i.test(selectedFile));
  const configPathGroups = useMemo(() => groupPathsByFolder(configPaths, "configs"), [configPaths]);
  const sumoConfigGroups = useMemo(() => groupPathsByFolder(sumoConfigPaths, "data"), [sumoConfigPaths]);
  const outputFolderGroups = useMemo(() => groupPathsByFolder(outputFolderPaths, "results"), [outputFolderPaths]);
  const configFolderChoices = useMemo(() => [...topLevelFolders(configPaths), "__new__"], [configPaths]);
  const computedNewConfigPath = useMemo(
    () => buildConfigPath(newConfigFolderChoice, newConfigCustomFolder, newConfigName),
    [newConfigFolderChoice, newConfigCustomFolder, newConfigName],
  );

  const workflowsById = useMemo(
    () => Object.fromEntries(workflowSpecs.map((workflow) => [workflow.id, workflow])) as Record<string, WorkflowSpec>,
    [workflowSpecs],
  );

  const workflowGroups = useMemo(() => {
    return workflowSpecs.reduce<Record<string, WorkflowSpec[]>>((acc, workflow) => {
      acc[workflow.category] = [...(acc[workflow.category] ?? []), workflow];
      return acc;
    }, {});
  }, [workflowSpecs]);

  const activeCategoryWorkflows = useMemo(() => {
    if (!(view in WORKFLOW_CATEGORY_BY_VIEW)) {
      return [];
    }
    return workflowGroups[WORKFLOW_CATEGORY_BY_VIEW[view as keyof typeof WORKFLOW_CATEGORY_BY_VIEW]] ?? [];
  }, [view, workflowGroups]);

  const logoSrc = assetPath(branding.logo_path, "/branding/antifragicity-logo-main-h.svg");
  const euLogoSrc = assetPath(branding.eu_logo_path, "/branding/eu-funded-by-eu.png");
  const isGreekSelection = selectedLocation?.country_code === "gr";
  const isThessalonikiSelection = isGreekSelection && `${selectedLocation?.display_name ?? ""}`.toLowerCase().includes("thessaloniki");
  const localityPolygons = useMemo(() => normalizeGeoJsonCoordinates(selectedLocation?.geojson), [selectedLocation]);
  const shapeBounds = useMemo(() => boundsFromPoints(customShapePoints), [customShapePoints]);
  const activeMapBounds = useMemo(() => {
    if (boundaryMode === "shape" && shapeBounds) {
      return shapeBounds;
    }
    return customBounds;
  }, [boundaryMode, shapeBounds, customBounds]);
  const configStudioInfo = {
    title: "Config Studio Guide",
    sections: [
      {
        heading: "How It Works",
        body: [
          "Config Studio edits the same YAML configuration files used by the CLI.",
          "Use the folder-grouped config picker to open an existing scenario, then save, clone, validate, or delete it from the same page.",
        ],
      },
      {
        heading: "Structured Editing",
        body: [
          "Structured mode is organized by simulation concern rather than raw YAML nesting.",
          "Risk, Accident, and Resilience tabs include deeper model dialogs because those parameters need interpretation, not just labels.",
        ],
      },
    ],
  };
  const dataIntegrationInfo = {
    title: "Data & Integrations Guide",
    sections: [
      {
        heading: "OSM Extracts",
        body: [
          "Use the OSM tab to search for a place, inspect the locality geometry, and decide whether to use the locality outline, a bounding box, or a custom drawn shape.",
          "The current download backend still performs extraction through a bounding box, so custom shapes are converted to their enclosing bbox for the actual fetch.",
        ],
      },
      {
        heading: "Traffic Feeds",
        body: [
          "The traffic-feed tab currently reflects the Thessaloniki integration already implemented in SAS.",
          "Additional city sources can be added later without changing the overall page structure.",
        ],
      },
    ],
  };

  useEffect(() => {
    document.documentElement.style.setProperty("--brand-primary", branding.colors.primary);
    document.documentElement.style.setProperty("--brand-secondary", branding.colors.secondary);
    document.documentElement.style.setProperty("--brand-ink", branding.colors.ink);
    document.documentElement.style.setProperty("--brand-surface", branding.colors.surface);
    document.documentElement.style.setProperty("--brand-surface-alt", branding.colors.surface_alt);
    document.documentElement.style.setProperty("--brand-border", branding.colors.border);
  }, [branding]);

  const refreshConfigs = async () => {
    const configData = await api.get<{ configs: Array<{ path: string }> }>("/api/configs");
    setConfigPaths(configData.configs.map((item) => item.path));
  };

  const refreshResults = async () => {
    const resultData = await api.get<{ entries: TreeNode[] }>("/api/results");
    setResultsTree(resultData.entries);
  };

  useEffect(() => {
    void Promise.all([
      api.get<{ workflows: WorkflowSpec[] }>("/api/workflows"),
      api.get<{ configs: Array<{ path: string }> }>("/api/configs"),
      api.get<{ sumo_configs: string[] }>("/api/sumo-configs"),
      api.get<{ output_folders: string[] }>("/api/output-folders"),
      api.get<{ docs: Array<{ path: string }> }>("/api/docs"),
      api.get<Branding>("/api/branding"),
      api.get<{ jobs: JobRecord[] }>("/api/jobs"),
      api.get<{ entries: TreeNode[] }>("/api/results"),
    ])
      .then(([workflowData, configData, sumoConfigData, outputFolderData, docsData, brandingData, jobData, resultData]) => {
        setWorkflowSpecs(workflowData.workflows);
        setConfigPaths(configData.configs.map((item) => item.path));
        setSumoConfigPaths(sumoConfigData.sumo_configs);
        setOutputFolderPaths(outputFolderData.output_folders);
        setDocPaths(docsData.docs.map((item) => item.path));
        setBranding(brandingData);
        setJobs(jobData.jobs);
        setResultsTree(resultData.entries);
        setWorkflowValues(
          Object.fromEntries(
            workflowData.workflows.map((workflow) => [
              workflow.id,
              Object.fromEntries(workflow.fields.map((field) => [field.name, field.default])),
            ]),
          ),
        );
        setSelectedJobId((current) => current ?? jobData.jobs[0]?.id ?? null);
        setMessage("Ready");
      })
      .catch((error) => setMessage(`Failed to load GUI metadata: ${String(error)}`));
  }, []);

  useEffect(() => {
    if (!selectedConfigPath) {
      return;
    }
    void api
      .get<ConfigDocument>(`/api/config?path=${encodeURIComponent(selectedConfigPath)}`)
      .then((doc) => {
        setConfigDoc(doc);
        setRawYaml(doc.raw_yaml);
      })
      .catch((error) => setMessage(`Failed to load config: ${String(error)}`));
  }, [selectedConfigPath]);

  useEffect(() => {
    if (!selectedDocPath) {
      setSelectedDocText("");
      return;
    }
    void fetch(api.textUrl(selectedDocPath))
      .then((response) => response.text())
      .then((text) => setSelectedDocText(text))
      .catch(() => setSelectedDocText("Unable to load documentation preview."));
  }, [selectedDocPath]);

  useEffect(() => {
    if (boundaryMode === "shape" && shapeBounds) {
      updateWorkflowValue("integration.fetch_osm", "south", shapeBounds[0]);
      updateWorkflowValue("integration.fetch_osm", "west", shapeBounds[1]);
      updateWorkflowValue("integration.fetch_osm", "north", shapeBounds[2]);
      updateWorkflowValue("integration.fetch_osm", "east", shapeBounds[3]);
    }
  }, [boundaryMode, shapeBounds]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      void api
        .get<{ jobs: JobRecord[] }>("/api/jobs")
        .then((data) => {
          setJobs(data.jobs);
          setSelectedJobId((current) => current ?? data.jobs[0]?.id ?? null);
        })
        .catch(() => undefined);
      void refreshResults().catch(() => undefined);
    }, 2000);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!selectedFile) {
      setSelectedFileText("");
      setSelectedRunSummary(null);
      return;
    }
    const isTextLike = /\.(json|csv|log|txt|yaml|yml|md|xml)$/i.test(selectedFile);
    if (isTextLike) {
      void fetch(api.textUrl(selectedFile))
        .then((response) => response.text())
        .then((text) => setSelectedFileText(text))
        .catch(() => setSelectedFileText("Unable to load file preview."));
    } else {
      setSelectedFileText("");
    }
    void api
      .get<ResultRunSummary>(`/api/results/summary?path=${encodeURIComponent(selectedFile)}`)
      .then((summary) => setSelectedRunSummary(summary))
      .catch(() => setSelectedRunSummary(null));
  }, [selectedFile]);

  const updateWorkflowValue = (workflowId: string, name: string, value: unknown) => {
    setWorkflowValues((current) => ({
      ...current,
      [workflowId]: {
        ...(current[workflowId] ?? {}),
        [name]: value,
      },
    }));
  };

  const launchWorkflow = async (workflowId: string) => {
    const workflow = workflowsById[workflowId];
    if (!workflow) {
      return;
    }
    try {
      const result = await api.post<JobRecord>("/api/jobs", {
        workflow_id: workflow.id,
        payload: workflowValues[workflow.id] ?? {},
      });
      setSelectedJobId(result.id);
      setMessage(`Started ${workflow.title}`);
      setView("jobs");
    } catch (error) {
      setMessage(`Failed to start job: ${String(error)}`);
    }
  };

  const saveStructuredConfig = async () => {
    if (!configDoc) {
      return;
    }
    try {
      await api.post("/api/config/save", {
        path: configDoc.path,
        config: configDoc.config,
      });
      setMessage(`Saved ${configDoc.path}`);
    } catch (error) {
      setMessage(`Save failed: ${String(error)}`);
    }
  };

  const saveRawConfig = async () => {
    if (!configDoc) {
      return;
    }
    try {
      await api.post("/api/config/save", {
        path: configDoc.path,
        raw_yaml: rawYaml,
      });
      const next = await api.get<ConfigDocument>(`/api/config?path=${encodeURIComponent(configDoc.path)}`);
      setConfigDoc(next);
      setRawYaml(next.raw_yaml);
      setMessage(`Saved ${configDoc.path}`);
    } catch (error) {
      setMessage(`Raw save failed: ${String(error)}`);
    }
  };

  const validateCurrentConfig = async () => {
    if (!configDoc) {
      return;
    }
    try {
      if (configMode === "raw") {
        await api.post("/api/config/validate", { raw_yaml: rawYaml });
      } else {
        await api.post("/api/config/validate", { config: configDoc.config });
      }
      setMessage("Config validation passed");
    } catch (error) {
      setMessage(`Validation failed: ${String(error)}`);
    }
  };

  const createConfig = async (mode: "clean" | "clone") => {
    if (!computedNewConfigPath.trim()) {
      setMessage("Set a target config path first");
      return;
    }
    try {
      const result = await api.post<{ path: string }>("/api/config/create", {
        path: computedNewConfigPath,
        source_path: mode === "clone" ? selectedConfigPath : null,
      });
      await refreshConfigs();
      setSelectedConfigPath(result.path);
      setMessage(mode === "clone" ? `Cloned config into ${result.path}` : `Created clean starter config at ${result.path}`);
    } catch (error) {
      setMessage(`Config creation failed: ${String(error)}`);
    }
  };

  const deleteCurrentConfig = async () => {
    if (!selectedConfigPath) {
      return;
    }
    if (!window.confirm(`Delete ${selectedConfigPath}?`)) {
      return;
    }
    try {
      await api.post("/api/config/delete", { path: selectedConfigPath });
      await refreshConfigs();
      const remaining = configPaths.filter((path) => path !== selectedConfigPath);
      setSelectedConfigPath(remaining[0] ?? "");
      setConfigDoc(null);
      setRawYaml("");
      setMessage(`Deleted ${selectedConfigPath}`);
    } catch (error) {
      setMessage(`Delete failed: ${String(error)}`);
    }
  };

  const handleStructuredConfigChange = (fieldPath: string, nextValue: unknown) => {
    if (!configDoc) {
      return;
    }
    setConfigDoc({
      ...configDoc,
      config: setPathValue(configDoc.config, fieldPath, nextValue),
    });
  };

  const searchLocations = async () => {
    if (locationQuery.trim().length < 2) {
      setMessage("Enter at least two characters to search OSM");
      return;
    }
    try {
      const result = await api.get<{ results: LocationSearchResult[] }>(`/api/locations/search?query=${encodeURIComponent(locationQuery.trim())}`);
      setLocationResults(result.results);
      setMessage(result.results.length ? `Found ${result.results.length} location result(s)` : "No matching OSM locations found");
    } catch (error) {
      setMessage(`Location search failed: ${String(error)}`);
    }
  };

  const selectLocation = (location: LocationSearchResult) => {
    setSelectedLocation(location);
    const [south, north, west, east] = location.boundingbox;
    setCustomBounds([south, west, north, east]);
    setCustomShapePoints([]);
    setBoundaryMode(location.geojson ? "locality" : "bbox");
    updateWorkflowValue("integration.fetch_osm", "place", location.display_name);
    updateWorkflowValue("integration.fetch_osm", "out", searchLocationOutputPath(location));
    updateWorkflowValue("integration.fetch_osm", "south", south);
    updateWorkflowValue("integration.fetch_osm", "west", west);
    updateWorkflowValue("integration.fetch_osm", "north", north);
    updateWorkflowValue("integration.fetch_osm", "east", east);
    setMessage(`Selected ${location.display_name}`);
  };

  const updateBounds = (index: number, nextValue: number) => {
    setCustomBounds((current) => {
      const next = [...(current ?? [0, 0, 0, 0])] as [number, number, number, number];
      next[index] = nextValue;
      updateWorkflowValue("integration.fetch_osm", ["south", "west", "north", "east"][index], nextValue);
      return next;
    });
  };

  const addCustomShapePoint = (point: [number, number]) => {
    setCustomShapePoints((current) => [...current, point]);
  };

  const clearCustomShape = () => {
    setCustomShapePoints([]);
    if (customBounds) {
      updateWorkflowValue("integration.fetch_osm", "south", customBounds[0]);
      updateWorkflowValue("integration.fetch_osm", "west", customBounds[1]);
      updateWorkflowValue("integration.fetch_osm", "north", customBounds[2]);
      updateWorkflowValue("integration.fetch_osm", "east", customBounds[3]);
    }
  };

  const renderStructuredSection = (section: ConfigSectionSpec) => {
    if (!configDoc) {
      return null;
    }
    const fieldsForGroups = section.key === "resilience_assessment"
      ? section.fields.filter((field) => !field.path.startsWith("resilience_assessment.incident_configs."))
      : section.fields;
    const grouped = fieldsForGroups.reduce<Record<string, ConfigFieldSpec[]>>((acc, field) => {
      const group = field.group ?? "Parameters";
      acc[group] = [...(acc[group] ?? []), field];
      return acc;
    }, {});
    const incidentConfigs = getPathValue(configDoc.config, "resilience_assessment.incident_configs");
    const incidentRows = Array.isArray(incidentConfigs) ? incidentConfigs : [];
    return (
      <div className="structured-section">
        <div className="structured-section-head">
          <div className="section-heading-row">
            <h3>{section.title}</h3>
            {section.guide ? (
              <button type="button" className="secondary-button" onClick={() => setSectionGuide(section.guide)}>
                About This Model
              </button>
            ) : null}
          </div>
          <p>{section.intro}</p>
        </div>
        <div className="structured-groups">
          {Object.entries(grouped).map(([group, fields]) => (
            <section key={group} className="structured-group-card">
              <h4>{group}</h4>
              <div className="structured-fields-grid">
                {fields.map((field) => (
                  <StructuredFieldInput
                    key={field.path}
                    field={field}
                    value={getPathValue(configDoc.config, field.path)}
                    onChange={(next) => handleStructuredConfigChange(field.path, next)}
                    sumoConfigGroups={sumoConfigGroups}
                    outputFolderGroups={outputFolderGroups}
                  />
                ))}
              </div>
            </section>
          ))}
        </div>
        {section.key === "resilience_assessment" ? (
          <section className="structured-group-card">
            <h4>Incident Scenarios</h4>
            <p className="muted">These rows define the named incident regimes used in the assessment batch. The left column keeps the common scenario ladder visible, while the editable columns control the actual label and probability used by the run.</p>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Preset</th>
                    <th>Scenario Name</th>
                    <th>Base Probability</th>
                  </tr>
                </thead>
                <tbody>
                  {incidentRows.map((item, index) => (
                    <tr key={index}>
                      <td>{["Baseline", "Low Incident", "Default Incident", "High Incident", "Extreme Incident"][index] ?? `Scenario ${index + 1}`}</td>
                      <td>
                        <input
                          type="text"
                          value={String((item as Record<string, unknown>).name ?? "")}
                          onChange={(event: ChangeEvent<HTMLInputElement>) => handleStructuredConfigChange(`resilience_assessment.incident_configs.${index}.name`, event.target.value)}
                        />
                      </td>
                      <td>
                        <input
                          type="number"
                          value={String((item as Record<string, unknown>).base_probability ?? "")}
                            onChange={(event: ChangeEvent<HTMLInputElement>) =>
                              handleStructuredConfigChange(
                                `resilience_assessment.incident_configs.${index}.base_probability`,
                                event.target.value === "" ? undefined : Number(event.target.value),
                            )
                          }
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        ) : null}
      </div>
    );
  };

  const renderWorkflowSection = (workflows: WorkflowSpec[], heading: string, intro: string) => (
    <section className="workflow-stack">
      <article className="panel">
        <div className="section-header">
          <div>
            <h2>{heading}</h2>
            <p className="muted">{intro}</p>
          </div>
          <span className="chip">{workflows.length} workflows</span>
        </div>
        <div className="workflow-grid">
          {workflows.map((workflow) => (
            <WorkflowCard
              key={workflow.id}
              workflow={workflow}
              values={workflowValues[workflow.id] ?? {}}
              onChange={(name, value) => updateWorkflowValue(workflow.id, name, value)}
              onLaunch={() => void launchWorkflow(workflow.id)}
              configPaths={configPaths}
            />
          ))}
        </div>
      </article>
    </section>
  );

  const metricsSeries = selectedRunSummary?.metrics.series;
  const metricsStats = selectedRunSummary?.metrics.stats;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <img src={logoSrc} alt="AntifragiCity" className="brand-logo" />
        <nav className="nav-list">
          {VIEW_LABELS.map((item) => (
            <button key={item.key} className={`nav-item ${view === item.key ? "is-active" : ""}`} onClick={() => setView(item.key)}>
              {item.label}
            </button>
          ))}
        </nav>
      </aside>

      <main className="main-panel">
        <header className="hero">
          <div>
            <p className="eyebrow">Simulation Control Console</p>
            <h1>{branding.name}</h1>
            <p className="hero-copy">
              Manage YAML configs, search and ingest source data, launch generators, run simulations and assessments,
              and inspect interactive results from one operator-facing workspace.
            </p>
          </div>
          <div className="status-card">
            <span className="status-dot" />
            <div>
              <strong>{message}</strong>
              <p>{jobs.filter((job) => job.status === "running").length} job(s) currently running</p>
            </div>
          </div>
        </header>

        {view === "overview" ? (
          <section className="content-grid overview-grid">
            <article className="panel metric-panel">
              <h2>Current Surface</h2>
              <div className="metric-list">
                <div>
                  <span>Configs</span>
                  <strong>{configPaths.length}</strong>
                </div>
                <div>
                  <span>Workflows</span>
                  <strong>{workflowSpecs.length}</strong>
                </div>
                <div>
                  <span>Jobs</span>
                  <strong>{jobs.length}</strong>
                </div>
                <div>
                  <span>Result Roots</span>
                  <strong>{resultsTree.length}</strong>
                </div>
              </div>
            </article>
            <article className="panel">
              <h2>Active Job</h2>
              {selectedJob ? (
                <>
                  <p className="job-title">{selectedJob.title}</p>
                  <div className="progress-track">
                    <div className="progress-fill" style={{ width: `${(selectedJob.progress ?? 0) * 100}%` }} />
                  </div>
                  <p className="job-status">
                    {selectedJob.status} · {selectedJob.progress_label}
                  </p>
                  {selectedJob.live_progress_path ? (
                    <img className="live-image" src={`${api.fileUrl(selectedJob.live_progress_path)}&t=${Date.now()}`} alt="Live progress" />
                  ) : (
                    <p className="muted">No live image available yet.</p>
                  )}
                </>
              ) : (
                <p className="muted">No jobs have been started yet.</p>
              )}
            </article>
            <article className="panel">
              <h2>Workflow Groups</h2>
              <div className="chip-list">
                {Object.entries(workflowGroups).map(([category, items]) => (
                  <span key={category} className="chip">
                    {category}: {items.length}
                  </span>
                ))}
              </div>
            </article>
          </section>
        ) : null}

        {view === "configs" ? (
          <section className="content-grid config-grid">
            <article className="panel">
              <div className="section-header">
                <div>
                  <h2>Config Studio</h2>
                  <p className="muted">Create new scenario files from a clean starter template or from an existing config, then edit them in structured or raw YAML mode.</p>
                </div>
                <div className="button-row">
                  <button className="secondary-button" onClick={() => setInfoModal(configStudioInfo)}>
                    About This Page
                  </button>
                  <button className="secondary-button" onClick={() => void validateCurrentConfig()}>
                    Validate
                  </button>
                  <button className="secondary-button" onClick={() => void deleteCurrentConfig()}>
                    Delete
                  </button>
                  {configMode === "structured" ? (
                    <button className="primary-button" onClick={() => void saveStructuredConfig()}>
                      Save Structured
                    </button>
                  ) : (
                    <button className="primary-button" onClick={() => void saveRawConfig()}>
                      Save Raw YAML
                    </button>
                  )}
                </div>
              </div>
              <div className="config-topbar">
                <div className="config-create-card">
                  <label className="field">
                    <span>Config File</span>
                    <select value={selectedConfigPath} onChange={(event) => setSelectedConfigPath(event.target.value)}>
                      {configPathGroups.map((group) => (
                        <optgroup key={group.folder} label={group.folder}>
                          {group.items.map((path) => (
                            <option key={path} value={path}>
                              {path.split("/").slice(-1)[0]}
                            </option>
                          ))}
                        </optgroup>
                      ))}
                    </select>
                    <small>Configs are grouped by folder so city-specific and custom scenarios stay easier to scan.</small>
                  </label>
                </div>
                <div className="config-create-card">
                  <label className="field">
                    <span>Target Folder</span>
                    <select value={newConfigFolderChoice} onChange={(event) => setNewConfigFolderChoice(event.target.value)}>
                      {configFolderChoices.map((folder) => (
                        <option key={folder} value={folder}>
                          {folder === "__new__" ? "Create New Folder" : folder}
                        </option>
                      ))}
                    </select>
                  </label>
                  {newConfigFolderChoice === "__new__" ? (
                    <label className="field">
                      <span>New Folder Name</span>
                      <input type="text" value={newConfigCustomFolder} onChange={(event) => setNewConfigCustomFolder(event.target.value)} placeholder="my_city" />
                    </label>
                  ) : null}
                  <label className="field">
                    <span>Config File Name</span>
                    <input type="text" value={newConfigName} onChange={(event) => setNewConfigName(event.target.value)} placeholder="my_scenario" />
                    <small>{computedNewConfigPath || "Choose a folder and filename to build the target path."}</small>
                  </label>
                  <div className="button-row">
                    <button className="secondary-button" onClick={() => void createConfig("clean")}>
                      Create Clean
                    </button>
                    <button className="primary-button" onClick={() => void createConfig("clone")}>
                      Clone Selected
                    </button>
                  </div>
                </div>
              </div>
              <div className="tab-row">
                <button className={configMode === "structured" ? "tab-active" : ""} onClick={() => setConfigMode("structured")}>
                  Structured
                </button>
                <button className={configMode === "raw" ? "tab-active" : ""} onClick={() => setConfigMode("raw")}>
                  Raw YAML
                </button>
              </div>
              {configDoc && configMode === "structured" ? (
                <>
                  <div className="secondary-tab-row">
                    {CONFIG_SECTIONS.map((section) => (
                      <button
                        key={section.key}
                        className={configSection === section.key ? "tab-active" : ""}
                        onClick={() => setConfigSection(section.key)}
                      >
                        {section.title}
                      </button>
                    ))}
                  </div>
                  {renderStructuredSection(CONFIG_SECTIONS.find((section) => section.key === configSection)!)}
                </>
              ) : null}
              {configDoc && configMode === "raw" ? (
                <textarea className="raw-editor" rows={30} value={rawYaml} onChange={(event) => setRawYaml(event.target.value)} />
              ) : null}
            </article>
          </section>
        ) : null}

        {view === "data_integrations" ? (
          <section className="workflow-stack">
            <article className="panel">
              <div className="section-header">
                <div>
                  <h2>Data & Integrations</h2>
                  <p className="muted">Search a place in OpenStreetMap, preview and tune its boundary, then launch the acquisition pipeline. Greek traffic-data integration is surfaced separately below.</p>
                </div>
                <div className="button-row">
                  <button className="secondary-button" onClick={() => setInfoModal(dataIntegrationInfo)}>
                    About This Page
                  </button>
                </div>
              </div>
              <div className="secondary-tab-row">
                <button className={dataTab === "osm" ? "tab-active" : ""} onClick={() => setDataTab("osm")}>
                  OSM Extract
                </button>
                <button className={dataTab === "feeds" ? "tab-active" : ""} onClick={() => setDataTab("feeds")}>
                  Traffic Feeds
                </button>
              </div>

              {dataTab === "osm" ? (
                <div className="data-grid">
                  <section className="workflow-card">
                    <div className="workflow-head">
                      <h3>Download OSM Extract</h3>
                      <code>{workflowsById["integration.fetch_osm"]?.module}</code>
                    </div>
                    <div className="search-row">
                      <input type="text" value={locationQuery} onChange={(event) => setLocationQuery(event.target.value)} placeholder="Search city, district, corridor, or municipality" />
                      <button className="primary-button" onClick={() => void searchLocations()}>
                        Search OSM
                      </button>
                    </div>
                    {locationResults.length > 0 ? (
                      <div className="location-results">
                        {locationResults.map((location) => (
                          <button key={`${location.osm_type}-${location.osm_id}`} className={`location-row ${selectedLocation?.osm_id === location.osm_id ? "is-selected" : ""}`} onClick={() => selectLocation(location)}>
                            <strong>{location.country || "Unknown country"} / {location.city || location.state || "Unknown locality"}</strong>
                            <span>{location.display_name}</span>
                          </button>
                        ))}
                      </div>
                    ) : (
                      <p className="muted">Search results will appear here, grouped implicitly by locality text.</p>
                    )}
                    <div className="workflow-fields compact-grid">
                      {(workflowsById["integration.fetch_osm"]?.fields ?? [])
                        .filter((field) => !["south", "west", "north", "east"].includes(field.name))
                        .map((field) => (
                          <WorkflowInput
                            key={field.name}
                            field={field}
                            value={workflowValues["integration.fetch_osm"]?.[field.name]}
                            onChange={(next) => updateWorkflowValue("integration.fetch_osm", field.name, next)}
                            configPaths={configPaths}
                          />
                        ))}
                    </div>
                    <div className="bounds-card">
                      <div className="bounds-head">
                        <h4>Boundary Mode</h4>
                        <p>Select the source of the extraction boundary. Locality uses the returned OSM geometry, bounding box uses the numeric limits, and custom shape lets you sketch a shape on the map.</p>
                      </div>
                      <div className="secondary-tab-row">
                        <button className={boundaryMode === "locality" ? "tab-active" : ""} onClick={() => setBoundaryMode("locality")} disabled={!localityPolygons.length}>
                          Locality Boundary
                        </button>
                        <button className={boundaryMode === "bbox" ? "tab-active" : ""} onClick={() => setBoundaryMode("bbox")}>
                          Bounding Box
                        </button>
                        <button className={boundaryMode === "shape" ? "tab-active" : ""} onClick={() => setBoundaryMode("shape")}>
                          Custom Shape
                        </button>
                      </div>
                      <div className="bounds-grid">
                        {(["South", "West", "North", "East"] as const).map((label, index) => (
                          <label key={label} className="field">
                            <span>{label}</span>
                            <input
                              type="number"
                              value={activeMapBounds?.[index] ?? ""}
                              onChange={(event) => updateBounds(index, Number(event.target.value))}
                            />
                          </label>
                        ))}
                      </div>
                      {boundaryMode === "shape" ? (
                        <div className="button-row">
                          <button className="secondary-button" onClick={clearCustomShape}>
                            Clear Shape
                          </button>
                          <span className="muted">Click on the map to add vertices. The enclosing bbox is used for the actual OSM request.</span>
                        </div>
                      ) : null}
                    </div>
                    <div className="button-row">
                      <button className="primary-button" onClick={() => void launchWorkflow("integration.fetch_osm")}>
                        Launch OSM Download
                      </button>
                    </div>
                  </section>

                  <section className="workflow-card">
                    <h3>Boundary Preview</h3>
                    {selectedLocation ? (
                      <>
                        <MapContainer
                          {...({
                            className: "map-frame",
                            center: [selectedLocation.lat, selectedLocation.lon],
                            zoom: 12,
                            scrollWheelZoom: true,
                          } as any)}
                        >
                          <TileLayer {...({ attribution: "&copy; OpenStreetMap contributors", url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" } as any)} />
                          <MapViewport bounds={activeMapBounds} center={[selectedLocation.lat, selectedLocation.lon]} mode={boundaryMode} />
                          <ShapeEditor enabled={boundaryMode === "shape"} onAddPoint={addCustomShapePoint} />
                          {boundaryMode === "locality" && selectedLocation.geojson ? <GeoJSON data={selectedLocation.geojson as never} /> : null}
                          {boundaryMode === "bbox" && customBounds ? (
                            <Rectangle bounds={[[customBounds[0], customBounds[1]], [customBounds[2], customBounds[3]]]} />
                          ) : null}
                          {boundaryMode === "shape" && customShapePoints.length >= 2 ? <Polygon positions={customShapePoints} /> : null}
                        </MapContainer>
                        <div className="location-summary">
                          <strong>{selectedLocation.display_name}</strong>
                          <span>
                            {selectedLocation.country_code.toUpperCase()} · {selectedLocation.class}/{selectedLocation.type} · Mode: {boundaryMode}
                          </span>
                        </div>
                      </>
                    ) : (
                      <p className="muted">Select a location to preview the extracted map boundary.</p>
                    )}
                  </section>
                </div>
              ) : (
                <div className="workflow-stack">
                  <section className="workflow-card">
                    <h3>Traffic Feeds</h3>
                    <p className="muted">
                      The feed structure is intentionally split into its own tab so future cities can be added under the same pattern. At the moment, the active operational source remains Thessaloniki.
                    </p>
                  </section>
                  <section className="structured-group-card">
                    <h4>Thessaloniki</h4>
                    <p className="muted">
                      The current gov.gr integration in this repository is backed by IMET/CERTH Thessaloniki feeds. It is appropriate for Thessaloniki today; other Greek cities need a new source mapping before this becomes generic.
                    </p>
                    <div className="workflow-grid">
                      {["integration.govgr_download", "integration.govgr_targets"].map((id) => {
                        const workflow = workflowsById[id];
                        if (!workflow) {
                          return null;
                        }
                        return (
                          <WorkflowCard
                            key={workflow.id}
                            workflow={workflow}
                            values={workflowValues[workflow.id] ?? {}}
                            onChange={(name, value) => updateWorkflowValue(workflow.id, name, value)}
                            onLaunch={() => void launchWorkflow(workflow.id)}
                            configPaths={configPaths}
                            disabled={!isThessalonikiSelection}
                            extraNote={
                              isThessalonikiSelection
                                ? "Current feed/source alignment matches Thessaloniki."
                                : "Select Thessaloniki in the OSM tab before launching this workflow."
                            }
                          />
                        );
                      })}
                    </div>
                  </section>
                </div>
              )}
            </article>
          </section>
        ) : null}

        {view === "generators"
          ? renderWorkflowSection(
              activeCategoryWorkflows,
              "Generators",
              "Bundled generator workflows remain available, but the page is framed around reusable generation tasks rather than city-specific one-off scripts.",
            )
          : null}

        {view === "simulations"
          ? renderWorkflowSection(
              activeCategoryWorkflows,
              "Simulations",
              "Run single simulations or resilience assessments against the currently managed config set.",
            )
          : null}

        {view === "analysis"
          ? renderWorkflowSection(
              activeCategoryWorkflows,
              "Analysis",
              "Post-process batches, sweeps, reports, and calibration comparisons from the same interface.",
            )
          : null}

        {view === "jobs" ? (
          <section className="content-grid jobs-grid">
            <article className="panel">
              <h2>Job Queue</h2>
              <div className="job-list">
                {jobs.map((job) => (
                  <button key={job.id} className={`job-row ${selectedJob?.id === job.id ? "is-selected" : ""}`} onClick={() => setSelectedJobId(job.id)}>
                    <div>
                      <strong>{job.title}</strong>
                      <p>{job.status}</p>
                    </div>
                    <span>{job.progress !== null ? `${Math.round(job.progress * 100)}%` : "…"}</span>
                  </button>
                ))}
              </div>
            </article>
            <article className="panel">
              <div className="section-header">
                <h2>Job Console</h2>
                {selectedJob?.status === "running" ? (
                  <button className="secondary-button" onClick={() => void api.post(`/api/jobs/${selectedJob.id}/cancel`, {})}>
                    Cancel
                  </button>
                ) : null}
              </div>
              {selectedJob ? (
                <>
                  <div className="progress-track">
                    <div className="progress-fill" style={{ width: `${(selectedJob.progress ?? 0) * 100}%` }} />
                  </div>
                  <p className="job-status">
                    {selectedJob.status} · {selectedJob.progress_label}
                  </p>
                  <code className="command-preview">{selectedJob.command.join(" ")}</code>
                  <div className="job-detail-stack">
                    <section className="job-media-panel">
                      <h3>Live / Reports</h3>
                      <div className="job-media-grid">
                        {selectedJob.live_progress_path ? (
                          <img className="live-image" src={`${api.fileUrl(selectedJob.live_progress_path)}&t=${Date.now()}`} alt="Live progress" />
                        ) : null}
                        {selectedJob.report_path ? <iframe className="report-frame" src={api.fileUrl(selectedJob.report_path)} title="Generated report" /> : null}
                      </div>
                      {!selectedJob.live_progress_path && !selectedJob.report_path ? <p className="muted">No live image or HTML report available yet.</p> : null}
                    </section>
                    <section className="job-log-panel">
                      <h3>Logs</h3>
                      <pre className="log-console">{selectedJob.log_lines.join("\n")}</pre>
                    </section>
                  </div>
                </>
              ) : (
                <p className="muted">Select a job to inspect its logs and outputs.</p>
              )}
            </article>
          </section>
        ) : null}

        {view === "results" ? (
          <section className="results-layout">
            <article className="panel results-tree-panel">
              <h2>Results Browser</h2>
              <TreeView nodes={resultsTree} onSelect={setSelectedFile} />
            </article>
            <div className="results-main-stack">
              <article className="panel">
                <div className="section-header">
                  <div>
                    <h2>Interactive Run Summary</h2>
                    <p className="muted">{selectedRunSummary ? selectedRunSummary.run_root : "Select any file or folder within a run to load its metrics and artifacts."}</p>
                  </div>
                </div>
                {selectedRunSummary ? (
                  <>
                    <div className="summary-card-grid">
                      <div className="summary-card">
                        <span>Total Accidents</span>
                        <strong>{formatNumber(selectedRunSummary.summary.total_accidents, 0)}</strong>
                      </div>
                      <div className="summary-card">
                        <span>Antifragility Index</span>
                        <strong>{formatNumber(selectedRunSummary.summary.antifragility_index, 4)}</strong>
                      </div>
                      <div className="summary-card">
                        <span>Peak Vehicles</span>
                        <strong>{formatNumber(metricsStats?.peak_vehicle_count, 0)}</strong>
                      </div>
                      <div className="summary-card">
                        <span>Peak Throughput / h</span>
                        <strong>{formatNumber(metricsStats?.peak_throughput_per_hour, 0)}</strong>
                      </div>
                      <div className="summary-card">
                        <span>Peak Active Accidents</span>
                        <strong>{formatNumber(metricsStats?.peak_active_accidents, 0)}</strong>
                      </div>
                      <div className="summary-card">
                        <span>Mean Delay (s)</span>
                        <strong>{formatNumber(metricsStats?.mean_delay_seconds)}</strong>
                      </div>
                    </div>

                    <div className="chart-grid">
                      <SimpleLineChart title="Active Vehicles" values={metricsSeries?.vehicle_count ?? []} accentClass="chart-primary" />
                      <SimpleLineChart title="Mean Speed (km/h)" values={metricsSeries?.mean_speed_kmh ?? []} accentClass="chart-secondary" />
                      <SimpleLineChart title="Throughput per Hour" values={metricsSeries?.throughput_per_hour ?? []} accentClass="chart-primary" />
                      <SimpleLineChart title="Mean Delay (s)" values={metricsSeries?.mean_delay_seconds ?? []} accentClass="chart-secondary" />
                      <SimpleLineChart title="Concurrent Active Accidents" values={metricsSeries?.active_accidents ?? []} accentClass="chart-primary" />
                      <SimpleLineChart title="Speed Ratio" values={metricsSeries?.speed_ratio ?? []} accentClass="chart-secondary" />
                    </div>

                    <div className="results-detail-grid">
                      <section className="detail-card">
                        <h3>Accident Distribution</h3>
                        {selectedRunSummary.accidents ? (
                          <>
                            <SeverityBars counts={selectedRunSummary.accidents.by_severity} />
                            <div className="chip-list">
                              <span className="chip">Max duration: {formatNumber(selectedRunSummary.accidents.max_duration_seconds, 0)} s</span>
                              <span className="chip">Max queue: {formatNumber(selectedRunSummary.accidents.max_queue_length_vehicles, 0)}</span>
                              <span className="chip">Max affected: {formatNumber(selectedRunSummary.accidents.max_vehicles_affected, 0)}</span>
                            </div>
                          </>
                        ) : (
                          <p className="muted">No accident report found for this run.</p>
                        )}
                      </section>

                      <section className="detail-card">
                        <h3>Antifragility</h3>
                        {selectedRunSummary.antifragility ? (
                          <>
                            <div className="chip-list">
                              <span className="chip">AI: {formatNumber(selectedRunSummary.antifragility.antifragility_index, 4)}</span>
                              <span className="chip">Events measured: {formatNumber(selectedRunSummary.antifragility.n_events_measured, 0)}</span>
                              <span className="chip">
                                95% CI: {formatNumber(selectedRunSummary.antifragility.ci_95_low, 3)} to {formatNumber(selectedRunSummary.antifragility.ci_95_high, 3)}
                              </span>
                            </div>
                            <p className="muted">{selectedRunSummary.antifragility.interpretation}</p>
                          </>
                        ) : (
                          <p className="muted">No antifragility file found for this run.</p>
                        )}
                      </section>
                    </div>

                    {selectedRunSummary.accidents?.items?.length ? (
                      <section className="detail-card">
                        <h3>Accident Table</h3>
                        <div className="table-wrap">
                          <table className="data-table">
                            <thead>
                              <tr>
                                <th>ID</th>
                                <th>Severity</th>
                                <th>Trigger</th>
                                <th>Duration (s)</th>
                                <th>Edge</th>
                                <th>Affected</th>
                                <th>Peak Queue</th>
                              </tr>
                            </thead>
                            <tbody>
                              {selectedRunSummary.accidents.items.slice(0, 20).map((item) => (
                                <tr key={String(item.accident_id)}>
                                  <td>{String(item.accident_id)}</td>
                                  <td>{String(item.severity)}</td>
                                  <td>{formatNumber(item.trigger_step, 0)}</td>
                                  <td>{formatNumber(item.duration_seconds, 0)}</td>
                                  <td>{String(item.edge_id ?? "–")}</td>
                                  <td>{formatNumber(item.vehicles_affected, 0)}</td>
                                  <td>{formatNumber(item.peak_queue_length_vehicles, 0)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </section>
                    ) : null}

                    {selectedRunSummary.antifragility?.per_event?.length ? (
                      <section className="detail-card">
                        <h3>Per-event Antifragility</h3>
                        <div className="table-wrap">
                          <table className="data-table">
                            <thead>
                              <tr>
                                <th>Accident</th>
                                <th>Event AI</th>
                                <th>Pre-speed</th>
                                <th>Post-speed</th>
                                <th>Pre Samples</th>
                                <th>Post Samples</th>
                              </tr>
                            </thead>
                            <tbody>
                              {selectedRunSummary.antifragility.per_event.map((event) => (
                                <tr key={String(event.accident_id)}>
                                  <td>{String(event.accident_id)}</td>
                                  <td>{formatNumber(event.event_ai, 4)}</td>
                                  <td>{formatNumber(event.pre_mean_speed_kmh)}</td>
                                  <td>{formatNumber(event.post_mean_speed_kmh)}</td>
                                  <td>{formatNumber(event.n_pre_samples, 0)}</td>
                                  <td>{formatNumber(event.n_post_samples, 0)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </section>
                    ) : null}

                    <section className="detail-card">
                      <h3>Run Artifacts</h3>
                      <div className="artifact-grid">
                        {selectedRunSummary.artifacts.report_path ? (
                          <iframe className="report-frame" src={api.fileUrl(selectedRunSummary.artifacts.report_path)} title="Run report" />
                        ) : null}
                        {selectedRunSummary.artifacts.image_paths.map((path) => (
                          <img key={path} className="preview-image" src={api.fileUrl(path)} alt={path} />
                        ))}
                      </div>
                    </section>
                  </>
                ) : (
                  <p className="muted">No interactive run summary is available for the current selection.</p>
                )}
              </article>

              <article className="panel">
                <h2>Selected File Preview</h2>
                {selectedFile ? <p className="muted">{selectedFile}</p> : null}
                {selectedFileIsImage && selectedFile ? <img className="preview-image" src={api.fileUrl(selectedFile)} alt={selectedFile} /> : null}
                {selectedFileIsHtml && selectedFile ? <iframe className="report-frame" src={api.fileUrl(selectedFile)} title={selectedFile} /> : null}
                {!selectedFileIsImage && !selectedFileIsHtml && selectedFileText ? <pre className="file-preview">{selectedFileText}</pre> : null}
                {!selectedFile && <p className="muted">Select a file or directory from the results tree.</p>}
              </article>
            </div>
          </section>
        ) : null}

        {view === "documentation" ? (
          <section className="content-grid config-grid">
            <article className="panel">
              <div className="section-header">
                <div>
                  <h2>Documentation</h2>
                  <p className="muted">Browse the project markdown documentation directly from the GUI. Each document is available as its own tab for quick reference while configuring or running workflows.</p>
                </div>
              </div>
              <div className="secondary-tab-row">
                {docPaths.map((path) => (
                  <button key={path} className={selectedDocPath === path ? "tab-active" : ""} onClick={() => setSelectedDocPath(path)}>
                    {path.split("/").slice(-1)[0].replace(/\.md$/i, "")}
                  </button>
                ))}
              </div>
              <div className="markdown-preview">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{selectedDocText}</ReactMarkdown>
              </div>
            </article>
          </section>
        ) : null}

        <footer className="app-footer">
          <div className="footer-links">
            {VIEW_LABELS.filter((item) => item.key !== "overview").map((item) => (
              <button key={item.key} className="footer-link" onClick={() => setView(item.key)}>
                {item.label}
              </button>
            ))}
            <a className="footer-link external-link" href={branding.project_url} target="_blank" rel="noreferrer">
              Project Website
            </a>
          </div>
          <div className="footer-funding">
            <img src={euLogoSrc} alt="Funded by the European Union" className="eu-logo" />
            <p>{branding.footer_disclaimer}</p>
          </div>
          <div className="footer-meta">
            <span>{branding.copyright}</span>
            <a href={branding.project_url} target="_blank" rel="noreferrer">
              antifragicity.eu
            </a>
          </div>
        </footer>

        {sectionGuide ? (
          <div className="modal-backdrop" onClick={() => setSectionGuide(null)}>
            <div className="modal-card modal-card-wide" onClick={(event) => event.stopPropagation()}>
              <div className="section-header">
                <h2>{sectionGuide.title}</h2>
                <button type="button" className="secondary-button" onClick={() => setSectionGuide(null)}>
                  Close
                </button>
              </div>
              <section className="modal-section">
                <h3>What This Controls</h3>
                {sectionGuide.overview.map((paragraph) => (
                  <p key={paragraph}>{paragraph}</p>
                ))}
              </section>
              <section className="modal-section">
                <h3>How To Read The Values</h3>
                {sectionGuide.interpretation.map((paragraph) => (
                  <p key={paragraph}>{paragraph}</p>
                ))}
              </section>
              {sectionGuide.references?.length ? (
                <section className="modal-section">
                  <h3>References</h3>
                  <ul className="modal-list">
                    {sectionGuide.references.map((reference) => (
                      <li key={reference}>{reference}</li>
                    ))}
                  </ul>
                </section>
              ) : null}
            </div>
          </div>
        ) : null}

        {infoModal ? (
          <div className="modal-backdrop" onClick={() => setInfoModal(null)}>
            <div className="modal-card modal-card-wide" onClick={(event) => event.stopPropagation()}>
              <div className="section-header">
                <h2>{infoModal.title}</h2>
                <button type="button" className="secondary-button" onClick={() => setInfoModal(null)}>
                  Close
                </button>
              </div>
              {infoModal.sections.map((section) => (
                <section key={section.heading} className="modal-section">
                  <h3>{section.heading}</h3>
                  {section.body.map((paragraph) => (
                    <p key={paragraph}>{paragraph}</p>
                  ))}
                </section>
              ))}
            </div>
          </div>
        ) : null}
      </main>
    </div>
  );
}
