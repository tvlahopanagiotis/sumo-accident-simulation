import { useEffect, useMemo, useState } from "react";
import { GeoJSON, MapContainer, Polygon, Rectangle, TileLayer, useMap, useMapEvents } from "react-leaflet";
import type { ChangeEvent } from "react";
import type { LeafletMouseEvent } from "leaflet";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { CityNetworkMap, classifyRoadGroup, type NetworkSelectionMode, type NetworkViewMode } from "./cityNetworkMap";
import { CONFIG_SECTIONS, type ConfigFieldSpec, type ConfigSectionSpec } from "./configStudio";
import { api } from "./lib/api";
import { ODDemandMap } from "./odDemandMap";
import { AccidentImpactScatter, SeverityDistributionChart, TimeSeriesChart } from "./resultsCharts";
import { TrafficFeedMap } from "./trafficFeedMap";
import type {
  Branding,
  CityDemandPreview,
  CityNetworkPreview,
  CityRecord,
  ConfigDocument,
  JobRecord,
  LocationSearchResult,
  ResultRunSummary,
  TrafficFeedPreview,
  TrafficFeedSourceRecord,
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
type OSMSubtab = "new" | "extracted";
type FeedSubtab = "new" | "exported";
type GeneratorSubtab = "build" | "view";
type InfoModal = {
  title: string;
  sections: Array<{ heading: string; body: string[] }>;
} | null;
const APP_VERSION = "0.2.3";

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
  { key: "documentation", label: "Documentation" },
  { key: "configs", label: "Config Studio" },
  { key: "data_integrations", label: "Data & Integrations" },
  { key: "generators", label: "Generators" },
  { key: "simulations", label: "Simulations" },
  { key: "analysis", label: "Analysis" },
  { key: "results", label: "Results" },
  { key: "jobs", label: "Jobs" },
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

function isBlankValue(value: unknown): boolean {
  return value === undefined || value === null || String(value).trim() === "";
}

function buildGovgrDownloadsRoot(citySlug: string): string {
  return `data/cities/${citySlug}/govgr/downloads`;
}

function buildGovgrTargetsOutput(citySlug: string, calibrationYear: number, validationYear: number): string {
  return `data/cities/${citySlug}/govgr/targets/calibration_${calibrationYear}_validation_${validationYear}`;
}

function isAutoGovgrDownloadsPath(value: unknown): boolean {
  return typeof value === "string" && /^data\/cities\/[^/]+\/govgr\/downloads\/?$/.test(value.trim());
}

function isAutoGovgrTargetsPath(value: unknown): boolean {
  return (
    typeof value === "string"
    && /^data\/cities\/[^/]+\/govgr\/targets\/(calibration_\d{4}_validation_\d{4}|post_metro_\d{4}_\d{4})\/?$/.test(value.trim())
  );
}

function pickPreferredTrafficFeedSource(
  feeds: TrafficFeedSourceRecord[],
  currentSlug: string,
): string {
  if (feeds.some((source) => source.slug === currentSlug)) {
    return currentSlug;
  }
  return feeds.find((source) => source.catalog_count > 0)?.slug ?? feeds[0]?.slug ?? "";
}

function humanizeDocLabel(path: string): string {
  const fileName = path.split("/").slice(-1)[0].replace(/\.md$/i, "");
  return fileName
    .replace(/_/g, " ")
    .replace(/-/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function humanizeRoadGroup(group: string): string {
  if (group === "local_other") {
    return "Local / Other";
  }
  return group
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function workflowSlotStatusLabel(status: string): string {
  if (status === "ready") {
    return "Ready Now";
  }
  if (status === "planned") {
    return "Planned";
  }
  return status.replace(/_/g, " ");
}

const DOC_LABELS = new Map<string, string>([
  ["README.md", "Project Overview"],
  ["docs/README.md", "Documentation Index"],
  ["docs/STRUCTURE.md", "Repository Structure"],
  ["docs/REFERENCE.md", "Reference"],
  ["docs/GUI.md", "GUI Guide"],
  ["docs/modules/README.md", "Module Guides Index"],
  ["docs/modules/simulator.md", "Simulator Module"],
  ["docs/modules/generators.md", "Generators Module"],
  ["docs/modules/data-integrations.md", "Data & Integrations Module"],
  ["docs/modules/analysis.md", "Analysis Module"],
  ["docs/operations/README.md", "Command Guides Index"],
  ["docs/operations/new-location-workflow.md", "Add A New Location"],
  ["docs/operations/simulation.md", "Simulation Operations"],
  ["docs/operations/generators.md", "Generator Operations"],
  ["docs/operations/data-integrations.md", "Data & Integrations Operations"],
  ["docs/operations/analysis.md", "Analysis Operations"],
  ["docs/THESSALONIKI_OPERATOR_GUIDE.md", "Thessaloniki Operator Guide"],
  ["docs/SEATTLE_DATA.md", "Seattle Data Notes"],
  ["docs/SUMO_ACCIDENT_SIMULATOR_REVIEW.md", "SUMO Incident Model Review"],
  ["docs/MACOS_INSTALL.md", "macOS Install Guide"],
  ["docs/WORKTREES.md", "Git Worktrees Guide"],
  ["docs/CHANGELOG.md", "Changelog"],
]);

function docLabel(path: string): string {
  return DOC_LABELS.get(path) ?? humanizeDocLabel(path);
}

function groupDocumentationPaths(paths: string[]): Array<{ title: string; description: string; items: string[] }> {
  const sections: Array<{ title: string; description: string; items: string[] }> = [
    {
      title: "Start Here",
      description: "Read these first to understand the project and the overall documentation map.",
      items: ["README.md", "docs/README.md"],
    },
    {
      title: "Foundations",
      description: "Core repository, interface, and reference material that defines how SAS is organized and how it behaves.",
      items: ["docs/STRUCTURE.md", "docs/REFERENCE.md", "docs/GUI.md"],
    },
    {
      title: "Module Guides",
      description: "Narrative guides for the main parts of the system: simulator, generators, data intake, and analysis.",
      items: [
        "docs/modules/README.md",
        "docs/modules/simulator.md",
        "docs/modules/generators.md",
        "docs/modules/data-integrations.md",
        "docs/modules/analysis.md",
      ],
    },
    {
      title: "Command Guides",
      description: "Consolidated operator runbooks, including the end-to-end new-location path.",
      items: [
        "docs/operations/README.md",
        "docs/operations/new-location-workflow.md",
        "docs/operations/data-integrations.md",
        "docs/operations/generators.md",
        "docs/operations/simulation.md",
        "docs/operations/analysis.md",
      ],
    },
    {
      title: "City Notes And Reviews",
      description: "City-specific notes and technical review material for the current incident model.",
      items: [
        "docs/THESSALONIKI_OPERATOR_GUIDE.md",
        "docs/SEATTLE_DATA.md",
        "docs/SUMO_ACCIDENT_SIMULATOR_REVIEW.md",
      ],
    },
    {
      title: "Maintenance",
      description: "Environment setup, repository workflow guidance, and release history.",
      items: ["docs/MACOS_INSTALL.md", "docs/WORKTREES.md", "docs/CHANGELOG.md"],
    },
  ];

  const pathSet = new Set(paths);
  const used = new Set<string>();
  const groups = sections
    .map((section) => {
      const items = section.items.filter((path) => pathSet.has(path));
      items.forEach((path) => used.add(path));
      return { ...section, items };
    })
    .filter((section) => section.items.length > 0);

  const leftovers = paths
    .filter((path) => !used.has(path))
    .sort((left, right) => docLabel(left).localeCompare(docLabel(right)));
  if (leftovers.length > 0) {
    groups.push({
      title: "Other Docs",
      description: "Markdown files that are available in the repository but are not yet part of the curated documentation path.",
      items: leftovers,
    });
  }

  return groups;
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

function citySlugFromLocation(location: LocationSearchResult): string {
  return slugify(location.city || location.state || location.country || location.display_name);
}

function buildCityConfigPath(citySlug: string): string {
  return `configs/${citySlug}/default.yaml`;
}

function buildCityNetworkDir(citySlug: string): string {
  return `data/cities/${citySlug}/network`;
}

function HelpBadge({ field }: { field: ConfigFieldSpec | WorkflowField }) {
  const hasContent = hasTooltipContent(field);
  if (!hasContent) {
    return null;
  }
  return <span className="help-badge">?</span>;
}

function hasTooltipContent(field: ConfigFieldSpec | WorkflowField): boolean {
  return Boolean(
    ("help" in field ? field.help ?? "" : "") ||
      ("description" in field ? field.description : "") ||
      ("example" in field ? field.example ?? "" : "") ||
      ("impact" in field ? field.impact ?? "" : ""),
  );
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
        <>
          <input
            type="text"
            list="sumo-config-paths"
            value={value === undefined || value === null ? "" : String(value)}
            onChange={(event) => onChange(event.target.value)}
            placeholder="data/cities/<city>/network/<city>.sumocfg"
          />
          <datalist id="sumo-config-paths">
            {sumoConfigGroups.flatMap((group) =>
              group.items.map((path) => (
                <option key={path} value={path}>
                  {group.folder}
                </option>
              )),
            )}
          </datalist>
        </>
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
    </label>
  );
}

function WorkflowInput({
  field,
  value,
  onChange,
  configPaths,
  cities,
}: {
  field: WorkflowField;
  value: unknown;
  onChange: (value: unknown) => void;
  configPaths: string[];
  cities: CityRecord[];
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
      ) : field.type === "city" ? (
        <select value={String(value ?? "")} onChange={(event) => onChange(event.target.value)}>
          <option value="">Select extracted city</option>
          {cities.map((city) => (
            <option key={city.slug} value={city.slug}>
              {city.display_name}
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
      ) : field.type === "choice_list" ? (
        <div className="choice-list-grid">
          {(field.options ?? []).map((option) => {
            const selected = Array.isArray(value) ? (value as string[]).includes(option) : false;
            return (
              <label key={option} className={`choice-pill ${selected ? "is-selected" : ""}`}>
                <input
                  type="checkbox"
                  checked={selected}
                  onChange={(event) => {
                    const current = Array.isArray(value) ? [...(value as string[])] : [];
                    if (event.target.checked) {
                      onChange([...current, option]);
                    } else {
                      onChange(current.filter((item) => item !== option));
                    }
                  }}
                />
                <span>{option}</span>
              </label>
            );
          })}
        </div>
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

function WorkflowCard({
  workflow,
  values,
  onChange,
  onLaunch,
  configPaths,
  cities,
  extraNote,
  disabled,
}: {
  workflow: WorkflowSpec;
  values: Record<string, unknown>;
  onChange: (name: string, value: unknown) => void;
  onLaunch: () => void;
  configPaths: string[];
  cities: CityRecord[];
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
            cities={cities}
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
  const [osmSubtab, setOsmSubtab] = useState<OSMSubtab>("new");
  const [feedSubtab, setFeedSubtab] = useState<FeedSubtab>("new");
  const [generatorSubtab, setGeneratorSubtab] = useState<GeneratorSubtab>("build");
  const [cities, setCities] = useState<CityRecord[]>([]);
  const [selectedGeneratorCitySlug, setSelectedGeneratorCitySlug] = useState<string>("");
  const [selectedGeneratorDemandPreview, setSelectedGeneratorDemandPreview] = useState<CityDemandPreview | null>(null);
  const [selectedCitySlug, setSelectedCitySlug] = useState<string>("");
  const [selectedCityPreview, setSelectedCityPreview] = useState<CityNetworkPreview | null>(null);
  const [networkSelectionMode, setNetworkSelectionMode] = useState<NetworkSelectionMode>("click");
  const [trafficFeedSources, setTrafficFeedSources] = useState<TrafficFeedSourceRecord[]>([]);
  const [selectedTrafficFeedCitySlug, setSelectedTrafficFeedCitySlug] = useState<string>("");
  const [selectedTrafficFeedTargetCitySlug, setSelectedTrafficFeedTargetCitySlug] = useState<string>("");
  const [selectedTrafficFeedPreview, setSelectedTrafficFeedPreview] = useState<TrafficFeedPreview | null>(null);
  const [selectedTrafficFeedPath, setSelectedTrafficFeedPath] = useState<string | null>(null);
  const [trafficFeedTree, setTrafficFeedTree] = useState<TreeNode[]>([]);
  const [selectedTrafficFeedFile, setSelectedTrafficFeedFile] = useState<string | null>(null);
  const [selectedTrafficFeedFileText, setSelectedTrafficFeedFileText] = useState<string>("");
  const [networkViewMode, setNetworkViewMode] = useState<NetworkViewMode>("speed");
  const [selectedWayIds, setSelectedWayIds] = useState<string[]>([]);
  const [bulkRoadGroups, setBulkRoadGroups] = useState<string[]>([]);
  const [bulkSpeedClass, setBulkSpeedClass] = useState<string>("any");
  const [bulkLaneClass, setBulkLaneClass] = useState<string>("any");
  const [bulkDirectionClass, setBulkDirectionClass] = useState<string>("any");
  const [bulkSpeedValue, setBulkSpeedValue] = useState<number | "">("");
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
  const documentationGroups = useMemo(() => groupDocumentationPaths(docPaths), [docPaths]);
  const configFolderChoices = useMemo(() => [...topLevelFolders(configPaths), "__new__"], [configPaths]);
  const computedNewConfigPath = useMemo(
    () => buildConfigPath(newConfigFolderChoice, newConfigCustomFolder, newConfigName),
    [newConfigFolderChoice, newConfigCustomFolder, newConfigName],
  );

  const workflowsById = useMemo(
    () => Object.fromEntries(workflowSpecs.map((workflow) => [workflow.id, workflow])) as Record<string, WorkflowSpec>,
    [workflowSpecs],
  );
  const osmFieldsByName = useMemo(
    () =>
      Object.fromEntries(
        ((workflowsById["integration.fetch_osm"]?.fields ?? []) as WorkflowField[]).map((field) => [field.name, field]),
      ) as Record<string, WorkflowField>,
    [workflowsById],
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
  const generatorWorkflows = useMemo(
    () => activeCategoryWorkflows.filter((workflow) => workflow.category === "Generators"),
    [activeCategoryWorkflows],
  );

  const logoSrc = assetPath(branding.logo_path, "/branding/antifragicity-logo-main-h.svg");
  const euLogoSrc = assetPath(branding.eu_logo_path, "/branding/eu-funded-by-eu.png");
  const localityPolygons = useMemo(() => normalizeGeoJsonCoordinates(selectedLocation?.geojson), [selectedLocation]);
  const shapeBounds = useMemo(() => boundsFromPoints(customShapePoints), [customShapePoints]);
  const selectedCity = useMemo(
    () => cities.find((city) => city.slug === selectedCitySlug) ?? null,
    [cities, selectedCitySlug],
  );
  const selectedTrafficFeedSource = useMemo(
    () => trafficFeedSources.find((source) => source.slug === selectedTrafficFeedCitySlug) ?? null,
    [trafficFeedSources, selectedTrafficFeedCitySlug],
  );
  const selectedTrafficFeedTargetCity = useMemo(
    () => cities.find((city) => city.slug === selectedTrafficFeedTargetCitySlug) ?? null,
    [cities, selectedTrafficFeedTargetCitySlug],
  );
  const govgrTargetCalibrationYear = Number(workflowValues["integration.govgr_targets"]?.calibration_year ?? 2025);
  const govgrTargetValidationYear = Number(workflowValues["integration.govgr_targets"]?.validation_year ?? 2026);
  const extractedRoadGroups = useMemo(() => {
    const groups = new Set<string>();
    for (const feature of selectedCityPreview?.features ?? []) {
      groups.add(classifyRoadGroup(feature.road_type));
    }
    return Array.from(groups).sort();
  }, [selectedCityPreview]);
  const extractedSpeedClasses = useMemo(() => {
    const values = new Set<string>();
    for (const feature of selectedCityPreview?.features ?? []) {
      values.add(feature.speed_kph === null ? "unknown" : String(feature.speed_kph));
    }
    return ["any", ...Array.from(values).sort((left, right) => {
      if (left === "unknown") {
        return -1;
      }
      if (right === "unknown") {
        return 1;
      }
      return Number(left) - Number(right);
    })];
  }, [selectedCityPreview]);
  const extractedLaneClasses = useMemo(() => {
    const values = new Set<string>();
    for (const feature of selectedCityPreview?.features ?? []) {
      values.add(feature.lane_count === null ? "unknown" : String(feature.lane_count));
    }
    return ["any", ...Array.from(values).sort((left, right) => {
      if (left === "unknown") {
        return -1;
      }
      if (right === "unknown") {
        return 1;
      }
      return Number(left) - Number(right);
    })];
  }, [selectedCityPreview]);
  const selectedWayNames = useMemo(() => {
    const selected = new Set(selectedWayIds);
    return (selectedCityPreview?.features ?? [])
      .filter((feature) => selected.has(feature.id))
      .slice(0, 5)
      .map((feature) => feature.name || feature.id);
  }, [selectedCityPreview, selectedWayIds]);
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
          "The SUMO config field now accepts both discovered .sumocfg suggestions and a manual future path for a newly bootstrapped city.",
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
          "Use `New Extract` to search for a place, inspect the locality geometry, bootstrap a new city folder and default config, and launch the raw OSM download.",
          "Use `Extracted Network` to inspect saved city `.osm` files, review speed tags, visualize signalized intersections, and clean up speed limits before generator work begins.",
          "Road editing supports direct map selection, Shift multi-select, filter-based selection, and expansion to the full connected road when segments share the same road name.",
        ],
      },
      {
        heading: "Traffic Feeds",
        body: [
          "Use `New Feed Pull` to launch the current Thessaloniki gov.gr downloader and target-builder workflows while choosing the target city folder separately from the feed source.",
          "Use `Exported Feeds` to inspect source catalogs separately from target-city download runs and target folders, which is especially useful for alternate Thessaloniki network variants.",
          "When feed `Link_id` values match OSM way IDs, the page also renders a feed-alignment map so you can inspect current speed, congestion, and coverage directly on the city network.",
          "The feed page is now structured around provider workflow slots so additional city-specific adapters can be added later without changing the operator flow again.",
        ],
      },
    ],
  };
  const generatorInfo = {
    title: "Generators Guide",
    sections: [
      {
        heading: "What This Page Does",
        body: [
          "Use the generic city generator to turn an extracted city .osm into a runnable SUMO network, routes, and .sumocfg inside data/cities/<slug>/network/.",
          "Use Sioux Falls and Riverside separately, because they are benchmark and synthetic cases rather than extracted city folders.",
        ],
      },
      {
        heading: "Random Demand",
        body: [
          "For random demand, the main volume control is the route period. A smaller period requests departures more often, so the requested trip count grows roughly like end_time divided by period.",
          "Network size does not directly set the requested departure count, but it does affect how many valid trips can be constructed and how many vehicles remain active at once. Larger or better-connected networks usually sustain more simultaneous vehicles for the same period.",
          "If a network is sparse or disconnected, randomTrips validation can remove trips, so the final generated trip count can be lower than the rough end_time divided by period estimate.",
        ],
      },
      {
        heading: "OD Demand",
        body: [
          "Use OD demand when the city folder has a compatible OD matrix and centroid node file, or when you explicitly point the generator at those files.",
          "The View Inputs tab lets you inspect the detected OD files, sample rows, and top centroid-to-centroid flows before launching the build.",
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

  const refreshConfigMetadata = async () => {
    const [configData, sumoConfigData, outputFolderData] = await Promise.all([
      api.get<{ configs: Array<{ path: string }> }>("/api/configs"),
      api.get<{ sumo_configs: string[] }>("/api/sumo-configs"),
      api.get<{ output_folders: string[] }>("/api/output-folders"),
    ]);
    setConfigPaths(configData.configs.map((item) => item.path));
    setSumoConfigPaths(sumoConfigData.sumo_configs);
    setOutputFolderPaths(outputFolderData.output_folders);
  };

  const refreshCities = async () => {
    const cityData = await api.get<{ cities: CityRecord[] }>("/api/cities");
    setCities(cityData.cities);
    setSelectedCitySlug((current) =>
      current && cityData.cities.some((city) => city.slug === current) ? current : "",
    );
    setSelectedGeneratorCitySlug((current) =>
      current && cityData.cities.some((city) => city.slug === current) ? current : cityData.cities[0]?.slug || "",
    );
  };

  const refreshTrafficFeeds = async () => {
    const feedData = await api.get<{ feeds: TrafficFeedSourceRecord[] }>("/api/traffic-feeds");
    setTrafficFeedSources(feedData.feeds);
    setSelectedTrafficFeedCitySlug((current) => pickPreferredTrafficFeedSource(feedData.feeds, current));
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
      api.get<{ cities: CityRecord[] }>("/api/cities"),
      api.get<{ feeds: TrafficFeedSourceRecord[] }>("/api/traffic-feeds"),
      api.get<Branding>("/api/branding"),
      api.get<{ jobs: JobRecord[] }>("/api/jobs"),
      api.get<{ entries: TreeNode[] }>("/api/results"),
    ])
      .then(([workflowData, configData, sumoConfigData, outputFolderData, docsData, cityData, feedData, brandingData, jobData, resultData]) => {
        setWorkflowSpecs(workflowData.workflows);
        setConfigPaths(configData.configs.map((item) => item.path));
        setSumoConfigPaths(sumoConfigData.sumo_configs);
        setOutputFolderPaths(outputFolderData.output_folders);
        setDocPaths(docsData.docs.map((item) => item.path));
        setCities(cityData.cities);
        setSelectedCitySlug((current) =>
          current && cityData.cities.some((city) => city.slug === current) ? current : "",
        );
        setSelectedGeneratorCitySlug((current) =>
          current && cityData.cities.some((city) => city.slug === current) ? current : cityData.cities[0]?.slug || "",
        );
        setTrafficFeedSources(feedData.feeds);
        setSelectedTrafficFeedCitySlug((current) => pickPreferredTrafficFeedSource(feedData.feeds, current));
        setSelectedTrafficFeedTargetCitySlug((current) =>
          current && cityData.cities.some((city) => city.slug === current)
            ? current
            : pickPreferredTrafficFeedSource(feedData.feeds, "") || cityData.cities[0]?.slug || "",
        );
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
      void refreshConfigMetadata().catch(() => undefined);
      void refreshResults().catch(() => undefined);
      void refreshCities().catch(() => undefined);
      void refreshTrafficFeeds().catch(() => undefined);
    }, 2000);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!selectedCitySlug) {
      setSelectedCityPreview(null);
      setSelectedWayIds([]);
      return;
    }
    void api
      .get<CityNetworkPreview>(`/api/cities/${encodeURIComponent(selectedCitySlug)}/network-preview`)
      .then((preview) => {
        setSelectedCityPreview(preview);
        setSelectedWayIds((current) =>
          current.filter((wayId) => preview.features.some((feature) => feature.id === wayId)),
        );
      })
      .catch(() => setSelectedCityPreview(null));
  }, [selectedCitySlug]);

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

  useEffect(() => {
    if (!selectedTrafficFeedCitySlug) {
      setSelectedTrafficFeedPreview(null);
      return;
    }
    void api
      .get<TrafficFeedPreview>(
        `/api/traffic-feeds/${encodeURIComponent(selectedTrafficFeedCitySlug)}?target_city_slug=${encodeURIComponent(
          selectedTrafficFeedTargetCitySlug || selectedTrafficFeedCitySlug,
        )}`,
      )
      .then((preview) => {
        setSelectedTrafficFeedPreview(preview);
        setSelectedTrafficFeedPath(
          preview.catalog_datasets[0]?.path
            ?? preview.download_runs[0]?.path
            ?? preview.target_exports[0]?.path
            ?? null,
        );
        setSelectedTrafficFeedFile(null);
      })
      .catch(() => setSelectedTrafficFeedPreview(null));
  }, [selectedTrafficFeedCitySlug, selectedTrafficFeedTargetCitySlug]);

  useEffect(() => {
    if (selectedTrafficFeedTargetCitySlug && cities.some((city) => city.slug === selectedTrafficFeedTargetCitySlug)) {
      return;
    }
    if (selectedTrafficFeedCitySlug && cities.some((city) => city.slug === selectedTrafficFeedCitySlug)) {
      setSelectedTrafficFeedTargetCitySlug(selectedTrafficFeedCitySlug);
      return;
    }
    if (cities.length > 0) {
      setSelectedTrafficFeedTargetCitySlug(cities[0].slug);
    }
  }, [cities, selectedTrafficFeedCitySlug, selectedTrafficFeedTargetCitySlug]);

  useEffect(() => {
    if (!selectedTrafficFeedTargetCitySlug) {
      return;
    }
    applyTrafficFeedDefaults(selectedTrafficFeedTargetCitySlug);
  }, [selectedTrafficFeedTargetCitySlug, govgrTargetCalibrationYear, govgrTargetValidationYear]);

  useEffect(() => {
    if (!selectedGeneratorCitySlug) {
      setSelectedGeneratorDemandPreview(null);
      return;
    }
    applyGeneratorCityDefaults(selectedGeneratorCitySlug);
    void api
      .get<CityDemandPreview>(`/api/cities/${encodeURIComponent(selectedGeneratorCitySlug)}/demand-preview`)
      .then((preview) => setSelectedGeneratorDemandPreview(preview))
      .catch(() => setSelectedGeneratorDemandPreview(null));
  }, [selectedGeneratorCitySlug]);

  useEffect(() => {
    if (!selectedTrafficFeedPath) {
      setTrafficFeedTree([]);
      return;
    }
    void api
      .get<{ entries: TreeNode[] }>(`/api/fs/tree?path=${encodeURIComponent(selectedTrafficFeedPath)}&depth=3`)
      .then((data) => setTrafficFeedTree(data.entries))
      .catch(() => setTrafficFeedTree([]));
  }, [selectedTrafficFeedPath]);

  useEffect(() => {
    if (!selectedTrafficFeedFile) {
      setSelectedTrafficFeedFileText("");
      return;
    }
    void fetch(api.textUrl(selectedTrafficFeedFile))
      .then((response) => response.text())
      .then((text) => setSelectedTrafficFeedFileText(text))
      .catch(() => setSelectedTrafficFeedFileText("Unable to load feed file preview."));
  }, [selectedTrafficFeedFile]);

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
      await refreshConfigMetadata();
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
      await refreshConfigMetadata();
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

  const applyCityScaffoldDefaults = (citySlug: string) => {
    if (!citySlug) {
      return;
    }
    updateWorkflowValue("integration.fetch_osm", "city_slug", citySlug);
    updateWorkflowValue("integration.fetch_osm", "out", `data/cities/${citySlug}/network/${citySlug}.osm`);
    updateWorkflowValue("integration.fetch_osm", "config_out", buildCityConfigPath(citySlug));
  };

  const applyGeneratorCityDefaults = (citySlug: string) => {
    if (!citySlug) {
      return;
    }
    setWorkflowValues((current) => {
      const nextValues = { ...current };
      const generatorValues = { ...(nextValues["generator.city"] ?? {}) };
      const currentOutDir = generatorValues.out_dir;
      const currentConfig = generatorValues.config;
      if (isBlankValue(currentOutDir) || /^data\/cities\/[^/]+\/network\/?$/.test(String(currentOutDir))) {
        generatorValues.out_dir = buildCityNetworkDir(citySlug);
      }
      if (isBlankValue(currentConfig) || /^configs\/[^/]+\/default\.yaml$/.test(String(currentConfig))) {
        generatorValues.config = buildCityConfigPath(citySlug);
      }
      generatorValues.city_slug = citySlug;
      nextValues["generator.city"] = generatorValues;
      return nextValues;
    });
  };

  const applyTrafficFeedDefaults = (citySlug: string) => {
    if (!citySlug) {
      return;
    }
    setWorkflowValues((current) => {
      const nextValues = { ...current };
      const downloadValues = { ...(nextValues["integration.govgr_download"] ?? {}) };
      const targetValues = { ...(nextValues["integration.govgr_targets"] ?? {}) };
      const calibrationYear = Number(targetValues.calibration_year ?? 2025);
      const validationYear = Number(targetValues.validation_year ?? 2026);
      const nextDownloadsRoot = buildGovgrDownloadsRoot(citySlug);
      const nextTargetsOutput = buildGovgrTargetsOutput(citySlug, calibrationYear, validationYear);
      const currentDownloadOutput = downloadValues.output_dir;
      const currentDownloadsRoot = targetValues.downloads_root;
      const currentTargetsOutput = targetValues.output_dir;

      if (isBlankValue(currentDownloadOutput) || isAutoGovgrDownloadsPath(currentDownloadOutput)) {
        downloadValues.output_dir = nextDownloadsRoot;
      }
      if (isBlankValue(currentDownloadsRoot) || isAutoGovgrDownloadsPath(currentDownloadsRoot)) {
        targetValues.downloads_root = nextDownloadsRoot;
      }
      if (isBlankValue(currentTargetsOutput) || isAutoGovgrTargetsPath(currentTargetsOutput)) {
        targetValues.output_dir = nextTargetsOutput;
      }

      nextValues["integration.govgr_download"] = downloadValues;
      nextValues["integration.govgr_targets"] = targetValues;
      return nextValues;
    });
  };

  const toggleWaySelection = (featureId: string, additive: boolean) => {
    setSelectedWayIds((current) => {
      if (!additive) {
        return current.length === 1 && current[0] === featureId ? current : [featureId];
      }
      return current.includes(featureId) ? current.filter((item) => item !== featureId) : [...current, featureId];
    });
  };

  const selectWaysInBounds = (bounds: [number, number, number, number], additive: boolean) => {
    const matched = (selectedCityPreview?.features ?? []).filter((feature) =>
      feature.coords.some(([lat, lon]) => lat >= bounds[0] && lat <= bounds[2] && lon >= bounds[1] && lon <= bounds[3]),
    );
    const matchedIds = matched.map((feature) => feature.id);
    setSelectedWayIds((current) => {
      if (!additive) {
        return matchedIds;
      }
      return Array.from(new Set([...current, ...matchedIds]));
    });
    setMessage(`Selected ${matchedIds.length} road segment(s) from the map box`);
  };

  const selectWaysByFilters = () => {
    const matched = (selectedCityPreview?.features ?? []).filter((feature) => {
      if (bulkRoadGroups.length > 0 && !bulkRoadGroups.includes(classifyRoadGroup(feature.road_type))) {
        return false;
      }
      if (
        bulkSpeedClass !== "any" &&
        !(
          (bulkSpeedClass === "unknown" && feature.speed_kph === null) ||
          (feature.speed_kph !== null && String(feature.speed_kph) === bulkSpeedClass)
        )
      ) {
        return false;
      }
      if (
        bulkLaneClass !== "any" &&
        !(
          (bulkLaneClass === "unknown" && feature.lane_count === null) ||
          (feature.lane_count !== null && String(feature.lane_count) === bulkLaneClass)
        )
      ) {
        return false;
      }
      if (bulkDirectionClass === "oneway" && !feature.oneway) {
        return false;
      }
      if (bulkDirectionClass === "bidirectional" && feature.oneway) {
        return false;
      }
      return true;
    });
    setSelectedWayIds(matched.map((feature) => feature.id));
    setMessage(`Selected ${matched.length} road segment(s) in ${selectedCity?.display_name ?? "city"}`);
  };

  const selectConnectedNamedRoad = () => {
    const features = selectedCityPreview?.features ?? [];
    if (features.length === 0 || selectedWayIds.length === 0) {
      setMessage("Select at least one road segment first");
      return;
    }

    const byId = new Map(features.map((feature) => [feature.id, feature]));
    const matchingName = (name: string | null | undefined) =>
      (name ?? "").trim().toLowerCase();
    const adjacency = new Map<string, string[]>();

    for (const feature of features) {
      adjacency.set(feature.id, []);
    }

    for (let i = 0; i < features.length; i += 1) {
      const left = features[i];
      const leftName = matchingName(left.name);
      if (!leftName) {
        continue;
      }
      const leftNodes = new Set(left.node_ids);
      for (let j = i + 1; j < features.length; j += 1) {
        const right = features[j];
        if (matchingName(right.name) !== leftName) {
          continue;
        }
        if (!right.node_ids.some((nodeId) => leftNodes.has(nodeId))) {
          continue;
        }
        adjacency.get(left.id)?.push(right.id);
        adjacency.get(right.id)?.push(left.id);
      }
    }

    const queue = [...selectedWayIds];
    const visited = new Set<string>(selectedWayIds);
    const namedSeeds = selectedWayIds.filter((wayId) => Boolean(byId.get(wayId)?.name?.trim()));

    if (namedSeeds.length === 0) {
      setMessage("The current selection has no named road segments to expand");
      return;
    }

    while (queue.length > 0) {
      const current = queue.shift()!;
      const feature = byId.get(current);
      if (!feature || !feature.name) {
        continue;
      }
      for (const neighborId of adjacency.get(current) ?? []) {
        if (visited.has(neighborId)) {
          continue;
        }
        visited.add(neighborId);
        queue.push(neighborId);
      }
    }

    setSelectedWayIds(Array.from(visited));
    const namedCount = Array.from(visited).filter((wayId) => byId.get(wayId)?.name).length;
    setMessage(`Expanded selection to ${visited.size} connected segment(s)${namedCount ? " with the same road name" : ""}`);
  };

  const applySpeedLimitUpdate = async () => {
    if (!selectedCitySlug) {
      setMessage("Select an extracted city first");
      return;
    }
    if (selectedWayIds.length === 0) {
      setMessage("Select one or more road segments first");
      return;
    }
    if (bulkSpeedValue === "" || Number(bulkSpeedValue) <= 0) {
      setMessage("Set a positive speed limit value first");
      return;
    }
    try {
      const result = await api.post<{ updated_way_count: number }>(
        `/api/cities/${encodeURIComponent(selectedCitySlug)}/speed-limits`,
        {
          way_ids: selectedWayIds,
          speed_kph: Number(bulkSpeedValue),
        },
      );
      const preview = await api.get<CityNetworkPreview>(
        `/api/cities/${encodeURIComponent(selectedCitySlug)}/network-preview`,
      );
      setSelectedCityPreview(preview);
      setMessage(`Updated ${result.updated_way_count} road segment(s) to ${Number(bulkSpeedValue)} km/h`);
    } catch (error) {
      setMessage(`Speed limit update failed: ${String(error)}`);
    }
  };

  const deleteSelectedWays = async () => {
    if (!selectedCitySlug) {
      setMessage("Select an extracted city first");
      return;
    }
    if (selectedWayIds.length === 0) {
      setMessage("Select one or more road segments first");
      return;
    }
    if (!window.confirm(`Delete ${selectedWayIds.length} selected road segment(s) from the raw OSM extract?`)) {
      return;
    }
    try {
      const result = await api.post<{ deleted_way_count: number }>(
        `/api/cities/${encodeURIComponent(selectedCitySlug)}/delete-ways`,
        { way_ids: selectedWayIds },
      );
      const preview = await api.get<CityNetworkPreview>(
        `/api/cities/${encodeURIComponent(selectedCitySlug)}/network-preview`,
      );
      setSelectedCityPreview(preview);
      setSelectedWayIds([]);
      setMessage(`Deleted ${result.deleted_way_count} road segment(s) from the OSM extract`);
    } catch (error) {
      setMessage(`Road deletion failed: ${String(error)}`);
    }
  };

  const expandSelectionToNamedRoad = () => {
    if (selectedWayIds.length === 0) {
      setMessage("Select a road segment first");
      return;
    }
    selectConnectedNamedRoad();
  };

  const selectLocation = (location: LocationSearchResult) => {
    const citySlug = citySlugFromLocation(location);
    setSelectedLocation(location);
    setSelectedCitySlug(citySlug);
    const [south, north, west, east] = location.boundingbox;
    setCustomBounds([south, west, north, east]);
    setCustomShapePoints([]);
    setBoundaryMode(location.geojson ? "locality" : "bbox");
    updateWorkflowValue("integration.fetch_osm", "place", location.display_name);
    applyCityScaffoldDefaults(citySlug);
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
              cities={cities}
            />
          ))}
        </div>
      </article>
    </section>
  );

  const renderGeneratorViewSection = () => (
    <>
      <div className="section-header">
        <div>
          <h2>Generator Inputs</h2>
          <p className="muted">Inspect OD support files before generating routes. When no OD files exist for a city, the generic generator can still run with random demand.</p>
        </div>
        <span className="chip">{selectedGeneratorDemandPreview?.supported ? "OD ready" : "Random only"}</span>
      </div>
      <div className="generator-view-stack">
        <div className="structured-fields-grid">
          <label className="field">
            <div className="field-heading">
              <span>City</span>
            </div>
            <select value={selectedGeneratorCitySlug} onChange={(event) => setSelectedGeneratorCitySlug(event.target.value)}>
              <option value="">Select extracted city</option>
              {cities.map((city) => (
                <option key={city.slug} value={city.slug}>
                  {city.display_name}
                </option>
              ))}
            </select>
            <small>Choose the city folder whose OD and node support files you want to inspect.</small>
          </label>
          <div className="workflow-note-box">
            <strong>Detected Inputs</strong>
            <p>
              OD file: {selectedGeneratorDemandPreview?.od_file ?? "Not found"}
              <br />
              Node file: {selectedGeneratorDemandPreview?.node_file ?? "Not found"}
            </p>
          </div>
        </div>

        {selectedGeneratorDemandPreview?.issues.length ? (
          <div className="workflow-note-grid">
            {selectedGeneratorDemandPreview.issues.map((issue) => (
              <div key={issue} className="workflow-note-box danger-note">
                <strong>Input Issue</strong>
                <p>{issue}</p>
              </div>
            ))}
          </div>
        ) : null}

        {selectedGeneratorDemandPreview?.summary ? (
          <div className="network-stat-grid">
            <div className="summary-card">
              <span>OD Rows</span>
              <strong>{selectedGeneratorDemandPreview.summary.od_row_count.toLocaleString()}</strong>
            </div>
            <div className="summary-card">
              <span>Zones</span>
              <strong>{selectedGeneratorDemandPreview.summary.zone_count.toLocaleString()}</strong>
            </div>
            <div className="summary-card">
              <span>Total OD</span>
              <strong>{formatNumber(selectedGeneratorDemandPreview.summary.total_od, 0)}</strong>
            </div>
            <div className="summary-card">
              <span>Top Mapped Flows</span>
              <strong>{selectedGeneratorDemandPreview.summary.mapped_top_flow_count.toLocaleString()}</strong>
            </div>
          </div>
        ) : null}

        <div className="generator-view-grid">
          <section className="structured-group-card">
            <h4>OD Sample Table</h4>
            {selectedGeneratorDemandPreview?.sample_rows.length ? (
              <div className="table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Origin</th>
                      <th>Destination</th>
                      <th>OD Number</th>
                      <th>Intrazonal</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedGeneratorDemandPreview.sample_rows.map((row, index) => (
                      <tr key={`${row.origin}-${row.destination}-${index}`}>
                        <td>{row.origin}</td>
                        <td>{row.destination}</td>
                        <td>{formatNumber(row.od_number, 2)}</td>
                        <td>{row.intrazonal ? "Yes" : "No"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="muted">No OD rows are available for the selected city.</p>
            )}
          </section>
          <ODDemandMap preview={selectedGeneratorDemandPreview} />
        </div>
      </div>
    </>
  );

  const renderGeneratorsSection = () => (
    <section className="workflow-stack">
      <article className="panel">
        <div className="section-header">
          <div>
            <h2>Generators</h2>
            <p className="muted">Use the generic city generator for extracted locations under data/cities/, and keep the benchmark/synthetic generators separate for Sioux Falls and Riverside.</p>
          </div>
          <div className="button-row">
            <span className="chip">{generatorWorkflows.length} workflows</span>
            <button className="secondary-button" onClick={() => setInfoModal(generatorInfo)}>
              About This Page
            </button>
          </div>
        </div>
        <div className="subtab-row" aria-label="Generator tabs">
          <button className={`subtab-button ${generatorSubtab === "build" ? "is-active" : ""}`} onClick={() => setGeneratorSubtab("build")}>
            Build
          </button>
          <button className={`subtab-button ${generatorSubtab === "view" ? "is-active" : ""}`} onClick={() => setGeneratorSubtab("view")}>
            View Inputs
          </button>
        </div>
        {generatorSubtab === "build" ? (
          <>
            <div className="workflow-note-grid">
              <div className="workflow-note-box">
                <strong>Random Demand Logic</strong>
                <p>
                  Lower `Random Route Period` means more requested departures. With the current settings, the rough requested trip count is about {estimatedRandomTrips?.toLocaleString() ?? "n/a"} over {cityGeneratorEnd.toLocaleString()} seconds.
                </p>
              </div>
              <div className="workflow-note-box">
                <strong>Why Network Size Still Matters</strong>
                <p>
                  The request rate comes from time and period, but the final generated count and simultaneous vehicles still depend on connectivity, valid routes, and average trip length. Larger or better-connected networks often keep more vehicles active for the same period.
                </p>
              </div>
              <div className={`workflow-note-box ${selectedGeneratorDemandPreview?.supported ? "" : "danger-note"}`}>
                <strong>Current City Input Status</strong>
                <p>
                  {selectedGeneratorDemandPreview?.supported
                    ? `OD support files are available for ${selectedGeneratorCitySlug || "the selected city"}. You can use the View Inputs tab to inspect them before switching demand source to OD.`
                    : `No complete OD support set is currently detected for ${selectedGeneratorCitySlug || "the selected city"}, so random demand is the practical default unless you provide OD and node files explicitly.`}
                </p>
              </div>
            </div>
            <div className="workflow-grid">
              {generatorWorkflows.map((workflow) => (
                <WorkflowCard
                  key={workflow.id}
                  workflow={workflow}
                  values={workflowValues[workflow.id] ?? {}}
                  onChange={(name, value) => {
                    updateWorkflowValue(workflow.id, name, value);
                    if (workflow.id === "generator.city" && name === "city_slug" && typeof value === "string") {
                      setSelectedGeneratorCitySlug(value);
                      applyGeneratorCityDefaults(value);
                    }
                  }}
                  onLaunch={() => void launchWorkflow(workflow.id)}
                  configPaths={configPaths}
                  cities={cities}
                />
              ))}
            </div>
          </>
        ) : (
          renderGeneratorViewSection()
        )}
      </article>
    </section>
  );

  const osmWorkflow = workflowsById["integration.fetch_osm"];
  const osmValues = workflowValues["integration.fetch_osm"] ?? {};
  const cityGeneratorValues = workflowValues["generator.city"] ?? {};
  const cityGeneratorDemandSource = String(cityGeneratorValues.demand_source ?? "random");
  const cityGeneratorPeriod = Number(cityGeneratorValues.period ?? 1.5);
  const cityGeneratorEnd = Number(cityGeneratorValues.end ?? 7200);
  const estimatedRandomTrips =
    cityGeneratorDemandSource === "random" && cityGeneratorPeriod > 0
      ? Math.round(cityGeneratorEnd / cityGeneratorPeriod)
      : null;
  const metricsStats = selectedRunSummary?.metrics.stats;
  const metricsRows = selectedRunSummary?.metrics.rows ?? [];
  const simulationNetworkSummary =
    (selectedRunSummary?.simulation_summary?.network as Record<string, unknown> | undefined) ?? {};

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
                <div className="workflow-stack">
                  <div className="subtab-row" aria-label="OSM workflow tabs">
                    <button className={`subtab-button ${osmSubtab === "new" ? "is-active" : ""}`} onClick={() => setOsmSubtab("new")}>
                      New Extract
                    </button>
                    <button className={`subtab-button ${osmSubtab === "extracted" ? "is-active" : ""}`} onClick={() => setOsmSubtab("extracted")}>
                      Extracted Network
                    </button>
                  </div>

                  {osmSubtab === "new" ? (
                    <div className="osm-workspace">
                      <section className="workflow-stack">
                        <section className="workflow-card">
                          <div className="workflow-head">
                            <div>
                              <h3>Search And Bootstrap</h3>
                              <p className="workflow-description">Resolve a place with Nominatim, choose the city slug, and scaffold the standard SAS folders before downloading the OSM extract.</p>
                            </div>
                            <code>{osmWorkflow?.module}</code>
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
                            <p className="muted">Search results appear here. Selecting one fills the city scaffold and boundary controls automatically.</p>
                          )}
                        </section>
                        <section className="structured-group-card">
                          <h4>City Bootstrap</h4>
                          <div className="structured-fields-grid">
                            {["place", "city_slug", "out", "config_out", "config_template"].map((name) => {
                              const field = osmFieldsByName[name];
                              if (!field) {
                                return null;
                              }
                              return (
                                <WorkflowInput
                                  key={name}
                                  field={field}
                                  value={osmValues[name]}
                                  onChange={(next) => {
                                    updateWorkflowValue("integration.fetch_osm", name, next);
                                    if (name === "city_slug" && typeof next === "string") {
                                      setSelectedCitySlug(next);
                                      applyCityScaffoldDefaults(next);
                                    }
                                  }}
                                  configPaths={configPaths}
                                  cities={cities}
                                />
                              );
                            })}
                          </div>
                          <div className="chip-list">
                            <span className="chip">Creates `data/cities/&lt;slug&gt;/network`</span>
                            <span className="chip">Creates `configs/&lt;slug&gt;/default.yaml`</span>
                            <span className="chip">Template: `configs/templates/city_default.yaml`</span>
                          </div>
                        </section>
                        <section className="structured-group-card">
                          <h4>Extraction Settings</h4>
                          <div className="structured-fields-grid">
                            {["pad_km", "road_types", "all_features", "bootstrap_layout", "bootstrap_config", "email"].map((name) => {
                              const field = osmFieldsByName[name];
                              if (!field) {
                                return null;
                              }
                              return (
                                <WorkflowInput
                                  key={name}
                                  field={field}
                                  value={osmValues[name]}
                                  onChange={(next) => updateWorkflowValue("integration.fetch_osm", name, next)}
                                  configPaths={configPaths}
                                  cities={cities}
                                />
                              );
                            })}
                          </div>
                          <div className="workflow-note-grid">
                            <div className="workflow-note-box">
                              <strong>Road Type Filter</strong>
                              <p>The default road-type set keeps the highway classes that are usually useful for SUMO network generation, such as arterials, local streets, and service roads. Pedestrian-only classes stay out by default to keep the extract focused and lighter.</p>
                            </div>
                            <div className="workflow-note-box">
                              <strong>All Features Override</strong>
                              <p>Keep `All Features` off for normal SAS work. Turn it on only when you explicitly want the full raw OSM node, way, and relation set and do not want the road-type filter to apply.</p>
                            </div>
                            <div className="workflow-note-box">
                              <strong>Endpoint Defaults</strong>
                              <p>`Nominatim URL` resolves the place name and boundary. `Overpass URL` downloads the raw OSM XML. The default public OSM services are already populated for normal use.</p>
                            </div>
                          </div>
                        </section>
                        <section className="structured-group-card">
                          <h4>Service Endpoints</h4>
                          <div className="structured-fields-grid">
                            {["nominatim_url", "overpass_url", "user_agent"].map((name) => {
                              const field = osmFieldsByName[name];
                              if (!field) {
                                return null;
                              }
                              return (
                                <WorkflowInput
                                  key={name}
                                  field={field}
                                  value={osmValues[name]}
                                  onChange={(next) => updateWorkflowValue("integration.fetch_osm", name, next)}
                                  configPaths={configPaths}
                                  cities={cities}
                                />
                              );
                            })}
                          </div>
                        </section>
                      </section>

                      <section className="workflow-card">
                        <div className="workflow-head">
                          <div>
                            <h3>Boundary Preview</h3>
                            <p className="workflow-description">Boundary mode, numeric bounds, and map preview stay together here so the extraction area can be adjusted visually.</p>
                          </div>
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
                        <div className="button-row">
                          <button className="primary-button" onClick={() => void launchWorkflow("integration.fetch_osm")}>
                            Launch OSM Download
                          </button>
                        </div>
                      </section>
                    </div>
                  ) : (
                    <div className="extracted-workspace">
                      <section className="workflow-stack">
                        <section className="workflow-card">
                          <div className="workflow-head">
                            <div>
                              <h3>Extracted Cities</h3>
                              <p className="workflow-description">Choose an already extracted city and inspect or clean up the raw OSM network before moving to network generation.</p>
                            </div>
                          </div>
                          <label className="field">
                            <span>City</span>
                            <select
                              value={selectedCitySlug}
                              onChange={(event) => {
                                setSelectedCitySlug(event.target.value);
                                setSelectedWayIds([]);
                              }}
                            >
                              <option value="">Select an extracted city…</option>
                              {cities.map((city) => (
                                <option key={city.slug} value={city.slug}>
                                  {city.display_name}
                                </option>
                              ))}
                            </select>
                            <small>
                              {selectedCity?.config_path
                                ? `Config: ${selectedCity.config_path}`
                                : "No default config found for this city yet."}
                            </small>
                          </label>
                          {selectedCityPreview ? (
                            <div className="integration-snapshot-grid">
                              <div className="summary-card">
                                <span>Road Segments</span>
                                <strong>{selectedCityPreview.stats.feature_count.toLocaleString()}</strong>
                              </div>
                              <div className="summary-card">
                                <span>Speed Tagged</span>
                                <strong>{selectedCityPreview.stats.with_speed_limit.toLocaleString()}</strong>
                              </div>
                              <div className="summary-card">
                                <span>Signalized Nodes</span>
                                <strong>{selectedCityPreview.stats.signalized_intersection_count.toLocaleString()}</strong>
                              </div>
                            </div>
                          ) : (
                            <p className="muted">The map preview stays unloaded until you choose a city from the dropdown.</p>
                          )}
                        </section>
                        <section className="structured-group-card">
                          <h4>Select Roads</h4>
                          <div className="structured-fields-grid">
                            <label className="field">
                              <span>Road Group</span>
                              <div className="choice-list-grid">
                                {extractedRoadGroups.map((option) => {
                                  const selected = bulkRoadGroups.includes(option);
                                  return (
                                    <label key={option} className={`choice-pill ${selected ? "is-selected" : ""}`}>
                                      <input
                                        type="checkbox"
                                        checked={selected}
                                        onChange={(event) => {
                                          if (event.target.checked) {
                                            setBulkRoadGroups((current) => [...current, option]);
                                          } else {
                                            setBulkRoadGroups((current) => current.filter((item) => item !== option));
                                          }
                                        }}
                                      />
                                      <span>{humanizeRoadGroup(option)}</span>
                                    </label>
                                  );
                                })}
                              </div>
                              <small>
                                {bulkRoadGroups.length > 0
                                  ? `Filtering by: ${bulkRoadGroups.map(humanizeRoadGroup).join(", ")}`
                                  : "No road-group filter applied."}
                              </small>
                            </label>
                            <label className="field">
                              <span>Current Speed</span>
                              <select value={bulkSpeedClass} onChange={(event) => setBulkSpeedClass(event.target.value)}>
                                {extractedSpeedClasses.map((option) => (
                                  <option key={option} value={option}>
                                    {option === "any" ? "Any" : option === "unknown" ? "Unknown" : `${option} km/h`}
                                  </option>
                                ))}
                              </select>
                            </label>
                            <label className="field">
                              <span>Lanes</span>
                              <select value={bulkLaneClass} onChange={(event) => setBulkLaneClass(event.target.value)}>
                                {extractedLaneClasses.map((option) => (
                                  <option key={option} value={option}>
                                    {option === "any" ? "Any" : option === "unknown" ? "Unknown" : option}
                                  </option>
                                ))}
                              </select>
                            </label>
                            <label className="field">
                              <span>Direction</span>
                              <select value={bulkDirectionClass} onChange={(event) => setBulkDirectionClass(event.target.value)}>
                                <option value="any">Any</option>
                                <option value="oneway">One-way</option>
                                <option value="bidirectional">Bidirectional / unspecified</option>
                              </select>
                            </label>
                          </div>
                          <div className="filter-action-row">
                            <button className="secondary-button" onClick={selectWaysByFilters}>
                              Select Matches
                            </button>
                            <button className="secondary-button" onClick={() => setBulkRoadGroups([])}>
                              Reset Road Groups
                            </button>
                            <button className="secondary-button" onClick={expandSelectionToNamedRoad} disabled={selectedWayIds.length === 0}>
                              Select Connected Named Road
                            </button>
                            <button className="secondary-button" onClick={() => setNetworkSelectionMode(networkSelectionMode === "box" ? "click" : "box")}>
                              {networkSelectionMode === "box" ? "Exit Box Select" : "Enter Box Select"}
                            </button>
                            <button className="secondary-button" onClick={() => setSelectedWayIds([])} disabled={selectedWayIds.length === 0}>
                              Clear Selection
                            </button>
                          </div>
                          <p className="muted">Use the filters for bulk selection, click a road on the map, hold Shift to add or remove multiple roads, drag a box in `Box Select` mode without moving the map, or expand the current selection to the full connected road when the segments share the same name.</p>
                        </section>
                        <section className="structured-group-card">
                          <h4>Edit Speed Limits</h4>
                          <div className="structured-fields-grid">
                            <label className="field">
                              <span>New Speed Limit (km/h)</span>
                              <input
                                type="number"
                                value={bulkSpeedValue}
                                onChange={(event) => setBulkSpeedValue(event.target.value === "" ? "" : Number(event.target.value))}
                                placeholder="50"
                              />
                            </label>
                            <label className="field">
                              <span>Selection Summary</span>
                              <div className="selection-summary">
                                <strong>{selectedWayIds.length} road segment(s)</strong>
                                <small>{selectedWayNames.length ? selectedWayNames.join(", ") : "No roads selected yet."}</small>
                              </div>
                            </label>
                          </div>
                          <div className="button-row">
                            <button className="primary-button" onClick={() => void applySpeedLimitUpdate()}>
                              Apply To Selected
                            </button>
                            <button className="secondary-button danger-button" onClick={() => void deleteSelectedWays()} disabled={selectedWayIds.length === 0}>
                              Delete Selected
                            </button>
                          </div>
                          <div className="workflow-note-box">
                            <strong>Typical Cleanup Pattern</strong>
                            <p>Example: choose `Road Group = Local / Other` and `Current Speed = Unknown`, click `Select Matches`, then set a default speed and apply it to the selected roads.</p>
                          </div>
                          <div className="workflow-note-box danger-note">
                            <strong>Deletion Is Destructive</strong>
                            <p>Deleting selected road segments removes those OSM ways from the raw city extract. Use this when you intentionally want to exclude links before network generation.</p>
                          </div>
                        </section>
                      </section>

                      <CityNetworkMap
                        preview={selectedCityPreview}
                        mode={networkViewMode}
                        onModeChange={setNetworkViewMode}
                        selectionMode={networkSelectionMode}
                        onSelectionModeChange={setNetworkSelectionMode}
                        selectedWayIds={selectedWayIds}
                        onFeatureClick={toggleWaySelection}
                        onBoxSelect={selectWaysInBounds}
                      />
                    </div>
                  )}
                </div>
              ) : (
                <div className="workflow-stack">
                  <div className="subtab-row" aria-label="Traffic feed tabs">
                    <button className={`subtab-button ${feedSubtab === "new" ? "is-active" : ""}`} onClick={() => setFeedSubtab("new")}>
                      New Feed Pull
                    </button>
                    <button className={`subtab-button ${feedSubtab === "exported" ? "is-active" : ""}`} onClick={() => setFeedSubtab("exported")}>
                      Exported Feeds
                    </button>
                  </div>

                  {feedSubtab === "new" ? (
                    <>
                      <section className="workflow-card">
                        <div className="workflow-head">
                          <div>
                            <h3>Feed Source Setup</h3>
                            <p className="workflow-description">
                              The current operational feed integration is the Thessaloniki gov.gr source. The page is now structured around provider workflow slots so additional city adapters can be added later without changing the operator flow.
                            </p>
                          </div>
                        </div>
                        <div className="structured-fields-grid">
                          <label className="field">
                            <span>City Feed Integration</span>
                            <select
                              value={selectedTrafficFeedCitySlug}
                              onChange={(event) => {
                                setSelectedTrafficFeedCitySlug(event.target.value);
                                if (!selectedTrafficFeedTargetCitySlug) {
                                  setSelectedTrafficFeedTargetCitySlug(event.target.value);
                                  applyTrafficFeedDefaults(event.target.value);
                                }
                              }}
                            >
                              {trafficFeedSources.map((source) => (
                                <option key={source.slug} value={source.slug}>
                                  {source.display_name}
                                </option>
                              ))}
                            </select>
                            <small>
                              {selectedTrafficFeedSource
                                ? `${selectedTrafficFeedSource.provider.toUpperCase()} · ${selectedTrafficFeedSource.provider_root}`
                                : "No feed integrations discovered yet."}
                            </small>
                          </label>
                          <label className="field">
                            <span>Target Network / Output City</span>
                            <select
                              value={selectedTrafficFeedTargetCitySlug}
                              onChange={(event) => {
                                setSelectedTrafficFeedTargetCitySlug(event.target.value);
                                applyTrafficFeedDefaults(event.target.value);
                              }}
                            >
                              {cities.map((city) => (
                                <option key={city.slug} value={city.slug}>
                                  {city.display_name}
                                </option>
                              ))}
                            </select>
                            <small>
                              {selectedTrafficFeedTargetCity
                                ? `Feed outputs will be written under data/cities/${selectedTrafficFeedTargetCity.slug}/govgr/`
                                : "Choose the city folder whose network/config this feed should support."}
                            </small>
                          </label>
                        </div>
                        <div className="chip-list">
                          <span className="chip">Published catalogs: {selectedTrafficFeedSource?.catalog_count ?? 0}</span>
                          <span className="chip">Downloaded runs: {selectedTrafficFeedSource?.download_run_count ?? 0}</span>
                          <span className="chip">Target exports: {selectedTrafficFeedSource?.target_export_count ?? 0}</span>
                        </div>
                        {selectedTrafficFeedSource ? (
                          <div className="workflow-note-box">
                            <strong>{selectedTrafficFeedSource.provider_label}</strong>
                            <p>{selectedTrafficFeedSource.coverage_note}</p>
                          </div>
                        ) : null}
                        {selectedTrafficFeedSource && selectedTrafficFeedTargetCitySlug && selectedTrafficFeedTargetCitySlug !== selectedTrafficFeedSource.slug ? (
                          <div className="workflow-note-box danger-note">
                            <strong>Cross-Network Feed Reuse</strong>
                            <p>
                              You are using the {selectedTrafficFeedSource.display_name} feed workflow for a different target city folder. This is allowed for alternate network variants such as smaller or metropolitan Thessaloniki extracts, but you should validate the spatial and operational fit carefully.
                            </p>
                          </div>
                        ) : null}
                      </section>
                      <section className="structured-group-card">
                        <h4>Provider Workflow Slots</h4>
                        <div className="provider-slot-grid">
                          {(selectedTrafficFeedSource?.workflow_slots ?? []).map((slot) => (
                            <article key={slot.id} className="provider-slot-card">
                              <header>
                                <h4>{slot.title}</h4>
                                <span className={`status-chip ${slot.status}`}>{workflowSlotStatusLabel(slot.status)}</span>
                              </header>
                              <p>{slot.description}</p>
                            </article>
                          ))}
                        </div>
                        <div className="workflow-note-grid">
                          <div className="workflow-note-box">
                            <strong>Current Scope</strong>
                            <p>
                              Today the downloader and target builder are aligned to Thessaloniki IMET/CERTH feeds. The GUI keeps the city selector anyway so the same operator pattern can be reused when another feed source is added.
                            </p>
                          </div>
                          <div className="workflow-note-box">
                            <strong>Export Layout</strong>
                            <p>
                              Downloader runs normally write timestamped folders under `data/cities/&lt;city&gt;/govgr/downloads/`, unless you override the output path with a more specific run folder. Target-building writes scenario folders under `data/cities/&lt;city&gt;/govgr/targets/`.
                            </p>
                          </div>
                        </div>
                      </section>
                      <section className="structured-group-card">
                        <h4>Thessaloniki gov.gr Workflows</h4>
                        <p className="muted">
                          Use the downloader first for realtime and/or historical pulls, then build calibration and validation targets from the historical exports.
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
                                cities={cities}
                                disabled={selectedTrafficFeedSource?.provider !== "govgr" || !selectedTrafficFeedTargetCitySlug}
                                extraNote={
                                  selectedTrafficFeedSource?.provider === "govgr"
                                    ? selectedTrafficFeedTargetCitySlug === selectedTrafficFeedSource.slug
                                      ? "This provider currently matches the selected target city directly."
                                      : "This provider is currently Thessaloniki-specific, but it can still be used for alternate target-network variants with caution."
                                    : "This workflow is not ready for the selected provider yet."
                                }
                              />
                            );
                          })}
                        </div>
                      </section>
                    </>
                  ) : (
                    <div className="feed-workspace">
                      <section className="workflow-stack feed-left-stack">
                        <section className="workflow-card">
                          <div className="workflow-head">
                            <div>
                              <h3>Feed Exports</h3>
                              <p className="workflow-description">
                                Inspect the current feed catalogs from the selected source integration, then compare them with the download runs and target exports stored under the selected target city folder.
                              </p>
                            </div>
                          </div>
                          <div className="structured-fields-grid">
                            <label className="field">
                              <span>Catalog Source Integration</span>
                              <select
                                value={selectedTrafficFeedCitySlug}
                                onChange={(event) => {
                                  setSelectedTrafficFeedCitySlug(event.target.value);
                                  setSelectedTrafficFeedPath(null);
                                  setSelectedTrafficFeedFile(null);
                                }}
                              >
                                {trafficFeedSources.map((source) => (
                                  <option key={source.slug} value={source.slug}>
                                    {source.display_name}
                                  </option>
                                ))}
                              </select>
                              <small>
                                {selectedTrafficFeedPreview
                                  ? `${selectedTrafficFeedPreview.source.provider_root}`
                                  : "Select a discovered feed source to inspect its published catalog."}
                              </small>
                            </label>
                            <label className="field">
                              <span>Target City Exports</span>
                              <select
                                value={selectedTrafficFeedTargetCitySlug}
                                onChange={(event) => {
                                  setSelectedTrafficFeedTargetCitySlug(event.target.value);
                                  setSelectedTrafficFeedPath(null);
                                  setSelectedTrafficFeedFile(null);
                                }}
                              >
                                {cities.map((city) => (
                                  <option key={city.slug} value={city.slug}>
                                    {city.display_name}
                                  </option>
                                ))}
                              </select>
                              <small>
                                {selectedTrafficFeedPreview
                                  ? selectedTrafficFeedPreview.target_city.provider_root
                                  : "Select the city folder whose download runs and target exports you want to inspect."}
                              </small>
                            </label>
                          </div>
                          {selectedTrafficFeedPreview ? (
                            <div className="workflow-note-box">
                              <strong>{selectedTrafficFeedPreview.source.provider_label}</strong>
                              <p>{selectedTrafficFeedPreview.source.coverage_note}</p>
                            </div>
                          ) : null}
                          {selectedTrafficFeedPreview && selectedTrafficFeedPreview.source.slug !== selectedTrafficFeedPreview.target_city.slug ? (
                            <div className="workflow-note-box danger-note">
                              <strong>Source / Target Split</strong>
                              <p>
                                Published catalog bundles are being read from {selectedTrafficFeedPreview.source.display_name}, while download runs, target exports, and the alignment map are being inspected against {selectedTrafficFeedPreview.target_city.display_name}.
                              </p>
                            </div>
                          ) : null}
                          <div className="chip-list">
                            <span className="chip">Catalogs: {selectedTrafficFeedPreview?.catalog_datasets.length ?? 0}</span>
                            <span className="chip">Download runs: {selectedTrafficFeedPreview?.download_runs.length ?? 0}</span>
                            <span className="chip">Target exports: {selectedTrafficFeedPreview?.target_exports.length ?? 0}</span>
                          </div>
                        </section>

                        <section className="structured-group-card">
                          <h4>Published Feed Catalog</h4>
                          {selectedTrafficFeedPreview?.catalog_datasets.length ? (
                            <div className="feed-card-list">
                              {selectedTrafficFeedPreview.catalog_datasets.map((dataset) => (
                                <button
                                  key={dataset.id}
                                  type="button"
                                  className={`feed-summary-card ${selectedTrafficFeedPath === dataset.path ? "is-selected" : ""}`}
                                  onClick={() => {
                                    setSelectedTrafficFeedPath(dataset.path);
                                    setSelectedTrafficFeedFile(dataset.sample_csv?.path ?? dataset.datapackage_path);
                                  }}
                                >
                                  <div className="feed-summary-head">
                                    <strong>{dataset.title}</strong>
                                    {dataset.version ? <span className="chip">{dataset.version}</span> : null}
                                  </div>
                                  <p>{dataset.description.split("\n")[0]}</p>
                                  <div className="chip-list">
                                    <span className="chip">{dataset.resources.length} resources</span>
                                    <span className="chip">{dataset.sample_csv?.columns.length ?? 0} columns</span>
                                  </div>
                                </button>
                              ))}
                            </div>
                          ) : (
                            <p className="muted">No published feed catalog bundles were found under the selected source integration.</p>
                          )}
                        </section>

                        <section className="structured-group-card">
                          <h4>Downloaded Export Runs</h4>
                          {selectedTrafficFeedPreview?.download_runs.length ? (
                            <div className="feed-card-list">
                              {selectedTrafficFeedPreview.download_runs.map((run) => (
                                <button
                                  key={run.path}
                                  type="button"
                                  className={`feed-summary-card ${selectedTrafficFeedPath === run.path ? "is-selected" : ""}`}
                                  onClick={() => {
                                    setSelectedTrafficFeedPath(run.path);
                                    setSelectedTrafficFeedFile(run.quality_report_path);
                                  }}
                                >
                                  <div className="feed-summary-head">
                                    <strong>{run.name}</strong>
                                    <span className="chip">{run.datasets.length} datasets</span>
                                  </div>
                                  <p>
                                    {run.started_utc
                                      ? `Started ${run.started_utc}`
                                      : "Timestamped downloader export folder"}
                                  </p>
                                  <div className="feed-metric-grid">
                                    {run.datasets.map((dataset) => (
                                      <div key={dataset.name}>
                                        <span>{dataset.name}</span>
                                        <strong>{dataset.realtime_rows_clean ?? dataset.historical_files_downloaded ?? 0}</strong>
                                      </div>
                                    ))}
                                  </div>
                                </button>
                              ))}
                            </div>
                          ) : (
                            <p className="muted">No downloader run folders were found yet under the selected target city’s `govgr/downloads` root.</p>
                          )}
                        </section>

                        <section className="structured-group-card">
                          <h4>Built Target Exports</h4>
                          {selectedTrafficFeedPreview?.target_exports.length ? (
                            <div className="feed-card-list">
                              {selectedTrafficFeedPreview.target_exports.map((targetExport) => (
                                <button
                                  key={targetExport.path}
                                  type="button"
                                  className={`feed-summary-card ${selectedTrafficFeedPath === targetExport.path ? "is-selected" : ""}`}
                                  onClick={() => {
                                    setSelectedTrafficFeedPath(targetExport.path);
                                    setSelectedTrafficFeedFile(targetExport.summary_path);
                                  }}
                                >
                                  <div className="feed-summary-head">
                                    <strong>{targetExport.name}</strong>
                                    <span className="chip">{targetExport.sets.length} sets</span>
                                  </div>
                                  <p>
                                    Calibration {targetExport.calibration_year ?? "?"} · Validation {targetExport.validation_year ?? "?"}
                                  </p>
                                  <div className="chip-list">
                                    {targetExport.sets.map((setInfo) => (
                                      <span key={setInfo.name} className="chip">
                                        {setInfo.name}: {setInfo.files.length} files
                                      </span>
                                    ))}
                                  </div>
                                </button>
                              ))}
                            </div>
                          ) : (
                            <p className="muted">No target-export folders were found yet under the selected target city’s `govgr/targets` root.</p>
                          )}
                        </section>
                      </section>

                      <section className="workflow-stack feed-right-stack">
                        <TrafficFeedMap preview={selectedTrafficFeedPreview} />
                        <section className="workflow-card feed-browser-card">
                          <div className="workflow-head">
                            <div>
                              <h3>Export Browser</h3>
                              <p className="workflow-description">
                                Select a catalog, downloader run, or target export on the left to inspect its file tree and preview the generated files here.
                              </p>
                            </div>
                          </div>
                          <div className="feed-browser-grid">
                            <div className="feed-browser-tree">
                              <h4>{selectedTrafficFeedPath ?? "No export selected"}</h4>
                              {trafficFeedTree.length ? (
                                <TreeView nodes={trafficFeedTree} onSelect={setSelectedTrafficFeedFile} />
                              ) : (
                                <p className="muted">Select an export card on the left to load its files.</p>
                              )}
                            </div>
                            <div className="feed-browser-preview">
                              <h4>{selectedTrafficFeedFile ?? "File preview"}</h4>
                              {selectedTrafficFeedFile ? (
                                <pre className="file-preview">{selectedTrafficFeedFileText || "Loading…"}</pre>
                              ) : (
                                <p className="muted">Select a file from the export browser to preview it here.</p>
                              )}
                            </div>
                          </div>
                        </section>
                      </section>
                    </div>
                  )}
                </div>
              )}
            </article>
          </section>
        ) : null}

        {view === "generators" ? renderGeneratorsSection() : null}

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
                        <span>Peak Blocked Lanes</span>
                        <strong>{formatNumber(metricsStats?.peak_active_blocked_lanes, 0)}</strong>
                      </div>
                      <div className="summary-card">
                        <span>Mean Speed (km/h)</span>
                        <strong>{formatNumber(simulationNetworkSummary.mean_speed_kmh ?? metricsStats?.mean_speed_kmh)}</strong>
                      </div>
                      <div className="summary-card">
                        <span>Mean Delay (s)</span>
                        <strong>{formatNumber(metricsStats?.mean_delay_seconds)}</strong>
                      </div>
                    </div>

                    <div className="chart-grid">
                      <TimeSeriesChart
                        title="Traffic State"
                        data={metricsRows}
                        leftAxisLabel="Vehicles"
                        rightAxisLabel="Incidents / lanes"
                        series={[
                          { key: "vehicle_count", label: "Active vehicles", color: "#f93262", yAxisId: "left" },
                          { key: "active_accidents", label: "Active accidents", color: "#ff8a64", yAxisId: "right" },
                          { key: "active_blocked_lanes", label: "Blocked lanes", color: "#29131b", yAxisId: "right" },
                        ]}
                      />
                      <TimeSeriesChart
                        title="Speed And Delay"
                        data={metricsRows}
                        leftAxisLabel="Speed (km/h)"
                        rightAxisLabel="Delay (s)"
                        series={[
                          { key: "mean_speed_kmh", label: "Mean speed", color: "#f93262", yAxisId: "left", valueSuffix: "km/h" },
                          { key: "mean_delay_seconds", label: "Mean delay", color: "#ff8a64", yAxisId: "right", valueSuffix: "s" },
                        ]}
                      />
                      <TimeSeriesChart
                        title="Throughput And Speed Ratio"
                        data={metricsRows}
                        leftAxisLabel="Throughput / h"
                        rightAxisLabel="Speed ratio"
                        referenceY={{ value: 1.0, label: "Baseline ratio", color: "#29131b" }}
                        series={[
                          { key: "throughput_per_hour", label: "Throughput", color: "#f93262", yAxisId: "left" },
                          { key: "speed_ratio", label: "Speed ratio", color: "#ff8a64", yAxisId: "right" },
                        ]}
                      />
                      <TimeSeriesChart
                        title="Incident Timeline"
                        data={metricsRows}
                        leftAxisLabel="Accidents"
                        series={[
                          { key: "cumulative_accidents", label: "Triggered", color: "#f93262", yAxisId: "left" },
                          { key: "resolved_accidents", label: "Resolved", color: "#2ecc71", yAxisId: "left" },
                        ]}
                      />
                    </div>

                    <div className="results-detail-grid">
                      <section className="detail-card">
                        <h3>Accident Distribution</h3>
                        {selectedRunSummary.accidents ? (
                          <>
                            <SeverityDistributionChart counts={selectedRunSummary.accidents.by_severity} />
                            <div className="chip-list">
                              <span className="chip">Max duration: {formatNumber(selectedRunSummary.accidents.max_duration_seconds, 0)} s</span>
                              <span className="chip">Max queue: {formatNumber(selectedRunSummary.accidents.max_queue_length_vehicles, 0)}</span>
                              <span className="chip">Max affected: {formatNumber(selectedRunSummary.accidents.max_vehicles_affected, 0)}</span>
                              <span className="chip">Total rerouted: {formatNumber(selectedRunSummary.accidents.total_rerouted_vehicles, 0)}</span>
                              <span className="chip">Total blocked lanes: {formatNumber(selectedRunSummary.accidents.total_blocked_lanes, 0)}</span>
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
                        <h3>Accident Impact Scatter</h3>
                        <p className="muted">
                          Duration on the x-axis and vehicles affected on the y-axis. Use this to spot long incidents that also propagate strongly through the network.
                        </p>
                        <AccidentImpactScatter items={selectedRunSummary.accidents.items} />
                      </section>
                    ) : null}

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
                                <th>Rerouted</th>
                                <th>Peak Queue</th>
                                <th>Blocked Lanes</th>
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
                                  <td>{formatNumber(item.rerouted_vehicles, 0)}</td>
                                  <td>{formatNumber(item.peak_queue_length_vehicles, 0)}</td>
                                  <td>{formatNumber(item.blocked_lanes, 0)}</td>
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
                        <p className="muted">
                          This run now includes a richer machine-readable summary in `simulation_summary.json` alongside the CSV and event reports.
                        </p>
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
          <section className="docs-layout">
            <article className="panel docs-nav-panel">
              <div className="section-header">
                <div>
                  <h2>Documentation</h2>
                  <p className="muted">Browse the project guides in reading order. The library panel and the open document now scroll independently.</p>
                </div>
              </div>
              <div className="docs-nav-scroll">
                <div className="docs-group-stack">
                  {documentationGroups.map((group) => (
                    <section key={group.title} className="docs-group">
                      <h3>{group.title}</h3>
                      <p className="muted">{group.description}</p>
                      <div className="docs-group-list">
                        {group.items.map((path) => (
                          <button
                            key={path}
                            type="button"
                            className={selectedDocPath === path ? "doc-link doc-link-active" : "doc-link"}
                            onClick={() => setSelectedDocPath(path)}
                          >
                            <span>{docLabel(path)}</span>
                            <small>{path}</small>
                          </button>
                        ))}
                      </div>
                    </section>
                  ))}
                </div>
              </div>
            </article>

            <article className="panel docs-content-panel">
              <div className="section-header">
                <div>
                  <h2>{docLabel(selectedDocPath)}</h2>
                  <p className="muted">{selectedDocPath}</p>
                </div>
              </div>
              <div className="docs-content-scroll">
                <div className="markdown-preview">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{selectedDocText}</ReactMarkdown>
                </div>
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
            <span>Version {APP_VERSION}</span>
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
