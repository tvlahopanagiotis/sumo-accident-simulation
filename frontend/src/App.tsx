import { useEffect, useMemo, useState } from "react";
import { api } from "./lib/api";
import type { Branding, ConfigDocument, JobRecord, TreeNode, WorkflowField, WorkflowSpec } from "./types";

type ViewKey = "overview" | "configs" | "workflows" | "jobs" | "results";

const DEFAULT_BRANDING: Branding = {
  name: "AntifragiCity SAS",
  project: "Horizon Europe AntifragiCity",
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
  font_note:
    "Official font files were not readable from this environment; the first version uses a clean fallback stack.",
};

const VIEW_LABELS: Array<{ key: ViewKey; label: string }> = [
  { key: "overview", label: "Overview" },
  { key: "configs", label: "Config Studio" },
  { key: "workflows", label: "Workflows" },
  { key: "jobs", label: "Jobs" },
  { key: "results", label: "Results" },
];

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

function PrimitiveField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  if (typeof value === "boolean") {
    return (
      <label className="field">
        <span>{label}</span>
        <input type="checkbox" checked={value} onChange={(event) => onChange(event.target.checked)} />
      </label>
    );
  }
  if (typeof value === "number") {
    return (
      <label className="field">
        <span>{label}</span>
        <input type="number" value={value} onChange={(event) => onChange(Number(event.target.value))} />
      </label>
    );
  }
  if (Array.isArray(value) && value.every((item) => ["string", "number"].includes(typeof item))) {
    const numeric = value.every((item) => typeof item === "number");
    return (
      <label className="field">
        <span>{label}</span>
        <textarea value={valueToListText(value)} rows={4} onChange={(event) => onChange(parseListValue(event.target.value, numeric))} />
      </label>
    );
  }
  return (
    <label className="field">
      <span>{label}</span>
      <input type="text" value={String(value ?? "")} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function NestedConfigEditor({
  value,
  path = [],
  onChange,
}: {
  value: unknown;
  path?: string[];
  onChange: (path: string[], value: unknown) => void;
}) {
  if (Array.isArray(value)) {
    return (
      <div className="nested-block">
        {value.map((item, index) => (
          <div key={`${path.join(".")}.${index}`} className="nested-entry">
            <div className="nested-label">[{index}]</div>
            {typeof item === "object" && item !== null ? (
              <NestedConfigEditor value={item} path={[...path, String(index)]} onChange={onChange} />
            ) : (
              <PrimitiveField
                label={`Item ${index + 1}`}
                value={item}
                onChange={(next) => onChange([...path, String(index)], next)}
              />
            )}
          </div>
        ))}
      </div>
    );
  }

  if (typeof value === "object" && value !== null) {
    return (
      <div className="nested-block">
        {Object.entries(value as Record<string, unknown>).map(([key, child]) => (
          <details key={[...path, key].join(".")} className="config-section" open={path.length < 2}>
            <summary>{key}</summary>
            {typeof child === "object" && child !== null ? (
              <NestedConfigEditor value={child} path={[...path, key]} onChange={onChange} />
            ) : (
              <PrimitiveField
                label={key}
                value={child}
                onChange={(next) => onChange([...path, key], next)}
              />
            )}
          </details>
        ))}
      </div>
    );
  }

  return <PrimitiveField label={path[path.length - 1] ?? "value"} value={value} onChange={(next) => onChange(path, next)} />;
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
  const commonProps = {
    placeholder: field.placeholder ?? field.help ?? "",
  };

  if (field.type === "boolean") {
    return (
      <label className="field">
        <span>{field.label}</span>
        <input type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(event.target.checked)} />
      </label>
    );
  }
  if (field.type === "choice") {
    return (
      <label className="field">
        <span>{field.label}</span>
        <select value={String(value ?? field.default ?? "")} onChange={(event) => onChange(event.target.value)}>
          {(field.options ?? []).map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </label>
    );
  }
  if (field.type === "config") {
    return (
      <label className="field">
        <span>{field.label}</span>
        <select value={String(value ?? field.default ?? "")} onChange={(event) => onChange(event.target.value)}>
          {configPaths.map((path) => (
            <option key={path} value={path}>
              {path}
            </option>
          ))}
        </select>
      </label>
    );
  }
  if (field.type === "number") {
    return (
      <label className="field">
        <span>{field.label}</span>
        <input
          type="number"
          value={value === undefined || value === null ? "" : String(value)}
          onChange={(event) => onChange(event.target.value === "" ? undefined : Number(event.target.value))}
          {...commonProps}
        />
      </label>
    );
  }
  if (field.type === "number_list") {
    return (
      <label className="field">
        <span>{field.label}</span>
        <textarea
          rows={4}
          value={valueToListText(value)}
          onChange={(event) => onChange(parseListValue(event.target.value, true))}
          placeholder={commonProps.placeholder}
        />
      </label>
    );
  }
  return (
    <label className="field">
      <span>{field.label}</span>
      <input type="text" value={value === undefined || value === null ? "" : String(value)} onChange={(event) => onChange(event.target.value)} {...commonProps} />
    </label>
  );
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

export default function App() {
  const [branding, setBranding] = useState<Branding>(DEFAULT_BRANDING);
  const [view, setView] = useState<ViewKey>("overview");
  const [workflowSpecs, setWorkflowSpecs] = useState<WorkflowSpec[]>([]);
  const [workflowValues, setWorkflowValues] = useState<Record<string, Record<string, unknown>>>({});
  const [configPaths, setConfigPaths] = useState<string[]>([]);
  const [selectedConfigPath, setSelectedConfigPath] = useState<string>("configs/thessaloniki/default.yaml");
  const [configDoc, setConfigDoc] = useState<ConfigDocument | null>(null);
  const [rawYaml, setRawYaml] = useState<string>("");
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [resultsTree, setResultsTree] = useState<TreeNode[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [selectedFileText, setSelectedFileText] = useState<string>("");
  const [message, setMessage] = useState<string>("Loading AntifragiCity SAS Console…");
  const [configMode, setConfigMode] = useState<"structured" | "raw">("structured");

  const selectedJob = useMemo(
    () => jobs.find((job) => job.id === selectedJobId) ?? jobs[0] ?? null,
    [jobs, selectedJobId],
  );

  useEffect(() => {
    document.documentElement.style.setProperty("--brand-primary", branding.colors.primary);
    document.documentElement.style.setProperty("--brand-secondary", branding.colors.secondary);
    document.documentElement.style.setProperty("--brand-ink", branding.colors.ink);
    document.documentElement.style.setProperty("--brand-surface", branding.colors.surface);
    document.documentElement.style.setProperty("--brand-surface-alt", branding.colors.surface_alt);
    document.documentElement.style.setProperty("--brand-border", branding.colors.border);
  }, [branding]);

  useEffect(() => {
    void Promise.all([
      api.get<{ workflows: WorkflowSpec[] }>("/api/workflows"),
      api.get<{ configs: Array<{ path: string }> }>("/api/configs"),
      api.get<Branding>("/api/branding"),
      api.get<{ jobs: JobRecord[] }>("/api/jobs"),
      api.get<{ entries: TreeNode[] }>("/api/results"),
    ])
      .then(([workflowData, configData, brandingData, jobData, resultData]) => {
        setWorkflowSpecs(workflowData.workflows);
        setConfigPaths(configData.configs.map((item) => item.path));
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
        setMessage("Ready");
      })
      .catch((error) => {
        setMessage(`Failed to load GUI metadata: ${String(error)}`);
      });
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
      .catch((error) => {
        setMessage(`Failed to load config: ${String(error)}`);
      });
  }, [selectedConfigPath]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      void api
        .get<{ jobs: JobRecord[] }>("/api/jobs")
        .then((data) => {
          setJobs(data.jobs);
          if (!selectedJobId && data.jobs[0]) {
            setSelectedJobId(data.jobs[0].id);
          }
        })
        .catch(() => undefined);
      void api
        .get<{ entries: TreeNode[] }>("/api/results")
        .then((data) => setResultsTree(data.entries))
        .catch(() => undefined);
    }, 2000);
    return () => window.clearInterval(interval);
  }, [selectedJobId]);

  useEffect(() => {
    if (!selectedFile) {
      setSelectedFileText("");
      return;
    }
    const isTextLike = /\.(json|csv|log|txt|yaml|yml|md|xml|html)$/i.test(selectedFile);
    if (!isTextLike) {
      setSelectedFileText("");
      return;
    }
    void fetch(api.textUrl(selectedFile))
      .then((response) => response.text())
      .then((text) => setSelectedFileText(text))
      .catch(() => setSelectedFileText("Unable to load file preview."));
  }, [selectedFile]);

  const workflowGroups = useMemo(() => {
    return workflowSpecs.reduce<Record<string, WorkflowSpec[]>>((acc, workflow) => {
      acc[workflow.category] = [...(acc[workflow.category] ?? []), workflow];
      return acc;
    }, {});
  }, [workflowSpecs]);

  const launchWorkflow = async (workflow: WorkflowSpec) => {
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

  const handleConfigChange = (path: string[], value: unknown) => {
    if (!configDoc) {
      return;
    }
    setConfigDoc({
      ...configDoc,
      config: cloneWithUpdate(configDoc.config, path, value) as Record<string, unknown>,
    });
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <img src="/branding/antifragicity-logo-main-h.svg" alt="AntifragiCity" className="brand-logo" />
          <p className="brand-project">{branding.project}</p>
        </div>
        <nav className="nav-list">
          {VIEW_LABELS.map((item) => (
            <button
              key={item.key}
              className={`nav-item ${view === item.key ? "is-active" : ""}`}
              onClick={() => setView(item.key)}
            >
              {item.label}
            </button>
          ))}
        </nav>
        <div className="sidebar-note">
          <strong>Visual system</strong>
          <p>{branding.font_note}</p>
        </div>
      </aside>

      <main className="main-panel">
        <header className="hero">
          <div>
            <p className="eyebrow">Simulation Control Console</p>
            <h1>{branding.name}</h1>
            <p className="hero-copy">
              Manage YAML configs, run simulations and assessments, launch generators and data tasks,
              and inspect outputs from one operator-focused interface.
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
                    <img
                      className="live-image"
                      src={`${api.fileUrl(selectedJob.live_progress_path)}&t=${Date.now()}`}
                      alt="Live progress"
                    />
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
                <h2>Config Studio</h2>
                <div className="button-row">
                  <button className="secondary-button" onClick={validateCurrentConfig}>
                    Validate
                  </button>
                  {configMode === "structured" ? (
                    <button className="primary-button" onClick={saveStructuredConfig}>
                      Save Structured
                    </button>
                  ) : (
                    <button className="primary-button" onClick={saveRawConfig}>
                      Save Raw YAML
                    </button>
                  )}
                </div>
              </div>
              <label className="field">
                <span>Config File</span>
                <select value={selectedConfigPath} onChange={(event) => setSelectedConfigPath(event.target.value)}>
                  {configPaths.map((path) => (
                    <option key={path} value={path}>
                      {path}
                    </option>
                  ))}
                </select>
              </label>
              <div className="tab-row">
                <button className={configMode === "structured" ? "tab-active" : ""} onClick={() => setConfigMode("structured")}>
                  Structured
                </button>
                <button className={configMode === "raw" ? "tab-active" : ""} onClick={() => setConfigMode("raw")}>
                  Raw YAML
                </button>
              </div>
              {configDoc && configMode === "structured" ? (
                <NestedConfigEditor value={configDoc.config} onChange={handleConfigChange} />
              ) : null}
              {configDoc && configMode === "raw" ? (
                <textarea className="raw-editor" rows={28} value={rawYaml} onChange={(event) => setRawYaml(event.target.value)} />
              ) : null}
            </article>
          </section>
        ) : null}

        {view === "workflows" ? (
          <section className="workflow-stack">
            {Object.entries(workflowGroups).map(([category, items]) => (
              <article key={category} className="panel">
                <div className="section-header">
                  <h2>{category}</h2>
                  <span className="chip">{items.length} workflows</span>
                </div>
                <div className="workflow-grid">
                  {items.map((workflow) => (
                    <section key={workflow.id} className="workflow-card">
                      <div className="workflow-head">
                        <h3>{workflow.title}</h3>
                        <code>{workflow.module}</code>
                      </div>
                      <p className="workflow-description">{workflow.description}</p>
                      <div className="workflow-fields">
                        {workflow.fields.map((field) => (
                          <WorkflowInput
                            key={`${workflow.id}.${field.name}`}
                            field={field}
                            value={workflowValues[workflow.id]?.[field.name]}
                            onChange={(next) =>
                              setWorkflowValues((current) => ({
                                ...current,
                                [workflow.id]: {
                                  ...(current[workflow.id] ?? {}),
                                  [field.name]: next,
                                },
                              }))
                            }
                            configPaths={configPaths}
                          />
                        ))}
                      </div>
                      <div className="button-row">
                        <button className="primary-button" onClick={() => void launchWorkflow(workflow)}>
                          Launch
                        </button>
                      </div>
                    </section>
                  ))}
                </div>
              </article>
            ))}
          </section>
        ) : null}

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
                          <img
                            className="live-image"
                            src={`${api.fileUrl(selectedJob.live_progress_path)}&t=${Date.now()}`}
                            alt="Live progress"
                          />
                        ) : null}
                        {selectedJob.report_path ? (
                          <iframe className="report-frame" src={api.fileUrl(selectedJob.report_path)} title="Generated report" />
                        ) : null}
                      </div>
                      {!selectedJob.live_progress_path && !selectedJob.report_path ? (
                        <p className="muted">No live image or HTML report available yet.</p>
                      ) : null}
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
          <section className="content-grid results-grid">
            <article className="panel">
              <h2>Results Explorer</h2>
              <TreeView nodes={resultsTree} onSelect={setSelectedFile} />
            </article>
            <article className="panel">
              <h2>Preview</h2>
              {selectedFile ? (
                <>
                  <p className="muted">{selectedFile}</p>
                  {/\.(png|jpg|jpeg|svg)$/i.test(selectedFile) ? (
                    <img className="preview-image" src={api.fileUrl(selectedFile)} alt={selectedFile} />
                  ) : null}
                  {/\.html$/i.test(selectedFile) ? (
                    <iframe className="report-frame" src={api.fileUrl(selectedFile)} title={selectedFile} />
                  ) : null}
                  {!/\.(png|jpg|jpeg|svg|html)$/i.test(selectedFile) ? (
                    <pre className="file-preview">{selectedFileText}</pre>
                  ) : null}
                </>
              ) : (
                <p className="muted">Select a file or directory from the results tree.</p>
              )}
            </article>
          </section>
        ) : null}
      </main>
    </div>
  );
}
