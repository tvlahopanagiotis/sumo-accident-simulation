import type { ReactNode } from "react";

export type IconName =
  | "analysis"
  | "config"
  | "data"
  | "docs"
  | "generators"
  | "info"
  | "jobs"
  | "menu"
  | "overview"
  | "results"
  | "search"
  | "simulations";

const ICON_PATHS: Record<IconName, string> = {
  overview: "M4 13.5h7.5V4H4v9.5Zm0 6.5h7.5v-4H4v4Zm10 0h6v-9h-6v9Zm0-11.5h6V4h-6v4.5Z",
  docs: "M6 3.5h8l4 4V20H6V3.5Zm7.25 1.75V8.5h3.25L13.25 5.25ZM8.5 11h7v1.5h-7V11Zm0 3h7v1.5h-7V14Zm0 3h5v1.5h-5V17Z",
  config: "M12 2.75 14.1 6l3.8.85-.35 3.9 2.45 3-3.1 2.35.15 3.9-3.75-1.05L10 21.25l-2.1-3.25-3.8-.85.35-3.9-2.45-3 3.1-2.35-.15-3.9 3.75 1.05L12 2.75Zm0 6.25a3 3 0 1 0 0 6 3 3 0 0 0 0-6Z",
  data: "M4 5.25C4 3.45 7.55 2 12 2s8 1.45 8 3.25v13.5C20 20.55 16.45 22 12 22s-8-1.45-8-3.25V5.25Zm2 4.15v2.85c1.3.9 3.5 1.5 6 1.5s4.7-.6 6-1.5V9.4c-1.45.7-3.55 1.1-6 1.1s-4.55-.4-6-1.1Zm0 6v3.35c.4.55 2.55 1.75 6 1.75s5.6-1.2 6-1.75V15.4c-1.45.7-3.55 1.1-6 1.1s-4.55-.4-6-1.1Z",
  generators: "M5 4h7a4 4 0 0 1 4 4v1h3l-4.25 4.25L10.5 9H14V8a2 2 0 0 0-2-2H5V4Zm14 16h-7a4 4 0 0 1-4-4v-1H5l4.25-4.25L13.5 15H10v1a2 2 0 0 0 2 2h7v2Z",
  simulations: "M5 5.5A2.5 2.5 0 1 1 7.5 8H7v3.25l4 2.35V10h2v3.6l4-2.35V8h-.5A2.5 2.5 0 1 1 19 5.5 2.5 2.5 0 0 1 17.05 8H19v4.4l-6 3.55V16.5h3.5a2.5 2.5 0 1 1 0 2H7.5a2.5 2.5 0 1 1 0-2H11v-.55l-6-3.55V8h1.95A2.5 2.5 0 0 1 5 5.5Z",
  analysis: "M4 19h16v2H4v-2Zm1-2V9h3v8H5Zm5 0V4h3v13h-3Zm5 0v-6h3v6h-3Z",
  results: "M5 4h14v16H5V4Zm2 2v12h10V6H7Zm1.5 8h2v2h-2v-2Zm3.25-4h2v6h-2v-6Zm3.25-2h2v8h-2V8Z",
  jobs: "M5 4h14v4H5V4Zm0 6h14v4H5v-4Zm0 6h14v4H5v-4Zm2-10.5V6h2v-.5H7Zm0 6V12h2v-.5H7Zm0 6V18h2v-.5H7Z",
  menu: "M4 6h16v2H4V6Zm0 5h16v2H4v-2Zm0 5h16v2H4v-2Z",
  info: "M12 2.75a9.25 9.25 0 1 1 0 18.5 9.25 9.25 0 0 1 0-18.5Zm-1 7.5V17h2v-6.75h-2ZM11 7v2h2V7h-2Z",
  search: "M10.5 4a6.5 6.5 0 0 1 5.1 10.54l4.18 4.18-1.42 1.42-4.18-4.18A6.5 6.5 0 1 1 10.5 4Zm0 2a4.5 4.5 0 1 0 0 9 4.5 4.5 0 0 0 0-9Z",
};

export function Icon({ name }: { name: IconName }) {
  return (
    <svg className="ui-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path d={ICON_PATHS[name]} />
    </svg>
  );
}

export function PageInfoButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button type="button" className="page-info-button" aria-label={label} title={label} onClick={onClick}>
      <Icon name="info" />
    </button>
  );
}

export function GuideButton({
  label = "Guide",
  icon = "search",
  onClick,
}: {
  label?: string;
  icon?: IconName;
  onClick: () => void;
}) {
  return (
    <button type="button" className="secondary-button guide-button" onClick={onClick}>
      <Icon name={icon} />
      <span>{label}</span>
    </button>
  );
}

export function TitleWithInfo({
  children,
  label,
  onClick,
}: {
  children: ReactNode;
  label: string;
  onClick: () => void;
}) {
  return (
    <span className="title-with-info">
      {children}
      <PageInfoButton label={label} onClick={onClick} />
    </span>
  );
}
