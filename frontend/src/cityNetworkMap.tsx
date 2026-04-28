import { useEffect, useMemo } from "react";
import { CircleMarker, MapContainer, Polyline, Popup, TileLayer, useMap } from "react-leaflet";
import type { LeafletMouseEvent } from "leaflet";
import type { CityNetworkPreview } from "./types";

export type NetworkViewMode = "speed" | "road_type" | "lanes" | "signals";

function formatMetric(value: number | null | undefined, suffix = ""): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "Unknown";
  }
  return `${value.toLocaleString(undefined, { maximumFractionDigits: value >= 100 ? 0 : 1 })}${suffix}`;
}

function FitPreviewBounds({ bbox }: { bbox: [number, number, number, number] | null }) {
  const map = useMap();

  useEffect(() => {
    if (!bbox) {
      return;
    }
    map.fitBounds(
      [
        [bbox[0], bbox[1]],
        [bbox[2], bbox[3]],
      ],
      { padding: [24, 24] },
    );
  }, [bbox, map]);

  return null;
}

function roadClassGroup(roadType: string): string {
  if (["motorway", "motorway_link"].includes(roadType)) {
    return "motorway";
  }
  if (["trunk", "trunk_link"].includes(roadType)) {
    return "trunk";
  }
  if (["primary", "primary_link"].includes(roadType)) {
    return "primary";
  }
  if (["secondary", "secondary_link"].includes(roadType)) {
    return "secondary";
  }
  if (["tertiary", "tertiary_link"].includes(roadType)) {
    return "tertiary";
  }
  return "local_other";
}

function colorForSpeed(speedKph: number | null): string {
  if (speedKph === null) {
    return "#8c8c8c";
  }
  if (speedKph < 30) {
    return "#1b9e77";
  }
  if (speedKph < 50) {
    return "#66a61e";
  }
  if (speedKph < 70) {
    return "#e6ab02";
  }
  return "#d95f02";
}

function colorForRoadType(roadType: string): string {
  const palette: Record<string, string> = {
    motorway: "#8b0000",
    trunk: "#d7301f",
    primary: "#ef6548",
    secondary: "#fc8d59",
    tertiary: "#fdbb84",
    local_other: "#6baed6",
  };
  return palette[roadClassGroup(roadType)] ?? "#7f7f7f";
}

function styleForFeature(
  feature: CityNetworkPreview["features"][number],
  mode: NetworkViewMode,
  selected: boolean,
): { color: string; weight: number; opacity: number; dashArray?: string } {
  const selectionBoost = selected ? 2.4 : 0;

  if (mode === "speed") {
    return {
      color: selected ? "#f93262" : colorForSpeed(feature.speed_kph),
      weight: 2.6 + selectionBoost,
      opacity: 0.95,
      dashArray: !selected && feature.speed_kph === null ? "4 4" : undefined,
    };
  }
  if (mode === "road_type") {
    return {
      color: selected ? "#f93262" : colorForRoadType(feature.road_type),
      weight: 2.6 + selectionBoost,
      opacity: 0.95,
    };
  }
  if (mode === "signals") {
    return {
      color: selected ? "#f93262" : "rgba(41, 19, 27, 0.28)",
      weight: 2.2 + selectionBoost,
      opacity: selected ? 1 : 0.65,
    };
  }
  return {
    color: selected ? "#f93262" : feature.oneway ? "#29131b" : "#2b7aaf",
    weight: Math.max(2.2, Math.min(7.6, (feature.lane_count ?? 1) + 1 + selectionBoost)),
    opacity: 0.95,
    dashArray: !selected && feature.oneway ? undefined : !selected ? "10 6" : undefined,
  };
}

function legendForMode(mode: NetworkViewMode): Array<{ label: string; color: string; note?: string }> {
  if (mode === "speed") {
    return [
      { label: "< 30 km/h", color: "#1b9e77" },
      { label: "30-49 km/h", color: "#66a61e" },
      { label: "50-69 km/h", color: "#e6ab02" },
      { label: ">= 70 km/h", color: "#d95f02" },
      { label: "Unknown", color: "#8c8c8c", note: "Dashed" },
      { label: "Selected", color: "#f93262" },
    ];
  }
  if (mode === "road_type") {
    return [
      { label: "Motorway", color: "#8b0000" },
      { label: "Trunk", color: "#d7301f" },
      { label: "Primary", color: "#ef6548" },
      { label: "Secondary", color: "#fc8d59" },
      { label: "Tertiary", color: "#fdbb84" },
      { label: "Local / Other", color: "#6baed6" },
      { label: "Selected", color: "#f93262" },
    ];
  }
  if (mode === "signals") {
    return [
      { label: "Signalized intersection", color: "#165d7a" },
      { label: "Road context", color: "rgba(41, 19, 27, 0.28)" },
      { label: "Selected road", color: "#f93262" },
    ];
  }
  return [
    { label: "One-way", color: "#29131b", note: "Solid" },
    { label: "Bidirectional", color: "#2b7aaf", note: "Dashed" },
    { label: "Lane count", color: "#2b7aaf", note: "Line width" },
    { label: "Selected", color: "#f93262" },
  ];
}

export function CityNetworkMap({
  preview,
  mode,
  onModeChange,
  selectedWayIds,
  onFeatureClick,
}: {
  preview: CityNetworkPreview | null;
  mode: NetworkViewMode;
  onModeChange: (mode: NetworkViewMode) => void;
  selectedWayIds: string[];
  onFeatureClick: (featureId: string, additive: boolean) => void;
}) {
  const legend = useMemo(() => legendForMode(mode), [mode]);
  const selectedSet = useMemo(() => new Set(selectedWayIds), [selectedWayIds]);

  if (!preview) {
    return (
      <section className="workflow-card network-card">
        <div className="workflow-head">
          <div>
            <h3>Extracted Network Preview</h3>
            <p className="workflow-description">Select an extracted city to inspect its OSM network footprint.</p>
          </div>
        </div>
        <p className="muted">No extracted city is selected yet, or the selected city does not have an `.osm` file.</p>
      </section>
    );
  }

  const signalMode = mode === "signals";

  return (
    <section className="workflow-card network-card">
      <div className="workflow-head">
        <div>
          <h3>{preview.city.display_name}</h3>
          <p className="workflow-description">
            OSM preview from <code>{preview.source_path}</code>
          </p>
        </div>
        <div className="network-mode-row">
          <button className={mode === "speed" ? "tab-active" : ""} onClick={() => onModeChange("speed")}>
            Speed Limits
          </button>
          <button className={mode === "road_type" ? "tab-active" : ""} onClick={() => onModeChange("road_type")}>
            Road Type
          </button>
          <button className={mode === "lanes" ? "tab-active" : ""} onClick={() => onModeChange("lanes")}>
            Lanes / Direction
          </button>
          <button className={mode === "signals" ? "tab-active" : ""} onClick={() => onModeChange("signals")}>
            Signals
          </button>
        </div>
      </div>
      <div className="network-stat-grid">
        <div className="summary-card">
          <span>Road Segments</span>
          <strong>{preview.stats.feature_count.toLocaleString()}</strong>
        </div>
        <div className="summary-card">
          <span>With Speed Tag</span>
          <strong>{preview.stats.with_speed_limit.toLocaleString()}</strong>
        </div>
        <div className="summary-card">
          <span>Signalized Intersections</span>
          <strong>{preview.stats.signalized_intersection_count.toLocaleString()}</strong>
        </div>
        <div className="summary-card">
          <span>Selected Roads</span>
          <strong>{selectedWayIds.length.toLocaleString()}</strong>
        </div>
      </div>
      <div className="network-map-frame">
        <MapContainer
          center={preview.bbox ? [preview.bbox[0], preview.bbox[1]] : [40.0, 22.0]}
          zoom={12}
          scrollWheelZoom={true}
          preferCanvas={true}
          className="map-frame network-map"
        >
          <TileLayer
            attribution="&copy; OpenStreetMap contributors"
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <FitPreviewBounds bbox={preview.bbox} />
          {preview.features.map((feature) => {
            const selected = selectedSet.has(feature.id);
            return (
              <Polyline
                key={feature.id}
                positions={feature.coords}
                pathOptions={styleForFeature(feature, mode, selected)}
                eventHandlers={{
                  click: (event: LeafletMouseEvent) => {
                    onFeatureClick(feature.id, Boolean(event.originalEvent.shiftKey));
                  },
                }}
              >
                <Popup>
                  <div className="network-popup">
                    <strong>{feature.name || feature.id}</strong>
                    <span>Type: {feature.road_type}</span>
                    <span>Speed: {formatMetric(feature.speed_kph, " km/h")}</span>
                    <span>Lanes: {feature.lane_count ?? "Unknown"}</span>
                    <span>
                      Direction:{" "}
                      {feature.reverse_oneway
                        ? "One-way (reverse)"
                        : feature.oneway
                          ? "One-way"
                          : "Bidirectional / unspecified"}
                    </span>
                    <span>{selected ? "Selected for editing" : "Click to select · Shift-click to multi-select"}</span>
                  </div>
                </Popup>
              </Polyline>
            );
          })}
          {(signalMode ? preview.intersections : []).map((intersection) => (
            <CircleMarker
              key={intersection.id}
              center={intersection.coords}
              radius={Math.max(5, 4 + intersection.connected_road_count)}
              pathOptions={{
                color: "#165d7a",
                fillColor: "#1ea6d6",
                fillOpacity: 0.72,
                weight: 1.4,
              }}
            >
              <Popup>
                <div className="network-popup">
                  <strong>Signalized Intersection {intersection.id}</strong>
                  <span>Connected roads: {intersection.connected_road_count}</span>
                  <span>
                    Types: {intersection.connected_road_types.length ? intersection.connected_road_types.join(", ") : "Unknown"}
                  </span>
                </div>
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>
      <div className="network-legend-card">
        <h4>View Legend</h4>
        <div className="network-legend-grid">
          {legend.map((item) => (
            <div key={item.label} className="network-legend-row">
              <span className="network-legend-swatch" style={{ background: item.color }} />
              <div>
                <strong>{item.label}</strong>
                {item.note ? <small>{item.note}</small> : null}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export function classifyRoadGroup(roadType: string): string {
  return roadClassGroup(roadType);
}
