import { useEffect, useMemo, useState } from "react";
import { MapContainer, Polyline, Popup, TileLayer, useMap } from "react-leaflet";
import type { TrafficFeedPreview } from "./types";

type FeedMapMode = "speed" | "congestion" | "coverage";

function FitBounds({ bbox }: { bbox: [number, number, number, number] | null }) {
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

function speedColor(value: number | null): string {
  if (value === null) {
    return "#8c8c8c";
  }
  if (value <= 15) {
    return "#7f0000";
  }
  if (value <= 25) {
    return "#a50f15";
  }
  if (value <= 35) {
    return "#de2d26";
  }
  if (value <= 45) {
    return "#fb6a4a";
  }
  if (value <= 55) {
    return "#fdae6b";
  }
  return "#31a354";
}

function congestionColor(value: string | null): string {
  const normalized = (value || "").toLowerCase();
  if (normalized === "high") {
    return "#a50f15";
  }
  if (normalized === "medium") {
    return "#ef6548";
  }
  if (normalized === "low") {
    return "#31a354";
  }
  return "#8c8c8c";
}

function coverageColor(): string {
  return "#165d7a";
}

function formatValue(value: number | null | undefined, suffix = ""): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "Unknown";
  }
  return `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })}${suffix}`;
}

function legend(mode: FeedMapMode): Array<{ label: string; color: string }> {
  if (mode === "speed") {
    return [
      { label: "<= 15 km/h", color: "#7f0000" },
      { label: "16-25 km/h", color: "#a50f15" },
      { label: "26-35 km/h", color: "#de2d26" },
      { label: "36-45 km/h", color: "#fb6a4a" },
      { label: "46-55 km/h", color: "#fdae6b" },
      { label: "> 55 km/h", color: "#31a354" },
      { label: "Unknown", color: "#8c8c8c" },
    ];
  }
  if (mode === "congestion") {
    return [
      { label: "High", color: "#a50f15" },
      { label: "Medium", color: "#ef6548" },
      { label: "Low", color: "#31a354" },
      { label: "Unknown", color: "#8c8c8c" },
    ];
  }
  return [{ label: "Matched feed-linked OSM ways", color: "#165d7a" }];
}

export function TrafficFeedMap({ preview }: { preview: TrafficFeedPreview | null }) {
  const [mode, setMode] = useState<FeedMapMode>("speed");
  const linked = preview?.linked_network ?? null;
  const legendItems = useMemo(() => legend(mode), [mode]);

  if (!linked) {
    return (
      <section className="workflow-card network-card">
        <div className="workflow-head">
          <div>
            <h3>Feed Alignment Map</h3>
            <p className="workflow-description">No OSM/feed alignment preview is available for the selected source.</p>
          </div>
        </div>
        <p className="muted">This source either has no compatible OSM way IDs or no city `.osm` extract to join against.</p>
      </section>
    );
  }

  return (
    <section className="workflow-card network-card">
      <div className="workflow-head">
        <div>
          <h3>Feed Alignment Map</h3>
          <p className="workflow-description">
            Matched feed `Link_id` values overlaid on the city OSM geometry. This shows only the subset that actually joins to current OSM way IDs.
          </p>
        </div>
        <div className="network-mode-row">
          <button className={mode === "speed" ? "tab-active" : ""} onClick={() => setMode("speed")}>
            Current Speed
          </button>
          <button className={mode === "congestion" ? "tab-active" : ""} onClick={() => setMode("congestion")}>
            Congestion
          </button>
          <button className={mode === "coverage" ? "tab-active" : ""} onClick={() => setMode("coverage")}>
            Coverage
          </button>
        </div>
      </div>
      <div className="network-stat-grid">
        <div className="summary-card">
          <span>Matched Feed Links</span>
          <strong>{linked.stats.matched_link_count.toLocaleString()}</strong>
        </div>
        <div className="summary-card">
          <span>Unmatched Feed Links</span>
          <strong>{linked.stats.unmatched_link_count.toLocaleString()}</strong>
        </div>
        <div className="summary-card">
          <span>Match Ratio</span>
          <strong>{(linked.stats.match_ratio * 100).toFixed(1)}%</strong>
        </div>
        <div className="summary-card">
          <span>Feed Speed Links</span>
          <strong>{linked.stats.feed_speed_link_count.toLocaleString()}</strong>
        </div>
      </div>
      <div className="network-map-frame">
        <MapContainer
          center={linked.bbox ? [linked.bbox[0], linked.bbox[1]] : [40.0, 22.0]}
          zoom={12}
          scrollWheelZoom={true}
          preferCanvas={true}
          className="map-frame network-map"
        >
          <TileLayer
            attribution="&copy; OpenStreetMap contributors"
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <FitBounds bbox={linked.bbox} />
          {linked.features.map((feature) => {
            const color =
              mode === "speed"
                ? speedColor(feature.speed_current_kph)
                : mode === "congestion"
                  ? congestionColor(feature.congestion_level)
                  : coverageColor();
            return (
              <Polyline
                key={feature.id}
                positions={feature.coords}
                pathOptions={{
                  color,
                  weight: 3.4,
                  opacity: 0.95,
                }}
              >
                <Popup>
                  <div className="network-popup">
                    <strong>{feature.name || feature.id}</strong>
                    <span>OSM way id: {feature.id}</span>
                    <span>Road type: {feature.road_type}</span>
                    <span>Speed limit: {formatValue(feature.speed_limit_kph, " km/h")}</span>
                    <span>Current feed speed: {formatValue(feature.speed_current_kph, " km/h")}</span>
                    <span>Current congestion: {feature.congestion_level || "Unknown"}</span>
                    <span>Latest timestamp: {feature.latest_timestamp || "Unknown"}</span>
                  </div>
                </Popup>
              </Polyline>
            );
          })}
        </MapContainer>
      </div>
      <div className="network-legend-card">
        <h4>View Legend</h4>
        <div className="network-legend-grid">
          {legendItems.map((item) => (
            <div key={item.label} className="network-legend-row">
              <span className="network-legend-swatch" style={{ background: item.color }} />
              <div>
                <strong>{item.label}</strong>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
