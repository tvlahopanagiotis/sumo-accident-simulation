import { useEffect } from "react";
import { CircleMarker, MapContainer, Polyline, Popup, TileLayer, useMap } from "react-leaflet";
import type { CityDemandPreview } from "./types";

function DemandBounds({ preview }: { preview: CityDemandPreview }) {
  const map = useMap();

  useEffect(() => {
    const points = preview.top_flows.flatMap((flow) => [flow.origin_coords, flow.destination_coords]);
    if (points.length === 0) {
      return;
    }
    const lats = points.map((point) => point[0]);
    const lons = points.map((point) => point[1]);
    map.fitBounds(
      [
        [Math.min(...lats), Math.min(...lons)],
        [Math.max(...lats), Math.max(...lons)],
      ],
      { padding: [24, 24] },
    );
  }, [map, preview]);

  return null;
}

function flowWeight(value: number, maxValue: number): number {
  if (maxValue <= 0) {
    return 2.2;
  }
  return 1.8 + (value / maxValue) * 5.2;
}

export function ODDemandMap({
  preview,
  selectedZoneId,
  onSelectedZoneChange,
}: {
  preview: CityDemandPreview | null;
  selectedZoneId?: string;
  onSelectedZoneChange?: (zoneId: string) => void;
}) {
  if (!preview?.supported || preview.top_flows.length === 0) {
    return (
      <section className="workflow-card network-card">
        <div className="workflow-head">
          <div>
            <h3>OD Flow Map</h3>
            <p className="workflow-description">No OD flow map is available for the selected city.</p>
          </div>
        </div>
        <p className="muted">OD and node support files are required before flows can be drawn on the map.</p>
      </section>
    );
  }

  const maxFlow = Math.max(...preview.top_flows.map((flow) => flow.od_number));
  const nodeMap = new Map(preview.nodes.map((node) => [node.zone_id, node.coords] as const));
  const selectedZone = selectedZoneId ? preview.zone_demands.find((zone) => zone.zone_id === selectedZoneId) : null;
  const visibleFlows = selectedZoneId
    ? preview.top_flows.filter((flow) => flow.origin === selectedZoneId || flow.destination === selectedZoneId)
    : preview.top_flows;

  return (
    <section className="workflow-card network-card">
      <div className="workflow-head">
        <div>
          <h3>OD Flow Map</h3>
          <p className="workflow-description">Top OD pairs drawn as desire lines between centroid nodes. This is an input inspection map, not a routed assignment output.</p>
        </div>
      </div>
      <div className="network-map-frame">
        <MapContainer
          center={preview.top_flows[0]?.origin_coords ?? [40.63, 22.95]}
          zoom={11}
          scrollWheelZoom={true}
          preferCanvas={true}
          className="map-frame network-map"
        >
          <TileLayer
            attribution="&copy; OpenStreetMap contributors"
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <DemandBounds preview={preview} />
          {visibleFlows.map((flow) => (
            <Polyline
              key={`${flow.origin}-${flow.destination}`}
              positions={[flow.origin_coords, flow.destination_coords]}
              pathOptions={{
                color: "#165d7a",
                opacity: 0.68,
                weight: flowWeight(flow.od_number, maxFlow),
              }}
            >
              <Popup>
                <div className="network-popup">
                  <strong>
                    {flow.origin} → {flow.destination}
                  </strong>
                  <span>OD volume: {flow.od_number.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
                </div>
              </Popup>
            </Polyline>
          ))}
          {(selectedZoneId ? preview.zone_demands.map((zone) => zone.zone_id) : Array.from(new Set(preview.top_flows.flatMap((flow) => [flow.origin, flow.destination])))).map((zoneId) => {
            const coords = nodeMap.get(zoneId);
            if (!coords) {
              return null;
            }
            const isSelected = zoneId === selectedZoneId;
            return (
              <CircleMarker
                key={zoneId}
                center={coords}
                radius={isSelected ? 8 : 4.5}
                pathOptions={{
                  color: isSelected ? "#165d7a" : "#f93262",
                  weight: isSelected ? 2 : 1,
                  fillColor: isSelected ? "#8bd3e6" : "#ffbea1",
                  fillOpacity: 0.92,
                }}
                eventHandlers={{
                  click: () => onSelectedZoneChange?.(zoneId),
                }}
              >
                <Popup>
                  <div className="network-popup">
                    <strong>Zone {zoneId}</strong>
                    <span>
                      Lat/Lon: {coords[0].toFixed(5)}, {coords[1].toFixed(5)}
                    </span>
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}
        </MapContainer>
      </div>
      {selectedZone ? (
        <div className="workflow-note-grid">
          <div className="workflow-note-box">
            <strong>Selected Zone {selectedZone.zone_id}</strong>
            <p>Origin demand: {selectedZone.origin_demand.toLocaleString(undefined, { maximumFractionDigits: 2 })}</p>
            <p>Destination demand: {selectedZone.destination_demand.toLocaleString(undefined, { maximumFractionDigits: 2 })}</p>
          </div>
          <div className="workflow-note-box">
            <strong>Displayed Flows</strong>
            <p>{visibleFlows.length.toLocaleString()} top mapped OD pair(s) connected to the selected zone.</p>
          </div>
        </div>
      ) : null}
      <div className="network-legend-card">
        <h4>How To Read This</h4>
        <div className="workflow-note-grid">
          <div className="workflow-note-box">
            <strong>Lines</strong>
            <p>Thicker lines represent larger OD volumes among the top mapped pairs.</p>
          </div>
          <div className="workflow-note-box">
            <strong>Nodes</strong>
            <p>Markers are centroid zones from the node file, not intersections or routed paths.</p>
          </div>
        </div>
      </div>
    </section>
  );
}
