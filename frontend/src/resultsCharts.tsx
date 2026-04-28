import {
  Bar,
  BarChart,
  Brush,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type SeriesConfig = {
  key: string;
  label: string;
  color: string;
  yAxisId?: "left" | "right";
  valueSuffix?: string;
  strokeWidth?: number;
};

function formatValue(value: unknown, suffix?: string): string {
  const numeric = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(numeric)) {
    return "–";
  }
  const rendered = numeric.toLocaleString(undefined, {
    maximumFractionDigits: numeric >= 100 ? 0 : 2,
  });
  return suffix ? `${rendered} ${suffix}` : rendered;
}

function EmptyChart({ message }: { message: string }) {
  return <p className="muted">{message}</p>;
}

export function TimeSeriesChart({
  title,
  data,
  series,
  xLabel = "Time (min)",
  leftAxisLabel,
  rightAxisLabel,
  referenceY,
}: {
  title: string;
  data: Array<Record<string, number>>;
  series: SeriesConfig[];
  xLabel?: string;
  leftAxisLabel?: string;
  rightAxisLabel?: string;
  referenceY?: { value: number; label: string; color?: string };
}) {
  if (data.length < 2 || series.length === 0) {
    return (
      <div className="chart-card chart-card-large">
        <div className="chart-head">
          <h3>{title}</h3>
        </div>
        <EmptyChart message="Not enough time-series data yet." />
      </div>
    );
  }

  return (
    <div className="chart-card chart-card-large">
      <div className="chart-head">
        <h3>{title}</h3>
        <span>{data.length} samples</span>
      </div>
      <div className="interactive-chart">
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={data} margin={{ top: 12, right: 20, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(41, 19, 27, 0.10)" />
            <XAxis
              dataKey="timestamp_minutes"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(value) => `${Number(value).toFixed(0)}`}
              label={{ value: xLabel, position: "insideBottom", offset: -4 }}
            />
            <YAxis
              yAxisId="left"
              tickFormatter={(value) => formatValue(value)}
              label={
                leftAxisLabel
                  ? { value: leftAxisLabel, angle: -90, position: "insideLeft" }
                  : undefined
              }
            />
            {series.some((item) => item.yAxisId === "right") ? (
              <YAxis
                yAxisId="right"
                orientation="right"
                tickFormatter={(value) => formatValue(value)}
                label={
                  rightAxisLabel
                    ? { value: rightAxisLabel, angle: 90, position: "insideRight" }
                    : undefined
                }
              />
            ) : null}
            <Tooltip
              formatter={(value, _name, item) => {
                const itemName = typeof item?.name === "string" ? item.name : "";
                return [
                  formatValue(
                    value,
                    series.find((entry) => entry.label === itemName)?.valueSuffix,
                  ),
                  itemName,
                ];
              }}
              labelFormatter={(value) => `Time: ${Number(value).toFixed(2)} min`}
              contentStyle={{
                borderRadius: "0.9rem",
                border: "1px solid rgba(41, 19, 27, 0.12)",
                boxShadow: "0 14px 34px rgba(41, 19, 27, 0.12)",
              }}
            />
            <Legend />
            {referenceY ? (
              <ReferenceLine
                yAxisId="left"
                y={referenceY.value}
                stroke={referenceY.color ?? "#f93262"}
                strokeDasharray="4 4"
                label={referenceY.label}
              />
            ) : null}
            {series.map((item) => (
              <Line
                key={item.key}
                yAxisId={item.yAxisId ?? "left"}
                type="monotone"
                dataKey={item.key}
                name={item.label}
                stroke={item.color}
                strokeWidth={item.strokeWidth ?? 2.4}
                dot={false}
                activeDot={{ r: 4 }}
              />
            ))}
            <Brush
              dataKey="timestamp_minutes"
              height={24}
              stroke="rgba(249, 50, 98, 0.5)"
              travellerWidth={10}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export function SeverityDistributionChart({ counts }: { counts: Record<string, number> }) {
  const data = Object.entries(counts).map(([severity, count]) => ({ severity, count }));
  const palette: Record<string, string> = {
    MINOR: "#2ecc71",
    MODERATE: "#f39c12",
    MAJOR: "#e74c3c",
    CRITICAL: "#8b0000",
    UNKNOWN: "#7f8c8d",
  };

  if (data.length === 0) {
    return <EmptyChart message="No accident severity data found for this run." />;
  }

  return (
    <div className="interactive-chart">
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ top: 12, right: 20, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(41, 19, 27, 0.10)" />
          <XAxis dataKey="severity" />
          <YAxis allowDecimals={false} />
          <Tooltip
            formatter={(value) => [formatValue(value), "Accidents"]}
            contentStyle={{
              borderRadius: "0.9rem",
              border: "1px solid rgba(41, 19, 27, 0.12)",
              boxShadow: "0 14px 34px rgba(41, 19, 27, 0.12)",
            }}
          />
          <Bar dataKey="count" name="Accidents" radius={[8, 8, 0, 0]}>
            {data.map((entry) => (
              <Cell key={entry.severity} fill={palette[entry.severity] ?? "#f93262"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function AccidentImpactScatter({
  items,
}: {
  items: Array<Record<string, unknown>>;
}) {
  const data = items
    .map((item) => ({
      accident_id: String(item.accident_id ?? ""),
      severity: String(item.severity ?? "UNKNOWN"),
      duration_seconds: Number(item.duration_seconds ?? 0),
      vehicles_affected: Number(item.vehicles_affected ?? 0),
      peak_queue_length_vehicles: Number(item.peak_queue_length_vehicles ?? 0),
    }))
    .filter((item) => Number.isFinite(item.duration_seconds) && Number.isFinite(item.vehicles_affected));

  const palette: Record<string, string> = {
    MINOR: "#2ecc71",
    MODERATE: "#f39c12",
    MAJOR: "#e74c3c",
    CRITICAL: "#8b0000",
    UNKNOWN: "#7f8c8d",
  };

  if (data.length === 0) {
    return <EmptyChart message="No accident impact points are available for this run." />;
  }

  return (
    <div className="interactive-chart">
      <ResponsiveContainer width="100%" height={320}>
        <ScatterChart margin={{ top: 12, right: 20, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(41, 19, 27, 0.10)" />
          <XAxis
            type="number"
            dataKey="duration_seconds"
            name="Duration"
            tickFormatter={(value) => formatValue(value, "s")}
            label={{ value: "Duration (s)", position: "insideBottom", offset: -4 }}
          />
          <YAxis
            type="number"
            dataKey="vehicles_affected"
            name="Affected vehicles"
            allowDecimals={false}
            label={{ value: "Vehicles affected", angle: -90, position: "insideLeft" }}
          />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            formatter={(value, name) => [formatValue(value), String(name ?? "")]}
            labelFormatter={(_label, payload) => {
              const point = payload?.[0]?.payload as
                | { accident_id?: string; severity?: string; peak_queue_length_vehicles?: number }
                | undefined;
              return point
                ? `${point.accident_id} · ${point.severity} · queue ${formatValue(point.peak_queue_length_vehicles)}`
                : "";
            }}
            contentStyle={{
              borderRadius: "0.9rem",
              border: "1px solid rgba(41, 19, 27, 0.12)",
              boxShadow: "0 14px 34px rgba(41, 19, 27, 0.12)",
            }}
          />
          <Scatter name="Accidents" data={data}>
            {data.map((entry) => (
              <Cell key={entry.accident_id} fill={palette[entry.severity] ?? "#f93262"} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
