"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ForecastPoint } from "@/lib/types";
import { categoriesInOrder, pivotForecast } from "@/lib/stats";
import { ChartCard } from "./ChartCard";
import { ChartTooltip } from "./ChartTooltip";

const CHART_COLORS = [
  "var(--color-chart-1)",
  "var(--color-chart-2)",
  "var(--color-chart-3)",
  "var(--color-chart-4)",
  "var(--color-chart-5)",
];

export function OutdoorInterestForecastChart({ data }: { data: ForecastPoint[] }) {
  if (data.length === 0) {
    return (
      <ChartCard title="Weather-driven outdoor interest forecast">
        <p className="text-sm text-ink-muted">
          No forecast data in range — the forecast window may need refreshing.
        </p>
      </ChartCard>
    );
  }

  const pivoted = pivotForecast(data);
  // Fixed order, assigned once — color follows the category, never its
  // rank, so it stays stable across re-renders even if the set changes.
  const categories = categoriesInOrder(data);

  return (
    <ChartCard
      title="Weather-driven outdoor interest forecast"
      caption="Outdoor Interest Index (0-100) per category, current ~7-day forecast window."
    >
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={pivoted} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
          <CartesianGrid stroke="var(--color-line)" vertical={false} />
          <XAxis
            dataKey="forecast_date"
            tick={{ fill: "var(--color-ink-faint)", fontSize: 11 }}
            axisLine={{ stroke: "var(--color-line-strong)" }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: "var(--color-ink-faint)", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={32}
          />
          <Tooltip content={<ChartTooltip />} />
          <Legend wrapperStyle={{ fontSize: 12, color: "var(--color-ink-muted)" }} />
          {categories.map((category, i) => (
            <Line
              key={category}
              type="monotone"
              dataKey={category}
              name={category}
              stroke={CHART_COLORS[i % CHART_COLORS.length]}
              strokeWidth={2}
              dot={{ r: 4 }}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}
