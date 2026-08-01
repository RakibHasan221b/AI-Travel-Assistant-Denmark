"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ChartTooltip } from "./ChartTooltip";

export interface BarDatum {
  label: string;
  value: number;
}

// Shared by 3 of the 4 dashboard charts (neighborhood/category/cluster
// size) — all single-series nominal-category bars, so one hue (--accent)
// throughout; the axis label already carries identity, no per-bar color
// variety needed.
export function SingleSeriesBarChart({
  data,
  valueName,
}: {
  data: BarDatum[];
  valueName: string;
}) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
        <CartesianGrid stroke="var(--color-line)" vertical={false} />
        <XAxis
          dataKey="label"
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
        <Tooltip
          cursor={{ fill: "var(--color-line)", opacity: 0.4 }}
          content={<ChartTooltip />}
        />
        <Bar
          dataKey="value"
          name={valueName}
          fill="var(--color-accent)"
          radius={[4, 4, 0, 0]}
          maxBarSize={48}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}
