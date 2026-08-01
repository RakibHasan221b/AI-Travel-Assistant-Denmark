import type { ForecastPoint } from "./types";

// TS equivalent of app/pages/3_Stats_Dashboard.py's df.pivot(): the API
// returns forecast rows in long format (one row per date+category), this
// reshapes to one row per date with one column per category, which is what
// a multi-series line chart wants. Kept as a pure function, not baked into
// the API response, so the API stays a thin passthrough of the SQL result.
export interface PivotedForecastRow {
  forecast_date: string;
  [category: string]: string | number;
}

export function pivotForecast(rows: ForecastPoint[]): PivotedForecastRow[] {
  const byDate = new Map<string, PivotedForecastRow>();
  for (const row of rows) {
    let entry = byDate.get(row.forecast_date);
    if (!entry) {
      entry = { forecast_date: row.forecast_date };
      byDate.set(row.forecast_date, entry);
    }
    entry[row.category] = row.avg_interest;
  }
  return [...byDate.values()].sort((a, b) =>
    a.forecast_date < b.forecast_date ? -1 : a.forecast_date > b.forecast_date ? 1 : 0
  );
}

// The fixed category order charts assign colors by — "color follows the
// entity, never its rank": categories keep the same color across renders
// even if which ones are present in a given forecast window changes.
export function categoriesInOrder(rows: ForecastPoint[]): string[] {
  const seen = new Set<string>();
  for (const row of rows) seen.add(row.category);
  return [...seen].sort();
}
