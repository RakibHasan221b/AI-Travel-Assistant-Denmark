import type { NeighborhoodQuality } from "@/lib/types";
import { ChartCard } from "./ChartCard";
import { SingleSeriesBarChart } from "./SingleSeriesBarChart";

export function QualityByNeighborhoodChart({ data }: { data: NeighborhoodQuality[] }) {
  return (
    <ChartCard
      title="Avg quality score by neighborhood"
      caption="Neighborhoods with fewer than 10 places omitted to avoid noisy averages."
    >
      <SingleSeriesBarChart
        data={data.map((d) => ({ label: d.neighborhood, value: Math.round(d.avg_quality_score) }))}
        valueName="Avg quality score"
      />
    </ChartCard>
  );
}
