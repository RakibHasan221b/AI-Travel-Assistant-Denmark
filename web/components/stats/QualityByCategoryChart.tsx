import type { CategoryQuality } from "@/lib/types";
import { ChartCard } from "./ChartCard";
import { SingleSeriesBarChart } from "./SingleSeriesBarChart";

export function QualityByCategoryChart({ data }: { data: CategoryQuality[] }) {
  return (
    <ChartCard title="Avg quality score by category">
      <SingleSeriesBarChart
        data={data.map((d) => ({ label: d.category, value: Math.round(d.avg_quality_score) }))}
        valueName="Avg quality score"
      />
    </ChartCard>
  );
}
