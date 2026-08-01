import type { VibeClusterSize } from "@/lib/types";
import { ChartCard } from "./ChartCard";
import { SingleSeriesBarChart } from "./SingleSeriesBarChart";

export function VibeClusterSizesChart({ data }: { data: VibeClusterSize[] }) {
  return (
    <ChartCard title="Vibe clusters">
      <SingleSeriesBarChart
        data={data.map((d) => ({ label: d.label, value: d.n_places }))}
        valueName="Places"
      />
    </ChartCard>
  );
}
