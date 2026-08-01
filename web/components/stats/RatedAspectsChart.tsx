import type { RatedAspect } from "@/lib/types";
import { ChartCard } from "./ChartCard";
import { SingleSeriesBarChart } from "./SingleSeriesBarChart";

export function RatedAspectsChart({ data }: { data: RatedAspect[] }) {
  const totalMentions = data.reduce((sum, d) => sum + d.total_mentions, 0);

  return (
    <ChartCard
      title="Rated aspects (avg score, 1-5)"
      caption={`Based on ${totalMentions} total rated mentions.`}
    >
      <SingleSeriesBarChart
        data={data.map((d) => ({
          label: d.aspect_category,
          value: Math.round(d.mean_score * 10) / 10,
        }))}
        valueName="Avg score"
      />
    </ChartCard>
  );
}
