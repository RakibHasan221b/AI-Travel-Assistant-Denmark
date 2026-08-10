import type { Metadata } from "next";
import { getStats } from "@/lib/api";
import { QualityByNeighborhoodChart } from "@/components/stats/QualityByNeighborhoodChart";
import { QualityByCategoryChart } from "@/components/stats/QualityByCategoryChart";
import { VibeClusterSizesChart } from "@/components/stats/VibeClusterSizesChart";
import { RatedAspectsChart } from "@/components/stats/RatedAspectsChart";
import { OutdoorInterestForecastChart } from "@/components/stats/OutdoorInterestForecastChart";
import { ModelEvaluationSection } from "@/components/stats/ModelEvaluationSection";

export const metadata: Metadata = {
  title: "Stats Dashboard",
};

// Force dynamic rendering: without this, Next.js statically prerenders
// this page at BUILD time and bakes in whatever /stats returned then,
// which (a) contradicts the page's own "computed live" copy — stats would
// only update on the next redeploy — and (b) makes the production build
// itself depend on Render's free-tier API being warm and reachable at
// Vercel's build step, a real fragility a low-traffic dashboard doesn't
// need to accept just to save one fast per-request SQL round-trip.
export const dynamic = "force-dynamic";

// Async Server Component: no user input on this page, so the data fetch
// happens server-side at request time and the page renders with data
// already present at first paint — no client loading state needed here,
// unlike Explore/Trip Planner which are driven by a form.
export default async function StatsPage() {
  const stats = await getStats();

  return (
    <main className="flex-1 px-6 py-12">
      <div className="mx-auto max-w-4xl">
        <h1 className="text-3xl font-semibold tracking-tight mb-2">Stats Dashboard</h1>
        <p className="text-ink-muted mb-8">
          Real aggregate analytics, computed live from Postgres — every number here is
          computed, not invented.
        </p>

        <div className="grid sm:grid-cols-2 gap-4">
          <QualityByNeighborhoodChart data={stats.quality_by_neighborhood} />
          <QualityByCategoryChart data={stats.quality_by_category} />
          <VibeClusterSizesChart data={stats.vibe_cluster_sizes} />
          <RatedAspectsChart data={stats.rated_aspects} />
          <ModelEvaluationSection data={stats.model_evaluation} />
        </div>

        <div className="mt-4">
          <OutdoorInterestForecastChart data={stats.outdoor_interest_forecast} />
        </div>
      </div>
    </main>
  );
}
