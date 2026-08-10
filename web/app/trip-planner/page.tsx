import type { Metadata } from "next";
import { TripPlannerClient } from "@/components/trip-planner/TripPlannerClient";

export const metadata: Metadata = {
  title: "Trip Planner",
};

export default function TripPlannerPage() {
  return (
    <main className="flex-1 px-6 py-12">
      <div className="mx-auto max-w-2xl">
        <h1 className="text-3xl font-semibold tracking-tight mb-2">Plan your Copenhagen day</h1>
        <p className="text-ink-muted mb-8">
          Tell us what you&apos;d like to see and do, and we&apos;ll create a personalized plan
          with nearby places, weather and recommendations.
        </p>

        <TripPlannerClient />
      </div>
    </main>
  );
}
