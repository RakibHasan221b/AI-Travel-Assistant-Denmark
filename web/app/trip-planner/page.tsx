import type { Metadata } from "next";
import { TripPlannerClient } from "@/components/trip-planner/TripPlannerClient";

export const metadata: Metadata = {
  title: "Trip Planner",
};

export default function TripPlannerPage() {
  return (
    <main className="flex-1 px-6 py-12">
      <div className="mx-auto max-w-2xl">
        <h1 className="text-3xl font-semibold tracking-tight mb-2">Trip Planner</h1>
        <p className="text-ink-muted mb-8">
          Two agents collaborate on this: a Place Scout finds candidates, and a Concierge checks
          real weather and writes the final recommendation grounded in what the Scout actually
          found. First request after a period of idle time can take longer — the server is
          waking up.
        </p>

        <TripPlannerClient />
      </div>
    </main>
  );
}
