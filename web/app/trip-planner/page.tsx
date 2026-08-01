import type { Metadata } from "next";
import Link from "next/link";
import { TripPlannerClient } from "@/components/trip-planner/TripPlannerClient";

export const metadata: Metadata = {
  title: "Trip Planner — AI Denmark Explorer",
};

export default function TripPlannerPage() {
  return (
    <main className="flex-1 px-6 py-12">
      <div className="mx-auto max-w-2xl">
        <Link href="/" className="text-sm text-ink-muted hover:text-ink">
          ← AI Denmark Explorer
        </Link>

        <h1 className="text-3xl font-semibold tracking-tight mt-4 mb-2">Trip Planner</h1>
        <p className="text-ink-muted mb-8">
          Three agents collaborate on this: a Place Scout finds candidates, a Conditions Analyst
          checks real weather, and a Concierge writes the final recommendation grounded in what
          the other two actually found. First request after a period of idle time can take
          longer — the server is waking up.
        </p>

        <TripPlannerClient />
      </div>
    </main>
  );
}
