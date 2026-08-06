import Link from "next/link";
import { CopenhagenSkyline } from "@/components/CopenhagenSkyline";

export default function Home() {
  return (
    <main className="flex flex-1 flex-col items-center justify-center px-6 py-16">
      <div className="w-full max-w-2xl">
        <CopenhagenSkyline className="w-full h-auto mb-10" />
        <p className="text-xs font-mono tracking-widest uppercase text-ink-faint mb-3">
          AI Denmark Explorer
        </p>
        <h1 className="text-4xl font-semibold tracking-tight mb-4">
          Plan a real Copenhagen trip.
        </h1>
        <p className="text-lg text-ink-muted mb-8 max-w-prose">
          Two AI agents collaborate live — a Place Scout and a Concierge —
          grounded in real places, real weather, and a real recommendation
          confidence. Nothing invented.
        </p>
        <Link
          href="/trip-planner"
          className="inline-flex items-center gap-2 rounded-sm bg-accent px-5 py-3 text-white font-medium hover:opacity-90 transition-opacity"
        >
          Plan a trip
        </Link>
      </div>
    </main>
  );
}
