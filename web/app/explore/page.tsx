import type { Metadata } from "next";
import { ExploreClient } from "@/components/explore/ExploreClient";

export const metadata: Metadata = {
  title: "Explore",
};

export default function ExplorePage() {
  return (
    <main className="flex-1 px-6 py-12">
      <div className="mx-auto max-w-3xl">
        <h1 className="text-3xl font-semibold tracking-tight mb-2">Explore Copenhagen places</h1>
        <p className="text-ink-muted mb-8">
          Semantic search over real Copenhagen places — matched by meaning, not just keywords,
          enriched with a real quality score and vibe cluster where available.
        </p>

        <ExploreClient />
      </div>
    </main>
  );
}
