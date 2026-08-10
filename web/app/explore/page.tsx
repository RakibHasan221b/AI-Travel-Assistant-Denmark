import type { Metadata } from "next";
import { ExploreClient } from "@/components/explore/ExploreClient";

export const metadata: Metadata = {
  title: "Explore",
};

export default function ExplorePage() {
  return (
    <main className="flex-1 px-6 py-12">
      <div className="mx-auto max-w-3xl">
        <h1 className="text-3xl font-semibold tracking-tight mb-2">Discover places in Copenhagen</h1>
        <p className="text-ink-muted mb-8">
          Find restaurants, cafés, hotels, landmarks and more based on what you&apos;re looking for.
        </p>

        <ExploreClient />
      </div>
    </main>
  );
}
