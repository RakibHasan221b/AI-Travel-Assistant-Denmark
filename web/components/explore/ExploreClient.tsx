"use client";

import { useState } from "react";
import { getExplore } from "@/lib/api";
import type { ExplorePlaceResult } from "@/lib/types";
import { ExploreForm, type ExploreFormValues } from "./ExploreForm";
import { ExploreResultCard } from "./ExploreResultCard";

type Status = "idle" | "loading" | "error";

export function ExploreClient() {
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState("");
  const [results, setResults] = useState<ExplorePlaceResult[] | null>(null);

  async function handleSubmit(values: ExploreFormValues) {
    setStatus("loading");
    setError("");
    try {
      const response = await getExplore({
        q: values.query,
        category: values.category,
        neighborhood: values.neighborhood,
      });
      setResults(response.results);
      setStatus("idle");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
    }
  }

  return (
    <div>
      <ExploreForm disabled={status === "loading"} onSubmit={handleSubmit} />

      {status === "error" && (
        <div className="mt-6 rounded-sm border border-accent bg-accent-soft px-4 py-3 text-sm text-accent-ink">
          {error}
        </div>
      )}

      {results && (
        <div className="mt-6 space-y-4">
          {results.length === 0 ? (
            <p className="text-ink-muted text-sm">
              No matching places found. Try a broader query or fewer filters.
            </p>
          ) : (
            results.map((place) => <ExploreResultCard key={place.name} place={place} />)
          )}
        </div>
      )}
    </div>
  );
}
