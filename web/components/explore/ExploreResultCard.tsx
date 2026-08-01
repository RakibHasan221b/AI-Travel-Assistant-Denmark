import type { ExplorePlaceResult } from "@/lib/types";

export function ExploreResultCard({ place }: { place: ExplorePlaceResult }) {
  const metaBits: string[] = [];
  if (place.vibe_cluster) metaBits.push(`Vibe: ${place.vibe_cluster}`);
  if (place.opening_hours) metaBits.push(`Hours: ${place.opening_hours}`);

  return (
    <div className="border border-line rounded-sm bg-surface p-5 border-l-2 border-l-accent">
      <div className="flex items-start justify-between gap-4">
        <h3 className="font-semibold text-lg leading-snug">
          {place.name}
          <span className="font-normal text-ink-muted">
            {" "}
            — {place.category}, {place.neighborhood}
          </span>
        </h3>
        <div className="shrink-0 text-right">
          <div className="font-mono text-sm text-ink-muted tabular-nums">
            {place.similarity.toFixed(2)}
          </div>
          <div className="text-[10px] tracking-widest uppercase text-ink-faint">Match</div>
        </div>
      </div>

      {place.quality_score !== null && (
        <p className="text-sm font-mono mt-1 tabular-nums text-accent-ink">
          {Math.round(place.quality_score)}/100 quality
        </p>
      )}

      {metaBits.length > 0 && (
        <p className="text-sm text-ink-muted mt-2 font-mono">{metaBits.join(" · ")}</p>
      )}

      {place.summary && (
        <p className="mt-3 text-sm text-ink-muted leading-relaxed">{place.summary}</p>
      )}

      {place.sources.length > 0 && (
        <p className="mt-2 text-xs text-ink-faint">Sources: {place.sources.join(", ")}</p>
      )}
    </div>
  );
}
