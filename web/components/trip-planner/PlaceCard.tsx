import type { PlaceRecommendation } from "@/lib/types";

// Near-literal port of app/pages/2_Trip_Planner.py's render_trip_plan meta
// line logic. Every branch here mirrors a null-check there — this is a
// direct expression of the backend's honesty-over-confidence principle
// (a null field means "we don't know," not "zero"), not just a layout to
// reproduce, so the conditional structure must stay exact.
function buildMetaBits(place: PlaceRecommendation): string[] {
  const bits: string[] = [];

  if (place.vibe_cluster) {
    bits.push(`Vibe: ${place.vibe_cluster}`);
  }
  if (place.opening_hours) {
    bits.push(`Hours: ${place.opening_hours}`);
  }
  if (place.near_place && place.near_distance_km !== null) {
    bits.push(`${place.near_distance_km.toFixed(2)} km from ${place.near_place}`);
  }
  if (place.distance_km !== null) {
    if (place.travel_note) {
      bits.push(`${place.distance_km.toFixed(1)} km from your start — ${place.travel_note}`);
    } else {
      const timeBits: string[] = [];
      if (place.walk_minutes !== null) timeBits.push(`${place.walk_minutes} min walk`);
      if (place.bike_minutes !== null) timeBits.push(`${place.bike_minutes} min bike`);
      const timeStr = timeBits.length > 0 ? ` (${timeBits.join(" / ")})` : "";
      bits.push(`${place.distance_km.toFixed(1)} km from your start${timeStr}`);
    }
  }

  return bits;
}

export function PlaceCard({ place }: { place: PlaceRecommendation }) {
  const metaBits = buildMetaBits(place);

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
        {place.quality_score !== null && (
          <div className="shrink-0 text-right">
            <div className="font-mono text-lg font-medium tabular-nums">
              {Math.round(place.quality_score)}
              <span className="text-ink-faint text-sm">/100</span>
            </div>
            <div className="text-[10px] tracking-widest uppercase text-ink-faint">
              Quality
            </div>
          </div>
        )}
      </div>

      {metaBits.length > 0 && (
        <p className="text-sm text-ink-muted mt-2 font-mono">{metaBits.join(" · ")}</p>
      )}

      <p className="mt-3 text-[15px] leading-relaxed">{place.why_recommended}</p>

      {place.summary && (
        <p className="mt-2 text-sm text-ink-muted leading-relaxed">{place.summary}</p>
      )}

      {place.sources.length > 0 && (
        <p className="mt-2 text-xs text-ink-faint">
          Sources: {place.sources.join(", ")}
        </p>
      )}
    </div>
  );
}
