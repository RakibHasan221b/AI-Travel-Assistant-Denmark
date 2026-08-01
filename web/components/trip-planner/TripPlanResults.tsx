import type { TripPlanResponse } from "@/lib/types";
import { PlaceCard } from "./PlaceCard";

export function TripPlanResults({ result }: { result: TripPlanResponse }) {
  return (
    <div className="mt-8 space-y-4 animate-[fadeIn_0.3s_ease-in]">
      <div className="rounded-sm bg-secondary-soft border border-line px-4 py-3 text-sm">
        <span className="text-[10px] tracking-widest uppercase text-ink-faint mr-2">
          Conditions
        </span>
        {result.weather_summary}
      </div>

      {result.places.map((place) => (
        <PlaceCard key={place.name} place={place} />
      ))}

      {result.overall_note && (
        <div className="border-t border-line pt-5 mt-6">
          <p className="text-[10px] tracking-widest uppercase text-ink-faint mb-2">
            Overall
          </p>
          <p className="text-[15px] leading-relaxed">{result.overall_note}</p>
        </div>
      )}
    </div>
  );
}
