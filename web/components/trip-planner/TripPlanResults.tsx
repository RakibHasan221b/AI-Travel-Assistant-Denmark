import type { TripPlanResponse } from "@/lib/types";
import { PlaceCard } from "./PlaceCard";

function ConditionsBox({ weatherSummary }: { weatherSummary: string }) {
  return (
    <div className="rounded-sm bg-secondary-soft border border-line px-4 py-3 flex items-start gap-3">
      <span className="text-secondary text-lg leading-none mt-0.5 shrink-0" aria-hidden="true">
        ☁
      </span>
      <div>
        <span className="text-[10px] tracking-widest uppercase text-ink-faint block mb-0.5">
          Conditions
        </span>
        <span className="text-sm text-ink">{weatherSummary}</span>
      </div>
    </div>
  );
}

export function TripPlanResults({ result }: { result: TripPlanResponse }) {
  return (
    <div className="mt-8 space-y-4 animate-[fadeIn_0.3s_ease-in]">
      <ConditionsBox weatherSummary={result.weather_summary} />

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
