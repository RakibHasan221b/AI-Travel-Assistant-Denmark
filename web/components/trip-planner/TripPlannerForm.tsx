"use client";

import { useState } from "react";

const AREAS = ["Any area", "Vesterbro", "Norrebro", "Osterbro", "Frederiksberg", "Indre By"];

export interface TripPlannerFormValues {
  requestText: string;
  area: string;
  startLocation: string;
  targetDate: string;
}

function tomorrowIso(): string {
  const d = new Date();
  d.setDate(d.getDate() + 1);
  return d.toISOString().slice(0, 10);
}

export function TripPlannerForm({
  disabled,
  onSubmit,
}: {
  disabled: boolean;
  onSubmit: (values: TripPlannerFormValues) => void;
}) {
  const [requestText, setRequestText] = useState("");
  const [area, setArea] = useState(AREAS[0]);
  const [startLocation, setStartLocation] = useState("");
  const [targetDate, setTargetDate] = useState(tomorrowIso());
  const [validationError, setValidationError] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!requestText.trim()) {
      setValidationError("Describe what you're looking for first.");
      return;
    }
    setValidationError("");
    onSubmit({ requestText, area, startLocation, targetDate });
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      <div>
        <label htmlFor="request" className="block text-sm font-medium mb-1.5">
          What are you looking for?
        </label>
        <textarea
          id="request"
          value={requestText}
          onChange={(e) => setRequestText(e.target.value)}
          disabled={disabled}
          placeholder="I want a cozy quiet cafe to work from and one landmark to visit nearby, I like calm places"
          rows={3}
          className="w-full rounded-sm border border-line bg-surface px-3 py-2.5 text-[15px] outline-none focus:border-accent disabled:opacity-60"
        />
      </div>

      <div className="grid sm:grid-cols-2 gap-5">
        <div>
          <label htmlFor="area" className="block text-sm font-medium mb-1.5">
            Area (optional)
          </label>
          <select
            id="area"
            value={area}
            onChange={(e) => setArea(e.target.value)}
            disabled={disabled}
            className="w-full rounded-sm border border-line bg-surface px-3 py-2.5 text-[15px] outline-none focus:border-accent disabled:opacity-60"
          >
            {AREAS.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label htmlFor="target-date" className="block text-sm font-medium mb-1.5">
            Target date
          </label>
          <input
            id="target-date"
            type="date"
            value={targetDate}
            onChange={(e) => setTargetDate(e.target.value)}
            disabled={disabled}
            className="w-full rounded-sm border border-line bg-surface px-3 py-2.5 text-[15px] font-mono outline-none focus:border-accent disabled:opacity-60"
          />
        </div>
      </div>

      <div>
        <label htmlFor="start-location" className="block text-sm font-medium mb-1.5">
          Starting point (optional)
        </label>
        <input
          id="start-location"
          type="text"
          value={startLocation}
          onChange={(e) => setStartLocation(e.target.value)}
          disabled={disabled}
          placeholder="Copenhagen Central Station, or your hotel name/address"
          className="w-full rounded-sm border border-line bg-surface px-3 py-2.5 text-[15px] outline-none focus:border-accent disabled:opacity-60"
        />
        <p className="mt-1 text-xs text-ink-faint">
          Add your starting point to see estimated travel distances.
        </p>
      </div>

      {validationError && <p className="text-sm text-accent-ink">{validationError}</p>}

      <button
        type="submit"
        disabled={disabled}
        className="rounded-sm bg-accent px-5 py-2.5 text-white font-medium hover:opacity-90 disabled:opacity-50 transition-opacity"
      >
        {disabled ? "Planning…" : "Plan my trip"}
      </button>
    </form>
  );
}
