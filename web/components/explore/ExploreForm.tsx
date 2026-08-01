"use client";

import { useState } from "react";

const CATEGORIES = ["", "restaurant", "cafe", "hotel", "landmark", "bar"];

export interface ExploreFormValues {
  query: string;
  category: string;
  neighborhood: string;
}

export function ExploreForm({
  disabled,
  onSubmit,
}: {
  disabled: boolean;
  onSubmit: (values: ExploreFormValues) => void;
}) {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("");
  const [neighborhood, setNeighborhood] = useState("");
  const [validationError, setValidationError] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) {
      setValidationError("Type something to search for.");
      return;
    }
    setValidationError("");
    onSubmit({ query, category, neighborhood });
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-wrap gap-3 items-end">
      <div className="flex-1 min-w-[220px]">
        <label htmlFor="explore-query" className="block text-sm font-medium mb-1.5">
          What are you looking for?
        </label>
        <input
          id="explore-query"
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={disabled}
          placeholder="cozy quiet cafe good for working"
          className="w-full rounded-sm border border-line bg-surface px-3 py-2.5 text-[15px] outline-none focus:border-accent disabled:opacity-60"
        />
      </div>

      <div>
        <label htmlFor="explore-category" className="block text-sm font-medium mb-1.5">
          Category
        </label>
        <select
          id="explore-category"
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          disabled={disabled}
          className="rounded-sm border border-line bg-surface px-3 py-2.5 text-[15px] outline-none focus:border-accent disabled:opacity-60"
        >
          {CATEGORIES.map((c) => (
            <option key={c} value={c}>
              {c || "Any"}
            </option>
          ))}
        </select>
      </div>

      <div>
        <label htmlFor="explore-neighborhood" className="block text-sm font-medium mb-1.5">
          Neighborhood (optional)
        </label>
        <input
          id="explore-neighborhood"
          type="text"
          value={neighborhood}
          onChange={(e) => setNeighborhood(e.target.value)}
          disabled={disabled}
          placeholder="Vesterbro"
          className="rounded-sm border border-line bg-surface px-3 py-2.5 text-[15px] outline-none focus:border-accent disabled:opacity-60"
        />
      </div>

      <button
        type="submit"
        disabled={disabled}
        className="rounded-sm bg-accent px-5 py-2.5 text-white font-medium hover:opacity-90 disabled:opacity-50 transition-opacity"
      >
        {disabled ? "Searching…" : "Search"}
      </button>

      {validationError && (
        <p className="w-full text-sm text-accent-ink">{validationError}</p>
      )}
    </form>
  );
}
