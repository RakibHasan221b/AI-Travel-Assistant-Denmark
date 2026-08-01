"use client";

import { useRef, useState } from "react";
import { planTrip } from "@/lib/api";
import type { TripPlanResponse } from "@/lib/types";
import { TripPlannerForm, type TripPlannerFormValues } from "./TripPlannerForm";
import { LoadingStatus } from "./LoadingStatus";
import { TripPlanResults } from "./TripPlanResults";

type Status = "idle" | "loading" | "error";

export function TripPlannerClient() {
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState("");
  // Only overwrite on success — a failed new attempt shouldn't wipe out a
  // plan that already worked, matching the existing Streamlit page's
  // session_state behavior.
  const [result, setResult] = useState<TripPlanResponse | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  async function handleSubmit(values: TripPlannerFormValues) {
    const fullRequest =
      values.area === "Any area" ? values.requestText : `In ${values.area}: ${values.requestText}`;

    const controller = new AbortController();
    controllerRef.current = controller;
    setStatus("loading");
    setError("");

    try {
      const response = await planTrip(
        {
          request: fullRequest,
          target_date: values.targetDate,
          start_location: values.startLocation.trim(),
        },
        controller.signal
      );
      setResult(response);
      setStatus("idle");
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        setStatus("idle");
        return;
      }
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
    }
  }

  function handleCancel() {
    controllerRef.current?.abort();
  }

  return (
    <div>
      <TripPlannerForm disabled={status === "loading"} onSubmit={handleSubmit} />

      {status === "loading" && <LoadingStatus onCancel={handleCancel} />}

      {status === "error" && (
        // The backend's error detail already reads "Trip planning failed:
        // ..." (see api/main.py) — display it as-is rather than doubling
        // the prefix.
        <div className="mt-8 rounded-sm border border-accent bg-accent-soft px-4 py-3 text-sm text-accent-ink">
          {error}
        </div>
      )}

      {result && <TripPlanResults result={result} />}
    </div>
  );
}
