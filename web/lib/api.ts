import type { TripPlanRequest, TripPlanResponse } from "./types";

// NEXT_PUBLIC_* vars are inlined into the client bundle at build time, not
// read live like Streamlit's env-var config — changing this on Vercel
// requires a redeploy, not just a settings save.
const API_URL =
  process.env.NEXT_PUBLIC_TRIP_PLANNER_API_URL ?? "http://localhost:8000";

// Deliberately a plain browser fetch straight to the Render API, not a
// Next.js API route proxy: /trip-plan genuinely takes 1-2 minutes (three
// real sequential LLM agent calls), which exceeds Vercel serverless
// function timeout limits. A direct client fetch has no such ceiling.
export async function planTrip(
  body: TripPlanRequest,
  signal?: AbortSignal
): Promise<TripPlanResponse> {
  const res = await fetch(`${API_URL}/trip-plan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const payload = await res.json();
      detail = payload.detail ?? detail;
    } catch {
      // body wasn't JSON — keep the statusText fallback
    }
    throw new Error(detail);
  }

  return res.json();
}
