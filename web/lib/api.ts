import type {
  ExploreResponse,
  StatsResponse,
  TripPlanRequest,
  TripPlanResponse,
} from "./types";

// NEXT_PUBLIC_* vars are inlined into the client bundle at build time, not
// read live like Streamlit's env-var config — changing this on Vercel
// requires a redeploy, not just a settings save.
const API_URL =
  process.env.NEXT_PUBLIC_TRIP_PLANNER_API_URL ?? "http://localhost:8000";

async function errorDetail(res: Response): Promise<string> {
  try {
    const payload = await res.json();
    return payload.detail ?? res.statusText;
  } catch {
    return res.statusText;
  }
}

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
  if (!res.ok) throw new Error(await errorDetail(res));
  return res.json();
}

export interface ExploreParams {
  q: string;
  category?: string;
  neighborhood?: string;
}

// /explore and /stats are fast (SQL + a local embedding model, no LLM call)
// — the Vercel-timeout reasoning above doesn't apply to them, but there's
// no actual benefit a proxy would add either (no secret to hide, nothing to
// transform), so they stay on the same direct-fetch pattern for one
// consistent request path across every endpoint.
export async function getExplore(
  params: ExploreParams,
  signal?: AbortSignal
): Promise<ExploreResponse> {
  const qs = new URLSearchParams({
    q: params.q,
    category: params.category ?? "",
    neighborhood: params.neighborhood ?? "",
  });
  const res = await fetch(`${API_URL}/explore?${qs}`, { signal });
  if (!res.ok) throw new Error(await errorDetail(res));
  return res.json();
}

export async function getStats(): Promise<StatsResponse> {
  const res = await fetch(`${API_URL}/stats`);
  if (!res.ok) throw new Error(await errorDetail(res));
  return res.json();
}
