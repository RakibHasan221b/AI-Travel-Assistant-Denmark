// Mirrors api/main.py's Pydantic models field-for-field.

export interface TripPlanRequest {
  request: string;
  target_date: string; // ISO date, "YYYY-MM-DD"
  start_location: string;
}

export interface PlaceRecommendation {
  name: string;
  category: string;
  neighborhood: string;
  opening_hours: string | null;
  quality_score: number | null;
  vibe_cluster: string | null;
  summary: string | null;
  sources: string[];
  distance_km: number | null;
  walk_minutes: number | null;
  bike_minutes: number | null;
  travel_note: string | null;
  near_place: string | null;
  near_distance_km: number | null;
  why_recommended: string;
}

export interface TripPlanResponse {
  places: PlaceRecommendation[];
  weather_summary: string;
  overall_note: string;
}
