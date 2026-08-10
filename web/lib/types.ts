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
  recommendation_confidence: number | null;
  recommendation_label: string | null;
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

export interface ExplorePlaceResult {
  name: string;
  category: string;
  neighborhood: string;
  opening_hours: string | null;
  similarity: number;
  quality_score: number | null;
  vibe_cluster: string | null;
  summary: string | null;
  sources: string[];
}

export interface ExploreResponse {
  results: ExplorePlaceResult[];
}

export interface NeighborhoodQuality {
  neighborhood: string;
  avg_quality_score: number;
  n_places: number;
}

export interface CategoryQuality {
  category: string;
  avg_quality_score: number;
  n_places: number;
}

export interface VibeClusterSize {
  label: string;
  n_places: number;
}

export interface RatedAspect {
  aspect_category: string;
  mean_score: number;
  total_mentions: number;
}

export interface ForecastPoint {
  forecast_date: string;
  category: string;
  avg_interest: number;
}

export interface ModelEvaluationMetric {
  label: string;
  value: number;
  description: string;
}

export interface ModelEvaluation {
  metrics: ModelEvaluationMetric[];
}

export interface StatsResponse {
  quality_by_neighborhood: NeighborhoodQuality[];
  quality_by_category: CategoryQuality[];
  vibe_cluster_sizes: VibeClusterSize[];
  rated_aspects: RatedAspect[];
  outdoor_interest_forecast: ForecastPoint[];
  model_evaluation: ModelEvaluation;
}
