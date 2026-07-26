"""Phase 10 — weather-aware "Outdoor Interest Index" forecast.

Honest framing (see plan discussion): there's no real per-place visitor data
for this pilot, so the target is a documented, transparent heuristic - not a
claim of measured foot traffic. What's genuinely real: the weather history
(Open-Meteo), and the event calendar (confirmed public dates for Copenhagen's
major recurring events, not fabricated).

The actual time-series technique being demonstrated is the CHRONOLOGICAL
train/test split (train on earlier days, test on the most recent days) -
the real methodological difference from Phase 9's random split, and the
correct way to evaluate a forecasting model without leaking future
information into training.
"""

import logging
import os
from datetime import date

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("forecast_interest")

# Confirmed real dates only - 2026 Pride omitted since it wasn't confirmed
# anywhere, and guessing a date would violate the project's own "no faked
# data" rule. It also falls outside the 7-day forecast window regardless.
EVENTS = [
    ("Copenhagen Marathon", date(2025, 5, 11), date(2025, 5, 11)),
    ("Copenhagen Distortion", date(2025, 6, 4), date(2025, 6, 8)),
    ("Copenhagen Jazz Festival", date(2025, 7, 4), date(2025, 7, 13)),
    ("Copenhagen Pride", date(2025, 8, 9), date(2025, 8, 17)),
    ("Copenhagen Marathon", date(2026, 5, 10), date(2026, 5, 10)),
    ("Copenhagen Distortion", date(2026, 6, 3), date(2026, 6, 7)),
    ("Copenhagen Jazz Festival", date(2026, 7, 3), date(2026, 7, 12)),
]

CATEGORY_WEATHER_SENSITIVITY = {"landmark": 1.0, "restaurant": 0.5, "cafe": 0.4, "hotel": 0.3}
CATEGORY_EVENT_BOOST = {"landmark": 25, "hotel": 20, "restaurant": 15, "cafe": 10}
CATEGORIES = list(CATEGORY_WEATHER_SENSITIVITY.keys())

TEST_DAYS = 30
RANDOM_STATE = 42


def is_event_day(d: date) -> bool:
    return any(start <= d <= end for _, start, end in EVENTS)


def weather_comfort(temp_max_c: float, precip_mm: float, wind_kph: float) -> float:
    comfort_temp = np.clip(100 - abs(temp_max_c - 20) * 4, 0, 100)
    precip_penalty = min(precip_mm * 8, 60)
    wind_penalty = min(max(0, wind_kph - 15) * 2, 30)
    return float(np.clip(comfort_temp - precip_penalty - wind_penalty, 0, 100))


def outdoor_interest_index(comfort: float, category: str, event_day: bool) -> float:
    score = CATEGORY_WEATHER_SENSITIVITY[category] * comfort
    if event_day:
        score += CATEGORY_EVENT_BOOST[category]
    return float(np.clip(score, 0, 100))


def load_weather(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute("SELECT date, temp_max_c, temp_min_c, precip_mm, wind_kph FROM weather_daily ORDER BY date;")
        columns = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=columns)


def build_dataset(weather: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, w in weather.iterrows():
        d = w["date"]
        comfort = weather_comfort(w["temp_max_c"], w["precip_mm"] or 0, w["wind_kph"] or 0)
        event = is_event_day(d)
        for category in CATEGORIES:
            rows.append(
                {
                    "date": d,
                    "category": category,
                    "temp_max_c": w["temp_max_c"],
                    "temp_min_c": w["temp_min_c"],
                    "precip_mm": w["precip_mm"],
                    "wind_kph": w["wind_kph"],
                    "day_of_week": d.weekday(),
                    "month": d.month,
                    "is_event_day": int(event),
                    "target": outdoor_interest_index(comfort, category, event),
                }
            )
    return pd.DataFrame(rows)


def main():
    db_url = os.environ["DATABASE_URL"]
    with psycopg.connect(db_url, connect_timeout=15) as conn:
        weather = load_weather(conn)

    today = date.today()
    historical = weather[weather["date"] < today]
    forecast_days = weather[weather["date"] >= today]
    log.info(f"{len(historical)} historical days, {len(forecast_days)} forecast days")

    dataset = build_dataset(historical)
    dataset = pd.get_dummies(dataset, columns=["category"], prefix="cat")
    feature_cols = [c for c in dataset.columns if c not in ("date", "target")]

    cutoff = sorted(historical["date"].unique())[-TEST_DAYS]
    train = dataset[dataset["date"] < cutoff]
    test = dataset[dataset["date"] >= cutoff]
    log.info(f"Chronological split: train={len(train)} rows (before {cutoff}), test={len(test)} rows (from {cutoff})")

    model = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE)
    model.fit(train[feature_cols], train["target"])

    pred = model.predict(test[feature_cols])
    mse = mean_squared_error(test["target"], pred)
    mae = mean_absolute_error(test["target"], pred)
    log.info(f"Chronological held-out eval: MSE={mse:.2f}, MAE={mae:.2f}")

    # Retrain on all historical data for the actual forecast
    model.fit(dataset[feature_cols], dataset["target"])

    forecast_input = build_dataset(forecast_days)
    forecast_dates_categories = forecast_input[["date", "category"]].copy()
    forecast_input = pd.get_dummies(forecast_input, columns=["category"], prefix="cat")
    for col in feature_cols:
        if col not in forecast_input.columns:
            forecast_input[col] = 0
    forecast_pred = model.predict(forecast_input[feature_cols])
    forecast_dates_categories["predicted_interest_score"] = np.clip(forecast_pred, 0, 100)

    with psycopg.connect(db_url, connect_timeout=15) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM visit_time_forecast WHERE model_name = 'xgboost-category-level';")
            cur.execute("SELECT place_id, category FROM places;")
            places = cur.fetchall()

            rows_written = 0
            for place_id, category in places:
                cat_forecast = forecast_dates_categories[forecast_dates_categories["category"] == category]
                for _, row in cat_forecast.iterrows():
                    cur.execute(
                        """
                        INSERT INTO visit_time_forecast (place_id, forecast_date, predicted_interest_score, model_name)
                        VALUES (%s, %s, %s, 'xgboost-category-level');
                        """,
                        (place_id, row["date"], float(row["predicted_interest_score"])),
                    )
                    rows_written += 1
        conn.commit()

    log.info(f"Done. {rows_written} place-day forecasts stored ({len(places)} places x 7 days).")


if __name__ == "__main__":
    main()
