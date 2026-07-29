"""Phase 10 — real Copenhagen weather, historical + forecast, from Open-Meteo.
Free, no key. Historical window covers 2025-01-01 through today, giving
enough days to span all four confirmed 2025 event dates and to do a proper
chronological train/test split later.
"""

import logging
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import psycopg
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("weather")

LAT, LON = 55.6761, 12.5683  # Copenhagen city center
TZ = ZoneInfo("Europe/Copenhagen")
HISTORY_START = "2025-01-01"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_FIELDS = "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,weathercode"


def fetch_daily(url: str, params: dict) -> dict:
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["daily"]


def upsert_days(conn, daily: dict):
    dates = daily["time"]
    rows = list(
        zip(
            dates,
            daily["temperature_2m_max"],
            daily["temperature_2m_min"],
            daily["precipitation_sum"],
            daily["windspeed_10m_max"],
            daily["weathercode"],
        )
    )
    with conn.cursor() as cur:
        for d, tmax, tmin, precip, wind, code in rows:
            cur.execute(
                """
                INSERT INTO weather_daily (date, temp_max_c, temp_min_c, precip_mm, wind_kph, condition_code, fetched_at)
                VALUES (%s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (date) DO UPDATE SET
                    temp_max_c = EXCLUDED.temp_max_c, temp_min_c = EXCLUDED.temp_min_c,
                    precip_mm = EXCLUDED.precip_mm, wind_kph = EXCLUDED.wind_kph,
                    condition_code = EXCLUDED.condition_code, fetched_at = now();
                """,
                (d, tmax, tmin, precip, wind, code),
            )
    conn.commit()
    return len(rows)


def main():
    db_url = os.environ["DATABASE_URL"]
    yesterday = (datetime.now(TZ).date() - timedelta(days=1)).isoformat()

    with psycopg.connect(db_url, connect_timeout=15) as conn:
        log.info(f"Fetching historical weather {HISTORY_START} to {yesterday}...")
        historical = fetch_daily(
            ARCHIVE_URL,
            {
                "latitude": LAT, "longitude": LON,
                "start_date": HISTORY_START, "end_date": yesterday,
                "daily": DAILY_FIELDS, "timezone": "Europe/Copenhagen",
            },
        )
        n_hist = upsert_days(conn, historical)
        log.info(f"Stored {n_hist} historical days")

        log.info("Fetching 7-day forecast...")
        forecast = fetch_daily(
            FORECAST_URL,
            {
                "latitude": LAT, "longitude": LON,
                "forecast_days": 7,
                "daily": DAILY_FIELDS, "timezone": "Europe/Copenhagen",
            },
        )
        n_fcst = upsert_days(conn, forecast)
        log.info(f"Stored {n_fcst} forecast days")

    log.info("Done.")


if __name__ == "__main__":
    main()
