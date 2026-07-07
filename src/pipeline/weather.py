import os

import pandas as pd
import requests

from src.pipeline.config import RestaurantConfig

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "wind_speed_10m_max",
]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

MAX_FORECAST_DAYS = 16


def _parse_daily_response(data: dict) -> pd.DataFrame:
    daily = data["daily"]

    weather = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "temp_max": daily["temperature_2m_max"],
        "temp_min": daily["temperature_2m_min"],
        "precipitation": daily["precipitation_sum"],
        "rain": daily["rain_sum"],
        "snowfall": daily["snowfall_sum"],
        "wind_max": daily["wind_speed_10m_max"],
    })

    weather["is_rainy"] = (weather["rain"] > 0).astype(int)
    weather["is_heavy_rain"] = (weather["rain"] >= 10).astype(int)
    weather["avg_temp"] = (weather["temp_max"] + weather["temp_min"]) / 2

    return weather


def fetch_historical_weather(config: RestaurantConfig, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": config.location.latitude,
        "longitude": config.location.longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": DAILY_VARS,
        "timezone": config.location.timezone,
    }

    response = requests.get(ARCHIVE_URL, params=params, timeout=60)
    response.raise_for_status()

    return _parse_daily_response(response.json())


def fetch_forecast_weather(config: RestaurantConfig, horizon_days: int) -> pd.DataFrame:
    if horizon_days > MAX_FORECAST_DAYS:
        raise ValueError(
            f"horizon_days={horizon_days} supera il limite di {MAX_FORECAST_DAYS} giorni "
            f"dell'endpoint forecast di Open-Meteo."
        )

    params = {
        "latitude": config.location.latitude,
        "longitude": config.location.longitude,
        "forecast_days": horizon_days,
        "daily": DAILY_VARS,
        "timezone": config.location.timezone,
    }

    response = requests.get(FORECAST_URL, params=params, timeout=60)
    response.raise_for_status()

    return _parse_daily_response(response.json())


def merge_weather(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    weather = weather_df.copy()
    weather["date"] = pd.to_datetime(weather["date"])
    return out.merge(weather, on="date", how="left")


def main():
    import argparse

    from src.pipeline.clean import clean
    from src.pipeline.config import load_restaurant_config
    from src.pipeline.ingest import ingest

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_restaurant_config(args.config)
    master_df, audit_df = ingest(config)
    clean_df = clean(master_df, audit_df, config)

    start_date = clean_df["date"].min().strftime("%Y-%m-%d")
    end_date = clean_df["date"].max().strftime("%Y-%m-%d")

    print("Scarico meteo da", start_date, "a", end_date)
    weather_df = fetch_historical_weather(config, start_date, end_date)

    os.makedirs(config.processed_dir, exist_ok=True)
    weather_df.to_csv(config.processed_path("weather_historical.csv"), index=False)

    merged = merge_weather(clean_df, weather_df)
    print("Meteo unito. Righe:", len(merged))
    print(weather_df.tail())


if __name__ == "__main__":
    main()
