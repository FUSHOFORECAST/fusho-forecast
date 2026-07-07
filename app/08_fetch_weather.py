import os
import requests
import pandas as pd

INPUT_FILE = "data/processed/master_dataset_clean.csv"
OUTPUT_FILE = "data/processed/weather_milan.csv"

# Fusho Milano - coordinate indicative Milano
LATITUDE = 45.4642
LONGITUDE = 9.1900

df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])

start_date = df["date"].min().strftime("%Y-%m-%d")
end_date = df["date"].max().strftime("%Y-%m-%d")

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": start_date,
    "end_date": end_date,
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "Europe/Rome"
}

print("Scarico meteo da", start_date, "a", end_date)

response = requests.get(url, params=params, timeout=60)
response.raise_for_status()

data = response.json()["daily"]

weather = pd.DataFrame({
    "date": pd.to_datetime(data["time"]),
    "temp_max": data["temperature_2m_max"],
    "temp_min": data["temperature_2m_min"],
    "precipitation": data["precipitation_sum"],
    "rain": data["rain_sum"],
    "snowfall": data["snowfall_sum"],
    "wind_max": data["wind_speed_10m_max"],
})

weather["is_rainy"] = (weather["rain"] > 0).astype(int)
weather["is_heavy_rain"] = (weather["rain"] >= 10).astype(int)
weather["avg_temp"] = (weather["temp_max"] + weather["temp_min"]) / 2

os.makedirs("data/processed", exist_ok=True)
weather.to_csv(OUTPUT_FILE, index=False)

print("Meteo salvato:", OUTPUT_FILE)
print(weather.head())
print(weather.tail())
