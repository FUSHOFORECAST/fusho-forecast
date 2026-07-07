import pandas as pd
import os

INPUT_FILE = "data/processed/master_dataset_weather.csv"
OUTPUT_FILE = "data/processed/model_dataset_weather.csv"

df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# calendario
df["dayofweek"] = df["date"].dt.dayofweek
df["week"] = df["date"].dt.isocalendar().week.astype(int)
df["dayofyear"] = df["date"].dt.dayofyear
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

# storico
for col in ["total", "delivery", "digital", "cash"]:
    for lag in [1, 7, 14, 21, 28, 30]:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    df[f"{col}_rolling_7"] = df[col].shift(1).rolling(7).mean()
    df[f"{col}_rolling_14"] = df[col].shift(1).rolling(14).mean()
    df[f"{col}_rolling_30"] = df[col].shift(1).rolling(30).mean()

# trend
df["total_trend"] = df["total_rolling_7"] / df["total_rolling_30"]
df["delivery_trend"] = df["delivery_rolling_7"] / df["delivery_rolling_30"]
df["digital_trend"] = df["digital_rolling_7"] / df["digital_rolling_30"]
df["cash_trend"] = df["cash_rolling_7"] / df["cash_rolling_30"]

# mix storico
df["delivery_share_lag"] = df["delivery_lag_7"] / df["total_lag_7"].replace(0, pd.NA)
df["digital_share_lag"] = df["digital_lag_7"] / df["total_lag_7"].replace(0, pd.NA)
df["cash_share_lag"] = df["cash_lag_7"] / df["total_lag_7"].replace(0, pd.NA)

# meteo già presente:
# temp_max, temp_min, precipitation, rain, snowfall, wind_max, is_rainy, is_heavy_rain, avg_temp

df = df.replace([float("inf"), -float("inf")], pd.NA)
df = df.dropna().copy()

os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("MODEL DATASET WEATHER CREATO")
print("Righe:", len(df))
print("Date:", df["date"].min(), "→", df["date"].max())
print("Colonne:", len(df.columns))
print(df.tail())
