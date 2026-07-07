import pandas as pd
import os

sales = pd.read_csv("data/processed/master_dataset_clean.csv")
weather = pd.read_csv("data/processed/weather_milan.csv")

sales["date"] = pd.to_datetime(sales["date"])
weather["date"] = pd.to_datetime(weather["date"])

df = sales.merge(weather, on="date", how="left")

print("Righe vendita :", len(sales))
print("Righe meteo   :", len(weather))
print("Righe merge   :", len(df))

print("\nValori mancanti:")
print(df.isna().sum())

os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    "data/processed/master_dataset_weather.csv",
    index=False
)

print("\nDataset salvato:")
print("data/processed/master_dataset_weather.csv")

print("\nColonne:")
print(df.columns.tolist())
