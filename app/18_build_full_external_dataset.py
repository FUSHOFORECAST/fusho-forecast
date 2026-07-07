import pandas as pd
import os

SALES_FILE = "data/processed/master_dataset_weather_calendar.csv"
EVENTS_FILE = "data/processed/events_calendar.csv"
OUTPUT_FILE = "data/processed/master_dataset_full.csv"

sales = pd.read_csv(SALES_FILE)
events = pd.read_csv(EVENTS_FILE)

sales["date"] = pd.to_datetime(sales["date"])
events["date"] = pd.to_datetime(events["date"])

df = sales.merge(events, on="date", how="left")

event_cols = [c for c in df.columns if c.startswith("event_") or c == "big_event"]

for col in event_cols:
    df[col] = df[col].fillna(0).astype(int)

os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("DATASET FULL CREATO")
print("Righe:", len(df))
print("Colonne:", len(df.columns))
print("Event columns:", event_cols)
print(df[event_cols].sum())
