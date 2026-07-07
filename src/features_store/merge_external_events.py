import os
import pandas as pd

FEATURE_STORE_FILE = "data/processed/feature_store.csv"
EVENTS_FILE = "data/processed/external_events_calendar.csv"
OUTPUT_FILE = "data/processed/feature_store_events.csv"

features = pd.read_csv(FEATURE_STORE_FILE)
events = pd.read_csv(EVENTS_FILE)

features["date"] = pd.to_datetime(features["date"])
events["date"] = pd.to_datetime(events["date"])

df = features.merge(events, on="date", how="left")

event_cols = [c for c in df.columns if c.startswith("event_")]
impact_cols = [
    "external_event_count",
    "external_positive_impact",
    "external_negative_impact",
    "external_mixed_impact",
    "external_total_impact",
]

for col in event_cols + impact_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)

os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("=" * 80)
print("FEATURE STORE + EVENTS CREATO")
print("=" * 80)
print("File:", OUTPUT_FILE)
print("Righe:", len(df))
print("Colonne:", len(df.columns))
print()
print(df[df["external_event_count"] > 0][[
    "date",
    "total",
    "external_event_count",
    "external_positive_impact",
    "external_negative_impact",
    "external_mixed_impact",
    "external_total_impact",
]].to_string(index=False))
