import pandas as pd
import os

INPUT_FILE = "data/processed/master_dataset_weather_calendar.csv"
EVENTS_FILE = "data/processed/events_manual.csv"
OUTPUT_FILE = "data/processed/events_calendar.csv"

df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])

events = pd.read_csv(EVENTS_FILE)
events["start_date"] = pd.to_datetime(events["start_date"])
events["end_date"] = pd.to_datetime(events["end_date"])

calendar = pd.DataFrame({"date": df["date"].unique()})
calendar["date"] = pd.to_datetime(calendar["date"])

event_types = events["event_type"].unique()

for event_type in event_types:
    calendar[f"event_{event_type}"] = 0

calendar["big_event"] = 0
calendar["event_count"] = 0
calendar["event_impact"] = 0

for _, row in events.iterrows():
    mask = (calendar["date"] >= row["start_date"]) & (calendar["date"] <= row["end_date"])

    calendar.loc[mask, f"event_{row['event_type']}"] = 1
    calendar.loc[mask, "big_event"] = 1
    calendar.loc[mask, "event_count"] += 1
    calendar.loc[mask, "event_impact"] += row["impact"]

os.makedirs("data/processed", exist_ok=True)
calendar.to_csv(OUTPUT_FILE, index=False)

print("EVENTI CREATI")
print("File:", OUTPUT_FILE)
print("Giorni evento:", calendar["big_event"].sum())
print(calendar[calendar["big_event"] == 1].head(30))
