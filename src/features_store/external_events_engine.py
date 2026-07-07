import os
import pandas as pd

BASE_FILE = "data/processed/feature_store.csv"
EVENTS_FILE = "data/processed/external_events_manual.csv"
OUTPUT_FILE = "data/processed/external_events_calendar.csv"

base = pd.read_csv(BASE_FILE)
base["date"] = pd.to_datetime(base["date"])

events = pd.read_csv(EVENTS_FILE)
events["start_date"] = pd.to_datetime(events["start_date"])
events["end_date"] = pd.to_datetime(events["end_date"])

start_date = base["date"].min()
end_date = max(base["date"].max() + pd.Timedelta(days=60), events["end_date"].max())

calendar = pd.DataFrame({
    "date": pd.date_range(start_date, end_date, freq="D")
})

categories = sorted(events["event_category"].unique())

for category in categories:
    calendar[f"event_{category}"] = 0

calendar["external_event_count"] = 0
calendar["external_positive_impact"] = 0
calendar["external_negative_impact"] = 0
calendar["external_mixed_impact"] = 0
calendar["external_total_impact"] = 0

for _, row in events.iterrows():
    mask = (calendar["date"] >= row["start_date"]) & (calendar["date"] <= row["end_date"])

    category_col = f"event_{row['event_category']}"
    calendar.loc[mask, category_col] = 1
    calendar.loc[mask, "external_event_count"] += 1

    strength = int(row["impact_strength"])

    if row["impact_direction"] == "positive":
        calendar.loc[mask, "external_positive_impact"] += strength
        calendar.loc[mask, "external_total_impact"] += strength

    elif row["impact_direction"] == "negative":
        calendar.loc[mask, "external_negative_impact"] += strength
        calendar.loc[mask, "external_total_impact"] -= strength

    else:
        calendar.loc[mask, "external_mixed_impact"] += strength

os.makedirs("data/processed", exist_ok=True)
calendar.to_csv(OUTPUT_FILE, index=False)

print("=" * 80)
print("EXTERNAL EVENTS CALENDAR CREATO")
print("=" * 80)
print("File:", OUTPUT_FILE)
print("Date:", calendar["date"].min(), "→", calendar["date"].max())
print()
print(calendar[calendar["external_event_count"] > 0].to_string(index=False))
