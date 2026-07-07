import os
import pandas as pd

BASE_FILE = "data/processed/master_dataset_full.csv"
SIGNALS_FILE = "data/processed/intelligence_manual.csv"
OUTPUT_FILE = "data/processed/intelligence_calendar.csv"

base = pd.read_csv(BASE_FILE)
base["date"] = pd.to_datetime(base["date"])

signals = pd.read_csv(SIGNALS_FILE)
signals["start_date"] = pd.to_datetime(signals["start_date"])
signals["end_date"] = pd.to_datetime(signals["end_date"])

start_date = base["date"].min()
end_date = max(
    base["date"].max() + pd.Timedelta(days=30),
    signals["end_date"].max()
)

calendar = pd.DataFrame({
    "date": pd.date_range(start_date, end_date, freq="D")
})

for signal_type in signals["signal_type"].unique():
    calendar[f"signal_{signal_type}"] = 0

calendar["signal_count"] = 0
calendar["signal_impact"] = 0

for _, row in signals.iterrows():
    mask = (calendar["date"] >= row["start_date"]) & (calendar["date"] <= row["end_date"])

    col = f"signal_{row['signal_type']}"
    calendar.loc[mask, col] = 1
    calendar.loc[mask, "signal_count"] += 1
    calendar.loc[mask, "signal_impact"] += row["impact"]

os.makedirs("data/processed", exist_ok=True)
calendar.to_csv(OUTPUT_FILE, index=False)

print("INTELLIGENCE DATABASE CREATO")
print("File:", OUTPUT_FILE)
print("Date:", calendar["date"].min(), "→", calendar["date"].max())
print(calendar[calendar["signal_count"] > 0])
