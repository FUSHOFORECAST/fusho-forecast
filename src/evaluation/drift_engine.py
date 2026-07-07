import json
import os
import pandas as pd

INPUT_FILE = "data/processed/master_dataset_full.csv"
OUTPUT_FILE = "reports/drift_engine.json"

df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df[df["total"] > 0].copy()
df = df.sort_values("date")

df["delivery_share"] = df["delivery"] / df["total"]
df["digital_share"] = df["digital"] / df["total"]
df["cash_share"] = df["cash"] / df["total"]

last_date = df["date"].max()

windows = {
    "last_30_days": 30,
    "last_90_days": 90,
    "last_180_days": 180,
    "last_365_days": 365,
}

summary = {
    "last_date": str(last_date.date()),
    "windows": {},
    "drift": {},
}

for window_name, days in windows.items():
    start_date = last_date - pd.Timedelta(days=days)
    temp = df[df["date"] > start_date].copy()

    summary["windows"][window_name] = {
        "days": int(len(temp)),
        "avg_total": float(round(temp["total"].mean(), 2)),
        "avg_delivery": float(round(temp["delivery"].mean(), 2)),
        "avg_digital": float(round(temp["digital"].mean(), 2)),
        "avg_cash": float(round(temp["cash"].mean(), 2)),
        "avg_delivery_share": float(round(temp["delivery_share"].mean(), 4)),
        "avg_digital_share": float(round(temp["digital_share"].mean(), 4)),
        "avg_cash_share": float(round(temp["cash_share"].mean(), 4)),
    }

baseline = summary["windows"]["last_365_days"]
recent = summary["windows"]["last_30_days"]

def pct_change(recent_value, baseline_value):
    if baseline_value == 0 or pd.isna(baseline_value):
        return 0.0
    return float(round(((recent_value - baseline_value) / baseline_value) * 100, 2))

summary["drift"] = {
    "total_vs_365_pct": pct_change(recent["avg_total"], baseline["avg_total"]),
    "delivery_share_vs_365_pct": pct_change(
        recent["avg_delivery_share"],
        baseline["avg_delivery_share"],
    ),
    "digital_share_vs_365_pct": pct_change(
        recent["avg_digital_share"],
        baseline["avg_digital_share"],
    ),
    "cash_share_vs_365_pct": pct_change(
        recent["avg_cash_share"],
        baseline["avg_cash_share"],
    ),
}

summary["drift_flags"] = {
    "total_drifting": bool(abs(summary["drift"]["total_vs_365_pct"]) >= 8),
    "delivery_mix_drifting": bool(abs(summary["drift"]["delivery_share_vs_365_pct"]) >= 8),
    "digital_mix_drifting": bool(abs(summary["drift"]["digital_share_vs_365_pct"]) >= 8),
    "cash_mix_drifting": bool(abs(summary["drift"]["cash_share_vs_365_pct"]) >= 8),
}

os.makedirs("reports", exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(summary, f, indent=2)

print("=" * 70)
print("DRIFT ENGINE")
print("=" * 70)
print(json.dumps(summary, indent=2))
print("\nSalvato:", OUTPUT_FILE)
