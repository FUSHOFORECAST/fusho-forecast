import os
import pandas as pd

INPUT_FILE = "data/processed/forecast_adaptive_memory_v2.csv"
OUTPUT_FILE = "reports/channel_forecast_report.csv"

df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])

df["delivery_pct"] = df["delivery_pred"] / df["total_pred"] * 100
df["digital_pct"] = df["digital_pred"] / df["total_pred"] * 100
df["cash_pct"] = df["cash_pred"] / df["total_pred"] * 100

report = df[[
    "date",
    "total_pred",
    "delivery_pred",
    "digital_pred",
    "cash_pred",
    "delivery_pct",
    "digital_pct",
    "cash_pct",
]].copy()

for col in [
    "total_pred",
    "delivery_pred",
    "digital_pred",
    "cash_pred",
]:
    report[col] = report[col].round(2)

for col in [
    "delivery_pct",
    "digital_pct",
    "cash_pct",
]:
    report[col] = report[col].round(2)

totals = {
    "date": "TOTAL",
    "total_pred": round(report["total_pred"].sum(), 2),
    "delivery_pred": round(report["delivery_pred"].sum(), 2),
    "digital_pred": round(report["digital_pred"].sum(), 2),
    "cash_pred": round(report["cash_pred"].sum(), 2),
    "delivery_pct": round(report["delivery_pred"].sum() / report["total_pred"].sum() * 100, 2),
    "digital_pct": round(report["digital_pred"].sum() / report["total_pred"].sum() * 100, 2),
    "cash_pct": round(report["cash_pred"].sum() / report["total_pred"].sum() * 100, 2),
}

report_with_total = pd.concat(
    [report, pd.DataFrame([totals])],
    ignore_index=True,
)

os.makedirs("reports", exist_ok=True)
report_with_total.to_csv(OUTPUT_FILE, index=False)

print("=" * 90)
print("CHANNEL FORECAST REPORT")
print("=" * 90)

print(report_with_total.to_string(index=False))

print("\nSalvato:", OUTPUT_FILE)
