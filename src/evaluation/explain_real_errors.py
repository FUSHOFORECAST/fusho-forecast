import pandas as pd

FORECAST_FILE = "data/processed/forecast_weighted_hierarchical_days.csv"
REAL_FILE = "data/processed/june_real_totals.csv"
INTELLIGENCE_FILE = "data/processed/intelligence_calendar.csv"
OUTPUT_FILE = "reports/real_error_explained.csv"

forecast = pd.read_csv(FORECAST_FILE)
real = pd.read_csv(REAL_FILE)
intel = pd.read_csv(INTELLIGENCE_FILE)

forecast["date"] = pd.to_datetime(forecast["date"])
real["date"] = pd.to_datetime(real["date"])
intel["date"] = pd.to_datetime(intel["date"])

df = forecast.merge(real, on="date", how="inner")
df = df.merge(intel, on="date", how="left")

signal_cols = [c for c in df.columns if c.startswith("signal_")]

for col in signal_cols:
    df[col] = df[col].fillna(0)

df["error"] = df["total_pred"] - df["real_total"]
df["abs_error"] = df["error"].abs()
df["pct_error"] = df["abs_error"] / df["real_total"] * 100
df["weekday"] = df["date"].dt.day_name()

print("=" * 80)
print("REAL ERROR EXPLAINED")
print("=" * 80)

cols = [
    "date",
    "weekday",
    "total_pred",
    "real_total",
    "error",
    "pct_error",
    "signal_count",
    "signal_impact",
] + signal_cols

print(df[cols].to_string(index=False))

print("\nMEDIA ERRORE CON SEGNALI:")
print(df.groupby("signal_count")["pct_error"].mean())

print("\nGIORNI CON ERRORE > 20%:")
print(
    df[df["pct_error"] > 20][cols].to_string(index=False)
)

df.to_csv(OUTPUT_FILE, index=False)
print("\nSalvato:", OUTPUT_FILE)
