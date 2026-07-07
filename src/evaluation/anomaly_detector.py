import os
import pandas as pd

INPUT_FILE = "data/processed/model_dataset_share.csv"
OUTPUT_FILE = "reports/anomaly_profile.csv"

df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# baseline mobile: media degli ultimi 30 giorni, senza usare il giorno stesso
df["rolling_30_baseline"] = df["total"].shift(1).rolling(30).mean()

df = df.dropna(subset=["rolling_30_baseline"]).copy()

df["vs_baseline_pct"] = (
    (df["total"] - df["rolling_30_baseline"])
    / df["rolling_30_baseline"]
    * 100
)

def classify_day(x):
    if x <= -20:
        return "LOW"
    elif x >= 20:
        return "HIGH"
    return "NORMAL"

df["day_regime"] = df["vs_baseline_pct"].apply(classify_day)
df["weekday"] = df["date"].dt.day_name()
df["month"] = df["date"].dt.month

print("=" * 70)
print("ANOMALY DETECTOR V1")
print("=" * 70)

print("\nDistribuzione regimi:")
print(df["day_regime"].value_counts())

print("\nMedia incasso per regime:")
print(df.groupby("day_regime")["total"].mean().round(2))

print("\nRegimi per giorno settimana:")
print(pd.crosstab(df["weekday"], df["day_regime"], normalize="index").round(3))

print("\nTOP LOW DAYS:")
print(
    df.sort_values("vs_baseline_pct")
    [["date", "weekday", "total", "rolling_30_baseline", "vs_baseline_pct", "day_regime"]]
    .head(20)
    .to_string(index=False)
)

print("\nTOP HIGH DAYS:")
print(
    df.sort_values("vs_baseline_pct", ascending=False)
    [["date", "weekday", "total", "rolling_30_baseline", "vs_baseline_pct", "day_regime"]]
    .head(20)
    .to_string(index=False)
)

os.makedirs("reports", exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("\nSalvato:", OUTPUT_FILE)
