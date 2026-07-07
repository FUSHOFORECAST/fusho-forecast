import os
import json
import pandas as pd

INPUT_FILE = "data/processed/model_dataset_share.csv"
OUTPUT_DATASET = "data/processed/model_dataset_regime.csv"
OUTPUT_PROFILE = "reports/adaptive_regime_profile.json"

df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Baseline mobile: media ultimi 30 giorni senza usare il giorno stesso
df["rolling_30_baseline"] = df["total"].shift(1).rolling(30).mean()

df = df.dropna(subset=["rolling_30_baseline"]).copy()

df["vs_baseline_pct"] = (
    (df["total"] - df["rolling_30_baseline"])
    / df["rolling_30_baseline"]
    * 100
)

# Percentili adattivi del ristorante
p10 = df["vs_baseline_pct"].quantile(0.10)
p25 = df["vs_baseline_pct"].quantile(0.25)
p75 = df["vs_baseline_pct"].quantile(0.75)
p90 = df["vs_baseline_pct"].quantile(0.90)

def classify_regime(x):
    if x <= p10:
        return "VERY_LOW"
    elif x <= p25:
        return "LOW"
    elif x < p75:
        return "NORMAL"
    elif x < p90:
        return "HIGH"
    else:
        return "VERY_HIGH"

df["adaptive_regime"] = df["vs_baseline_pct"].apply(classify_regime)

df["weekday"] = df["date"].dt.day_name()
df["month"] = df["date"].dt.month
df["dayofweek"] = df["date"].dt.dayofweek

profile = {
    "method": "adaptive_percentile_regime",
    "baseline": "rolling_30_days_shifted",
    "thresholds": {
        "p10_very_low_max": round(float(p10), 2),
        "p25_low_max": round(float(p25), 2),
        "p75_high_min": round(float(p75), 2),
        "p90_very_high_min": round(float(p90), 2),
    },
    "regime_distribution": (
        df["adaptive_regime"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .to_dict()
    ),
    "avg_total_by_regime": (
        df.groupby("adaptive_regime")["total"]
        .mean()
        .round(2)
        .to_dict()
    ),
    "avg_vs_baseline_by_regime": (
        df.groupby("adaptive_regime")["vs_baseline_pct"]
        .mean()
        .round(2)
        .to_dict()
    ),
    "regime_by_weekday": (
        pd.crosstab(df["weekday"], df["adaptive_regime"], normalize="index")
        .mul(100)
        .round(1)
        .to_dict()
    ),
}

os.makedirs("data/processed", exist_ok=True)
os.makedirs("reports", exist_ok=True)

df.to_csv(OUTPUT_DATASET, index=False)

with open(OUTPUT_PROFILE, "w") as f:
    json.dump(profile, f, indent=2)

print("=" * 80)
print("ADAPTIVE REGIME ENGINE")
print("=" * 80)

print("\nThresholds:")
print(profile["thresholds"])

print("\nDistribuzione regimi:")
print(df["adaptive_regime"].value_counts())

print("\nMedia totale per regime:")
print(df.groupby("adaptive_regime")["total"].mean().round(2))

print("\nMedia vs baseline per regime:")
print(df.groupby("adaptive_regime")["vs_baseline_pct"].mean().round(2))

print("\nSalvato:")
print(OUTPUT_DATASET)
print(OUTPUT_PROFILE)
