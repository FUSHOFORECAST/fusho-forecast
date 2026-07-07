import json
import os
import numpy as np
import pandas as pd

INPUT = "data/processed/feature_store_state.csv"

OUTPUT_JSON = "reports/confidence_engine/confidence_summary.json"
OUTPUT_DATA = "reports/confidence_engine/confidence_dataset.csv"

os.makedirs("reports/confidence_engine", exist_ok=True)

df = pd.read_csv(INPUT)
df["date"] = pd.to_datetime(df["date"])

# ---------- CONFIDENCE COMPONENTS ----------

# Business stability
df["conf_business"] = (
    1
    - np.clip(np.abs(df["business_acceleration"]) / 0.15, 0, 1)
)

# Volatility
df["conf_volatility"] = (
    1
    - np.clip((df["volatility_ratio_14_60"] - 0.5) / 1.5, 0, 1)
)

# Momentum
df["conf_momentum"] = np.clip(
    1 - np.abs(df["business_momentum_30_90"] - 1),
    0,
    1,
)

# Restaurant health
df["conf_health"] = np.clip(
    df["restaurant_health_index"] / 100,
    0,
    1,
)

# External pressure
pressure_score = {
    "LOW_PRESSURE": 1.00,
    "MEDIUM_PRESSURE": 0.75,
    "HIGH_PRESSURE": 0.45,
}

df["conf_pressure"] = (
    df["market_pressure"]
    .map(pressure_score)
    .fillna(0.80)
)

# ---------- GLOBAL CONFIDENCE ----------

weights = {
    "conf_business": 0.20,
    "conf_volatility": 0.25,
    "conf_momentum": 0.20,
    "conf_health": 0.20,
    "conf_pressure": 0.15,
}

df["forecast_confidence"] = (
    df["conf_business"] * weights["conf_business"]
    + df["conf_volatility"] * weights["conf_volatility"]
    + df["conf_momentum"] * weights["conf_momentum"]
    + df["conf_health"] * weights["conf_health"]
    + df["conf_pressure"] * weights["conf_pressure"]
)

df["forecast_confidence_pct"] = (
    df["forecast_confidence"] * 100
).round(2)

# ---------- LABEL ----------

def confidence_label(x):

    if x >= 90:
        return "VERY_HIGH"

    if x >= 80:
        return "HIGH"

    if x >= 70:
        return "MEDIUM"

    if x >= 60:
        return "LOW"

    return "VERY_LOW"

df["forecast_confidence_label"] = (
    df["forecast_confidence_pct"]
    .apply(confidence_label)
)

df.to_csv(OUTPUT_DATA, index=False)

summary = {
    "engine": "confidence_engine_v1",
    "rows": int(len(df)),
    "mean_confidence": round(
        float(df["forecast_confidence_pct"].mean()),2
    ),
    "median_confidence": round(
        float(df["forecast_confidence_pct"].median()),2
    ),
    "distribution": (
        df["forecast_confidence_label"]
        .value_counts()
        .to_dict()
    ),
}

with open(OUTPUT_JSON,"w") as f:
    json.dump(summary,f,indent=2)

print("="*90)
print("CONFIDENCE ENGINE V1")
print("="*90)

print(json.dumps(summary,indent=2))

print("\nUltimi giorni:\n")

print(
    df[
        [
            "date",
            "total",
            "forecast_confidence_pct",
            "forecast_confidence_label",
            "business_momentum_30_90",
            "volatility_ratio_14_60",
            "restaurant_health_index",
            "market_pressure",
        ]
    ].tail(20).to_string(index=False)
)

print("\nSalvato:")
print(OUTPUT_DATA)
print(OUTPUT_JSON)
