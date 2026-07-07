import os
import pandas as pd

FORECAST_FILE = "data/processed/forecast_weighted_hierarchical_days.csv"
INTELLIGENCE_FILE = "data/processed/intelligence_calendar.csv"
OUTPUT_FILE = "data/processed/forecast_with_confidence.csv"

forecast = pd.read_csv(FORECAST_FILE)
intel = pd.read_csv(INTELLIGENCE_FILE)

forecast["date"] = pd.to_datetime(forecast["date"])
intel["date"] = pd.to_datetime(intel["date"])

df = forecast.merge(intel, on="date", how="left")

signal_cols = [c for c in df.columns if c.startswith("signal_")]

for col in signal_cols:
    df[col] = df[col].fillna(0)

df["dayofweek"] = df["date"].dt.dayofweek
df["weekday"] = df["date"].dt.day_name()

# confidence base
df["confidence_score"] = 90

# penalità segnali urbani
df["confidence_score"] -= df["signal_count"] * 8

# giorni più instabili
df.loc[df["dayofweek"].isin([1, 2]), "confidence_score"] -= 8  # martedì/mercoledì
df.loc[df["dayofweek"] == 5, "confidence_score"] -= 6          # sabato

df["confidence_score"] = df["confidence_score"].clip(lower=45, upper=95)

def risk_level(score):
    if score >= 80:
        return "LOW"
    elif score >= 65:
        return "MEDIUM"
    return "HIGH"

df["risk_level"] = df["confidence_score"].apply(risk_level)

# RANGE OPERATIVO STRETTO
df["range_pct"] = 0.06
df.loc[df["risk_level"] == "MEDIUM", "range_pct"] = 0.09
df.loc[df["risk_level"] == "HIGH", "range_pct"] = 0.12

df["forecast_min"] = (df["total_pred"] * (1 - df["range_pct"])).round(2)
df["forecast_max"] = (df["total_pred"] * (1 + df["range_pct"])).round(2)

df["warning"] = ""
df.loc[df["risk_level"] == "HIGH", "warning"] = (
    "Giornata instabile: possibile errore superiore al range operativo."
)

cols = [
    "date",
    "weekday",
    "total_pred",
    "forecast_min",
    "forecast_max",
    "confidence_score",
    "risk_level",
    "warning",
    "signal_count",
    "signal_impact",
] + signal_cols

print("=" * 80)
print("FORECAST WITH CONFIDENCE - RANGE OPERATIVO")
print("=" * 80)

print(df[cols].to_string(index=False))

os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("\nSalvato:", OUTPUT_FILE)
