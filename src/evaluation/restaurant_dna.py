import json
import os
import pandas as pd

INPUT_FILE = "data/processed/master_dataset_full.csv"
OUTPUT_FILE = "reports/restaurant_dna.json"

df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df[df["total"] > 0].copy()

df["delivery_share"] = df["delivery"] / df["total"]
df["digital_share"] = df["digital"] / df["total"]
df["cash_share"] = df["cash"] / df["total"]

df["dayofweek"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

def pct_effect(group_mean, base_mean):
    if base_mean == 0 or pd.isna(base_mean) or pd.isna(group_mean):
        return 0.0
    return float(round(((group_mean - base_mean) / base_mean) * 100, 2))

avg_total = df["total"].mean()
delivery_share = df["delivery_share"].mean()
cash_share = df["cash_share"].mean()

weekend_avg = df[df["is_weekend"] == 1]["total"].mean()
weekday_avg = df[df["is_weekend"] == 0]["total"].mean()

delivery_by_year = df.groupby("year")["delivery_share"].mean()
cash_by_year = df.groupby("year")["cash_share"].mean()

dna = {
    "restaurant_type": {
        "delivery_driven": bool(delivery_share >= 0.60),
        "cash_light": bool(cash_share <= 0.15),
        "weekend_sensitive": bool(abs(pct_effect(weekend_avg, weekday_avg)) >= 8),
        "delivery_growing": bool(delivery_by_year.iloc[-1] > delivery_by_year.iloc[0]),
        "cash_declining": bool(cash_by_year.iloc[-1] < cash_by_year.iloc[0]),
    },
    "baseline": {
        "avg_total": float(round(avg_total, 2)),
        "avg_delivery_share": float(round(df["delivery_share"].mean(), 4)),
        "avg_digital_share": float(round(df["digital_share"].mean(), 4)),
        "avg_cash_share": float(round(df["cash_share"].mean(), 4)),
    },
    "channel_trends": {
        "delivery_share_by_year": {str(k): float(round(v, 4)) for k, v in delivery_by_year.items()},
        "cash_share_by_year": {str(k): float(round(v, 4)) for k, v in cash_by_year.items()},
        "digital_share_by_year": {
            str(k): float(round(v, 4))
            for k, v in df.groupby("year")["digital_share"].mean().items()
        },
    },
    "weekday_behavior": {
        "avg_total_by_weekday": {
            str(k): float(round(v, 2))
            for k, v in df.groupby("dayofweek")["total"].mean().items()
        },
        "delivery_share_by_weekday": {
            str(k): float(round(v, 4))
            for k, v in df.groupby("dayofweek")["delivery_share"].mean().items()
        },
        "weekend_effect_pct": pct_effect(weekend_avg, weekday_avg),
    },
    "weather_sensitivity": {},
    "seasonality": {
        "avg_total_by_month": {
            str(k): float(round(v, 2))
            for k, v in df.groupby("month")["total"].mean().items()
        },
        "delivery_share_by_month": {
            str(k): float(round(v, 4))
            for k, v in df.groupby("month")["delivery_share"].mean().items()
        },
    },
}

if "rain" in df.columns:
    rainy = df[df["rain"] > 0]["total"].mean()
    dry = df[df["rain"] == 0]["total"].mean()
    dna["weather_sensitivity"]["rain_effect_pct"] = pct_effect(rainy, dry)

if "temp_max" in df.columns:
    hot = df[df["temp_max"] >= 30]["total"].mean()
    normal = df[df["temp_max"] < 30]["total"].mean()
    extreme = df[df["temp_max"] >= 35]["total"].mean()
    not_extreme = df[df["temp_max"] < 35]["total"].mean()

    dna["weather_sensitivity"]["heat_over_30_effect_pct"] = pct_effect(hot, normal)
    dna["weather_sensitivity"]["heat_over_35_effect_pct"] = pct_effect(extreme, not_extreme)

os.makedirs("reports", exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(dna, f, indent=2)

print("=" * 70)
print("RESTAURANT DNA CREATO")
print("=" * 70)
print(json.dumps(dna, indent=2))
print("\nSalvato:", OUTPUT_FILE)
