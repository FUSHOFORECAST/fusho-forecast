import json
import os
import pandas as pd

INPUT_FILE = "reports/anomaly_profile.csv"
SEASON_FILE = "reports/business_season_patterns.csv"
OUTPUT_FILE = "reports/business_model_profile.json"

df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])

season = pd.read_csv(SEASON_FILE)

profile = {}

# =========================
# BASE
# =========================

profile["dataset"] = {
    "start_date": str(df["date"].min().date()),
    "end_date": str(df["date"].max().date()),
    "days": int(len(df)),
}

profile["baseline"] = {
    "avg_total": round(float(df["total"].mean()), 2),
    "median_total": round(float(df["total"].median()), 2),
    "std_total": round(float(df["total"].std()), 2),
}

# =========================
# VOLATILITY
# =========================

cv = df["total"].std() / df["total"].mean()

profile["volatility"] = {
    "coefficient_of_variation": round(float(cv), 4),
    "business_type": "volatile" if cv >= 0.25 else "stable",
}

# =========================
# REGIME DISTRIBUTION
# =========================

regime_counts = df["day_regime"].value_counts(normalize=True) * 100

profile["regimes"] = {
    "normal_rate": round(float(regime_counts.get("NORMAL", 0)), 2),
    "low_rate": round(float(regime_counts.get("LOW", 0)), 2),
    "high_rate": round(float(regime_counts.get("HIGH", 0)), 2),
}

# =========================
# WEEKDAY SENSITIVITY
# =========================

weekday = (
    df.groupby("weekday")
    .agg(
        avg_total=("total", "mean"),
        low_rate=("day_regime", lambda x: (x == "LOW").mean() * 100),
        high_rate=("day_regime", lambda x: (x == "HIGH").mean() * 100),
    )
    .reset_index()
)

weekday["avg_total"] = weekday["avg_total"].round(2)
weekday["low_rate"] = weekday["low_rate"].round(2)
weekday["high_rate"] = weekday["high_rate"].round(2)

best_day = weekday.sort_values("avg_total", ascending=False).iloc[0]
worst_day = weekday.sort_values("avg_total").iloc[0]

weekday_spread = (
    (best_day["avg_total"] - worst_day["avg_total"])
    / df["total"].mean()
    * 100
)

profile["weekday_sensitivity"] = {
    "best_day": str(best_day["weekday"]),
    "worst_day": str(worst_day["weekday"]),
    "weekday_spread_pct": round(float(weekday_spread), 2),
    "weekday_sensitive": bool(abs(weekday_spread) >= 10),
    "details": weekday.to_dict(orient="records"),
}

# =========================
# SEASONALITY
# =========================

season_sorted = season.sort_values("avg_vs_baseline_pct")

weakest_season = season_sorted.iloc[0]
strongest_season = season_sorted.iloc[-1]

season_spread = (
    strongest_season["avg_vs_baseline_pct"]
    - weakest_season["avg_vs_baseline_pct"]
)

profile["seasonality"] = {
    "strongest_season": str(strongest_season["commercial_season"]),
    "strongest_effect_pct": round(float(strongest_season["avg_vs_baseline_pct"]), 2),
    "weakest_season": str(weakest_season["commercial_season"]),
    "weakest_effect_pct": round(float(weakest_season["avg_vs_baseline_pct"]), 2),
    "seasonality_spread_pct": round(float(season_spread), 2),
    "seasonal_sensitive": bool(abs(season_spread) >= 15),
    "details": season.to_dict(orient="records"),
}

# =========================
# LOW / HIGH PATTERN SENSITIVITY
# =========================

profile["pattern_sensitivity"] = {
    "low_days_exist": bool(profile["regimes"]["low_rate"] >= 10),
    "high_days_exist": bool(profile["regimes"]["high_rate"] >= 10),
    "requires_regime_model": bool(
        profile["regimes"]["low_rate"] >= 10
        or profile["regimes"]["high_rate"] >= 10
    ),
}

# =========================
# FINAL BUSINESS MODEL
# =========================

tags = []

if profile["volatility"]["business_type"] == "volatile":
    tags.append("volatile_business")
else:
    tags.append("stable_business")

if profile["seasonality"]["seasonal_sensitive"]:
    tags.append("seasonal_business")

if profile["weekday_sensitivity"]["weekday_sensitive"]:
    tags.append("weekday_sensitive")

if profile["pattern_sensitivity"]["requires_regime_model"]:
    tags.append("regime_sensitive")

profile["business_model"] = {
    "tags": tags,
    "recommended_forecast_strategy": (
        "regime_aware_forecast"
        if profile["pattern_sensitivity"]["requires_regime_model"]
        else "standard_forecast"
    ),
}

os.makedirs("reports", exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(profile, f, indent=2)

print("=" * 80)
print("BUSINESS MODEL PROFILE")
print("=" * 80)
print(json.dumps(profile, indent=2))
print("\nSalvato:", OUTPUT_FILE)
