import json
import os
import pandas as pd

from src.config import FULL_DATASET, REPORTS_DIR


PROFILE_OUTPUT = "reports/restaurant_profile.json"


def safe_pct_change(high, low):
    if low == 0 or pd.isna(low) or pd.isna(high):
        return None
    return round(((high - low) / low) * 100, 2)


def build_restaurant_profile(input_file=FULL_DATASET, output_file=PROFILE_OUTPUT):
    df = pd.read_csv(input_file)
    df["date"] = pd.to_datetime(df["date"])

    df = df[df["total"] > 0].copy()

    df["cash_share"] = df["cash"] / df["total"]
    df["delivery_share"] = df["delivery"] / df["total"]
    df["digital_share"] = df["digital"] / df["total"]

    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekday_name"] = df["date"].dt.day_name()
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    # temperature buckets
    df["temp_bucket"] = pd.cut(
        df["temp_max"],
        bins=[-50, 10, 15, 20, 25, 30, 35, 60],
        labels=[
            "very_cold_under_10",
            "cold_10_15",
            "mild_15_20",
            "warm_20_25",
            "hot_25_30",
            "very_hot_30_35",
            "extreme_heat_over_35",
        ],
    )

    overall_total_avg = df["total"].mean()
    overall_delivery_share = df["delivery_share"].mean()
    overall_cash_share = df["cash_share"].mean()

    weekday_total = (
        df.groupby("weekday_name")["total"]
        .mean()
        .sort_values(ascending=False)
        .round(2)
        .to_dict()
    )

    weekday_delivery_share = (
        df.groupby("weekday_name")["delivery_share"]
        .mean()
        .sort_values(ascending=False)
        .round(4)
        .to_dict()
    )

    temp_total = (
        df.groupby("temp_bucket", observed=True)["total"]
        .agg(["count", "mean"])
        .reset_index()
    )

    temp_profile = {}
    for _, row in temp_total.iterrows():
        temp_profile[str(row["temp_bucket"])] = {
            "days": int(row["count"]),
            "avg_total": round(row["mean"], 2),
            "effect_vs_avg_pct": safe_pct_change(row["mean"], overall_total_avg),
        }

    rainy_avg = df[df["is_rainy"] == 1]["total"].mean()
    dry_avg = df[df["is_rainy"] == 0]["total"].mean()

    heavy_rain_avg = df[df["is_heavy_rain"] == 1]["total"].mean()
    no_heavy_rain_avg = df[df["is_heavy_rain"] == 0]["total"].mean()

    weekend_avg = df[df["dayofweek"].isin([5, 6])]["total"].mean()
    weekday_avg = df[~df["dayofweek"].isin([5, 6])]["total"].mean()

    sunday_delivery_share = df[df["dayofweek"] == 6]["delivery_share"].mean()
    all_other_delivery_share = df[df["dayofweek"] != 6]["delivery_share"].mean()

    cash_by_year = (
        df.groupby("year")["cash_share"]
        .mean()
        .round(4)
        .to_dict()
    )

    delivery_by_year = (
        df.groupby("year")["delivery_share"]
        .mean()
        .round(4)
        .to_dict()
    )

    # simple inferred patterns
    heat_threshold = 30
    high_heat_avg = df[df["temp_max"] >= heat_threshold]["total"].mean()
    normal_temp_avg = df[df["temp_max"] < heat_threshold]["total"].mean()

    profile = {
        "dataset": {
            "start_date": str(df["date"].min().date()),
            "end_date": str(df["date"].max().date()),
            "days_used": int(len(df)),
        },
        "baseline": {
            "avg_total": round(overall_total_avg, 2),
            "avg_delivery_share": round(overall_delivery_share, 4),
            "avg_digital_share": round(df["digital_share"].mean(), 4),
            "avg_cash_share": round(overall_cash_share, 4),
        },
        "weekday_profile": {
            "avg_total_by_weekday": weekday_total,
            "avg_delivery_share_by_weekday": weekday_delivery_share,
        },
        "weather_profile": {
            "temperature_buckets": temp_profile,
            "rain_effect_pct": safe_pct_change(rainy_avg, dry_avg),
            "heavy_rain_effect_pct": safe_pct_change(heavy_rain_avg, no_heavy_rain_avg),
            "heat_over_30_effect_pct": safe_pct_change(high_heat_avg, normal_temp_avg),
        },
        "weekend_profile": {
            "weekend_effect_pct": safe_pct_change(weekend_avg, weekday_avg),
            "sunday_delivery_share": round(sunday_delivery_share, 4),
            "other_days_delivery_share": round(all_other_delivery_share, 4),
            "sunday_delivery_share_lift_pct": safe_pct_change(
                sunday_delivery_share,
                all_other_delivery_share,
            ),
        },
        "trend_profile": {
            "cash_share_by_year": cash_by_year,
            "delivery_share_by_year": delivery_by_year,
        },
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(profile, f, indent=2)

    print("=" * 70)
    print("RESTAURANT PROFILE CREATO")
    print("=" * 70)
    print(json.dumps(profile, indent=2))
    print("\nSalvato:", output_file)


if __name__ == "__main__":
    build_restaurant_profile()
