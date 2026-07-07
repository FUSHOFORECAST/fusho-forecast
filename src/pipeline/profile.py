import json
import os

import pandas as pd

from src.pipeline.config import RestaurantConfig

PRIMARY_CHANNEL_THRESHOLD = 0.60
CASH_LIGHT_THRESHOLD = 0.15
WEEKEND_SENSITIVE_THRESHOLD_PCT = 8
HEAT_THRESHOLD_1 = 30
HEAT_THRESHOLD_2 = 35


def safe_pct_change(high, low):
    if low == 0 or pd.isna(low) or pd.isna(high):
        return None
    return round(((high - low) / low) * 100, 2)


def _with_shares(df: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    out = df[df["total"] > 0].copy()
    for channel in channels:
        out[f"{channel}_share"] = out[channel] / out["total"]
    return out


def build_profile(df: pd.DataFrame, config: RestaurantConfig) -> dict:
    channels = config.channels
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = _with_shares(data, channels)

    data["dayofweek"] = data["date"].dt.dayofweek
    data["weekday_name"] = data["date"].dt.day_name()
    data["month"] = data["date"].dt.month
    data["year"] = data["date"].dt.year
    data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(int)

    overall_total_avg = data["total"].mean()
    baseline = {"avg_total": round(overall_total_avg, 2)}
    for channel in channels:
        baseline[f"avg_{channel}_share"] = round(data[f"{channel}_share"].mean(), 4)

    weekday_total = (
        data.groupby("weekday_name")["total"].mean().sort_values(ascending=False).round(2).to_dict()
    )
    weekday_share_by_channel = {
        channel: data.groupby("weekday_name")[f"{channel}_share"].mean().round(4).to_dict()
        for channel in channels
    }

    weather_profile = {}
    if "temp_max" in data.columns:
        data["temp_bucket"] = pd.cut(
            data["temp_max"],
            bins=[-50, 10, 15, 20, 25, 30, 35, 60],
            labels=[
                "very_cold_under_10", "cold_10_15", "mild_15_20", "warm_20_25",
                "hot_25_30", "very_hot_30_35", "extreme_heat_over_35",
            ],
        )
        temp_total = data.groupby("temp_bucket", observed=True)["total"].agg(["count", "mean"]).reset_index()
        temp_profile = {}
        for _, row in temp_total.iterrows():
            temp_profile[str(row["temp_bucket"])] = {
                "days": int(row["count"]),
                "avg_total": round(row["mean"], 2),
                "effect_vs_avg_pct": safe_pct_change(row["mean"], overall_total_avg),
            }
        weather_profile["temperature_buckets"] = temp_profile

    if "is_rainy" in data.columns:
        rainy_avg = data[data["is_rainy"] == 1]["total"].mean()
        dry_avg = data[data["is_rainy"] == 0]["total"].mean()
        weather_profile["rain_effect_pct"] = safe_pct_change(rainy_avg, dry_avg)

    if "is_heavy_rain" in data.columns:
        heavy_rain_avg = data[data["is_heavy_rain"] == 1]["total"].mean()
        no_heavy_rain_avg = data[data["is_heavy_rain"] == 0]["total"].mean()
        weather_profile["heavy_rain_effect_pct"] = safe_pct_change(heavy_rain_avg, no_heavy_rain_avg)

    if "temp_max" in data.columns:
        high_heat_avg = data[data["temp_max"] >= HEAT_THRESHOLD_1]["total"].mean()
        normal_temp_avg = data[data["temp_max"] < HEAT_THRESHOLD_1]["total"].mean()
        extreme_avg = data[data["temp_max"] >= HEAT_THRESHOLD_2]["total"].mean()
        not_extreme_avg = data[data["temp_max"] < HEAT_THRESHOLD_2]["total"].mean()
        weather_profile["heat_over_30_effect_pct"] = safe_pct_change(high_heat_avg, normal_temp_avg)
        weather_profile["heat_over_35_effect_pct"] = safe_pct_change(extreme_avg, not_extreme_avg)

    weekend_avg = data[data["is_weekend"] == 1]["total"].mean()
    weekday_avg = data[data["is_weekend"] == 0]["total"].mean()

    weekend_profile = {"weekend_effect_pct": safe_pct_change(weekend_avg, weekday_avg)}
    for channel in channels:
        sunday_share = data[data["dayofweek"] == 6][f"{channel}_share"].mean()
        other_share = data[data["dayofweek"] != 6][f"{channel}_share"].mean()
        weekend_profile[f"sunday_{channel}_share"] = round(sunday_share, 4)
        weekend_profile[f"other_days_{channel}_share"] = round(other_share, 4)
        weekend_profile[f"sunday_{channel}_share_lift_pct"] = safe_pct_change(sunday_share, other_share)

    share_by_year = {channel: data.groupby("year")[f"{channel}_share"].mean().round(4).to_dict() for channel in channels}
    share_by_month = {channel: data.groupby("month")[f"{channel}_share"].mean().round(4).to_dict() for channel in channels}

    restaurant_type = {}
    for channel in channels:
        restaurant_type[f"{channel}_driven"] = bool(baseline[f"avg_{channel}_share"] >= PRIMARY_CHANNEL_THRESHOLD)
    if "cash" in channels:
        restaurant_type["cash_light"] = bool(baseline["avg_cash_share"] <= CASH_LIGHT_THRESHOLD)
    restaurant_type["weekend_sensitive"] = bool(
        abs(safe_pct_change(weekend_avg, weekday_avg) or 0) >= WEEKEND_SENSITIVE_THRESHOLD_PCT
    )
    for channel in channels:
        by_year = data.groupby("year")[f"{channel}_share"].mean()
        if len(by_year) >= 2:
            restaurant_type[f"{channel}_growing"] = bool(by_year.iloc[-1] > by_year.iloc[0])
            restaurant_type[f"{channel}_declining"] = bool(by_year.iloc[-1] < by_year.iloc[0])

    profile = {
        "dataset": {
            "start_date": str(data["date"].min().date()),
            "end_date": str(data["date"].max().date()),
            "days_used": int(len(data)),
        },
        "baseline": baseline,
        "weekday_profile": {
            "avg_total_by_weekday": weekday_total,
            "share_by_weekday": weekday_share_by_channel,
        },
        "weather_profile": weather_profile,
        "weekend_profile": weekend_profile,
        "trend_profile": {"share_by_year": share_by_year},
        "restaurant_type": restaurant_type,
        "seasonality": {
            "avg_total_by_month": {
                str(k): float(round(v, 2)) for k, v in data.groupby("month")["total"].mean().items()
            },
            "share_by_month": {
                channel: {str(k): float(v) for k, v in vals.items()} for channel, vals in share_by_month.items()
            },
        },
    }

    return profile


def profile_features(df: pd.DataFrame, profile: dict, config: RestaurantConfig) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["dayofweek"] = out["date"].dt.dayofweek
    out["year"] = out["date"].dt.year

    out["is_sunday"] = (out["dayofweek"] == 6).astype(int)

    weekend_profile = profile.get("weekend_profile", {})
    weather_profile = profile.get("weather_profile", {})
    share_by_year = profile.get("trend_profile", {}).get("share_by_year", {})
    baseline = profile.get("baseline", {})

    for channel in config.channels:
        lift_pct = weekend_profile.get(f"sunday_{channel}_share_lift_pct") or 0
        out[f"{channel}_sunday_boost"] = out["is_sunday"] * lift_pct

    if "is_rainy" in out.columns and "rain_effect_pct" in weather_profile:
        out["rain_penalty"] = out["is_rainy"] * weather_profile["rain_effect_pct"]

    if "is_heavy_rain" in out.columns and "heavy_rain_effect_pct" in weather_profile:
        out["heavy_rain_penalty"] = out["is_heavy_rain"] * weather_profile["heavy_rain_effect_pct"]

    if "temp_max" in out.columns and "heat_over_30_effect_pct" in weather_profile:
        out["heat_over_30"] = (out["temp_max"] >= HEAT_THRESHOLD_1).astype(int)
        out["heat_over_30_effect"] = out["heat_over_30"] * weather_profile["heat_over_30_effect_pct"]

    for channel in config.channels:
        year_map = share_by_year.get(channel, {})
        year_map = {int(k): v for k, v in year_map.items()}
        out[f"profile_{channel}_share_year"] = out["year"].map(year_map)
        out[f"profile_{channel}_share_year"] = out[f"profile_{channel}_share_year"].fillna(
            baseline.get(f"avg_{channel}_share", 0)
        )

    return out


def main():
    import argparse

    from src.pipeline.config import load_restaurant_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_restaurant_config(args.config)
    df = pd.read_csv(config.processed_path("master_dataset_full.csv"))
    profile = build_profile(df, config)

    os.makedirs(config.reports_dir, exist_ok=True)
    with open(config.reports_path("restaurant_profile.json"), "w") as f:
        json.dump(profile, f, indent=2)

    print(json.dumps(profile, indent=2))
    print("\nSalvato:", config.reports_path("restaurant_profile.json"))


if __name__ == "__main__":
    main()
