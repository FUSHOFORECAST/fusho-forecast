import json
import pandas as pd

PROFILE_FILE = "reports/restaurant_profile.json"
INPUT_FILE = "data/processed/model_dataset_share.csv"
OUTPUT_FILE = "data/processed/model_dataset_profile.csv"


def main():
    with open(PROFILE_FILE, "r") as f:
        profile = json.load(f)

    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])

    df["dayofweek"] = df["date"].dt.dayofweek
    df["year"] = df["date"].dt.year

    # pattern automatici dal profilo
    df["is_sunday"] = (df["dayofweek"] == 6).astype(int)

    df["sunday_delivery_boost"] = (
        df["is_sunday"] *
        profile["weekend_profile"]["sunday_delivery_share_lift_pct"]
    )

    df["rain_penalty"] = (
        df["is_rainy"] *
        profile["weather_profile"]["rain_effect_pct"]
    )

    df["heavy_rain_penalty"] = (
        df["is_heavy_rain"] *
        profile["weather_profile"]["heavy_rain_effect_pct"]
    )

    df["heat_over_30"] = (df["temp_max"] >= 30).astype(int)

    df["heat_over_30_effect"] = (
        df["heat_over_30"] *
        profile["weather_profile"]["heat_over_30_effect_pct"]
    )

    cash_by_year = profile["trend_profile"]["cash_share_by_year"]
    delivery_by_year = profile["trend_profile"]["delivery_share_by_year"]

    df["profile_cash_share_year"] = df["year"].astype(str).map(cash_by_year)
    df["profile_delivery_share_year"] = df["year"].astype(str).map(delivery_by_year)

    df["profile_cash_share_year"] = df["profile_cash_share_year"].fillna(
        profile["baseline"]["avg_cash_share"]
    )

    df["profile_delivery_share_year"] = df["profile_delivery_share_year"].fillna(
        profile["baseline"]["avg_delivery_share"]
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("PROFILE FEATURES CREATE")
    print("File:", OUTPUT_FILE)
    print("Righe:", len(df))
    print("Colonne:", len(df.columns))
    print(df[[
        "date",
        "sunday_delivery_boost",
        "rain_penalty",
        "heavy_rain_penalty",
        "heat_over_30_effect",
        "profile_cash_share_year",
        "profile_delivery_share_year"
    ]].tail(10))


if __name__ == "__main__":
    main()
