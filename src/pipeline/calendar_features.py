from datetime import timedelta

import holidays
import pandas as pd

from src.pipeline.config import RestaurantConfig


def add_calendar_features(df: pd.DataFrame, config: RestaurantConfig) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    min_year = out["date"].dt.year.min()
    max_year = out["date"].dt.year.max()
    year_range = range(int(min_year), int(max_year) + 2)

    country_holidays = holidays.country_holidays(config.location.country_code, years=year_range)
    holiday_dates = set(country_holidays.keys())

    out["is_holiday"] = out["date"].dt.date.apply(lambda d: int(d in holiday_dates))
    out["holiday_name"] = out["date"].dt.date.apply(lambda d: country_holidays.get(d, ""))
    out["dayofweek"] = out["date"].dt.dayofweek

    out["is_pre_holiday"] = out["date"].dt.date.apply(
        lambda d: int((d + timedelta(days=1)) in holiday_dates)
    )
    out["is_post_holiday"] = out["date"].dt.date.apply(
        lambda d: int((d - timedelta(days=1)) in holiday_dates)
    )

    def is_bridge(row):
        d = row["date"].date()
        wd = row["dayofweek"]

        if wd == 0 and (d + timedelta(days=1)) in holiday_dates:
            return 1
        if wd == 4 and (d - timedelta(days=1)) in holiday_dates:
            return 1
        return 0

    out["is_bridge"] = out.apply(is_bridge, axis=1)

    return out


def main():
    import argparse

    from src.pipeline.clean import clean
    from src.pipeline.config import load_restaurant_config
    from src.pipeline.ingest import ingest
    from src.pipeline.weather import fetch_historical_weather, merge_weather

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_restaurant_config(args.config)
    master_df, audit_df = ingest(config)
    clean_df = clean(master_df, audit_df, config)

    weather_df = fetch_historical_weather(
        config, clean_df["date"].min().strftime("%Y-%m-%d"), clean_df["date"].max().strftime("%Y-%m-%d")
    )
    df = merge_weather(clean_df, weather_df)
    df = add_calendar_features(df, config)

    print(df[["date", "holiday_name", "is_holiday", "is_pre_holiday", "is_post_holiday", "is_bridge"]].tail(20))


if __name__ == "__main__":
    main()
