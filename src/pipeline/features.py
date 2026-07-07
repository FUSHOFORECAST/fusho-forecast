import os

import pandas as pd

from src.pipeline.config import RestaurantConfig

EXCLUDE_FROM_NA_CHECK = ["holiday_name"]


def add_calendar_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    out["week"] = out["date"].dt.isocalendar().week.astype(int)
    out["dayofyear"] = out["date"].dt.dayofyear
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)
    out["is_month_start"] = out["date"].dt.is_month_start.astype(int)
    out["is_month_end"] = out["date"].dt.is_month_end.astype(int)

    return out


def add_lag_rolling_features(
    df: pd.DataFrame, channels: list[str], lags: list[int], rolling_windows: list[int], trend_windows: list[int]
) -> pd.DataFrame:
    out = df.copy()

    short_window, long_window = trend_windows

    for channel in channels:
        for lag in lags:
            out[f"{channel}_lag_{lag}"] = out[channel].shift(lag)

        for window in rolling_windows:
            out[f"{channel}_rolling_{window}"] = out[channel].shift(1).rolling(window).mean()
            out[f"{channel}_rolling_median_{window}"] = out[channel].shift(1).rolling(window).median()

        out[f"{channel}_trend"] = (
            out[f"{channel}_rolling_{short_window}"] / out[f"{channel}_rolling_{long_window}"]
        )

    for channel in channels:
        if channel == "total":
            continue
        out[f"{channel}_share_lag"] = out[f"{channel}_lag_7"] / out["total_lag_7"].replace(0, pd.NA)

    return out


def build_features(df: pd.DataFrame, config: RestaurantConfig, drop_na: bool = True) -> pd.DataFrame:
    out = df.sort_values("date").reset_index(drop=True).copy()

    if "holiday_name" in out.columns:
        out["holiday_name"] = out["holiday_name"].fillna("")

    out = add_calendar_derived_fields(out)

    channels_plus_total = config.channels + ["total"]
    out = add_lag_rolling_features(
        out, channels_plus_total, config.features.lags, config.features.rolling_windows, config.features.trend_windows
    )

    out = out.replace([float("inf"), -float("inf")], pd.NA)

    if drop_na:
        cols_to_check = [c for c in out.columns if c not in EXCLUDE_FROM_NA_CHECK]
        out = out.dropna(subset=cols_to_check).reset_index(drop=True)

    return out


def main():
    import argparse

    from src.pipeline.config import load_restaurant_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_restaurant_config(args.config)
    df = pd.read_csv(config.processed_path("master_dataset_full.csv"))
    feat_df = build_features(df, config)

    os.makedirs(config.processed_dir, exist_ok=True)
    feat_df.to_csv(config.processed_path("model_dataset_full.csv"), index=False)

    print("FEATURE DATASET CREATO")
    print("Righe:", len(feat_df))
    print("Colonne:", len(feat_df.columns))
    print("Date:", feat_df["date"].min(), "->", feat_df["date"].max())


if __name__ == "__main__":
    main()
