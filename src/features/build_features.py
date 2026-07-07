import os
import pandas as pd

from src.config import FULL_DATASET, MODEL_DATASET


EXCLUDE_FROM_NA_CHECK = ["holiday_name"]


def build_features(input_file=FULL_DATASET, output_file=MODEL_DATASET):
    df = pd.read_csv(input_file)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "holiday_name" in df.columns:
        df["holiday_name"] = df["holiday_name"].fillna("")

    # Calendar
    df["dayofweek"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["dayofyear"] = df["date"].dt.dayofyear
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # Lags / rolling
    for col in ["total", "delivery", "digital", "cash"]:
        for lag in [1, 7, 14, 21, 28, 30]:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        df[f"{col}_rolling_7"] = df[col].shift(1).rolling(7).mean()
        df[f"{col}_rolling_14"] = df[col].shift(1).rolling(14).mean()
        df[f"{col}_rolling_30"] = df[col].shift(1).rolling(30).mean()

    # Trend
    df["total_trend"] = df["total_rolling_7"] / df["total_rolling_30"]
    df["delivery_trend"] = df["delivery_rolling_7"] / df["delivery_rolling_30"]
    df["digital_trend"] = df["digital_rolling_7"] / df["digital_rolling_30"]
    df["cash_trend"] = df["cash_rolling_7"] / df["cash_rolling_30"]

    # Mix
    df["delivery_share_lag"] = df["delivery_lag_7"] / df["total_lag_7"].replace(0, pd.NA)
    df["digital_share_lag"] = df["digital_lag_7"] / df["total_lag_7"].replace(0, pd.NA)
    df["cash_share_lag"] = df["cash_lag_7"] / df["total_lag_7"].replace(0, pd.NA)

    df = df.replace([float("inf"), -float("inf")], pd.NA)

    cols_to_check = [c for c in df.columns if c not in EXCLUDE_FROM_NA_CHECK]
    df = df.dropna(subset=cols_to_check).copy()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

    print("FEATURE DATASET CREATO")
    print("File:", output_file)
    print("Righe:", len(df))
    print("Colonne:", len(df.columns))
    print("Date:", df["date"].min(), "→", df["date"].max())


if __name__ == "__main__":
    build_features()
