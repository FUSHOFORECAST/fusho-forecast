import os
import pandas as pd

INPUT_FILE = "data/processed/master_dataset_full.csv"
OUTPUT_FILE = "data/processed/feature_store.csv"


def add_calendar_features(df):
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekday"] = df["date"].dt.day_name()
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["dayofyear"] = df["date"].dt.dayofyear
    df["week_of_month"] = ((df["day"] - 1) // 7) + 1
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    return df


def add_channel_features(df):
    df["delivery_share"] = df["delivery"] / df["total"]
    df["digital_share"] = df["digital"] / df["total"]
    df["cash_share"] = df["cash"] / df["total"]

    df[["delivery_share", "digital_share", "cash_share"]] = df[
        ["delivery_share", "digital_share", "cash_share"]
    ].fillna(0)

    return df


def add_lag_rolling_features(df):
    targets = ["total", "delivery", "digital", "cash"]

    for col in targets:
        for lag in [1, 7, 14, 21, 28, 30]:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        for window in [7, 14, 30, 60, 90, 180, 365]:
            df[f"{col}_rolling_{window}"] = df[col].shift(1).rolling(window).mean()

    return df


def add_trend_features(df):
    df["total_trend_7_30"] = df["total_rolling_7"] / df["total_rolling_30"]
    df["total_trend_30_90"] = df["total_rolling_30"] / df["total_rolling_90"]
    df["total_trend_90_365"] = df["total_rolling_90"] / df["total_rolling_365"]

    df["delivery_trend_7_30"] = df["delivery_rolling_7"] / df["delivery_rolling_30"]
    df["digital_trend_7_30"] = df["digital_rolling_7"] / df["digital_rolling_30"]
    df["cash_trend_7_30"] = df["cash_rolling_7"] / df["cash_rolling_30"]

    return df


def add_business_state_features(df):
    df["business_momentum_30"] = df["total_rolling_30"] / df["total_rolling_180"]
    df["business_momentum_90"] = df["total_rolling_90"] / df["total_rolling_365"]

    df["delivery_mix_recent"] = df["delivery"].shift(1).rolling(90).sum() / df["total"].shift(1).rolling(90).sum()
    df["digital_mix_recent"] = df["digital"].shift(1).rolling(90).sum() / df["total"].shift(1).rolling(90).sum()
    df["cash_mix_recent"] = df["cash"].shift(1).rolling(90).sum() / df["total"].shift(1).rolling(90).sum()

    return df


def add_commercial_season(df):
    def season(row):
        m = row["month"]
        d = row["day"]

        if m == 8:
            return "AUGUST_EMPTY"
        if m == 7 and d <= 15:
            return "EARLY_JULY"
        if m == 7 and d > 15:
            return "LATE_JULY"
        if m == 9 and d <= 15:
            return "BACK_TO_CITY"
        if m == 12 and d >= 20:
            return "CHRISTMAS_PERIOD"
        if m == 1 and d <= 7:
            return "NEW_YEAR_PERIOD"
        if m in [3, 4, 5]:
            return "SPRING"
        if m == 6:
            return "EARLY_SUMMER"

        return "NORMAL"

    df["commercial_season"] = df.apply(season, axis=1)

    season_dummies = pd.get_dummies(df["commercial_season"], prefix="season").astype(int)
    df = pd.concat([df, season_dummies], axis=1)

    return df


def main():
    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df = df[df["total"] > 0].copy()

    df = add_calendar_features(df)
    df = add_channel_features(df)
    df = add_lag_rolling_features(df)
    df = add_trend_features(df)
    df = add_business_state_features(df)
    df = add_commercial_season(df)

    df = df.replace([float("inf"), -float("inf")], pd.NA)

    # teniamo solo righe con almeno rolling 365 disponibile
    df = df.dropna(subset=["total_rolling_365"]).copy()

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("=" * 80)
    print("FEATURE STORE CREATO")
    print("=" * 80)
    print("File:", OUTPUT_FILE)
    print("Righe:", len(df))
    print("Colonne:", len(df.columns))
    print("Date:", df["date"].min(), "→", df["date"].max())

    print("\nColonne principali:")
    print([
        "date",
        "total",
        "delivery",
        "digital",
        "cash",
        "total_rolling_30",
        "total_rolling_90",
        "business_momentum_30",
        "delivery_mix_recent",
        "digital_mix_recent",
        "cash_mix_recent",
        "commercial_season",
    ])

    print("\nUltime righe:")
    print(df[[
        "date",
        "total",
        "delivery",
        "digital",
        "cash",
        "total_rolling_30",
        "total_rolling_90",
        "business_momentum_30",
        "delivery_mix_recent",
        "digital_mix_recent",
        "cash_mix_recent",
        "commercial_season",
    ]].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
