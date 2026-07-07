import os
import json
import pandas as pd

INPUT = "data/processed/feature_store_final.csv"
OUTPUT = "data/processed/feature_store_meta.csv"
SUMMARY = "reports/external_intelligence/meta_feature_summary.json"


def safe_div(a, b):
    return a / b if b != 0 else 0


def main():
    df = pd.read_csv(INPUT)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # =========================
    # BUSINESS MOMENTUM
    # =========================

    df["business_momentum_7_30"] = df["total_rolling_7"] / df["total_rolling_30"]
    df["business_momentum_30_90"] = df["total_rolling_30"] / df["total_rolling_90"]
    df["business_momentum_90_365"] = df["total_rolling_90"] / df["total_rolling_365"]

    # =========================
    # BUSINESS ACCELERATION
    # =========================

    df["business_acceleration"] = (
        df["business_momentum_7_30"] - df["business_momentum_30_90"]
    )

    df["business_acceleration_long"] = (
        df["business_momentum_30_90"] - df["business_momentum_90_365"]
    )

    # =========================
    # VOLATILITY
    # =========================

    df["volatility_14"] = df["total"].shift(1).rolling(14).std()
    df["volatility_30"] = df["total"].shift(1).rolling(30).std()
    df["volatility_60"] = df["total"].shift(1).rolling(60).std()

    df["volatility_ratio_14_60"] = df["volatility_14"] / df["volatility_60"]

    # =========================
    # CHANNEL STRENGTH
    # =========================

    df["delivery_strength"] = df["delivery_rolling_7"] / df["delivery_rolling_30"]
    df["digital_strength"] = df["digital_rolling_7"] / df["digital_rolling_30"]
    df["cash_strength"] = df["cash_rolling_7"] / df["cash_rolling_30"]

    df["delivery_vs_business"] = df["delivery_strength"] / df["business_momentum_7_30"]
    df["digital_vs_business"] = df["digital_strength"] / df["business_momentum_7_30"]
    df["cash_vs_business"] = df["cash_strength"] / df["business_momentum_7_30"]

    # =========================
    # CASH DECLINE SPEED
    # =========================

    df["cash_share_30"] = df["cash"].shift(1).rolling(30).sum() / df["total"].shift(1).rolling(30).sum()
    df["cash_share_90"] = df["cash"].shift(1).rolling(90).sum() / df["total"].shift(1).rolling(90).sum()
    df["cash_share_365"] = df["cash"].shift(1).rolling(365).sum() / df["total"].shift(1).rolling(365).sum()

    df["cash_decline_speed_30_365"] = df["cash_share_30"] - df["cash_share_365"]
    df["cash_decline_speed_90_365"] = df["cash_share_90"] - df["cash_share_365"]

    # =========================
    # DELIVERY GROWTH SPEED
    # =========================

    df["delivery_share_30"] = df["delivery"].shift(1).rolling(30).sum() / df["total"].shift(1).rolling(30).sum()
    df["delivery_share_90"] = df["delivery"].shift(1).rolling(90).sum() / df["total"].shift(1).rolling(90).sum()
    df["delivery_share_365"] = df["delivery"].shift(1).rolling(365).sum() / df["total"].shift(1).rolling(365).sum()

    df["delivery_growth_speed_30_365"] = df["delivery_share_30"] - df["delivery_share_365"]
    df["delivery_growth_speed_90_365"] = df["delivery_share_90"] - df["delivery_share_365"]

    # =========================
    # SEASON TRANSITION
    # =========================

    df["season_transition"] = (
        df["commercial_season"] != df["commercial_season"].shift(1)
    ).astype(int)

    df["days_from_season_start"] = (
        df.groupby("commercial_season").cumcount()
    )

    # =========================
    # RESTAURANT HEALTH INDEX
    # =========================

    # indice semplice 0-100
    # momentum positivo + delivery stabile + volatilità non estrema
    momentum_score = (df["business_momentum_30_90"] * 50).clip(0, 70)
    delivery_score = (df["delivery_share_90"] * 30).clip(0, 30)
    volatility_penalty = ((df["volatility_ratio_14_60"] - 1).abs() * 15).clip(0, 20)

    df["restaurant_health_index"] = (
        momentum_score + delivery_score + 20 - volatility_penalty
    ).clip(0, 100)

    # =========================
    # CLEANUP
    # =========================

    df = df.replace([float("inf"), -float("inf")], pd.NA)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports/external_intelligence", exist_ok=True)

    df.to_csv(OUTPUT, index=False)

    meta_features = [
        "business_momentum_7_30",
        "business_momentum_30_90",
        "business_momentum_90_365",
        "business_acceleration",
        "business_acceleration_long",
        "volatility_14",
        "volatility_30",
        "volatility_60",
        "volatility_ratio_14_60",
        "delivery_strength",
        "digital_strength",
        "cash_strength",
        "delivery_vs_business",
        "digital_vs_business",
        "cash_vs_business",
        "cash_share_30",
        "cash_share_90",
        "cash_share_365",
        "cash_decline_speed_30_365",
        "cash_decline_speed_90_365",
        "delivery_share_30",
        "delivery_share_90",
        "delivery_share_365",
        "delivery_growth_speed_30_365",
        "delivery_growth_speed_90_365",
        "season_transition",
        "days_from_season_start",
        "restaurant_health_index",
    ]

    summary = {
        "engine": "meta_feature_engine_v1",
        "input": INPUT,
        "output": OUTPUT,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "meta_features_created": meta_features,
    }

    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 90)
    print("META FEATURE ENGINE V1")
    print("=" * 90)
    print(json.dumps(summary, indent=2))

    print("\nUltime righe meta:")
    print(
        df[[
            "date",
            "total",
            "business_momentum_7_30",
            "business_momentum_30_90",
            "business_acceleration",
            "volatility_ratio_14_60",
            "delivery_strength",
            "cash_share_30",
            "cash_share_365",
            "cash_decline_speed_30_365",
            "restaurant_health_index",
        ]].tail(10).to_string(index=False)
    )

    print("\nSalvato:")
    print(OUTPUT)
    print(SUMMARY)


if __name__ == "__main__":
    main()
