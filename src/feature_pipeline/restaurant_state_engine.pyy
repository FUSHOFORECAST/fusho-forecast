import os
import json
import pandas as pd

INPUT = "data/processed/feature_store_meta.csv"
OUTPUT = "data/processed/feature_store_state.csv"
SUMMARY = "reports/external_intelligence/restaurant_state_summary.json"


def classify_growth(row):
    momentum = row["business_momentum_30_90"]
    acceleration = row["business_acceleration"]

    if momentum >= 1.08 and acceleration > 0.02:
        return "STRONG_GROWTH"
    elif momentum >= 1.03:
        return "GROWING"
    elif momentum <= 0.92 and acceleration < -0.02:
        return "DECLINING"
    elif momentum <= 0.97:
        return "COOLING"
    return "STABLE"


def classify_volatility(row):
    ratio = row["volatility_ratio_14_60"]

    if ratio >= 1.35:
        return "HIGH_VOLATILITY"
    elif ratio <= 0.75:
        return "LOW_VOLATILITY"
    return "NORMAL_VOLATILITY"


def classify_temperature(row):
    momentum = row["business_momentum_7_30"]
    volatility = row["volatility_ratio_14_60"]

    if momentum >= 1.08 and volatility >= 1.10:
        return "HOT"
    elif momentum >= 1.04:
        return "WARM"
    elif momentum <= 0.94 and volatility >= 1.10:
        return "COLD"
    elif momentum <= 0.97:
        return "COOL"
    return "NORMAL"


def classify_delivery_state(row):
    delivery_strength = row["delivery_strength"]
    delivery_growth = row["delivery_growth_speed_30_365"]

    if delivery_strength >= 1.08 and delivery_growth > 0:
        return "DELIVERY_ACCELERATING"
    elif delivery_strength >= 1.03:
        return "DELIVERY_STRONG"
    elif delivery_strength <= 0.94:
        return "DELIVERY_WEAK"
    return "DELIVERY_NORMAL"


def classify_cash_state(row):
    cash_decline = row["cash_decline_speed_30_365"]
    cash_strength = row["cash_strength"]

    if cash_decline <= -0.02:
        return "CASH_DECLINING_FAST"
    elif cash_decline < -0.005:
        return "CASH_DECLINING"
    elif cash_strength >= 1.15:
        return "CASH_SPIKE"
    return "CASH_NORMAL"


def classify_pressure(row):
    external_negative = row.get("external_negative_impact", 0)
    is_bridge = row.get("is_bridge_candidate", 0)
    is_holiday = row.get("is_public_holiday", 0)
    is_school = row.get("is_school_break", 0)
    volatility = row["volatility_ratio_14_60"]

    score = 0

    score += external_negative
    score += is_bridge * 2
    score += is_holiday * 2
    score += is_school * 1

    if volatility >= 1.25:
        score += 2

    if score >= 5:
        return "HIGH_PRESSURE"
    elif score >= 2:
        return "MEDIUM_PRESSURE"
    return "LOW_PRESSURE"


def main():
    df = pd.read_csv(INPUT)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    required_cols = [
        "business_momentum_30_90",
        "business_momentum_7_30",
        "business_acceleration",
        "volatility_ratio_14_60",
        "delivery_strength",
        "delivery_growth_speed_30_365",
        "cash_decline_speed_30_365",
        "cash_strength",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Colonna mancante: {col}")

    df["growth_state"] = df.apply(classify_growth, axis=1)
    df["volatility_state"] = df.apply(classify_volatility, axis=1)
    df["restaurant_temperature"] = df.apply(classify_temperature, axis=1)
    df["delivery_state"] = df.apply(classify_delivery_state, axis=1)
    df["cash_state"] = df.apply(classify_cash_state, axis=1)
    df["market_pressure"] = df.apply(classify_pressure, axis=1)

    df["restaurant_state"] = (
        df["growth_state"]
        + "__"
        + df["volatility_state"]
        + "__"
        + df["restaurant_temperature"]
    )

    state_cols = [
        "growth_state",
        "volatility_state",
        "restaurant_temperature",
        "delivery_state",
        "cash_state",
        "market_pressure",
        "restaurant_state",
    ]

    dummies = pd.get_dummies(df[state_cols], prefix=state_cols).astype(int)
    df = pd.concat([df, dummies], axis=1)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports/external_intelligence", exist_ok=True)

    df.to_csv(OUTPUT, index=False)

    summary = {
        "engine": "restaurant_state_engine_v1",
        "input": INPUT,
        "output": OUTPUT,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "state_columns": state_cols,
        "growth_state_distribution": df["growth_state"].value_counts().to_dict(),
        "volatility_state_distribution": df["volatility_state"].value_counts().to_dict(),
        "temperature_distribution": df["restaurant_temperature"].value_counts().to_dict(),
        "delivery_state_distribution": df["delivery_state"].value_counts().to_dict(),
        "cash_state_distribution": df["cash_state"].value_counts().to_dict(),
        "market_pressure_distribution": df["market_pressure"].value_counts().to_dict(),
    }

    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 90)
    print("RESTAURANT STATE ENGINE V1")
    print("=" * 90)
    print(json.dumps(summary, indent=2))

    print("\nUltime righe:")
    print(
        df[[
            "date",
            "total",
            "business_momentum_30_90",
            "business_acceleration",
            "volatility_ratio_14_60",
            "growth_state",
            "volatility_state",
            "restaurant_temperature",
            "delivery_state",
            "cash_state",
            "market_pressure",
            "restaurant_state",
        ]].tail(20).to_string(index=False)
    )

    print("\nSalvato:")
    print(OUTPUT)
    print(SUMMARY)


if __name__ == "__main__":
    main()
