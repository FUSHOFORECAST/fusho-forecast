import os
import json
import pandas as pd

INPUT = "data/processed/feature_store_state.csv"

OUTPUT_DIR = "reports/momentum_adjusted_candidate"
OUTPUT_FILE = f"{OUTPUT_DIR}/momentum_adjusted_candidate.csv"
OUTPUT_SUMMARY = f"{OUTPUT_DIR}/momentum_adjusted_summary.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    df = pd.read_csv(INPUT)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df = df.dropna(subset=[
        "total",
        "total_rolling_7",
        "total_rolling_30",
        "business_momentum_7_30",
        "business_acceleration",
        "volatility_ratio_14_60",
    ]).copy()

    # Base reattiva
    df["momentum_base"] = df["total_rolling_7"]

    # Momentum: quanto il breve periodo è sopra/sotto il medio periodo
    df["momentum_lift"] = df["business_momentum_7_30"] - 1

    # Acceleration: quanto il trend breve sta accelerando rispetto al medio
    df["acceleration_lift"] = df["business_acceleration"]

    # Volatility cap: se il ristorante è molto volatile, limitiamo la correzione
    df["volatility_damper"] = 1 / (1 + df["volatility_ratio_14_60"].clip(lower=0))

    # Formula universale, non specifica Fusho
    df["momentum_adjustment_pct"] = (
        df["momentum_lift"] * 0.70
        + df["acceleration_lift"] * 0.30
    ) * df["volatility_damper"]

    # Limite prudente universale
    df["momentum_adjustment_pct"] = df["momentum_adjustment_pct"].clip(
        lower=-0.18,
        upper=0.18,
    )

    df["momentum_adjusted_forecast"] = (
        df["momentum_base"] * (1 + df["momentum_adjustment_pct"])
    )

    df["error"] = df["momentum_adjusted_forecast"] - df["total"]
    df["abs_error"] = df["error"].abs()
    df["pct_error"] = df["abs_error"] / df["total"] * 100

    result = df[[
        "date",
        "total",
        "momentum_base",
        "business_momentum_7_30",
        "business_acceleration",
        "volatility_ratio_14_60",
        "momentum_adjustment_pct",
        "momentum_adjusted_forecast",
        "error",
        "abs_error",
        "pct_error",
        "growth_state",
        "restaurant_temperature",
        "delivery_state",
    ]].copy()

    summary = {
        "engine": "momentum_adjusted_candidate_v1",
        "rows": int(len(result)),
        "mae": round(float(result["abs_error"].mean()), 2),
        "mape": round(float(result["pct_error"].mean()), 2),
        "total_forecast": round(float(result["momentum_adjusted_forecast"].sum()), 2),
        "total_real": round(float(result["total"].sum()), 2),
        "total_error": round(float(result["momentum_adjusted_forecast"].sum() - result["total"].sum()), 2),
        "total_error_pct": round(
            float(
                (result["momentum_adjusted_forecast"].sum() - result["total"].sum())
                / result["total"].sum()
                * 100
            ),
            2,
        ),
        "avg_adjustment_pct": round(float(result["momentum_adjustment_pct"].mean() * 100), 2),
        "max_adjustment_pct": round(float(result["momentum_adjustment_pct"].max() * 100), 2),
        "min_adjustment_pct": round(float(result["momentum_adjustment_pct"].min() * 100), 2),
    }

    result.to_csv(OUTPUT_FILE, index=False)

    with open(OUTPUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 90)
    print("MOMENTUM ADJUSTED CANDIDATE V1")
    print("=" * 90)

    print(json.dumps(summary, indent=2))

    print("\nULTIMI 20 GIORNI")
    print(
        result[
            [
                "date",
                "total",
                "momentum_base",
                "momentum_adjusted_forecast",
                "error",
                "pct_error",
                "momentum_adjustment_pct",
                "growth_state",
                "restaurant_temperature",
                "delivery_state",
            ]
        ].tail(20).to_string(index=False)
    )

    print("\nSalvato:")
    print(OUTPUT_FILE)
    print(OUTPUT_SUMMARY)


if __name__ == "__main__":
    main()
