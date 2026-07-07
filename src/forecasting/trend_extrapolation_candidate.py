import os
import json
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

INPUT = "data/processed/feature_store_state.csv"

OUTPUT_DIR = "reports/trend_extrapolation_candidate"
OUTPUT_FILE = f"{OUTPUT_DIR}/trend_extrapolation_candidate.csv"
OUTPUT_SUMMARY = f"{OUTPUT_DIR}/trend_extrapolation_summary.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "total"

CATEGORICAL_FEATURES = [
    "weekday",
    "commercial_season",
    "growth_state",
    "volatility_state",
    "restaurant_temperature",
    "delivery_state",
    "cash_state",
    "market_pressure",
    "restaurant_state",
]

NUMERIC_FEATURES = [
    "month",
    "dayofweek",
    "dayofyear",
    "week_of_month",

    "total_lag_1",
    "total_lag_7",
    "total_lag_14",
    "total_lag_30",

    "total_rolling_7",
    "total_rolling_14",
    "total_rolling_30",
    "total_rolling_60",
    "total_rolling_90",
    "total_rolling_180",
    "total_rolling_365",

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

    "delivery_share_30",
    "delivery_share_90",
    "delivery_share_365",
    "delivery_growth_speed_30_365",
    "delivery_growth_speed_90_365",

    "cash_share_30",
    "cash_share_90",
    "cash_share_365",
    "cash_decline_speed_30_365",
    "cash_decline_speed_90_365",

    "restaurant_health_index",

    "is_public_holiday",
    "is_preholiday",
    "is_postholiday",
    "is_school_break",
    "is_bridge_candidate",
    "days_to_next_holiday",
    "days_from_previous_holiday",
]


def prepare_features(df):
    available_numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    available_categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    x_num = df[available_numeric].copy().fillna(0)

    x_cat = pd.get_dummies(
        df[available_categorical].fillna("UNKNOWN"),
        prefix=available_categorical,
    ).astype(int)

    return pd.concat([x_num, x_cat], axis=1)


def align_columns(train_x, test_x):
    return test_x.reindex(columns=train_x.columns, fill_value=0)


def main():
    df = pd.read_csv(INPUT)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df = df.dropna(subset=["total_rolling_365", TARGET]).copy()

    split_date = df["date"].max() - pd.Timedelta(days=90)

    train = df[df["date"] <= split_date].copy()
    test = df[df["date"] > split_date].copy()

    x_train = prepare_features(train)
    x_test = prepare_features(test)
    x_test = align_columns(x_train, x_test)

    y_train = train[TARGET]
    y_test = test[TARGET]

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=700,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=4,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=700,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=3,
        ),
    }

    best_model = None
    best_name = None
    best_mae = float("inf")
    best_pred = None

    model_scores = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        mae = mean_absolute_error(y_test, pred)
        mape = (abs(pred - y_test) / y_test * 100).mean()

        model_scores[name] = {
            "mae": round(float(mae), 2),
            "mape": round(float(mape), 2),
        }

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_name = name
            best_pred = pred

    result = test[[
        "date",
        "total",
        "weekday",
        "commercial_season",
        "growth_state",
        "restaurant_temperature",
        "delivery_state",
        "restaurant_state",
    ]].copy()

    result["trend_extrapolation"] = best_pred
    result["error"] = result["trend_extrapolation"] - result["total"]
    result["abs_error"] = result["error"].abs()
    result["pct_error"] = result["abs_error"] / result["total"] * 100

    result["trend_extrapolation"] = result["trend_extrapolation"].round(2)
    result["error"] = result["error"].round(2)
    result["abs_error"] = result["abs_error"].round(2)
    result["pct_error"] = result["pct_error"].round(2)

    summary = {
        "engine": "trend_extrapolation_candidate_v1",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "best_model": best_name,
        "model_scores": model_scores,
        "mae": round(float(result["abs_error"].mean()), 2),
        "mape": round(float(result["pct_error"].mean()), 2),
        "total_forecast": round(float(result["trend_extrapolation"].sum()), 2),
        "total_real": round(float(result["total"].sum()), 2),
        "total_error": round(float(result["trend_extrapolation"].sum() - result["total"].sum()), 2),
        "total_error_pct": round(
            float(
                (result["trend_extrapolation"].sum() - result["total"].sum())
                / result["total"].sum()
                * 100
            ),
            2,
        ),
    }

    result.to_csv(OUTPUT_FILE, index=False)

    with open(OUTPUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 90)
    print("TREND EXTRAPOLATION CANDIDATE V1")
    print("=" * 90)

    print(json.dumps(summary, indent=2))

    print("\nULTIMI 20 GIORNI TEST")
    print(
        result[
            [
                "date",
                "total",
                "trend_extrapolation",
                "error",
                "pct_error",
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
