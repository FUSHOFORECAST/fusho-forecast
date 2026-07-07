import os
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

INPUT = "reports/adaptive_weight_engine/forecast_candidates_backtest.csv"

OUTPUT_DIR = "reports/error_prediction_engine_v5"
OUTPUT_FORECAST = f"{OUTPUT_DIR}/error_prediction_forecast.csv"
OUTPUT_SUMMARY = f"{OUTPUT_DIR}/error_prediction_summary.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CANDIDATES = [
    "memory_short",
    "memory_medium",
    "memory_long",
    "adaptive_memory",
    "state_forecast",
    "calendar_forecast",
    "similarity_forecast",
]

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

    X_num = df[available_numeric].copy().fillna(0)

    X_cat = pd.get_dummies(
        df[available_categorical].fillna("UNKNOWN"),
        prefix=available_categorical,
    ).astype(int)

    X = pd.concat([X_num, X_cat], axis=1)

    return X


def align_columns(train_x, test_x):
    test_x = test_x.reindex(columns=train_x.columns, fill_value=0)
    return test_x


def inverse_error_weights(predicted_errors):
    weights = {}

    for candidate, error in predicted_errors.items():
        error = max(float(error), 1.0)
        weights[candidate] = 1 / error

    total = sum(weights.values())

    if total == 0:
        return {k: 1 / len(weights) for k in weights}

    return {k: v / total for k, v in weights.items()}


def main():
    df = pd.read_csv(INPUT)
    df["date"] = pd.to_datetime(df["date"])

    required_cols = ["date", "total"] + CANDIDATES
    df = df.dropna(subset=[c for c in required_cols if c in df.columns]).copy()

    for candidate in CANDIDATES:
        df[f"{candidate}_abs_error"] = (df[candidate] - df["total"]).abs()

    split_date = df["date"].max() - pd.Timedelta(days=90)

    train = df[df["date"] <= split_date].copy()
    test = df[df["date"] > split_date].copy()

    X_train = prepare_features(train)
    X_test = prepare_features(test)
    X_test = align_columns(X_train, X_test)

    error_models = {}
    model_scores = {}

    for candidate in CANDIDATES:
        target_col = f"{candidate}_abs_error"

        y_train = train[target_col]

        models = {
            "RandomForest": RandomForestRegressor(
                n_estimators=500,
                random_state=42,
                n_jobs=-1,
                min_samples_leaf=5,
            ),
            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=500,
                random_state=42,
                n_jobs=-1,
                min_samples_leaf=5,
            ),
        }

        best_model = None
        best_name = None
        best_mae = float("inf")

        for model_name, model in models.items():
            model.fit(X_train, y_train)

            pred_error = model.predict(X_test)
            real_error = test[target_col]

            mae = mean_absolute_error(real_error, pred_error)

            if mae < best_mae:
                best_mae = mae
                best_model = model
                best_name = model_name

        error_models[candidate] = best_model
        model_scores[candidate] = {
            "best_model": best_name,
            "error_prediction_mae": round(float(best_mae), 2),
        }

    rows = []

    for idx, row in test.iterrows():
        x_row = X_test.loc[[idx]]

        predicted_errors = {}

        for candidate in CANDIDATES:
            predicted_errors[candidate] = float(
                error_models[candidate].predict(x_row)[0]
            )

        weights = inverse_error_weights(predicted_errors)

        forecast = 0
        weight_total = 0

        out = {
            "date": row["date"],
            "real_total": row["total"],
        }

        for candidate in CANDIDATES:
            candidate_pred = row[candidate]
            weight = weights[candidate]
            predicted_error = predicted_errors[candidate]

            forecast += candidate_pred * weight
            weight_total += weight

            out[candidate] = round(float(candidate_pred), 2)
            out[f"{candidate}_predicted_error"] = round(float(predicted_error), 2)
            out[f"{candidate}_weight"] = round(float(weight), 4)

        final_forecast = forecast / weight_total if weight_total else row["memory_short"]

        out["v5_forecast"] = round(float(final_forecast), 2)
        out["error"] = round(float(final_forecast - row["total"]), 2)
        out["abs_error"] = round(float(abs(final_forecast - row["total"])), 2)
        out["pct_error"] = round(float(abs(final_forecast - row["total"]) / row["total"] * 100), 2)

        out["best_weighted_candidate"] = max(weights, key=weights.get)

        rows.append(out)

    result = pd.DataFrame(rows)

    summary = {
        "engine": "error_prediction_engine_v5",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "mae": round(float(result["abs_error"].mean()), 2),
        "mape": round(float(result["pct_error"].mean()), 2),
        "total_forecast": round(float(result["v5_forecast"].sum()), 2),
        "total_real": round(float(result["real_total"].sum()), 2),
        "total_error": round(float(result["v5_forecast"].sum() - result["real_total"].sum()), 2),
        "total_error_pct": round(
            float(
                (result["v5_forecast"].sum() - result["real_total"].sum())
                / result["real_total"].sum()
                * 100
            ),
            2,
        ),
        "error_models": model_scores,
        "best_weighted_candidate_distribution": result["best_weighted_candidate"].value_counts().to_dict(),
    }

    result.to_csv(OUTPUT_FORECAST, index=False)

    with open(OUTPUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 90)
    print("ERROR PREDICTION ENGINE V5")
    print("=" * 90)

    print(json.dumps(summary, indent=2))

    print("\nULTIMI 20 GIORNI TEST")
    print(
        result[
            [
                "date",
                "real_total",
                "v5_forecast",
                "error",
                "pct_error",
                "best_weighted_candidate",
            ]
        ].tail(20).to_string(index=False)
    )

    print("\nSalvato:")
    print(OUTPUT_FORECAST)
    print(OUTPUT_SUMMARY)


if __name__ == "__main__":
    main()
