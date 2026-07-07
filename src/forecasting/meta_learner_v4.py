import os
import json
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

INPUT = "reports/adaptive_weight_engine/forecast_candidates_backtest.csv"

OUTPUT_DIR = "reports/meta_learner_v4"
OUTPUT_PROBA = f"{OUTPUT_DIR}/meta_learner_probabilities.csv"
OUTPUT_SUMMARY = f"{OUTPUT_DIR}/meta_learner_summary.json"

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


def compute_best_candidate(df):
    ape_cols = [f"{c}_ape" for c in CANDIDATES if f"{c}_ape" in df.columns]

    valid = df[ape_cols].notna().any(axis=1)
    df = df[valid].copy()

    df["best_candidate"] = (
        df[ape_cols]
        .idxmin(axis=1)
        .str.replace("_ape", "", regex=False)
    )

    df["best_ape"] = df[ape_cols].min(axis=1)

    return df


def prepare_features(df):
    available_numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    available_categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    X_num = df[available_numeric].copy()
    X_num = X_num.fillna(0)

    X_cat = pd.get_dummies(
        df[available_categorical].fillna("UNKNOWN"),
        prefix=available_categorical,
    ).astype(int)

    X = pd.concat([X_num, X_cat], axis=1)

    return X


def main():
    df = pd.read_csv(INPUT)
    df["date"] = pd.to_datetime(df["date"])

    df = compute_best_candidate(df)

    X = prepare_features(df)
    y = df["best_candidate"]

    split_date = df["date"].max() - pd.Timedelta(days=90)

    train_mask = df["date"] <= split_date
    test_mask = df["date"] > split_date

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    test_df = df[test_mask].copy()

    model = RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
    )

    model.fit(X_train, y_train)

    pred_class = model.predict(X_test)
    proba = model.predict_proba(X_test)
    classes = model.classes_

    accuracy = accuracy_score(y_test, pred_class)

    result = test_df[["date", "total"] + CANDIDATES].copy()
    result["real_best_candidate"] = y_test.values
    result["predicted_best_candidate"] = pred_class

    for i, cls in enumerate(classes):
        result[f"prob_{cls}"] = proba[:, i]

    # forecast finale = media pesata dalle probabilità
    ensemble = []

    for _, row in result.iterrows():
        weighted_sum = 0
        weight_total = 0

        for cls in classes:
            if cls not in CANDIDATES:
                continue

            prob = row[f"prob_{cls}"]
            pred = row[cls]

            if pd.isna(pred):
                continue

            weighted_sum += pred * prob
            weight_total += prob

        if weight_total == 0:
            ensemble.append(row["memory_short"])
        else:
            ensemble.append(weighted_sum / weight_total)

    result["meta_forecast"] = ensemble

    result["error"] = result["meta_forecast"] - result["total"]
    result["abs_error"] = result["error"].abs()
    result["pct_error"] = result["abs_error"] / result["total"] * 100

    metrics = {
        "engine": "meta_learner_v4",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy_best_candidate": round(float(accuracy), 4),
        "mae": round(float(result["abs_error"].mean()), 2),
        "mape": round(float(result["pct_error"].mean()), 2),
        "total_forecast": round(float(result["meta_forecast"].sum()), 2),
        "total_real": round(float(result["total"].sum()), 2),
        "total_error": round(float(result["meta_forecast"].sum() - result["total"].sum()), 2),
        "total_error_pct": round(
            float(
                (result["meta_forecast"].sum() - result["total"].sum())
                / result["total"].sum()
                * 100
            ),
            2,
        ),
        "classes": list(classes),
    }

    result.to_csv(OUTPUT_PROBA, index=False)

    with open(OUTPUT_SUMMARY, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 90)
    print("META LEARNER V4")
    print("=" * 90)

    print(json.dumps(metrics, indent=2))

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, pred_class, zero_division=0))

    print("\nULTIMI 20 GIORNI TEST")
    print(
        result[
            [
                "date",
                "total",
                "meta_forecast",
                "error",
                "pct_error",
                "real_best_candidate",
                "predicted_best_candidate",
            ]
        ].tail(20).to_string(index=False)
    )

    print("\nSalvato:")
    print(OUTPUT_PROBA)
    print(OUTPUT_SUMMARY)


if __name__ == "__main__":
    main()
