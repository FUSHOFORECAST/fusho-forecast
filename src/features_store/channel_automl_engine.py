import json
import os
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURE_STORE = "data/processed/feature_store.csv"
SELECTED_FEATURES = "reports/selected_features.json"

OUTPUT_DIR = "reports/channel_automl"
OUTPUT_FILE = "reports/channel_automl/channel_automl_results.csv"
BEST_MODELS_FILE = "reports/channel_automl/best_channel_models.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS = ["delivery", "digital", "cash"]

TEST_DAYS = 30


def mape(y_true, y_pred):
    return (abs(y_true - y_pred) / y_true).mean() * 100


def bias_pct(y_true, y_pred):
    return ((y_pred.sum() - y_true.sum()) / y_true.sum()) * 100


def forecast_volatility(y_pred):
    return pd.Series(y_pred).std()


def score_model(mape_value, mae_value, bias_value, volatility_value):
    """
    Score più basso = modello migliore.
    Penalizza MAPE, MAE, bias e forecast troppo piatto.
    """
    return (
        mape_value * 0.50
        + abs(bias_value) * 0.30
        + (mae_value / 100) * 0.15
        - (volatility_value / 100) * 0.05
    )


def create_models():
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=3,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=3,
            random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=31,
            random_state=42,
        ),
    }


def main():
    df = pd.read_csv(FEATURE_STORE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    with open(SELECTED_FEATURES, "r") as f:
        selected = json.load(f)

    results = []
    best_models = {}

    print("=" * 90)
    print("CHANNEL AUTOML ENGINE")
    print("=" * 90)

    for target in TARGETS:
        print("\n" + "=" * 70)
        print(target.upper())
        print("=" * 70)

        features = selected["targets"][target]["features"]

        train = df.iloc[:-TEST_DAYS].copy()
        test = df.iloc[-TEST_DAYS:].copy()

        X_train = train[features].fillna(0)
        y_train = train[target]

        X_test = test[features].fillna(0)
        y_test = test[target]

        models = create_models()

        target_results = []

        for model_name, model in models.items():
            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, pred)
            mse = mean_squared_error(y_test, pred)
            rmse = mse ** 0.5
            mape_value = mape(y_test, pred)
            bias_value = bias_pct(y_test, pred)
            r2 = r2_score(y_test, pred)
            volatility = forecast_volatility(pred)

            score = score_model(
                mape_value=mape_value,
                mae_value=mae,
                bias_value=bias_value,
                volatility_value=volatility,
            )

            row = {
                "target": target,
                "model": model_name,
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "mape": round(mape_value, 2),
                "bias_pct": round(bias_value, 2),
                "r2": round(r2, 4),
                "forecast_volatility": round(volatility, 2),
                "score": round(score, 4),
            }

            results.append(row)
            target_results.append(row)

        target_df = pd.DataFrame(target_results).sort_values("score")

        print(target_df.to_string(index=False))

        best = target_df.iloc[0].to_dict()

        best_models[target] = {
            "best_model": best["model"],
            "score": best["score"],
            "mape": best["mape"],
            "mae": best["mae"],
            "bias_pct": best["bias_pct"],
            "features": features,
        }

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    with open(BEST_MODELS_FILE, "w") as f:
        json.dump(best_models, f, indent=2)

    print("\n" + "=" * 90)
    print("BEST CHANNEL MODELS")
    print("=" * 90)
    print(json.dumps(best_models, indent=2))

    print("\nSalvato:")
    print(OUTPUT_FILE)
    print(BEST_MODELS_FILE)


if __name__ == "__main__":
    main()
