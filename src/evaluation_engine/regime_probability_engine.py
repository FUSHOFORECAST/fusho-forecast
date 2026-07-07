import json
import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score

FEATURE_STORE = "data/processed/feature_store.csv"
REGIME_FILE = "data/processed/model_dataset_regime.csv"
FORECAST_FILE = "data/processed/forecast_adaptive_memory_v2.csv"

SELECTED_FEATURES_FILE = "reports/selected_features.json"

OUTPUT_DIR = "reports/regime_probability"
OUTPUT_FILE = f"{OUTPUT_DIR}/regime_probability_forecast.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_calendar_features(df):
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["dayofyear"] = df["date"].dt.dayofyear
    df["week_of_month"] = ((df["day"] - 1) // 7) + 1
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    return df


def main():
    feature_store = pd.read_csv(FEATURE_STORE)
    regimes = pd.read_csv(REGIME_FILE)
    forecast = pd.read_csv(FORECAST_FILE)

    feature_store["date"] = pd.to_datetime(feature_store["date"])
    regimes["date"] = pd.to_datetime(regimes["date"])
    forecast["date"] = pd.to_datetime(forecast["date"])

    regimes = regimes[["date", "adaptive_regime", "vs_baseline_pct"]].copy()

    df = feature_store.merge(regimes, on="date", how="inner")
    df = df.dropna(subset=["adaptive_regime"]).copy()

    with open(SELECTED_FEATURES_FILE, "r") as f:
        selected = json.load(f)

    # usiamo le feature del total forecast come base regime
    features = selected["targets"]["total"]["features"]

    available_features = [
        c for c in features
        if c in df.columns
    ]

    train = df.copy()

    X = train[available_features].fillna(0)
    y = train["adaptive_regime"]

    models = {
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=3,
            class_weight="balanced",
        ),
        "ExtraTreesClassifier": ExtraTreesClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
            class_weight="balanced",
        ),
    }

    print("=" * 90)
    print("REGIME PROBABILITY ENGINE V1")
    print("=" * 90)

    best_model = None
    best_name = None
    best_accuracy = -1

    # validazione semplice ultimi 60 giorni
    train_part = train.iloc[:-60].copy()
    test_part = train.iloc[-60:].copy()

    X_train = train_part[available_features].fillna(0)
    y_train = train_part["adaptive_regime"]

    X_test = test_part[available_features].fillna(0)
    y_test = test_part["adaptive_regime"]

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)

        print("\n" + name)
        print("Accuracy:", round(acc, 4))
        print(classification_report(y_test, pred, zero_division=0))

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_name = name

    print("\nBEST MODEL:", best_name)
    print("BEST ACCURACY:", round(best_accuracy, 4))

    # allena best model su tutto lo storico
    best_model.fit(X, y)

    forecast_features = forecast.copy()
    forecast_features = add_calendar_features(forecast_features)

    # copia le feature mancanti dal forecast V2 quando presenti,
    # altrimenti usa mediana storica
    for col in available_features:
        if col not in forecast_features.columns:
            forecast_features[col] = df[col].median()

        forecast_features[col] = forecast_features[col].fillna(df[col].median())

    X_future = forecast_features[available_features].fillna(0)

    proba = best_model.predict_proba(X_future)
    classes = best_model.classes_

    result = forecast[["date", "total_pred"]].copy()

    for i, cls in enumerate(classes):
        result[f"prob_{cls}"] = proba[:, i]

    result["predicted_regime"] = best_model.predict(X_future)

    # moltiplicatori medi reali per regime
    regime_multipliers = (
        df.groupby("adaptive_regime")["vs_baseline_pct"]
        .mean()
        .to_dict()
    )

    expected_delta = []

    for _, row in result.iterrows():
        delta = 0

        for cls in classes:
            delta += row[f"prob_{cls}"] * regime_multipliers.get(cls, 0)

        expected_delta.append(delta)

    result["expected_regime_delta_pct"] = expected_delta

    # correzione prudente: applichiamo solo il 30% del delta regime
    result["regime_corrected_pred"] = (
        result["total_pred"]
        * (1 + (result["expected_regime_delta_pct"] * 0.30) / 100)
    )

    result["total_pred"] = result["total_pred"].round(2)
    result["expected_regime_delta_pct"] = result["expected_regime_delta_pct"].round(2)
    result["regime_corrected_pred"] = result["regime_corrected_pred"].round(2)

    for col in result.columns:
        if col.startswith("prob_"):
            result[col] = (result[col] * 100).round(2)

    result.to_csv(OUTPUT_FILE, index=False)

    print("\nREGIME PROBABILITY FORECAST")
    print(result.to_string(index=False))

    print("\nTOTALE BASE:", round(result["total_pred"].sum(), 2))
    print("TOTALE CORRETTO:", round(result["regime_corrected_pred"].sum(), 2))
    print("\nSalvato:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
