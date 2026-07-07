import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

INPUT_FILE = "data/processed/feature_store_meta.csv"

OUTPUT_DIR = "reports/feature_importance"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS = [
    "total",
    "delivery",
    "digital",
    "cash",
]

LEAKAGE_COLUMNS = [
    "total",
    "delivery",
    "digital",
    "cash",
    "pos",
    "ticket",
    "satispay",
    "uber",
    "just_eat",
    "glovo",
    "deliveroo",
    "delivery_share",
    "digital_share",
    "cash_share",
]

NON_NUMERIC_COLUMNS = [
    "date",
    "weekday",
    "commercial_season",
    "country",
    "region",
    "city",
    "holiday_name",
    "school_break_name",
]

df = pd.read_csv(INPUT_FILE)

features = [
    c for c in df.columns
    if c not in LEAKAGE_COLUMNS + NON_NUMERIC_COLUMNS
]

features = [
    c for c in features
    if pd.api.types.is_numeric_dtype(df[c])
]

print("=" * 80)
print("FEATURE IMPORTANCE LAB - META FEATURES - NO LEAKAGE")
print("=" * 80)
print("Input:", INPUT_FILE)
print("Numero feature usate:", len(features))

summary = []

for target in TARGETS:
    train = df.dropna(subset=[target]).copy()

    X = train[features].fillna(0)
    y = train[target]

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=3,
    )

    model.fit(X, y)

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    output_path = f"{OUTPUT_DIR}/{target}_importance_no_leakage.csv"
    importance.to_csv(output_path, index=False)

    print("\n" + "=" * 50)
    print(target.upper())
    print("=" * 50)
    print(importance.head(35).to_string(index=False))

    for _, row in importance.head(35).iterrows():
        summary.append({
            "target": target,
            "feature": row["feature"],
            "importance": row["importance"],
        })

summary = pd.DataFrame(summary)

pivot = summary.pivot(
    index="feature",
    columns="target",
    values="importance",
).fillna(0)

pivot["global_score"] = pivot.mean(axis=1)

pivot = pivot.sort_values("global_score", ascending=False)

pivot.to_csv(f"{OUTPUT_DIR}/global_feature_importance_no_leakage.csv")

print("\n" + "=" * 80)
print("GLOBAL FEATURE IMPORTANCE - META FEATURES - NO LEAKAGE")
print("=" * 80)
print(pivot.head(60).to_string())

print("\nSalvato in:", OUTPUT_DIR)
